import math
from itertools import product
from logging import getLogger

import amplify
import numpy as np
from amplify import BaseClient
from amplify_bbopt import (
    BinaryVariable,
    DiscretizationSpec,
    EncodingMethod,
    Optimizer,
    RealVariable,
    blackbox,
)
from amplify_bbopt.amplify_model import AmplifyModel
from amplify_bbopt.blackbox import BlackBoxFuncBase
from amplify_bbopt.encoder import decode_values
from amplify_bbopt.optimizer import FlattenedSolution
from amplify_bbopt.trainer import Dataset, Trainer
from amplify_bbopt.bbopt_logging import AMPLIFY_BBOPT_LOGGER_NAME

from .training import BasicFMTrainer

logger = getLogger(AMPLIFY_BBOPT_LOGGER_NAME)

"""
ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
L以上M以下の値を取る実数変数n_iをN個用意する.
[n_1, n_2, ..., n_N]
これを１つのデータとし，D個のデータを縦に並べてデータセットを作る．
X_Real = [[n_1 , n_2 , ..., n_N ]
          [n_1', n_2', ..., n_N']
          [n_1", n_2", ..., n_N"]]
X_RealはD行N列行列となる．
ここで，変数n_iを整数・バイナリ変換する．n_iをLからMの範囲でd等分する．
L=-1，M=1，d=5とすると，
-1, -0.5, 0, 0.5, 1を取りうる．

n_iを，
One-hot表現にすると，一つの変数をd個のバイナリ変数に分解する．バイナリ変数の総数は Dd 個
Domain-wall表現では，(d-1)個のバイナリ変数に分解する．バイナリ変数の総数は D(d-1) 個
二進数表現では，ceil(log_2(d))個のバイナリ変数に分解する．バイナリ変数の総数は D×ceil(log_2(d)) 個
Xを整数，Oh,Dw,Biに変換するコードを記す．
ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー
各変数
D = データの数 : int, 

N = 実数変数の数 : int, 

L = 実数変数の最小値 : float, 
M = 実数変数の最大値 : float,
d = 実数変数の分割数 : int
ーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーーー

"""


def generate_initial_samples(
    N: int,
    L: float,
    M: float,
    q: int,
    n_samples: int,
    encoding: str,
    seed: int | None = None,
):
    """
    Generate initial training samples.

    Parameters
    ----------
    N : int
        Number of continuous variables.
    L, M : float
        Lower and upper bounds.
    q : int
        Number of discretization points.
    n_samples : int
        Number of initial samples.
    encoding : {"Oh", "Dw", "Bi", "OhDw", "DwOh"}
        Integer-binary encoding.
    seed : int, optional

    Returns
    -------
    X_real : ndarray, shape=(n_samples, N)
        Continuous variables.
    X_binary : ndarray
        Encoded binary variables.
    """

    rng = np.random.default_rng(seed)

    # q個の離散値
    grid = np.linspace(L, M, q)

    # 離散点からランダムサンプリング
    X_real = grid[rng.integers(0, q, size=(n_samples, N))]

    # エンコーディング
    X_binary = X_Real_to_Any(
        N,
        L,
        M,
        q,
        X_real,
        encoding,
    )

    return X_real, X_binary
    
#エンコーディング制約を満たしているか確認．
#制約を満たすときTrueを返す．
def Decode_const_check(EncodingMode, N, d, x):
    Value = 0
    if EncodingMode == "Oh" or EncodingMode == "OhDw":
        for n in range(N):
                const_n = 0
                for e in range(n*d, (n+1)*d):
                    const_n += x[e]
                Value += ((const_n-1)**2)             
    elif EncodingMode == "Dw" or EncodingMode == "DwOh":
        for n in range(N):
                for e in range(n*(d-1), (n+1)*(d-1)-1):
                    Value += x[e+1] * (1-x[e]) 
    if Value == 0:
        return True
    else:
        return False

    
# One-hotから整数へ変換
def X_Oh_to_Real(N, L, M, d, X_Oh): 
    if X_Oh.ndim == 1:
        X_Int = np.zeros(N) 
        if Decode_const_check("Oh", N, d, X_Oh) == False:
            raise ValueError(f"制約を満たしていないOhは整数に変換できません: {X_Oh}")
        for n in range(N):
            I = 0
            for i in range(0, d):
                I += i * X_Oh[n*d + i]
            X_Int[n] = I 
    elif X_Oh.ndim == 2:
        X_Int = np.zeros((X_Oh.shape[0], N))
        for n0 in range(X_Oh.shape[0]):
            if Decode_const_check("Oh", N, d, X_Oh[n0]) == False:
                raise ValueError(f"制約を満たしていないOhは整数に変換できません: {X_Oh[n0]}")
            for n in range(N):
                I = 0
                for i in range(0, d):
                    I += i * X_Oh[n0, n*d + i]
                X_Int[n0, n] = I
    else:
        raise ValueError("X_Ohは1次元または2次元の配列である必要があります。") 
    X_Real = X_Int*(M-L)/(d-1)+L
    return X_Real

#　Domain-wallから整数へ変換
def X_Dw_to_Real(N, L, M, d, X_Dw): 
    if X_Dw.ndim == 1:
        X_Int = np.zeros(N)
        if Decode_const_check("Dw", N, d, X_Dw) == False:
            raise ValueError(f"制約を満たしていないDwは整数に変換できません: {X_Dw}")
        for n in range(N):
            I = 0
            for i in range(n*(d-1), (n+1)*(d-1)):
                I += X_Dw[i]   
            X_Int[n] = I
    elif X_Dw.ndim == 2:
        X_Int = np.zeros((X_Dw.shape[0], N))
        for n0 in range((X_Dw.shape[0])):
            if Decode_const_check("Dw", N, d, X_Dw[n0]) == False:
                raise ValueError(f"制約を満たしていないDwは整数に変換できません: {X_Dw[n0]}")
            for n in range(N):
                I = 0
                for i in range(n*(d-1), (n+1)*(d-1)):
                    I += X_Dw[n0, i]
                X_Int[n0,n] = I
    else:
        raise ValueError("X_Dwは1次元または2次元の配列である必要があります。") 
    X_Real = X_Int*(M-L)/(d-1)+L
    return X_Real

#　二進数表現から整数へ変換
def X_Bi_to_Real(N, L, M, d, X_Bi): 
    binary_d = math.ceil(math.log2(d))
    if X_Bi.ndim == 1:
        X_Int = np.zeros(N)
        for n in range(N):
            I = 0
            for i in range(n*binary_d, (n+1)*binary_d):
                I += (2**(i-n*binary_d))*X_Bi[i]
            X_Int[n] = I
    elif X_Bi.ndim == 2:
        X_Int = np.zeros((X_Bi.shape[0], N))
        for n0 in range((X_Bi.shape[0])):
            for n in range(N):
                I = 0
                for i in range(n*binary_d, (n+1)*binary_d):
                    I += (2**(i-n*binary_d))*X_Bi[n0,i]
                X_Int[n0,n] = I
    else:
        raise ValueError("X_Biは1次元または2次元の配列である必要があります。") 
    X_Real = X_Int*(M-L)/(d-1)+L
    return X_Real

# 整数からOne-hot表現へ変換
def X_Real_to_Oh(N, L, M, d, X_Real):
    X_Int = np.round((X_Real - L) * (d - 1) / (M - L)).astype(int)
    if X_Int.ndim == 1:  # 1次元の場合
        X_Oh = np.zeros(N * d, dtype=int)  # One-hot表現を格納する配列を初期化
        for n in range(N):
            I = int(X_Int[n])  # 整数値を取得
            if I < 0 or I > d-1:
                raise ValueError(f"整数の値が範囲を超えています: {I}")
            X_Oh[n*d + I] = 1  # One-hot表現に変換
    elif X_Int.ndim == 2:  # 2次元の場合
        X_Oh = np.zeros((X_Int.shape[0], N * d), dtype=int)
        for n0 in range(X_Int.shape[0]):
            for n in range(N):
                I = X_Int[n0, n]  # 整数値を取得
                if I < 0 or I > d-1:
                    raise ValueError(f"整数の値が範囲を超えています: {I}")
                X_Oh[n0, n*d + I] = 1  # One-hot表現に変換
    else:
        raise ValueError("X_Realは1次元または2次元の配列である必要があります。") 
    return X_Oh

# 整数からDomain-wall表現へ変換
def X_Real_to_Dw(N, L, M, d, X_Real):
    X_Int = np.round((X_Real - L) * (d - 1) / (M - L)).astype(int)
    if X_Int.ndim == 1:  # 1次元の場合
        X_Dw = np.zeros(N * (d-1), dtype=int)  # Domain-wall表現を格納する配列を初期化
        for n in range(N):
            I = int(X_Int[n])  # 整数値を取得
            if I < 0 or I > d-1:
                raise ValueError(f"整数の値が範囲を超えています: {I}")
            for i in range(I):  # Domain-wall表現に変換
                X_Dw[n*(d-1) + i] = 1
    elif X_Int.ndim == 2:  # 2次元の場合
        X_Dw = np.zeros((X_Int.shape[0], N * (d-1)), dtype=int)
        for n0 in range(X_Int.shape[0]):
            for n in range(N):
                I = int(X_Int[n0, n])  # 整数値を取得
                if I < 0 or I > (d-1):
                    raise ValueError(f"整数の値が範囲を超えています: {I}")
                for i in range(I):  # Domain-wall表現に変換
                    X_Dw[n0, n*(d-1) + i] = 1
    else:
        raise ValueError("X_Intは1次元または2次元の配列である必要があります。") 
    return X_Dw

# 整数から二進数表現へ変換
def X_Real_to_Bi(N, L, M, d, X_Real):
    X_Int = np.round((X_Real - L) * (d - 1) / (M - L)).astype(int)
    binary_d = math.ceil(math.log2(d))
    if X_Int.ndim == 1:  # 1次元の場合
        X_Bi = np.zeros(N * binary_d, dtype=int)  # 二進数を格納する配列を初期化
        for n in range(N):
            # 範囲チェック：整数が0以上C以下かを確認
            if X_Int[n] < 0 or X_Int[n] > d-1:
                raise ValueError(f"整数の値が範囲外です: {X_Int[n]} (許可される範囲は0から{d-1})")
            # 整数を二進数に変換し、反転後にゼロパディング
            X_Bi[n * binary_d:(n+1) * binary_d] = [int(digit) for digit in bin(X_Int[n])[2:].zfill(binary_d)][::-1]
    elif X_Int.ndim == 2:  # 2次元の場合
        X_Bi = np.zeros((X_Int.shape[0], N * binary_d), dtype=int)
        for n0 in range(X_Int.shape[0]):
            for n in range(N):
                # 範囲チェック：整数が0以上C以下かを確認
                if X_Int[n0, n] < 0 or X_Int[n0, n] > d-1:
                    raise ValueError(f"整数の値が範囲外です: {X_Int[n0, n]} (許可される範囲は0から{d-1})")
                # 整数を二進数に変換し、反転後にゼロパディング
                X_Bi[n0, n * binary_d:(n+1) * binary_d] = [int(digit) for digit in bin(X_Int[n0, n])[2:].zfill(binary_d)][::-1]
    else:
        raise ValueError("X_Intは1次元または2次元の配列である必要があります。")    
    return X_Bi

def X_Oh_to_Dw(N, d, X_Oh):    
    if X_Oh.ndim == 1:
        X_Dw = np.zeros(N * (d-1), dtype=int)
        if Decode_const_check("Oh", N, d, X_Oh) == False:  # 制約チェック
            raise ValueError(f"制約を満たしていないOhデータが存在します: {X_Oh}")
        for n in range(N):
            I = int(np.argmax(X_Oh[n*d:(n+1)*d]))  # One-hotから整数に変換
            X_Dw[n*(d-1):(n+1)*(d-1)] = [1] * I + [0] * ((d-1) - I)
    elif X_Oh.ndim == 2:
        X_Dw = np.zeros((X_Oh.shape[0], N * (d-1)), dtype=int)
        for n0 in range(X_Oh.shape[0]):
            if Decode_const_check("Oh", N, d, X_Oh[n0]) == False:
                raise ValueError(f"制約を満たしていないOhデータが存在します: {X_Oh[n0]}")
            for n in range(N):
                I = int(np.argmax(X_Oh[n0, n*d:(n+1)*d]))
                X_Dw[n0, n*(d-1):(n+1)*(d-1)] = [1] * I + [0] * (d-1 - I)
    else:
        raise ValueError("X_Ohは1次元または2次元の配列である必要があります。")
    return X_Dw

def X_Oh_to_Bi(N, d, X_Oh):
    binary_d = math.ceil(math.log2(d))  # 必要なビット数を計算
    if X_Oh.ndim == 1:
        X_Bi = np.zeros(N * binary_d, dtype=int)
        if Decode_const_check("Oh", N, d, X_Oh) == False:  # 制約チェック
            raise ValueError(f"制約を満たしていないOhデータが存在します: {X_Oh}")
        for n in range(N):
            I = int(np.argmax(X_Oh[n*d:(n+1)*d]))  # One-hotから整数に変換
            X_Bi[n*binary_d:(n+1)*binary_d] = [int(b) for b in bin(I)[2:].zfill(binary_d)][::-1]
    elif X_Oh.ndim == 2:
        X_Bi = np.zeros((X_Oh.shape[0], N * binary_d), dtype=int)
        for n0 in range(X_Oh.shape[0]):
            if Decode_const_check("Oh", N, d, X_Oh[n0]) == False:
                raise ValueError(f"制約を満たしていないOhデータが存在します: {X_Oh[n0]}")
            for n in range(N):
                I = int(np.argmax(X_Oh[n0, n*d:(n+1)*d]))
                X_Bi[n0, n*binary_d:(n+1)*binary_d] = [int(b) for b in bin(I)[2:].zfill(binary_d)][::-1]
    else:
        raise ValueError("X_Ohは1次元または2次元の配列である必要があります。")
    return X_Bi

def X_Dw_to_Oh(N, d, X_Dw):
    if X_Dw.ndim == 1:  # 1次元の場合
        X_Oh = np.zeros(N * d, dtype=int)
        if Decode_const_check("Dw", N, d, X_Dw) == False:
            raise ValueError(f"制約を満たしていないDwは整数に変換できません: {X_Dw}")
        for n in range(N):
            y_nM1 = 1 
            for c in range(d-1):
                y_n = X_Dw[n * (d-1) + c]
                X_Oh[n * d + c] = y_nM1 + y_n - 2 * y_nM1 * y_n
                y_nM1 = y_n  
            X_Oh[(n + 1) * d - 1] = y_nM1
    elif X_Dw.ndim == 2:  # 2次元の場合
        X_Oh = np.zeros((X_Dw.shape[0], N * d), dtype=int)
        for n0 in range(X_Dw.shape[0]):
            if Decode_const_check("Dw", N, d, X_Dw[n0]) == False:
                raise ValueError(f"制約を満たしていないDwは整数に変換できません: {X_Dw[n0]}")
            for n in range(N):
                y_nM1 = 1 
                for c in range(d-1):
                    y_n = X_Dw[n0, n * (d-1) + c]
                    X_Oh[n0, n * d + c] = y_nM1 + y_n - 2 * y_nM1 * y_n
                    y_nM1 = y_n  
                X_Oh[n0, (n + 1) * d - 1] = y_nM1
    else:
        raise ValueError("X_Dwは1次元または2次元の配列である必要があります。")
    return X_Oh

def X_Dw_to_Bi(N, d, X_Dw): 
    binary_d = math.ceil(math.log2(d))  # C+1を表すために必要なビット数を計算
    if X_Dw.ndim == 1:  # 1次元の場合
        X_Bi = np.zeros(N * binary_d, dtype=int)
        if Decode_const_check("Dw", N, d, X_Dw) == False:
            raise ValueError(f"制約を満たしていないDwは整数に変換できません: {X_Dw}")
        for n in range(N):
            I = int(sum(X_Dw[n * (d-1):(n + 1) * (d-1)]))  # Domain-wall表現を整数に変換
            bin_str = bin(I)[2:].zfill(binary_d)
            X_Bi[n * binary_d:(n + 1) * binary_d] = [int(bit) for bit in bin_str[::-1]]
    elif X_Dw.ndim == 2:  # 2次元の場合
        X_Bi = np.zeros((X_Dw.shape[0], N * binary_d), dtype=int)
        for n0 in range(X_Dw.shape[0]):
            if Decode_const_check("Dw", N, d, X_Dw[n0]) == False:
                raise ValueError(f"制約を満たしていないDwは整数に変換できません: {X_Dw[n0]}")
            for n in range(N):
                I = int(sum(X_Dw[n0, n * (d-1):(n + 1) * (d-1)]))
                bin_str = bin(I)[2:].zfill(binary_d)
                X_Bi[n0, n * binary_d:(n + 1) * binary_d] = [int(bit) for bit in bin_str[::-1]]
    else:
        raise ValueError("X_Dwは1次元または2次元の配列である必要があります。")
    return X_Bi

def X_Bi_to_Oh(N, d, X_Bi):
    binary_d = math.ceil(math.log2(d))  # C+1を表すために必要なビット数を計算
    if X_Bi.ndim == 1:  # 1次元の場合
        X_Oh = np.zeros(N * (d), dtype=int)  # One-hot表現の配列を初期化
        for n in range(N):
            I = 0
            # バイナリから整数に変換
            for i in range(n * binary_d, (n + 1) * binary_d):
                I += (2 ** (i - n * binary_d)) * X_Bi[i]
            # 整数をOne-hot表現に変換
            X_Oh[n * (d) + int(I)] = 1  # 該当する位置に1をセット
            
    elif X_Bi.ndim == 2:  # 2次元の場合
        X_Oh = np.zeros((X_Bi.shape[0], N * (d)), dtype=int)
        for n0 in range(X_Bi.shape[0]):
            for n in range(N):
                I = 0
                # バイナリから整数に変換
                for i in range(n * binary_d, (n + 1) * binary_d):
                    I += (2 ** (i - n * binary_d)) * X_Bi[n0, i]
                # 整数をOne-hot表現に変換
                X_Oh[n0, n * (d) + int(I)] = 1  # 該当する位置に1をセット
    else:
        raise ValueError("X_Biは1次元または2次元の配列である必要があります。")
    return X_Oh

def X_Bi_to_Dw(N, d, X_Bi):
    binary_d = math.ceil(math.log2(d))  # C+1を表すために必要なビット数を計算   
    if X_Bi.ndim == 1:  # 1次元の場合
        X_Dw = np.zeros(N * (d-1), dtype=int)  # Domain-wall表現の配列を初期化
        for n in range(N):
            I = 0
            # バイナリから整数へ変換
            for i in range(n*binary_d, (n+1)*binary_d):
                I += (2**(i-n*binary_d)) * X_Bi[i]
            # 整数をDomain-wall表現に変換
            X_Dw[n*(d-1):(n+1)*(d-1)] = [1] * int(I) + [0] * ((d-1) - int(I))
    elif X_Bi.ndim == 2:  # 2次元の場合
        X_Dw = np.zeros((X_Bi.shape[0], N * (d-1)), dtype=int)
        for n0 in range(X_Bi.shape[0]):
            for n in range(N):
                I = 0
                # バイナリから整数へ変換
                for i in range(n*binary_d, (n+1)*binary_d):
                    I += (2**(i-n*binary_d)) * X_Bi[n0, i]
                # 整数をDomain-wall表現に変換
                X_Dw[n0, n*(d-1):(n+1)*(d-1)] = [1] * int(I) + [0] * (d-1 - int(I))
    else:
        raise ValueError("X_Biは1次元または2次元の配列である必要があります。")  # エラー処理
    return X_Dw

def X_Real_to_Any(N, L, M, d, X_Real, EncodingMode):
    if EncodingMode=="Oh" or EncodingMode=="OhDw":
        X_Any = X_Real_to_Oh(N, L, M, d, X_Real)
    elif EncodingMode=="Dw" or EncodingMode=="DwOh":
        X_Any = X_Real_to_Dw(N, L, M, d, X_Real)
    elif EncodingMode=="Bi":
        X_Any = X_Real_to_Bi(N, L, M, d, X_Real)
    elif EncodingMode=="Real":
        X_Any = X_Real
    else:
        raise ValueError(f"未対応のエンコーディング方式です: {EncodingMode}")
    return X_Any

def X_Any_to_Real(N, L, M, d, X_Any, EncodingMode): 
    if EncodingMode=="Oh" or EncodingMode=="OhDw":
        X_Real = X_Oh_to_Real(N, L, M, d, X_Any)
    elif EncodingMode=="Dw" or EncodingMode=="DwOh":
        X_Real = X_Dw_to_Real(N, L, M, d, X_Any)
    elif EncodingMode=="Bi":
        X_Real = X_Bi_to_Real(N, L, M, d, X_Any)
    elif EncodingMode=="Real":
        X_Real = X_Any
    else:
        raise ValueError(f"未対応のエンコーディング方式です: {EncodingMode}")
    return X_Real

# 上三角行列のQUBO行列をone-hot表現からdomain-wall表現へ変換
def QUBO_Oh_to_Dw(N, d, QUBO_Oh):       
    QUBO_Dw = np.zeros((N*(d-1),N*(d-1)))
    Q_0 = 0
    for l in range(N):
        for m in range(N):
            Q_0 += QUBO_Oh[d*l,d*m]
            for i in range(d-1):
                for j in range(d-1):
                    if l==m and i==j:                            
                        for n_num in range(N):
                            if d*l+i<d*n_num+i:
                                _i = d*l+i
                                _j = d*n_num
                                
                                _i2 = d*l+i+1
                                _j2 = d*n_num
                            elif d*l+i==d*n_num+i:
                                _i = d*l+i
                                _j = d*n_num+i
                                
                                _i2 = d*l+i+1
                                _j2 = d*n_num+i+1
                            else:
                                _j = d*l+i
                                _i = d*n_num
                                _j2 = d*l+i+1
                                _i2 = d*n_num
                            QUBO_Dw[(d-1)*l+i][(d-1)*m+j] += -QUBO_Oh[_i][_j] + QUBO_Oh[_i2][_j2]     
                    elif l==m and i!=j:
                        QUBO_Dw[(d-1)*l+i][(d-1)*m+j] = 0
                    elif l!=m:
                        QUBO_Dw[(d-1)*l+i][(d-1)*m+j] = QUBO_Oh[d*l+i][d*m+j]-QUBO_Oh[d*l+i][d*m+j+1]-QUBO_Oh[d*l+i+1][d*m+j]+QUBO_Oh[d*l+i+1][d*m+j+1]                            
    return QUBO_Dw, Q_0

# 上三角行列のQUBO行列をdomain-wall表現からone-hot表現へ変換
def QUBO_Dw_to_Oh(N, d, QUBO_Dw):
    QUBO_Oh = np.zeros((N*d, N*d))
    for l in range(N):
        for m in range(N):
            for i in range(d):
                for j in range(d):
                    if l == m:
                        if i != j: 
                            QUBO_Oh[d*l+i][d*m+j] = 0
                            continue
                        else:
                            for a in range(i):
                                for b in range(a, i):
                                    QUBO_Oh[d*l+i][d*m+j] += QUBO_Dw[(d-1)*l+a][(d-1)*l+b]
                    else:
                        if i == 0 or j == 0:
                            QUBO_Oh[d*l+i][d*m+j] = 0
                            continue
                        else:
                            QUBO_Oh[d*l+i][d*m+j] = \
                                  QUBO_Oh[d*l+i][d*m+j-1] \
                                + QUBO_Oh[d*l+i-1][d*m+j] \
                                - QUBO_Oh[d*l+i-1][d*m+j-1] \
                                + QUBO_Dw[(d-1)*l+i-1][(d-1)*m+j-1]
    return QUBO_Oh


def _verification_states(N: int, q: int, max_states: int = 256) -> np.ndarray:
    """QUBO変換の検証に用いる有効な離散状態を決定的に生成する。"""
    num_states = q**N
    if num_states <= max_states:
        return np.asarray(list(product(range(q), repeat=N)), dtype=int)

    rng = np.random.default_rng(0)
    states = rng.integers(0, q, size=(max_states, N))
    states[0] = 0
    states[1] = q - 1
    return states


def _verify_qubo_conversion(
    N: int,
    q: int,
    source_qubo: np.ndarray,
    target_qubo: np.ndarray,
    source_mode: str,
    constant: float = 0.0,
) -> tuple[int, float]:
    """有効状態上で変換前後のQUBOエネルギーが一致することを確認する。"""
    states = _verification_states(N, q)
    one_hot = np.zeros((len(states), N * q), dtype=int)
    domain_wall = np.zeros((len(states), N * (q - 1)), dtype=int)
    for row, state in enumerate(states):
        for n, level in enumerate(state):
            one_hot[row, n * q + level] = 1
            domain_wall[row, n * (q - 1) : n * (q - 1) + level] = 1

    if source_mode == "Oh":
        source_values, target_values = one_hot, domain_wall
    elif source_mode == "Dw":
        source_values, target_values = domain_wall, one_hot
    else:
        raise ValueError(f"未対応の変換元エンコーディングです: {source_mode}")

    source_energy = np.einsum(
        "bi,ij,bj->b", source_values, source_qubo, source_values
    )
    target_energy = np.einsum(
        "bi,ij,bj->b", target_values, target_qubo, target_values
    ) + constant
    max_error = float(np.max(np.abs(source_energy - target_energy)))
    if not np.allclose(source_energy, target_energy, rtol=1e-9, atol=1e-10):
        logger.error(
            "QUBO conversion verification failed: %s -> %s, "
            "states=%d, max_error=%.3e",
            source_mode,
            "Dw" if source_mode == "Oh" else "Oh",
            len(states),
            max_error,
        )
        raise RuntimeError("変換前後のQUBOエネルギーが一致しません。")

    logger.info(
        "QUBO conversion verification passed: %s -> %s, states=%d",
        source_mode,
        "Dw" if source_mode == "Oh" else "Oh",
        len(states),
    )
    return len(states), max_error


def _fm_qubo(trainer: BasicFMTrainer) -> np.ndarray:
    """学習済みFMのパラメータから上三角形式のQUBO行列を作成する。"""
    model = trainer.model
    if model is None:
        raise ValueError("FMがまだ学習されていません。")

    v = model.quadratic.detach().cpu().numpy()
    w = model.linear.detach().cpu().numpy()
    qubo = np.zeros((len(w), len(w)), dtype=float)
    np.fill_diagonal(qubo, w)

    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            qubo[i, j] = np.dot(v[i], v[j])

    scale = np.max(np.abs(qubo))
    return qubo / scale if scale > 0 else qubo


def _one_hot_penalty(N: int, q: int, weight: float) -> np.ndarray:
    """各One-hotブロックに対する制約ペナルティをQUBO行列として返す。"""
    penalty = np.zeros((N * q, N * q), dtype=float)
    for n in range(N):
        start = n * q
        stop = (n + 1) * q
        for i in range(start, stop):
            penalty[i, i] -= weight
            for j in range(i + 1, stop):
                penalty[i, j] += 2 * weight
    return penalty


def _domain_wall_penalty(N: int, q: int, weight: float) -> np.ndarray:
    """各Domain-wallブロック内の0から1への不正な遷移にペナルティを与える。"""
    block_size = q - 1
    penalty = np.zeros((N * block_size, N * block_size), dtype=float)
    for n in range(N):
        start = n * block_size
        stop = (n + 1) * block_size
        for i in range(start + 1, stop):
            penalty[i, i] += 2 * weight
        for i in range(start, stop - 1):
            penalty[i, i + 1] -= 2 * weight
    return penalty


def _complement_qubo(qubo: np.ndarray) -> tuple[np.ndarray, float]:
    """上三角QUBO y^T Q y を，y = 1 - z の置換で z^T Q' z + 定数 に書き換える。"""
    off_diagonal = np.triu(qubo, 1)
    linear = np.diag(qubo) + off_diagonal.sum(axis=0) + off_diagonal.sum(axis=1)
    return off_diagonal - np.diag(linear), float(qubo.sum())


_SWITCHABLE_ENCODINGS = (EncodingMethod.OneHot, EncodingMethod.DomainWall)


def _block_structure(blackbox: BlackBoxFuncBase) -> tuple[EncodingMethod, int, int]:
    """QUBO変換が前提とする (学習時エンコーディング, 変数の数, 分割数) を返す。"""
    encodings = [var.encoding for var in blackbox.flattened_variables_dict.values()]
    methods = {getattr(encoding, "method", None) for encoding in encodings}
    num_bins = {getattr(encoding, "num_bins", None) for encoding in encodings}
    if len(methods) != 1 or not methods <= set(_SWITCHABLE_ENCODINGS):
        raise ValueError(
            "All variables must share the OneHot or DomainWall encoding"
        )
    if len(num_bins) != 1:
        raise ValueError("All variables must have the same number of bins")
    return methods.pop(), len(encodings), num_bins.pop()  # pyright: ignore[reportReturnType]


class EncodingSwitchOptimizer(Optimizer):
    """FMの学習時とアニーリング時でエンコーディングを切り替える最適化器。

    学習時のエンコーディングと分割数はブラックボックス関数の変数定義から決まる。
    """

    def __init__(
        self,
        blackbox: BlackBoxFuncBase,
        trainer,
        client,
        *,
        search_encoding: EncodingMethod,
        encoding_penalty: float = 1000.0,
        training_data: Dataset | None = None,
        constraints: amplify.Constraint | amplify.ConstraintList | None = None,
        seed: int = 0,
    ) -> None:
        if search_encoding not in _SWITCHABLE_ENCODINGS:
            raise ValueError("search_encoding must be OneHot or DomainWall")
        if encoding_penalty <= 0:
            raise ValueError("encoding_penalty must be positive")
        training_encoding, N, q = _block_structure(blackbox)
        # 探索用モデルは元の変数を持たないため，ユーザー定義の制約を移せない
        if search_encoding != training_encoding and constraints is not None:
            raise ValueError("Constraints are not supported when switching encodings")
        super().__init__(
            blackbox,
            trainer,
            client,
            training_data=training_data,
            constraints=constraints,
            seed=seed,
        )
        self._N = N
        self._q = q
        self._training_encoding = training_encoding
        self._search_encoding = search_encoding
        self._encoding_penalty = encoding_penalty

    def minimize_surrogate(
        self, max_retries: int = 10
    ) -> tuple[list[FlattenedSolution], amplify.Result]:
        if self._search_encoding == self._training_encoding:
            return super().minimize_surrogate(max_retries)

        trainer = self.surrogate_model.trainer
        if not isinstance(trainer, BasicFMTrainer):
            raise TypeError(
                "エンコーディングを切り替えるにはBasicFMTrainerが必要です。"
            )

        training_qubo = _fm_qubo(trainer)
        if self._training_encoding == EncodingMethod.OneHot:
            logger.info(
                "Converting QUBO encoding: Oh -> Dw (N=%d, q=%d)",
                self._N,
                self._q,
            )
            converted_qubo, constant = QUBO_Oh_to_Dw(
                self._N, self._q, training_qubo
            )
            logger.debug("QUBO before conversion (Oh):\n%s", training_qubo)
            logger.debug("QUBO after conversion (Dw):\n%s", converted_qubo)
            _verify_qubo_conversion(
                self._N,
                self._q,
                training_qubo,
                converted_qubo,
                "Oh",
                constant,
            )
            search_qubo = converted_qubo.copy()
            search_qubo += _domain_wall_penalty(
                self._N, self._q, self._encoding_penalty
            )
            search_mode = "Dw"
        else:
            logger.info(
                "Converting QUBO encoding: Dw -> Oh (N=%d, q=%d)",
                self._N,
                self._q,
            )
            # amplify_bbopt の Domain-wall は本モジュールの規約のビット反転なので，
            # 学習済みQUBOを本モジュールの規約に揃えてから変換する（定数項は探索に不要）
            training_qubo, _ = _complement_qubo(training_qubo)
            converted_qubo = QUBO_Dw_to_Oh(
                self._N, self._q, training_qubo
            )
            logger.debug("QUBO before conversion (Dw):\n%s", training_qubo)
            logger.debug("QUBO after conversion (Oh):\n%s", converted_qubo)
            _verify_qubo_conversion(
                self._N,
                self._q,
                training_qubo,
                converted_qubo,
                "Dw",
            )
            search_qubo = converted_qubo.copy()
            search_qubo += _one_hot_penalty(
                self._N, self._q, self._encoding_penalty
            )
            search_mode = "Oh"

        generator = amplify.VariableGenerator()
        search_variables = generator.array("Binary", search_qubo.shape[0])
        search_model = AmplifyModel(search_variables, self.amplify_model.client)
        search_model.objective = search_variables @ search_qubo @ search_variables  # pyright: ignore[reportOperatorIssue, reportAttributeAccessIssue]
        search_solutions, amplify_result = search_model.solve(
            constraint_weight=1.0, max_retries=max_retries
        )

        solutions: list[FlattenedSolution] = []
        for search_solution in search_solutions:
            if not Decode_const_check(
                search_mode, self._N, self._q, search_solution
            ):
                continue

            if self._training_encoding == EncodingMethod.OneHot:
                surrogate_solution = X_Dw_to_Oh(
                    self._N, self._q, search_solution
                )
            else:
                # 学習データと同じ amplify_bbopt の Domain-wall 規約に戻す
                surrogate_solution = 1 - X_Oh_to_Dw(
                    self._N, self._q, search_solution
                )

            blackbox_solution = decode_values(
                self.encoding_info, surrogate_solution
            )
            solutions.append(
                FlattenedSolution(
                    np.asarray(blackbox_solution, dtype=float),
                    np.asarray(surrogate_solution, dtype=float),
                    np.asarray(search_solution, dtype=float),
                )
            )

        return solutions, amplify_result


def encoding_switch_optimizer(
    blackbox: BlackBoxFuncBase,
    trainer: Trainer,
    client: BaseClient,
    dataset: Dataset,
    *,
    search_encoding: EncodingMethod,
    encoding_penalty: float = 1000.0,
) -> EncodingSwitchOptimizer:
    return EncodingSwitchOptimizer(
        blackbox,
        trainer,
        client,
        search_encoding=search_encoding,
        encoding_penalty=encoding_penalty,
        training_data=dataset,
    )


def run(
    client,
    bb_func,
    N: int,
    q: int,
    L: float,
    M: float,
    encoding: str,
    k: int,
    n_iter: int,
    initial_data: np.ndarray,
    epochs: int,
    optimizer_params: dict,
    lr_scheduler_class=None,
    lr_scheduler_params: dict | None = None,
    *,
    seed: int = 0,
    encoding_penalty: float = 1000.0,
    max_deduplication_trials: int | None = 100,
):
    """指定されたエンコーディング方式でFMQAを実行する。"""

    valid_encodings = {"Oh", "Dw", "Bi", "OhDw", "DwOh"}
    if encoding not in valid_encodings:
        raise ValueError(
            f"encodingには{sorted(valid_encodings)}のいずれかを指定してください。"
            f"指定された値: {encoding!r}"
        )
    if N <= 0 or q <= 1:
        raise ValueError("Nには1以上、qには2以上の整数を指定してください。")
    if L >= M:
        raise ValueError("探索範囲の下限Lは上限Mより小さくしてください。")
    if encoding_penalty <= 0:
        raise ValueError("encoding_penaltyには正の値を指定してください。")
    if max_deduplication_trials is not None and max_deduplication_trials < 0:
        raise ValueError(
            "max_deduplication_trialsには0以上の整数またはNoneを"
            "指定してください。"
        )

    data = np.asarray(initial_data, dtype=float)
    if data.ndim != 2:
        raise ValueError("initial_dataは2次元配列にしてください。")

    if encoding == "Bi":
        expected_size = N * math.ceil(math.log2(q))
        if data.shape[1] == N:
            data = X_Real_to_Bi(N, L, M, q, data)
        elif data.shape[1] != expected_size:
            raise ValueError(
                f"Biのinitial_dataの列数は{N}または{expected_size}にしてください。"
            )

        @blackbox
        def optimization_blackbox(  # pyright: ignore[reportRedeclaration]
            x: list[int] = [BinaryVariable() for _ in range(expected_size)],
        ):
            real_x = X_Bi_to_Real(N, L, M, q, np.asarray(x))
            return float(bb_func(real_x))

        dataset_x = data
    else:
        training_mode = "Oh" if encoding in {"Oh", "OhDw"} else "Dw"
        expected_size = N * q if training_mode == "Oh" else N * (q - 1)
        if data.shape[1] == expected_size:
            dataset_x = X_Any_to_Real(N, L, M, q, data, training_mode)
        elif data.shape[1] == N:
            dataset_x = data
        else:
            raise ValueError(
                f"{training_mode}のinitial_dataの列数は{N}または"
                f"{expected_size}にしてください。"
            )

        variable_encoding = (
            EncodingMethod.OneHot
            if training_mode == "Oh"
            else EncodingMethod.DomainWall
        )

        @blackbox
        def optimization_blackbox(
            x: list[float] = [
                RealVariable(
                    bounds=(L, M),
                    encoding=variable_encoding,
                    discretization_spec=DiscretizationSpec(num_bins=q),
                )
                for _ in range(N)
            ],
        ):
            return float(bb_func(np.asarray(x, dtype=float)))

    dataset_y = np.asarray(
        [optimization_blackbox(x) for x in dataset_x], dtype=float
    )
    trainer = BasicFMTrainer()
    trainer.epochs = epochs
    trainer.optimizer_params = optimizer_params
    trainer.lr_scheduler_class = lr_scheduler_class
    trainer.lr_scheduler_params = lr_scheduler_params or {}
    trainer.num_factors = k

    if encoding == "Bi":
        optimizer = Optimizer(
            blackbox=optimization_blackbox,
            trainer=trainer,
            client=client,
            training_data=Dataset(dataset_x, dataset_y),
            seed=seed,
        )
    else:
        optimizer = EncodingSwitchOptimizer(
            blackbox=optimization_blackbox,
            trainer=trainer,
            client=client,
            training_data=Dataset(dataset_x, dataset_y),
            seed=seed,
            search_encoding=(
                EncodingMethod.DomainWall
                if encoding in {"Dw", "OhDw"}
                else EncodingMethod.OneHot
            ),
            encoding_penalty=encoding_penalty,
        )
    optimizer.optimize(
        n_iter,
        max_deduplication_trials=max_deduplication_trials,
    )
    return optimizer
