from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc


def generate(
    method: str | None,
    n_samples: int,
    *,
    lower_bounds: ArrayLike | None = None,
    upper_bounds: ArrayLike | None = None,
    variable_values: Sequence[Sequence[object]] | None = None,
    seed: int | np.random.Generator | None = None,
    scramble: bool = True,
) -> NDArray[np.generic]:
    """Generate initial samples using random, LHS, or Sobol' sequences."""
    method = (method or "random").lower()

    if variable_values is not None:
        n_dim = len(variable_values)
    elif lower_bounds is not None and upper_bounds is not None:
        n_dim = len(np.asarray(lower_bounds))
    else:
        raise ValueError("Either variable_values or (lower_bounds, upper_bounds) must be provided.")

    if method == "lhs":
        sampler = qmc.LatinHypercube(d=n_dim, rng=seed)
        unit_samples = sampler.random(n=n_samples)
    elif method == "sobol":
        sampler = qmc.Sobol(d=n_dim, scramble=scramble, rng=seed)
        if (n_samples & (n_samples - 1) == 0):
            unit_samples = sampler.random_base2(m=n_samples.bit_length() - 1)
        else:
            warnings.warn(
                "Sobol' balance properties are guaranteed for n_samples = 2**m; "
                f"generating the requested n_samples={n_samples} with random(n).",
                UserWarning,
                stacklevel=2,
            )
            unit_samples = sampler.random(n=n_samples)
    else:
        # Default to uniform random
        rng = np.random.default_rng(seed)
        unit_samples = rng.random((n_samples, n_dim))

    if variable_values is not None:
        numeric_only = all(np.asarray(v).dtype.kind in "biufc" for v in variable_values)
        dtype = np.result_type(*(np.asarray(v).dtype for v in variable_values)) if numeric_only else object
        initial_data = np.empty(unit_samples.shape, dtype=dtype)
        for column, values in enumerate(variable_values):
            val_arr = np.asarray(values)
            indices = np.minimum((unit_samples[:, column] * val_arr.size).astype(int), val_arr.size - 1)
            initial_data[:, column] = val_arr[indices]

        m = len(variable_values[0])
        if n_samples % m != 0 and method != "random":
            warnings.warn(
                "初期データ数と分割数が合っていないため，完全な網羅性は保証されません．",
                UserWarning,
                stacklevel=2,
            )
        return initial_data
    else:
        assert lower_bounds is not None and upper_bounds is not None
        return qmc.scale(unit_samples, lower_bounds, upper_bounds)


__all__ = ["generate"]
