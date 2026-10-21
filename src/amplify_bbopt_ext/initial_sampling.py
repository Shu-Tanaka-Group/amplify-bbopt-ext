from __future__ import annotations

import warnings
from collections.abc import Sequence
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats import qmc

if TYPE_CHECKING:
    from amplify_bbopt import BlackBoxFuncBase

# Same convention as `amplify_bbopt.Variable.possible_values`:
# a tuple `(lower, upper)` for a continuous variable, a list of candidates for a discrete variable.
PossibleValues = Union[tuple[float, float], Sequence[float]]


class SampleGenerator(ABC):
    @abstractmethod
    def generate(self, n_samples: int) -> NDArray[np.generic]:
        pass

class QRS(SampleGenerator):
    """Generates samples with Quasi-random sequence.

    The underlying sampler is created once and kept, so successive calls of
    :meth:`generate` continue the sequence instead of repeating it.

    Points in the unit hypercube are mapped to each variable as follows:
    a continuous variable is scaled to ``[lower, upper]``, and a discrete
    variable takes the candidate ``floor(u * k)`` out of its ``k`` candidates,
    so that every candidate (one-hot bit) is assigned an equal share of ``[0, 1)``.

    Constraints among variables are not taken into account.

    Args:
        method: Sampling method, either ``"lhs"`` or ``"sobol"``.
        possible_values: Possible values of each variable in the same format as
            ``amplify_bbopt.Variable.possible_values``: a tuple ``(lower, upper)``
            for a continuous variable, or a list of candidates for a discrete one.
        seed: Seed or generator passed to the underlying sampler.
        scramble: Whether to scramble the Sobol' sequence.
    """

    METHODS = ("lhs", "sobol")

    def __init__(
        self,
        method: str,
        possible_values: Sequence[PossibleValues],
        *,
        seed: int | np.random.Generator | None = None,
        scramble: bool = True,
    ) -> None:
        self.method = method.lower()
        if self.method not in self.METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {self.METHODS}.")

        if len(possible_values) == 0:
            raise ValueError("possible_values must contain at least one variable.")
        self.possible_values = list(possible_values)
        self.n_dim = len(self.possible_values)

        # None for a continuous variable, candidate values for a discrete variable.
        self._candidates: list[NDArray[np.float64] | None] = []
        for index, values in enumerate(self.possible_values):
            if isinstance(values, tuple):
                if len(values) != 2 or values[0] > values[1]:
                    raise ValueError(f"Variable {index}: bounds must be a tuple (lower, upper) with lower <= upper.")
                self._candidates.append(None)
            else:
                candidates = np.asarray(values, dtype=np.float64)
                if candidates.size == 0:
                    raise ValueError(f"Variable {index}: candidates must not be empty.")
                self._candidates.append(candidates)

        self._sampler: qmc.QMCEngine
        if self.method == "lhs":
            self._sampler = qmc.LatinHypercube(d=self.n_dim, rng=seed)
        else:
            self._sampler = qmc.Sobol(d=self.n_dim, scramble=scramble, rng=seed)

    @classmethod
    def from_blackbox(
        cls,
        method: str,
        blackbox: BlackBoxFuncBase,
        *,
        seed: int | np.random.Generator | None = None,
        scramble: bool = True,
    ) -> QRS:
        """Create a sampler for the variables of an ``amplify_bbopt`` black-box function.

        Columns of the generated samples follow the order of
        ``blackbox.flattened_variables_dict``, which is the input format of
        ``amplify_bbopt.Optimizer``.
        """
        possible_values = [var.possible_values for var in blackbox.flattened_variables_dict.values()]
        return cls(method, possible_values, seed=seed, scramble=scramble)

    def generate(self, n_samples: int) -> NDArray[np.float64]:
        """Generate the next ``n_samples`` samples with the configured method."""
        if n_samples < 1:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}.")

        if self.method == "sobol":
            # A block of 2**m points starting at a multiple of 2**m is balanced.
            n_generated = self._sampler.num_generated
            if n_samples & (n_samples - 1) != 0 or n_generated % n_samples != 0:
                warnings.warn(
                    "Sobol' balance properties are guaranteed for n_samples = 2**m drawn after "
                    f"a multiple of 2**m points; got n_samples={n_samples} after {n_generated} points.",
                    UserWarning,
                    stacklevel=2,
                )
            with warnings.catch_warnings():
                # Superseded by the warning above.
                warnings.filterwarnings("ignore", message="The balance properties of Sobol' points")
                unit_samples = self._sampler.random(n=n_samples)
        else:
            unit_samples = self._sampler.random(n=n_samples)

        samples = np.empty(unit_samples.shape, dtype=np.float64)
        uncovered: list[int] = []
        for column, candidates in enumerate(self._candidates):
            unit_column = unit_samples[:, column]
            if candidates is None:
                lower, upper = self.possible_values[column]
                samples[:, column] = lower + (upper - lower) * unit_column
            else:
                n_candidates = candidates.size
                indices = np.minimum((unit_column * n_candidates).astype(int), n_candidates - 1)
                samples[:, column] = candidates[indices]
                if not _covers_all_candidates(n_samples, n_candidates):
                    uncovered.append(column)

        if uncovered:
            warnings.warn(
                f"n_samples={n_samples} does not guarantee that every candidate of variables "
                f"{uncovered} appears at least once.",
                UserWarning,
                stacklevel=2,
            )
        return samples


def _covers_all_candidates(n_samples: int, n_candidates: int) -> bool:
    """Whether one point per ``[i/n, (i+1)/n)`` always hits every ``[j/k, (j+1)/k)``.

    This holds iff each candidate interval fully contains some stratum, i.e.
    ``ceil(j*n/k) + 1 <= (j+1)*n/k`` for all ``j``.
    """
    n, k = n_samples, n_candidates
    return all((-(-j * n // k) + 1) * k <= (j + 1) * n for j in range(k))


__all__ = ["QRS", "PossibleValues", "SampleGenerator"]
