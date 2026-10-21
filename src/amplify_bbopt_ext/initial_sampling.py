from __future__ import annotations

import warnings
from collections.abc import Sequence
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import qmc


class SampleGenerator(ABC):
    @abstractmethod
    def generate(self, n_samples: int) -> NDArray[np.generic]:
        pass

class QRS(SampleGenerator):
    """Generates samples with Quasi-random sequence.

    The underlying sampler is created once and kept, so successive calls of
    :meth:`generate` continue the sequence instead of repeating it.

    Args:
        method: Sampling method, either ``"lhs"`` or ``"sobol"``.
        lower_bounds: Lower bounds of continuous variables.
        upper_bounds: Upper bounds of continuous variables.
        variable_values: Candidate values of each discrete variable. Takes
            precedence over ``lower_bounds`` and ``upper_bounds``.
        seed: Seed or generator passed to the underlying sampler.
        scramble: Whether to scramble the Sobol' sequence.
    """

    METHODS = ("lhs", "sobol")

    def __init__(
        self,
        method: str,
        *,
        lower_bounds: ArrayLike | None = None,
        upper_bounds: ArrayLike | None = None,
        variable_values: Sequence[Sequence[object]] | None = None,
        seed: int | np.random.Generator | None = None,
        scramble: bool = True,
    ) -> None:
        self.method = method.lower()
        if self.method not in self.METHODS:
            raise ValueError(f"Unknown method {method!r}; expected one of {self.METHODS}.")

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.variable_values = variable_values

        if variable_values is not None:
            self.n_dim = len(variable_values)
        elif lower_bounds is not None and upper_bounds is not None:
            self.n_dim = len(np.asarray(lower_bounds))
        else:
            raise ValueError("Either variable_values or (lower_bounds, upper_bounds) must be provided.")

        if variable_values is not None:
            self._value_arrays = [np.asarray(v) for v in variable_values]
            if any(v.size == 0 for v in self._value_arrays):
                raise ValueError("Each entry of variable_values must have at least one candidate.")
            numeric_only = all(v.dtype.kind in "biufc" for v in self._value_arrays)
            self._dtype = np.result_type(*(v.dtype for v in self._value_arrays)) if numeric_only else object

        self._sampler: qmc.QMCEngine
        if self.method == "lhs":
            self._sampler = qmc.LatinHypercube(d=self.n_dim, rng=seed)
        else:
            self._sampler = qmc.Sobol(d=self.n_dim, scramble=scramble, rng=seed)

    def generate(self, n_samples: int) -> NDArray[np.generic]:
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

        if self.variable_values is not None:
            initial_data = np.empty(unit_samples.shape, dtype=self._dtype)
            for column, val_arr in enumerate(self._value_arrays):
                indices = np.minimum((unit_samples[:, column] * val_arr.size).astype(int), val_arr.size - 1)
                initial_data[:, column] = val_arr[indices]

            uncovered = [column for column, val_arr in enumerate(self._value_arrays) if n_samples % val_arr.size != 0]
            if uncovered:
                warnings.warn(
                    f"n_samples={n_samples} is not a multiple of the number of candidates of "
                    f"variables {uncovered}; full coverage of candidates is not guaranteed.",
                    UserWarning,
                    stacklevel=2,
                )
            return initial_data
        else:
            assert self.lower_bounds is not None and self.upper_bounds is not None
            return qmc.scale(unit_samples, self.lower_bounds, self.upper_bounds)


__all__ = ["QRS", "SampleGenerator"]
