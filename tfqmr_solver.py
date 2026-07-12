#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Matrix-free TFQMR solver used by PyGSC."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Optional, TextIO

import numpy as np
from scipy.sparse.linalg import LinearOperator, tfqmr

Array = np.ndarray
MatVec = Callable[[Array], Array]


@dataclass(frozen=True)
class TFQMRResult:
    solution: Array
    info: int
    iterations: int
    final_residual_norm: float

    @property
    def converged(self) -> bool:
        return self.info == 0


def _tolerance_kwargs(atol: float, rtol: float) -> dict:
    params = inspect.signature(tfqmr).parameters
    kwargs = {}

    # SciPy 1.8.1 uses tol; newer versions use rtol.
    if "rtol" in params:
        kwargs["rtol"] = rtol
    elif "tol" in params:
        kwargs["tol"] = rtol
    if "atol" in params:
        kwargs["atol"] = atol
    if "show" in params:
        kwargs["show"] = False

    return kwargs


def solve_tfqmr(
    matvec: MatVec,
    rhs: Array,
    size: int,
    maxiter: int = 500,
    atol: float = 1.0e-3,
    rtol: float = 0.0,
    iter_chk: int = 4,
    log_f: Optional[TextIO] = None,
    label: str = "TFQMR",
) -> TFQMRResult:
    rhs = np.asarray(rhs, dtype=np.float64).reshape(size)
    x0 = np.zeros_like(rhs)
    niter = 0

    def wrapped_matvec(x: Array) -> Array:
        y = matvec(np.asarray(x, dtype=np.float64).reshape(size))
        return np.asarray(y, dtype=np.float64).reshape(size)

    Aop = LinearOperator(
        shape=(size, size),
        matvec=wrapped_matvec,
        dtype=np.float64,
    )

    def callback(xk: Array) -> None:
        nonlocal niter
        niter += 1
        if log_f is not None and iter_chk > 0 and niter % iter_chk == 0:
            norm_watch = np.linalg.norm(wrapped_matvec(xk) - rhs)
            log_f.write(
                "{}/scipy_tfqmr, iter {} norm {:13.6e}\n".format(
                    label, niter, norm_watch
                )
            )

    kwargs = {
        "x0": x0,
        "maxiter": maxiter,
        "callback": callback,
    }
    kwargs.update(_tolerance_kwargs(atol, rtol))

    x, info = tfqmr(Aop, rhs, **kwargs)
    final_norm = float(np.linalg.norm(wrapped_matvec(x) - rhs))

    return TFQMRResult(
        solution=x,
        info=int(info),
        iterations=niter,
        final_residual_norm=final_norm,
    )
