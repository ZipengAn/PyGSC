#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""PyGSC: post-SCF perturbative correction of KS orbital energies."""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, TextIO, Tuple

import numpy as np
from numba import njit
from pyscf import dft, gto, lib

from input_mol import FileOptions, parse_command_line, read_inputfile, read_xyzfile
from tfqmr_solver import TFQMRResult, solve_tfqmr

Array = np.ndarray


@dataclass(frozen=True)
class GSCParams:

    dtmp: float = 0.015
    nSpin: int = 2
    au2ev: float = 27.2113834
    iter_max: int = 2000
    iter_chk: int = 4
    diff_v_cnvg: float = 1.0e-3
    scf_max_cycle: int = 200
    grid_level: int = 3

    @property
    def cx(self) -> float:
        return 0.46526286817455000879417246873674 * 2.0

    @property
    def cx_v1(self) -> float:
        return -self.cx * 4.0 / 3.0

    @property
    def cx_v2(self) -> float:
        return -self.cx * 4.0 / 9.0


@dataclass
class GSCData:

    mol: object
    mf: object
    settings: GSCParams
    CoefMatrix: Array
    EigValue: Array
    electron: Array
    iHOMO: Array
    iLUMO: Array
    iF: Array
    iflag_fmo: int
    HF_factor: float

    @property
    def nSpin(self) -> int:
        return self.settings.nSpin

    @property
    def nCBasis(self) -> int:
        return int(self.CoefMatrix.shape[-1])

    @property
    def lnum(self) -> int:
        return self.nSpin * self.nCBasis * self.nCBasis


@dataclass(frozen=True)
class GridData:

    ao: Array
    dm: Array
    weights: Array


@dataclass(frozen=True)
class DensityMatrices:

    DMat_1a: Array
    DMat_1b: Array
    DMat_1: Array
    DMat_2a: Array
    DMat_2b: Array
    DMat_2: Array


class CalculationError(RuntimeError):
    pass


def _required(input_info: Dict[str, object], key: str) -> str:
    try:
        value = input_info[key]
    except KeyError as exc:
        raise CalculationError(
            "Missing required input keyword: {}".format(key)
        ) from exc
    return str(value)


def resolve_xc_expression(input_info: Dict[str, object]) -> str:
    method = _required(input_info, "method")
    if method.upper() == "HF":
        return "hf"

    functional = input_info.get("func")
    if functional is not None:
        return str(functional)

    exchange = input_info.get("xfunc")
    if exchange is None:
        raise CalculationError(
            "Please check the method/function keywords in the input file."
        )

    correlation = input_info.get("cfunc")
    if correlation is None:
        raise CalculationError("The correlation functional is missing.")

    exchange_name = str(exchange)[1:]
    correlation_name = str(correlation)[1:]
    if correlation_name == "lda":
        correlation_name = "vwn5"
    return "{},{}".format(exchange_name, correlation_name)


def build_mol(
    input_info: Dict[str, object],
    atoms: object,
    output_file: Path,
) -> object:
    basis = input_info.get("basis")
    if not isinstance(basis, dict) or not basis:
        raise CalculationError("No basis-set definitions were found in the input.")

    mol = gto.Mole()
    mol.charge = int(_required(input_info, "charge"))
    mol.spin = int(_required(input_info, "mult")) - 1
    mol.verbose = 9
    mol.atom = atoms
    mol.basis = basis
    mol.output = str(output_file)
    mol.build(dump_input=False)
    return mol


def run_scf(
    mol: object,
    xc_expression: str,
    input_info: Dict[str, object],
    checkpoint_file: Path,
    settings: GSCParams,
) -> object:
    mf = dft.UKS(mol)
    mf.xc = xc_expression
    mf.chkfile = str(checkpoint_file)
    mf.max_cycle = settings.scf_max_cycle

    scf_shift = input_info.get("scfshift")
    if scf_shift is not None:
        mf.level_shift = float(str(scf_shift))

    initial_guess = input_info.get("guess")
    if initial_guess is not None:
        mf.init_guess = str(initial_guess)

    mf.kernel()
    if not mf.converged:
        raise CalculationError("The unrestricted SCF calculation did not converge.")
    return mf


def build_gsc_data(
    mol: object,
    mf: object,
    input_info: Dict[str, object],
    settings: GSCParams,
) -> GSCData:
    if settings.nSpin != 2:
        raise CalculationError("PyGSC currently uses two unrestricted spin channels.")

    electron = np.array(
        [
            (mol.nelectron + mol.spin) // 2,
            (mol.nelectron - mol.spin) // 2,
        ],
        dtype=np.int32,
    )
    iHOMOInt = electron - 1

    nShiftFukui = int(str(input_info.get("shiftfukui", 0)))
    iLUMO = electron + nShiftFukui
    iHOMO = iHOMOInt - nShiftFukui
    iflag_fmo = int(input_info.get("lumofukui") is not None)
    iF = iLUMO if iflag_fmo else iHOMO

    CoefMatrix = np.asarray(mf.mo_coeff, dtype=np.float64).transpose(0, 2, 1)
    EigValue = np.asarray(mf.mo_energy, dtype=np.float64)
    HF_factor = float(dft.libxc.hybrid_coeff(mf.xc))

    return GSCData(
        mol=mol,
        mf=mf,
        settings=settings,
        CoefMatrix=CoefMatrix,
        EigValue=EigValue,
        electron=electron,
        iHOMO=iHOMO,
        iLUMO=iLUMO,
        iF=iF,
        iflag_fmo=iflag_fmo,
        HF_factor=HF_factor,
    )

def build_grid_data(state: GSCData) -> GridData:
    grids = dft.Grids(state.mol)
    grids.level = state.settings.grid_level
    grids.build()

    ao = dft.numint.eval_ao(state.mol, grids.coords)
    dm = np.empty((state.nSpin, len(grids.weights)), dtype=np.float64)
    for spin in range(state.nSpin):
        dm[spin] = dft.numint.eval_rho2(
            state.mol,
            ao,
            state.mf.mo_coeff[spin],
            state.mf.mo_occ[spin],
        )

    return GridData(
        ao=ao,
        dm=dm,
        weights=np.asarray(grids.weights, dtype=np.float64).copy(),
    )


def _safe_divide(numerator: Array, denominator: Array, threshold: float) -> Array:
    result = np.zeros_like(numerator, dtype=np.float64)
    valid = np.abs(denominator) >= threshold
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _signed_cube_root(values: Array) -> Array:
    return np.power(np.abs(values), 1.0 / 3.0) * np.sign(values)


def get_j(state: GSCData, density_matrices: Array) -> Array:
    return state.mf.get_j(state.mol, density_matrices, hermi=0)


def _lda_exchange_matrix(
    state: GSCData,
    perturbation_density: Array,
    spin: int,
    reference_density: Optional[Array],
    initial: bool,
    hybrid_branch: bool,
) -> Array:
    nCBasis = state.nCBasis
    matrix = np.zeros((nCBasis, nCBasis), dtype=np.float64)
    numerical_integrator = state.mf._numint

    if hybrid_branch:
        scf_density = state.mf.make_rdm1()
    else:
        scf_density = state.mf.make_rdm1()[spin]

    make_rho = numerical_integrator._gen_rho_evaluator(
        state.mol, scf_density, hermi=1
    )[0]

    for ao_block, mask, weights, coordinates in numerical_integrator.block_loop(
        state.mol,
        state.mf.grids,
        nCBasis,
        0,
    ):
        del coordinates
        rho = make_rho(0, ao_block, mask, "LDA")
        if np.linalg.norm(rho) <= 1.0e-5:
            continue

        delta_rho = lib.einsum(
            "mn,rm,rn->r",
            perturbation_density[spin],
            ao_block,
            ao_block,
        )

        if initial:
            # p = 0: finite-density form of the modified XC perturbation, Eq. (25).
            weighted_prefactor = weights * state.settings.cx_v1
            if state.iflag_fmo:
                nonlinear_term = _signed_cube_root(rho + delta_rho) - _signed_cube_root(
                    rho
                )
            else:
                nonlinear_term = _signed_cube_root(rho) - _signed_cube_root(
                    rho - delta_rho
                )
            matrix += lib.einsum(
                "ri,rj,r,r->ij",
                ao_block,
                ao_block,
                weighted_prefactor,
                nonlinear_term,
            )
            continue

        if reference_density is None:
            raise ValueError("reference_density is required for a response matrix")

        reference_delta_rho = lib.einsum(
            "mn,rm,rn->r",
            reference_density[spin],
            ao_block,
            ao_block,
        )
        # Iterative update of the perturbative XC potential, Eqs. (26) and (28).
        base_density = rho + reference_delta_rho if state.iflag_fmo else rho
        weighted_prefactor = (
            weights
            * state.settings.cx_v2
            * np.power(np.abs(base_density), -2.0 / 3.0)
        )
        matrix += lib.einsum(
            "ri,rj,r,r->ij",
            ao_block,
            ao_block,
            weighted_prefactor,
            delta_rho,
        )

    return matrix


def get_xc(
    state: GSCData,
    perturbation_density: Array,
    spin: int,
    reference_density: Optional[Array] = None,
    initial: bool = False,
) -> Array:
    HF_factor = state.HF_factor
    if not 0.0 <= HF_factor <= 1.0:
        raise CalculationError(
            "Unsupported exact-exchange fraction: {}".format(HF_factor)
        )

    exact_exchange = None
    if HF_factor > 0.0:
        exact_exchange = -state.mf.get_k(
            state.mol, perturbation_density, hermi=0
        )[spin]

    local_exchange = None
    if HF_factor < 1.0:
        local_exchange = _lda_exchange_matrix(
            state=state,
            perturbation_density=perturbation_density,
            spin=spin,
            reference_density=reference_density,
            initial=initial,
            hybrid_branch=0.0 < HF_factor < 1.0,
        )

    if HF_factor == 1.0:
        assert exact_exchange is not None
        return exact_exchange
    if HF_factor == 0.0:
        assert local_exchange is not None
        return local_exchange

    assert exact_exchange is not None and local_exchange is not None
    return HF_factor * exact_exchange + (1.0 - HF_factor) * local_exchange


def _make_dmat_from_w(
    state: GSCData,
    mo_matrix: Array,
) -> Array:
    density_response = np.zeros(
        (state.nSpin, state.nCBasis, state.nCBasis), dtype=np.float64
    )

    for spin in range(state.nSpin):
        occupied = int(state.electron[spin])
        energies = state.EigValue[spin]
        gaps = energies[:occupied, None] - energies[None, :]

        scaled_mo_matrix = np.zeros((state.nCBasis, state.nCBasis), dtype=np.float64)
        # Small orbital gaps are omitted, as in the original implementation.
        scaled_mo_matrix[:occupied] = _safe_divide(
            mo_matrix[spin, :occupied],
            gaps,
            state.settings.dtmp,
        )

        CoefMatrix = state.CoefMatrix[spin]
        ao_matrix = CoefMatrix.T @ scaled_mo_matrix @ CoefMatrix
        density_response[spin] = ao_matrix + ao_matrix.T

    return density_response


def tfqmr_matvec(
    state: GSCData,
    reference_density: Array,
    vector: Array,
) -> Array:
    mo_input = np.asarray(vector, dtype=np.float64).reshape(
        state.nSpin, state.nCBasis, state.nCBasis
    )
    density_response = _make_dmat_from_w(state, mo_input)
    coulomb = get_j(state, density_response)

    mo_output = np.empty_like(mo_input)
    for spin in range(state.nSpin):
        xc_matrix = get_xc(
            state,
            perturbation_density=density_response,
            spin=spin,
            reference_density=reference_density,
            initial=False,
        )
        effective_response = coulomb[spin] + coulomb[1 - spin] + xc_matrix
        CoefMatrix = state.CoefMatrix[spin]
        mo_output[spin] = mo_input[spin] - (
            CoefMatrix @ effective_response @ CoefMatrix.T
        )

    return mo_output.reshape(state.lnum)


def make_dmat_front(data: GSCData) -> Array:
    # AO density matrix of the unrelaxed frontier term f_[0](r) in Eq. (25).
    DMatFront = np.empty(
        (data.nSpin, data.nCBasis, data.nCBasis), dtype=np.float64
    )
    for isFrac in range(data.nSpin):
        phi_f = data.CoefMatrix[isFrac, data.iF[isFrac]]
        DMatFront[isFrac] = np.outer(phi_f, phi_f)
    return DMatFront


def make_ae_mat(data: GSCData, DMatFront: Array) -> Array:
    AEMat = np.empty(
        (data.nSpin, data.nSpin, data.nCBasis, data.nCBasis),
        dtype=np.float64,
    )
    JMat1 = get_j(data, DMatFront)

    for isFrac in range(data.nSpin):
        XCMat1 = get_xc(
            data,
            perturbation_density=DMatFront,
            spin=isFrac,
            initial=True,
        )
        for ispin in range(data.nSpin):
            tmpMat = JMat1[ispin] + XCMat1 if ispin == isFrac else JMat1[isFrac]
            CoefMatrix = data.CoefMatrix[ispin]
            AEMat[isFrac, ispin] = CoefMatrix @ tmpMat @ CoefMatrix.T

    return AEMat


def _log_solver_result(
    out_f: TextIO,
    label: str,
    result: TFQMRResult,
    tolerance: float,
) -> None:
    converged = result.converged or result.final_residual_norm < tolerance
    if converged:
        out_f.write(
            "{} converged after {} callback iterations.\n".format(
                label, result.iterations
            )
        )
    else:
        out_f.write("Error! {} procedures NOT converged.\n".format(label))
        out_f.write("SciPy tfqmr info = {}\n".format(result.info))
    out_f.write("Final norm = {:14.6e}\n\n".format(result.final_residual_norm))


def solve_w(
    data: GSCData,
    AEMat: Array,
    DMat0: Array,
    out_f: TextIO,
    label: str,
    description: str,
) -> Array:
    # Matrix-free solution of the self-consistent w^(k) cycle in Eq. (12).
    AFMat = np.empty_like(AEMat)
    out_f.write("\n***********************************************************\n\n")

    for isFrac in range(data.nSpin):
        out_f.write("{} for isFrac = {}\n".format(description, isFrac))
        rhs = AEMat[isFrac].reshape(data.lnum)
        out_f.write("Initial norm {:14.6e}\n".format(np.linalg.norm(rhs)))

        result = solve_tfqmr(
            matvec=lambda vector, DMat0=DMat0: tfqmr_matvec(
                data, DMat0, vector
            ),
            rhs=rhs,
            size=data.lnum,
            maxiter=data.settings.iter_max,
            atol=data.settings.diff_v_cnvg,
            rtol=0.0,
            iter_chk=data.settings.iter_chk,
            log_f=out_f,
            label=label,
        )
        AFMat[isFrac] = result.solution.reshape(
            data.nSpin, data.nCBasis, data.nCBasis
        )
        _log_solver_result(
            out_f,
            label=label,
            result=result,
            tolerance=data.settings.diff_v_cnvg,
        )

    return AFMat


def make_dmat_fst(data: GSCData, AFMat: Array) -> Array:
    # Orbital-relaxation part of the first-order Fukui function in Eq. (9).
    DMatFst = np.empty_like(AFMat)
    for isFrac in range(data.nSpin):
        DMatFst[isFrac] = _make_dmat_from_w(data, AFMat[isFrac])
    return DMatFst


def make_qmat_front(data: GSCData, AFMat: Array) -> Array:
    # Frontier-orbital term entering the second-order Fukui function, Eq. (10).
    QMatFront = np.empty(
        (data.nSpin, data.nCBasis, data.nCBasis), dtype=np.float64
    )

    for isFrac in range(data.nSpin):
        iF = int(data.iF[isFrac])
        gaps = data.EigValue[isFrac] - data.EigValue[isFrac, iF]
        coeff = _safe_divide(
            AFMat[isFrac, isFrac, :, iF],
            gaps,
            data.settings.dtmp,
        )
        dphi_f = coeff @ data.CoefMatrix[isFrac]
        tmpMat = np.outer(data.CoefMatrix[isFrac, iF], dphi_f)
        QMatFront[isFrac] = -tmpMat - tmpMat.T

    return QMatFront


def make_qmat_fst(data: GSCData, AFMat: Array) -> Array:
    QMatFst = np.empty_like(AFMat)

    for isFrac in range(data.nSpin):
        for ispin in range(data.nSpin):
            nocc = int(data.electron[ispin])
            gaps = (
                data.EigValue[ispin, :, None]
                - data.EigValue[ispin, None, :nocc]
            )
            tmpMat = np.zeros((data.nCBasis, data.nCBasis), dtype=np.float64)
            tmpMat[:, :nocc] = _safe_divide(
                AFMat[isFrac, ispin, :, :nocc],
                gaps,
                data.settings.dtmp,
            )
            CoefMatrix = data.CoefMatrix[ispin]
            QMatFst[isFrac, ispin] = (
                CoefMatrix.T @ (tmpMat @ tmpMat.T) @ CoefMatrix
            )

    return QMatFst


# RMatFst and SMatFst contain the remaining orbital-relaxation terms
# used to assemble the second- and third-order Fukui functions in Eqs. (10)-(11).
def make_rmat_fst(
    data: GSCData,
    AFMat: Array,
    AFMat2: Optional[Array] = None,
) -> Array:
    RMatFst = np.empty_like(AFMat)
    dtmp = data.settings.dtmp

    for isFrac in range(data.nSpin):
        for ispin in range(data.nSpin):
            EigValue = data.EigValue[ispin]
            nocc = int(data.electron[ispin])
            w1 = AFMat[isFrac, ispin]

            tmpMat1 = np.zeros((data.nCBasis, data.nCBasis), dtype=np.float64)
            tmpMat2 = np.zeros_like(tmpMat1)
            tmpMat3 = np.zeros_like(tmpMat1)

            for m in range(nocc):
                gaps = EigValue - EigValue[m]
                valid = np.abs(gaps) >= dtmp
                w_over_gap = _safe_divide(w1[m], gaps, dtmp)

                tmpMat1[m, valid] = (w1[valid] @ w_over_gap) / gaps[valid]
                tmpMat2[m, valid] = (
                    w1[valid, m] * w1[m, m] / np.square(gaps[valid])
                )

                if AFMat2 is not None:
                    tmpMat3[valid, m] = AFMat2[isFrac, ispin, valid, m] / gaps[valid]

            tmpMat = tmpMat1 - tmpMat2 - tmpMat3
            CoefMatrix = data.CoefMatrix[ispin]
            tmpMat = CoefMatrix.T @ tmpMat @ CoefMatrix
            RMatFst[isFrac, ispin] = tmpMat + tmpMat.T

    return RMatFst


def make_smat_fst(data: GSCData, AFMat: Array) -> Array:
    SMatFst = np.empty_like(AFMat)

    for isFrac in range(data.nSpin):
        for ispin in range(data.nSpin):
            nocc = int(data.electron[ispin])
            gaps = (
                data.EigValue[ispin, :nocc, None]
                - data.EigValue[ispin, None, :]
            )
            tmpMat1 = _safe_divide(
                AFMat[isFrac, ispin, :nocc],
                gaps,
                data.settings.dtmp,
            )
            diagonal = np.sum(np.square(tmpMat1), axis=1)
            tmpMat2 = np.zeros((data.nCBasis, data.nCBasis), dtype=np.float64)
            tmpMat2[np.arange(nocc), np.arange(nocc)] = diagonal
            CoefMatrix = data.CoefMatrix[ispin]
            SMatFst[isFrac, ispin] = -(
                CoefMatrix.T @ tmpMat2 @ CoefMatrix
            )

    return SMatFst


def make_dmat_1(data: GSCData, DMatFront: Array, DMatFst: Array) -> Array:
    DMat_1 = np.empty_like(DMatFront)
    for isFrac in range(data.nSpin):
        DMat_1[isFrac] = (
            DMatFront[isFrac]
            + DMatFst[isFrac, isFrac]
            + DMatFst[isFrac, 1 - isFrac]
        )
    return DMat_1


def make_ae_mat2(
    data: GSCData,
    DMat_1: Array,
    QMatFront: Array,
    QMatFst: Array,
    RMatFst_tmp: Array,
    SMatFst: Array,
) -> Array:
    AEMat2 = np.empty_like(QMatFst)

    for isFrac in range(data.nSpin):
        tmpMat1 = (
            RMatFst_tmp[isFrac]
            + QMatFst[isFrac]
            + SMatFst[isFrac]
        ).copy()
        tmpMat1[isFrac] += QMatFront[isFrac]

        JMat2 = get_j(data, tmpMat1)
        for ispin in range(data.nSpin):
            XCMat2 = get_xc(
                data,
                perturbation_density=tmpMat1,
                spin=ispin,
                reference_density=DMat_1,
                initial=False,
            )
            tmpMat2 = JMat2[ispin] + JMat2[1 - ispin] + XCMat2
            CoefMatrix = data.CoefMatrix[ispin]
            AEMat2[isFrac, ispin] = CoefMatrix @ tmpMat2 @ CoefMatrix.T

    return AEMat2


def collect_density_matrices(
    data: GSCData,
    DMatFront: Array,
    DMatFst: Array,
    QMatFront: Array,
    QMatFst: Array,
    RMatFst: Array,
    SMatFst: Array,
) -> DensityMatrices:
    DMat_1a = np.empty_like(DMatFront)
    DMat_1b = np.empty_like(DMatFront)
    DMat_2a = np.empty_like(DMatFront)
    DMat_2b = np.empty_like(DMatFront)

    for ispin in range(data.nSpin):
        jspin = 1 - ispin
        DMat_1a[ispin] = DMatFront[ispin] + DMatFst[ispin, ispin]
        DMat_1b[ispin] = DMatFst[ispin, jspin]
        DMat_2a[ispin] = (
            QMatFront[ispin]
            + QMatFst[ispin, ispin]
            + RMatFst[ispin, ispin]
            + SMatFst[ispin, ispin]
        )
        DMat_2b[ispin] = (
            QMatFst[ispin, jspin]
            + RMatFst[ispin, jspin]
            + SMatFst[ispin, jspin]
        )

    return DensityMatrices(
        DMat_1a=DMat_1a,
        DMat_1b=DMat_1b,
        DMat_1=DMat_1a + DMat_1b,
        DMat_2a=DMat_2a,
        DMat_2b=DMat_2b,
        DMat_2=DMat_2a + DMat_2b,
    )


def get_coulomb_energy(
    data: GSCData,
    DMats: DensityMatrices,
) -> Tuple[Array, Array]:
    DMatJ1 = get_j(data, DMats.DMat_1)
    DMatJ2 = get_j(data, DMats.DMat_2)

    deltaEJ = np.empty(data.nSpin, dtype=np.float64)
    dE3rdJ = np.empty(data.nSpin, dtype=np.float64)
    for ispin in range(data.nSpin):
        deltaEJ[ispin] = 0.5 * np.einsum(
            "ij,ij->", DMats.DMat_1[ispin], DMatJ1[ispin]
        )
        dE3rdJ[ispin] = np.einsum(
            "ij,ij->", DMats.DMat_1[ispin], DMatJ2[ispin]
        )

    return deltaEJ, dE3rdJ


def get_de_2nd(data: GSCData, AFMat: Array) -> Array:
    dE2nd = np.zeros(data.nSpin, dtype=np.float64)

    for isFrac in range(data.nSpin):
        for ispin in range(data.nSpin):
            nocc = int(data.electron[ispin])
            gaps = (
                data.EigValue[ispin, None, nocc:]
                - data.EigValue[ispin, :nocc, None]
            )
            amplitudes = AFMat[isFrac, ispin, :nocc, nocc:]
            valid = np.abs(gaps) >= data.settings.dtmp
            dE2nd[isFrac] += np.sum(
                np.divide(
                    np.square(amplitudes),
                    gaps,
                    out=np.zeros_like(amplitudes),
                    where=valid,
                )
            )

    return dE2nd


@njit(cache=True)
def _get_de_3rd_a_kernel(
    AFMat: Array,
    AFMat2: Array,
    EigValue: Array,
    electron: Array,
    iF: Array,
    dtmp: float,
) -> Array:
    nSpin = AFMat.shape[0]
    nCBasis = AFMat.shape[-1]
    dE3rda = np.zeros(nSpin, dtype=np.float64)

    for isFrac in range(nSpin):
        for ispin in range(nSpin):
            nocc = electron[ispin]
            for m in range(nocc):
                for n in range(nCBasis):
                    dtmp1 = EigValue[ispin, n] - EigValue[ispin, m]
                    if abs(dtmp1) < dtmp:
                        continue
                    for l in range(nCBasis):
                        dtmp2 = EigValue[ispin, l] - EigValue[ispin, m]
                        if abs(dtmp2) < dtmp:
                            continue
                        dE3rda[isFrac] += (
                            -2.0
                            * AFMat[isFrac, ispin, m, l]
                            * AFMat[isFrac, ispin, l, n]
                            * AFMat[isFrac, ispin, n, m]
                            / dtmp1
                            / dtmp2
                        )
                    dE3rda[isFrac] += (
                        2.0
                        * AFMat[isFrac, ispin, n, m]
                        * AFMat[isFrac, ispin, n, m]
                        * AFMat[isFrac, ispin, m, m]
                        / dtmp1
                        / dtmp1
                        + 2.0
                        * AFMat[isFrac, ispin, n, m]
                        * AFMat2[isFrac, ispin, n, m]
                        / dtmp1
                    )

        for n in range(nCBasis):
            dtmp3 = EigValue[isFrac, iF[isFrac]] - EigValue[isFrac, n]
            if abs(dtmp3) < dtmp:
                continue
            dE3rda[isFrac] += (
                -AFMat[isFrac, isFrac, iF[isFrac], n]
                * AFMat[isFrac, isFrac, iF[isFrac], n]
                / dtmp3
            )

    return dE3rda


def get_de_3rd_a(data: GSCData, AFMat: Array, AFMat2: Array) -> Array:
    return _get_de_3rd_a_kernel(
        AFMat,
        AFMat2,
        data.EigValue,
        data.electron,
        data.iF,
        data.settings.dtmp,
    )


def get_delta_exc(
    data: GSCData,
    grid: GridData,
    DMats: DensityMatrices,
) -> Array:
    HF_factor = data.HF_factor
    deltaEXC_HF = np.zeros(data.nSpin, dtype=np.float64)
    deltaEXC_DFA = np.zeros(data.nSpin, dtype=np.float64)

    if HF_factor > 0.0:
        DMatK1a = data.mf.get_k(data.mol, DMats.DMat_1a, hermi=0)
        DMatK1b = data.mf.get_k(data.mol, DMats.DMat_1b, hermi=0)
        for ispin in range(data.nSpin):
            deltaEXC_HF[ispin] = 0.5 * (
                np.einsum("ij,ji->", DMats.DMat_1a[ispin], DMatK1a[ispin])
                + np.einsum("ij,ji->", DMats.DMat_1b[ispin], DMatK1b[ispin])
            )

    if HF_factor < 1.0:
        for ispin in range(data.nSpin):
            rho = grid.dm[ispin]
            rho13 = np.power(rho, 1.0 / 3.0)
            drho1 = lib.einsum(
                "ri,rj,ij->r",
                grid.ao,
                grid.ao,
                DMats.DMat_1[ispin],
            )
            vxc = data.settings.cx * grid.weights

            if data.iflag_fmo:
                tmp = (
                    4.0 / 3.0 * drho1 * rho13
                    + rho * rho13
                    - np.power(np.abs(rho + drho1), 4.0 / 3.0)
                )
            else:
                tmp = (
                    4.0 / 3.0 * drho1 * rho13
                    - rho * rho13
                    + np.power(np.abs(rho - drho1), 4.0 / 3.0)
                )
            deltaEXC_DFA[ispin] = np.einsum("r,r->", vxc, tmp)

    if HF_factor == 1.0:
        return -deltaEXC_HF if data.iflag_fmo else deltaEXC_HF
    if HF_factor == 0.0:
        return deltaEXC_DFA

    if data.iflag_fmo:
        deltaEXC_HF = -deltaEXC_HF
    return HF_factor * deltaEXC_HF + (1.0 - HF_factor) * deltaEXC_DFA


def get_de_3rd_k(data: GSCData, DMats: DensityMatrices) -> Array:
    if data.HF_factor == 0.0:
        return np.zeros(data.nSpin, dtype=np.float64)

    DMatK2a = data.mf.get_k(data.mol, DMats.DMat_2a, hermi=0)
    DMatK2b = data.mf.get_k(data.mol, DMats.DMat_2b, hermi=0)
    dE3rdK = np.empty(data.nSpin, dtype=np.float64)

    for ispin in range(data.nSpin):
        dE3rdK[ispin] = -(
            np.einsum("ij,ji->", DMats.DMat_1a[ispin], DMatK2a[ispin])
            + np.einsum("ij,ji->", DMats.DMat_1b[ispin], DMatK2b[ispin])
        )

    return data.HF_factor * dE3rdK


def write_corrected_energies(
    data: GSCData,
    out_f: TextIO,
    deltaEJ: Array,
    deltaEXC: Array,
    dE2nd: Array,
    dE3rda: Array,
    dE3rdJ: Array,
    dE3rdK: Array,
) -> None:
    out_f.write("***********************************************************\n\n")
    out_f.write("PyGSC calculation with the modified perturbative XC potential.\n\n")
    out_f.write("***********************************************************\n\n")

    # Eq. (13): Delta epsilon_f = Delta epsilon_f^(1)
    #            + Delta epsilon_f^(2) + Delta epsilon_f^(3).
    for ispin in range(data.nSpin):
        iorb = data.iLUMO[ispin] if data.iflag_fmo else data.iHOMO[ispin]
        e_ori = data.EigValue[ispin, iorb] * data.settings.au2ev

        if data.iflag_fmo:
            e_1st = e_ori + (deltaEXC[ispin] + deltaEJ[ispin]) * data.settings.au2ev
            e_2nd = e_1st + dE2nd[ispin] * data.settings.au2ev
        else:
            e_1st = e_ori + (deltaEXC[ispin] - deltaEJ[ispin]) * data.settings.au2ev
            e_2nd = e_1st - dE2nd[ispin] * data.settings.au2ev

        e_3rd = e_2nd + (
            dE3rda[ispin] + dE3rdJ[ispin] + dE3rdK[ispin]
        ) * data.settings.au2ev

        out_f.write("iSpin = {:1d}, e_ori = {:14.6e}\n".format(ispin, e_ori))
        out_f.write("iSpin = {:1d}, e_1st = {:14.6e}\n".format(ispin, e_1st))
        out_f.write("iSpin = {:1d}, e_2nd = {:14.6e}\n".format(ispin, e_2nd))
        out_f.write("iSpin = {:1d}, e_3rd = {:14.6e}\n\n".format(ispin, e_3rd))


def _split_elapsed(seconds: float) -> Tuple[int, int, int]:
    hours = int(seconds // 3600)
    minutes = int((seconds - hours * 3600) // 60)
    whole_seconds = int(seconds - hours * 3600 - minutes * 60)
    return hours, minutes, whole_seconds


def write_timing(
    output: TextIO,
    wall_seconds: float,
    cpu_seconds: float,
    settings: GSCParams,
) -> None:
    wall = _split_elapsed(wall_seconds)
    cpu = _split_elapsed(cpu_seconds)
    output.write("***********************************************************\n\n")
    output.write(
        "The wall time: {:4d} hour {:4d} min {:4d} sec\n".format(*wall)
    )
    output.write(
        "The CPU time: {:4d} hour {:4d} min {:4d} sec\n".format(*cpu)
    )
    output.write("\n\n***********************************************************\n\n")
    output.write(
        "Test: The delta of converge is {:.4f}\n\n".format(
            settings.dtmp
        )
    )


def run_calculation(options: FileOptions) -> None:
    params = GSCParams()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    input_info = read_inputfile(str(options.input_file))
    xyz_info = read_xyzfile(str(options.xyz_file))
    xc_info = resolve_xc_expression(input_info)

    mol = build_mol(input_info, xyz_info, options.output_file)
    mf = run_scf(
        mol,
        xc_info,
        input_info,
        options.checkpoint_file,
        params,
    )
    data = build_gsc_data(mol, mf, input_info, params)
    grid = build_grid_data(data)
    out_f = mol.stdout

    DMatFront = make_dmat_front(data)

    AEMat = make_ae_mat(data, DMatFront)
    AFMat = solve_w(
        data,
        AEMat,
        DMatFront,
        out_f,
        label="TFQMR",
        description="Solving perturbation Hamiltonian matrix",
    )
    print("TFQMR1 IS OVER")

    DMatFst = make_dmat_fst(data, AFMat)
    QMatFront = make_qmat_front(data, AFMat)
    QMatFst = make_qmat_fst(data, AFMat)
    RMatFst_tmp = make_rmat_fst(data, AFMat)
    SMatFst = make_smat_fst(data, AFMat)
    DMat_1 = make_dmat_1(data, DMatFront, DMatFst)

    AEMat2 = make_ae_mat2(
        data,
        DMat_1,
        QMatFront,
        QMatFst,
        RMatFst_tmp,
        SMatFst,
    )
    AFMat2 = solve_w(
        data,
        AEMat2,
        DMat_1,
        out_f,
        label="TFQMR2",
        description="Solving perturbation Hamiltonian matrix2",
    )

    RMatFst = make_rmat_fst(data, AFMat, AFMat2=AFMat2)
    DMats = collect_density_matrices(
        data,
        DMatFront,
        DMatFst,
        QMatFront,
        QMatFst,
        RMatFst,
        SMatFst,
    )

    deltaEJ, dE3rdJ = get_coulomb_energy(data, DMats)
    dE2nd = get_de_2nd(data, AFMat)
    dE3rda = get_de_3rd_a(data, AFMat, AFMat2)
    deltaEXC = get_delta_exc(data, grid, DMats)
    dE3rdK = get_de_3rd_k(data, DMats)

    write_corrected_energies(
        data,
        out_f,
        deltaEJ,
        deltaEXC,
        dE2nd,
        dE3rda,
        dE3rdJ,
        dE3rdK,
    )
    write_timing(
        out_f,
        wall_seconds=time.perf_counter() - wall_start,
        cpu_seconds=time.process_time() - cpu_start,
        settings=params,
    )
    out_f.flush()


def main(argv: Optional[Sequence[str]] = None) -> int:
    options = parse_command_line(argv)
    try:
        run_calculation(options)
    except Exception as exc:
        with options.error_file.open("w", encoding="utf-8") as error_output:
            error_output.write("FATAL ERROR: {}\n".format(exc))
        raise

    with options.error_file.open("w", encoding="utf-8") as error_output:
        error_output.write("The program ends normally, with no fatal error.\n")
    print("The program is end.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
