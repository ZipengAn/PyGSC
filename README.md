# PyGSC

PyGSC is a Python program for post-SCF correction of Kohn–Sham orbital energies using the perturbative global scaling correction (GSC) and QE-DFT framework. It is built on PySCF and includes orbital-relaxation corrections through third order.

The implementation follows the modified perturbative exchange–correlation potential described in:

> Z. An, X. Yang, X. Zheng, and W. Yang, *PyGSC: A Python tool for correcting Kohn–Sham orbital energies by mitigating the delocalization error of density functional approximations*.

## 1. Files

The release contains:

- `gsc.py`: main program;
- `input_mol.py`: command-line, molecular-input, and XYZ parser;
- `tfqmr_solver.py`: matrix-free TFQMR solver based on SciPy;
- `LICENSE`: Apache License 2.0.

Keep the three Python files in the same directory when running the program.

## 2. Requirements

The following versions are the current verified compatibility baseline:

| Package | Version |
|---|---:|
| Python | 3.8 or later |
| NumPy | 1.20 or later |
| SciPy | 1.8.1 or later |
| Numba | 0.55 or later |
| PySCF | 2.0.1 or later |

These are conservative compatibility baselines rather than exhaustively tested absolute minimum versions.

A typical installation is:

```bash
python -m pip install "numpy>=1.20" "scipy>=1.8.1" "numba>=0.55" "pyscf>=2.0.1"
```

## 3. Usage

The command-line form is:

```bash
python gsc.py -i INPUT [-x XYZ] [-o OUTPUT] [-e ERROR] [-c CHECKPOINT]
```

For example:

```bash
python gsc.py \
  -i sample.inp \
  -x sample.xyz \
  -o sample.out \
  -e sample.err \
  -c sample.chk
```

Only the input file is required. When the other filenames are omitted, PyGSC uses the same file stem automatically:

```bash
python gsc.py -i sample.inp
```

This command uses:

```text
sample.inp
sample.xyz
sample.out
sample.err
sample.chk
```

Command-line options:

| Option | Description |
|---|---|
| `-i`, `--input` | Molecular input file; required |
| `-x`, `--xyz` | XYZ geometry file |
| `-o`, `--output` | Main output file |
| `-e`, `--error` | Error/status file |
| `-c`, `--check`, `--chk` | PySCF checkpoint file |
| `-v`, `--version` | Print the program version |
| `-h`, `--help` | Show command-line help |

The filenames and paths may be chosen freely. The Python interpreter should be the one in which PySCF and the other required packages are installed.

## 4. Molecular input file

PyGSC reads the section between `$qm` and `end`. Blank lines and lines beginning with `#` are ignored.

### 4.1 Required keywords

| Keyword | Description |
|---|---|
| `method` | Calculation type. Use `HF` for Hartree–Fock; use a non-HF value such as `DFT` together with functional keywords for a DFA calculation |
| `basis ELEMENT BASIS` | Basis set for one element; repeat for every element type in the molecule |
| `charge` | Total molecular charge |
| `mult` | Spin multiplicity, `2S + 1` |

For a DFA calculation, provide either:

```text
func b3lyp
```

or the legacy split form:

```text
xfunc xb88
cfunc clyp
```

In the split form, the first character is removed before constructing the PySCF expression, so `xb88` and `clyp` become `b88,lyp`. The correlation name `lda` is converted to `vwn5`.

Both basis-set forms below are accepted:

```text
basis H basis.6-31g
basis H 6-31g
```

### 4.2 Optional keywords

| Keyword | Description |
|---|---|
| `guess VALUE` | Set the PySCF initial guess, such as `minao`, `atom`, `1e`, or `chk` |
| `scfshift VALUE` | Set the PySCF SCF level shift |
| `lumofukui` | Select the virtual-orbital/electron-addition branch instead of the occupied-orbital/electron-removal branch |
| `shiftfukui N` | Shift the selected orbital by integer `N` |

Orbital selection is controlled as follows:

```text
# HOMO
# no additional keyword

# HOMO-1
shiftfukui 1

# LUMO
lumofukui

# LUMO+2
lumofukui
shiftfukui 2
```

### 4.3 B3LYP example

```text
$qm
method DFT
func b3lyp
basis C basis.6-31g
basis H basis.6-31g
charge 0
mult 1
end
```

### 4.4 Hartree–Fock example

```text
$qm
method HF
basis Li basis.6-31g
basis H basis.6-31g
charge 0
mult 1
end
```

### 4.5 LUMO example

```text
$qm
method DFT
func b3lyp
basis C basis.6-31g
basis H basis.6-31g
charge 0
mult 1
lumofukui
end
```

## 5. XYZ geometry file

The geometry file uses standard XYZ format:

```text
5
methane
C   0.000000   0.000000   0.000000
H   0.000000   0.000000   1.089000
H   1.026719   0.000000  -0.363000
H  -0.513360  -0.889165  -0.363000
H  -0.513360   0.889165  -0.363000
```

PyGSC skips the first two lines and reads the element symbol and the first three Cartesian coordinates from each remaining non-empty line.

## 6. Program functions

PyGSC provides the following functions:

- builds molecular systems from charge, multiplicity, Cartesian coordinates, and element-specific basis sets;
- performs unrestricted two-spin-channel calculations with `pyscf.dft.UKS` for closed-shell and open-shell systems;
- supports Hartree–Fock, pure density functionals, and global hybrid functionals;
- accepts functional definitions through `func` or the legacy `xfunc`/`cfunc` form;
- supports user-selected PySCF initial guesses and SCF level shifts;
- writes and reads PySCF checkpoint files through the selected checkpoint filename and initial-guess setting;
- selects HOMO, LUMO, HOMO−`N`, or LUMO+`N` independently for the two spin channels;
- constructs the unrelaxed frontier-orbital density matrix `DMatFront`;
- evaluates the initial Hartree and exchange–correlation perturbation matrices;
- solves the first-order perturbation-Hamiltonian response equation;
- constructs `DMatFst`, `QMatFront`, `QMatFst`, `RMatFst`, and `SMatFst`;
- builds and solves the second-order perturbation-Hamiltonian response equation;
- uses a matrix-free TFQMR solver without explicitly assembling the full response supermatrix;
- supports both the older SciPy `tol` interface and the newer `rtol` interface;
- evaluates Hartree, local-exchange, and Hartree–Fock exchange contributions;
- combines local and exact-exchange contributions for global hybrid functionals;
- applies the modified perturbative exchange-potential expressions used in the accompanying manuscript;
- omits orbital pairs whose absolute energy gaps are smaller than the source-level threshold `dtmp`;
- evaluates first-, second-, and third-order corrections to the selected orbital energies;
- accelerates the most expensive third-order orbital-relaxation kernel with Numba;
- reports TFQMR iteration information, monitored residual norms, convergence status, and final residual norms;
- reports wall-clock time and CPU time.

## 7. Output

The main output file contains the PySCF SCF output, the response-equation convergence information, and the corrected orbital energies for both spin channels.

The principal energy labels are:

| Label | Description |
|---|---|
| `e_ori` | Original Kohn–Sham or generalized Kohn–Sham orbital energy |
| `e_1st` | Orbital energy after the first-order correction |
| `e_2nd` | Orbital energy after corrections through second order |
| `e_3rd` | Orbital energy after corrections through third order |

All four orbital energies are reported in electronvolts.

The error/status file contains either a normal-completion message or a fatal-error message. The checkpoint file is written by PySCF and may be reused through an appropriate initial-guess setting.

## 8. Default numerical settings

| Parameter | Default | Description |
|---|---:|---|
| `dtmp` | `0.015` | Minimum absolute orbital-energy gap retained in perturbative denominators |
| `nSpin` | `2` | Number of unrestricted spin channels |
| `au2ev` | `27.2113834` | Hartree-to-electronvolt conversion factor |
| `iter_max` | `2000` | Maximum TFQMR iterations |
| `iter_chk` | `4` | Interval for writing monitored residual norms |
| `diff_v_cnvg` | `1.0e-3` | TFQMR absolute convergence threshold |
| `scf_max_cycle` | `200` | Maximum PySCF SCF cycles |
| `grid_level` | `3` | PySCF numerical-integration grid level |

These values are defined in `GSCParams` in `gsc.py` and are not molecular-input keywords.
