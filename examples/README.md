# PyGSC examples

Each example is placed in a separate directory so that the default output, error, geometry, and checkpoint filenames can be used directly.

Run an example from its directory, for example:

```bash
cd examples/ch4_b3lyp_homo
python ../../gsc.py -i ch4_b3lyp_homo.inp
```

The examples are intended to demonstrate the input syntax and program options.

| Directory | Demonstrated features |
|---|---|
| `ch4_b3lyp_homo` | B3LYP calculation and the default HOMO branch |
| `ch4_b3lyp_lumo_plus1` | Electron-addition branch with `lumofukui` and `shiftfukui 1` |
| `li_hf_homo` | Open-shell Hartree–Fock calculation with `mult 2` |
| `hli_b3lyp_homo` | Closed-shell heteronuclear HLi molecule with B3LYP |
| `h2s_b3lyp_homo_minus1` | Occupied orbital below the HOMO with `shiftfukui 1` |
| `n2_b3lyp_homo_minus2` | Occupied orbital below the HOMO with `shiftfukui 2` |
| `h2_blyp_options` | Split `xfunc`/`cfunc` form, `guess`, and `scfshift` |

For a plain LUMO calculation, use `lumofukui` without `shiftfukui`. For a checkpoint restart, add `guess chk` to the input and pass the checkpoint file with `-c` when needed.
