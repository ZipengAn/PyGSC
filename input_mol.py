#!/usr/bin/env python3
# Copyright 2022-2026 An Zipeng
# SPDX-License-Identifier: Apache-2.0

"""Input handling for PyGSC."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

PROGRAM_VERSION = "0.00.02"


@dataclass(frozen=True)
class FileOptions:
    input_file: Path
    output_file: Path
    error_file: Path
    xyz_file: Path
    checkpoint_file: Path

    def as_legacy_list(self) -> List[str]:
        return [
            str(self.input_file),
            str(self.output_file),
            str(self.error_file),
            str(self.xyz_file),
            str(self.checkpoint_file),
        ]


def parse_command_line(argv: Optional[Sequence[str]] = None) -> FileOptions:
    parser = argparse.ArgumentParser(
        prog="gsc.py",
        description="Run a PyGSC orbital-energy correction calculation.",
    )
    parser.add_argument("-v", "--version", action="version", version=PROGRAM_VERSION)
    parser.add_argument("-i", "--input", required=True, dest="inputfile")
    parser.add_argument("-o", "--output", dest="outputfile")
    parser.add_argument("-e", "--error", dest="errorfile")
    parser.add_argument("-x", "--xyz", dest="xyzfile")
    parser.add_argument("-c", "--check", "--chk", dest="checkfile")

    args = parser.parse_args(argv)
    inputfile = Path(args.inputfile)

    return FileOptions(
        input_file=inputfile,
        output_file=Path(args.outputfile) if args.outputfile else inputfile.with_suffix(".out"),
        error_file=Path(args.errorfile) if args.errorfile else inputfile.with_suffix(".err"),
        xyz_file=Path(args.xyzfile) if args.xyzfile else inputfile.with_suffix(".xyz"),
        checkpoint_file=(
            Path(args.checkfile) if args.checkfile else inputfile.with_suffix(".chk")
        ),
    )


def read_command(argv: Optional[Sequence[str]] = None) -> List[str]:
    return parse_command_line(argv).as_legacy_list()


def _basis_name(raw_basis: str) -> str:
    return raw_basis.split(".", 1)[1] if "." in raw_basis else raw_basis


def read_inputfile(inputfile: str) -> Dict[str, object]:
    input_info: Dict[str, object] = {"basis": {}}
    read_file_flag = False

    with Path(inputfile).open("r", encoding="utf-8") as inp_f:
        for lineno, line in enumerate(inp_f, start=1):
            items = line.split()
            if not items or items[0].startswith("#"):
                continue

            key = items[0]
            if key == "$qm":
                read_file_flag = True
                continue
            if key == "end" and read_file_flag:
                break
            if not read_file_flag:
                continue

            if key == "basis":
                if len(items) < 3:
                    raise ValueError("Invalid basis specification on line {}".format(lineno))
                basis_info = input_info["basis"]
                assert isinstance(basis_info, dict)
                basis_info[items[1]] = _basis_name(items[2])
            else:
                input_info[key] = items[1] if len(items) > 1 else "no_args"

    if not read_file_flag:
        raise ValueError("Input file does not contain a '$qm' section.")

    return input_info


def read_xyzfile(xyzfile: str) -> List[Tuple[str, Tuple[float, float, float]]]:
    xyz_info: List[Tuple[str, Tuple[float, float, float]]] = []

    with Path(xyzfile).open("r", encoding="utf-8") as xyz_f:
        xyz_lines = xyz_f.readlines()

    for lineno, line in enumerate(xyz_lines[2:], start=3):
        items = line.split()
        if not items:
            continue
        if len(items) < 4:
            raise ValueError("Invalid XYZ record on line {}".format(lineno))

        atom = items[0]
        coord = tuple(float(value) for value in items[1:4])
        xyz_info.append((atom, coord))

    return xyz_info
