#!/usr/bin/env runaiida
"""Submit a wan2skeaf calculation using create_wan2skeaf_builder function."""
# pylint: disable=unused-import
import pathlib

from aiida import engine, orm

from aiida_skeaf.calculations.functions import create_bxsf_from_file
from aiida_skeaf.utils import create_wan2skeaf_builder

INPUT_DIR = pathlib.Path(__file__).absolute().parent / "input_files"


def submit():
    """Submit a Wan2skeafCalculation."""

    remote_path = orm.Str(str(INPUT_DIR / "bxsf"))
    computer = orm.Str("localhost")
    bxsf = create_bxsf_from_file(remote_path, computer)
    # bxsf = orm.load_node("e1f4857d-e6d0-4a7f-a25c-783d2b9ca24a")

    wan2skeaf_code = orm.load_code("skeaf-wan2skeaf-jl@localhost")

    parameters = {
        "num_electrons": 11,
        "band_index": -1,
        "smearing_type": "cold",
        "smearing_value": 0.1,
    }

    inputs = {
        "code": wan2skeaf_code,
        "parameters": parameters,
        "bxsf": bxsf,
        "bxsf_filename": "copper.7z",
        "settings": dict({"autolink_bxsf_filename": "copper.7z"}),
    }

    builder = create_wan2skeaf_builder(parameters, inputs)
    calc = engine.submit(builder)

    print(f"Submitted {calc}")


if __name__ == "__main__":
    submit()
