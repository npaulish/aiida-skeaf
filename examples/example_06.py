#!/usr/bin/env runaiida
"""Submit a wan2skeaf calculation using create_wan2skeaf_builder function."""
# pylint: disable=unused-import

from aiida import engine, orm

from aiida_skeaf.utils import create_wan2skeaf_builder


def submit():
    """Submit a Wan2skeafCalculation."""

    bxsf = orm.load_node(244145)

    wan2skeaf_code = orm.load_code("skeaf-wan2skeaf-jl@localhost-slurm")

    parameters = {
        "num_electrons": 31,
        "band_index": -1,
        "smearing_type": "cold",
        "smearing_value": 0.1,
    }

    inputs = {
        "code": wan2skeaf_code,
        "parameters": parameters,
        "bxsf": bxsf,
        "bxsf_filename": "wjl.7z",
        "settings": dict({"autolink_bxsf_filename": "wjl.7z"}),
    }

    builder = create_wan2skeaf_builder(parameters, inputs)
    calc = engine.submit(builder)

    print(f"Submitted {calc}")


if __name__ == "__main__":
    submit()
