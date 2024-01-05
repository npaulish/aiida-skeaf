#!/usr/bin/env runaiida
"""Examplary script to submit a Wan2skeafCalculation."""
# pylint: disable=unused-import
from aiida import engine, orm

from aiida_skeaf.calculations.wan2skeaf import Wan2skeafCalculation


def submit():
    """Submit a Wan2skeafCalculation."""

    # bxsf = orm.load_node(137298)
    bxsf = orm.load_node("e1f4857d-e6d0-4a7f-a25c-783d2b9ca24a")

    # skeaf_code = orm.load_code("skeaf@localhost")
    wan2skeaf_code = orm.load_code("skeaf-wan2skeaf@localhost")

    parameters = {
        "num_electrons": 11,
        "band_index": -1,
    }

    inputs = {
        "code": wan2skeaf_code,
        "parameters": parameters,
        "bxsf": bxsf,
        "bxsf_filename": "bxsf.7z",
        "settings": dict({"autolink_bxsf_filename": "bxsf.7z"}),
    }

    calc = engine.submit(Wan2skeafCalculation, **inputs)

    print(f"Submitted {calc}")


if __name__ == "__main__":
    submit()
