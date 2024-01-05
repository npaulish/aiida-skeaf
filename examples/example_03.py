#!/usr/bin/env runaiida
"""Examplary script to submit a SkeafWorkChain."""
from aiida import engine, orm

from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder

from aiida_skeaf.utils.plot import (  # pylint: disable=unused-import
    plot_frequency_workchain,
)
from aiida_skeaf.workflows import SkeafWorkChain


def submit():
    """Submit a SkeafWorkChain."""

    # bxsf = orm.load_node(137298)
    bxsf = orm.load_node("e1f4857d-e6d0-4a7f-a25c-783d2b9ca24a")

    codes = {
        "wan2skeaf": "skeaf-wan2skeaf-jl@localhost",
        "skeaf": "skeaf-skeaf@localhost",
    }

    builder = SkeafWorkChain.get_builder_from_protocol(
        codes,
        bxsf=bxsf,
        num_electrons=11,
        protocol="fast",
    )

    builder.wan2skeaf.settings.autolink_bxsf_filename = (  # pylint: disable=no-member
        "bxsf.7z"
    )
    builder.wan2skeaf.bxsf_filename = "bxsf.7z"  # pylint: disable=no-member

    print_builder(builder)

    wkchain = engine.submit(builder)

    print(f"Submitted {wkchain}")

    # Once workchain has finished, plot frequency vs angle
    # plot_frequency_workchain(wkchain, x="theta")


if __name__ == "__main__":
    submit()
