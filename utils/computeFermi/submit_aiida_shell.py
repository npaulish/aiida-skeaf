#!/usr/bin/env python
# pylint: skip-file
"""Python script to run compute_fermi.jl using aiida-shell."""
import re
import typing as ty

from aiida_shell import launch_shell_job
import click

from aiida import load_profile, orm

from aiida_wannier90_workflows.utils.pseudo import (
    get_number_of_electrons,
    get_pseudo_and_cutoff,
)

load_profile("main")


class BXSFFileNotFoundError(
    Exception
):  # Should be inherited from aiida.common.exceptions.NotExistent?
    """Raised when BXSF file is not found."""


class JobNotFinishedError(
    Exception
):  # Should be inherited from aiida.common.exceptions?
    """Raised when wan2skeaf job is not finished and end timestamp is not in the output."""


class NumElecNotWithinToleranceError(
    Exception
):  # Should be inherited from aiida.common.exceptions?
    """
    Raised when the bisection algorithm to compute Fermi level can't converge
    within the tolerance in number of electrons.
    """


def get_num_electrons(structure: orm.StructureData, params: ty.Union[orm.Dict, dict]):
    """Get number of electrons to run skeaf.

    :param structure: [description]
    :type structure: orm.StructureData
    :param params: [description]
    :type params: orm.Dict
    :return: [number of electrons]
    :rtype: float
    """
    pseudo_family = "SSSP/1.1/PBEsol/efficiency"
    pseudos, _, _ = get_pseudo_and_cutoff(pseudo_family, structure)
    num_elec_pw = get_number_of_electrons(structure, pseudos)
    if isinstance(params, orm.Dict):
        params = params.get_dict()
    num_excl_bands = params.get("exclude_bands", [])
    num_elec = num_elec_pw - 2 * len(num_excl_bands)

    # should be integer
    if abs(num_elec - int(num_elec)) > 1e-5:
        raise ValueError("Non-integer number of electrons?")

    return int(num_elec)


def parse_output(self, dirpath):  # pylint: disable=unused-argument
    """Parse the output of compute_fermi.jl and return the Fermi energy."""
    from aiida.orm import Dict, Float, Int

    parameters = {}

    regexs = {
        "input_file_not_found": re.compile(r"ERROR: input file\s*(.+) does not exist."),
        "failed_to_find_Fermi_energy_within_tolerance": re.compile(
            r"Error: tolerance for number of electrons exceeded the tol_n_electrons_upperbound. Exiting..."
        ),
        "timestamp_started": re.compile(r"Started on\s*(.+)"),
        "num_electrons": re.compile(
            r"Number of electrons:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "tol_n_electrons_initial": re.compile(
            r"Initial tolerance for number of electrons \(default 1e-6\):\s*([+-]?(?:[0-9]*[.])?[0-9]+e?[+-]?[0-9]*)"
        ),
        "fermi_energy_in_bxsf": re.compile(
            r"Fermi Energy from file:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "fermi_energy_computed": re.compile(
            r"Computed Fermi energy:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "fermi_energy_unit": re.compile(r"Fermi energy unit:\s*(.+)"),
        "closest_eigenvalue_below_fermi": re.compile(
            r"Closest eigenvalue below Fermi energy:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "closest_eigenvalue_above_fermi": re.compile(
            r"Closest eigenvalue above Fermi energy:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "num_bands": re.compile(r"Number of bands:\s*([0-9]+)"),
        "kpoint_mesh": re.compile(r"Grid shape:\s*(.+)"),
        "smearing_type": re.compile(r"Smearing type:\s*(.+)"),
        "smearing_width": re.compile(
            r"Smearing width:\s*([+-]?(?:[0-9]*[.])?[0-9]+e?[+-]?[0-9]*)"
        ),
        "occupation_prefactor": re.compile(
            r"Occupation prefactor:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "tol_n_electrons_final": re.compile(
            r"Final tolerance for number of electrons:\s*([+-]?(?:[0-9]*[.])?[0-9]+e?[+-]?[0-9]*)"
        ),
        "band_indexes_in_bxsf": re.compile(r"Bands in bxsf:\s*(.+)"),
        "bands_crossing_fermi": re.compile(r"Bands crossing Fermi energy:\s*(.*)"),
        "timestamp_end": re.compile(r"Job done at\s*(.+)"),
    }
    re_band_minmax = re.compile(
        r"Min and max of band\s*([0-9]*)\s*:\s*([+-]?(?:[0-9]*[.])?[0-9]+)\s+([+-]?(?:[0-9]*[.])?[0-9]+)"
    )
    band_minmax = {}

    output = (dirpath / "stdout").read_text().strip()
    for line in output.split("\n"):
        for key, reg in regexs.items():
            match = reg.match(line.strip())
            if match:
                parameters[key] = match.group(1)
                regexs.pop(key, None)
                break

        match = re_band_minmax.match(line.strip())
        if match:
            band = int(match.group(1))
            band_min = float(match.group(2))
            band_max = float(match.group(3))
            band_minmax[band] = (band_min, band_max)

    if "input_file_not_found" in parameters:
        import errno
        import os

        raise BXSFFileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), parameters["input_file_not_found"]
        )
    if "failed_to_find_Fermi_energy_within_tolerance" in parameters:
        raise NumElecNotWithinToleranceError(
            "Failed to find Fermi energy within tolerance, Δn_elec = "
            + f"{parameters['failed_to_find_Fermi_energy_within_tolerance']}"
        )
    if "timestamp_end" not in parameters:
        raise JobNotFinishedError("Job not finished!")

    parameters["kpoint_mesh"] = [int(_) for _ in parameters["kpoint_mesh"].split("x")]
    parameters["band_indexes_in_bxsf"] = [
        int(_) for _ in parameters["band_indexes_in_bxsf"].split()
    ]
    parameters["bands_crossing_fermi"] = [
        int(_) for _ in parameters["bands_crossing_fermi"].split()
    ]
    float_keys = [
        "smearing_width",
        "tol_n_electrons_initial",
        "tol_n_electrons_final",
        "fermi_energy_in_bxsf",
        "fermi_energy_computed",
        "closest_eigenvalue_below_fermi",
        "closest_eigenvalue_above_fermi",
    ]
    for key in float_keys:
        parameters[key] = float(parameters[key])
    parameters["fermi_energy_unit"] = parameters["fermi_energy_unit"]
    parameters["smearing_type"] = parameters["smearing_type"]
    parameters["num_bands"] = int(parameters["num_bands"])
    parameters["num_electrons"] = int(parameters["num_electrons"])
    parameters["occupation_prefactor"] = int(parameters["occupation_prefactor"])

    # make sure the order is the same as parameters["band_indexes_in_bxsf"]
    parameters["band_min"] = [
        band_minmax[_][0] for _ in parameters["band_indexes_in_bxsf"]
    ]
    parameters["band_max"] = [
        band_minmax[_][1] for _ in parameters["band_indexes_in_bxsf"]
    ]

    return {
        "output_parameters": Dict(parameters)
        # "fermi_energy": Float(parameters["fermi_energy"]),
        # "vbm": Float(parameters["vbm"]),
        # "cbm": Float(parameters["cbm"]),
        # "band_gap_eV": Float(parameters["band_gap_eV"]),
    }


def submit_job(
    bxsf, num_electrons, code
):  # pylint: disable=redefined-outer-name, missing-function-docstring
    bxsf_filename = "bxsf.7z"
    _, calc = launch_shell_job(  # pylint: disable=redefined-outer-name
        command=code,
        arguments=[
            bxsf_filename,
            f"-n {num_electrons}",
        ],  # "2>/dev/null"
        submit=True,
        parser=parse_output,
        nodes={
            "bxsf": bxsf,
        },
        metadata={
            "options": {
                "redirect_stderr": True,
                "use_symlinks": True,
                "resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1},
                # "resources": {"num_cores": 1, "num_mpiprocs": 1},
            }
        },
    )

    # print(f"Submitted job {calc.pk}")
    return calc


@click.command()
@click.option(
    "-r",
    "--run/--dry-run",
    default=False,
    show_default=True,
    help="Do the actual submission or only print builder",
)
@click.option(
    "-m",
    "--max_concurrent",
    type=int,
    default=1,
    show_default=True,
    help="Max number of concurrent running workflows",
)
def cmd_group(max_concurrent, run):
    """Submit SkeafWorkChain from a group of nodes with submission controller.

    Allowed parent nodes: WannierjlCalculation
    """
    from aiida_fermisurface.calculations.stash import unstash

    parent_group_label = "workchain/PBEsol/wannier/lumi/final/bxsf/cubic_HT_test"
    group_label = (
        "workchain/PBEsol/wannier/lumi/final/bxsf/cubic_HT_test/fermi_from_bxsf"
    )

    code = orm.load_code("compute-Fermi@localhost-slurm")

    qb = orm.QueryBuilder()
    qb.append(orm.Group, filters={"label": {"==": group_label}}, tag="group")
    qb.append(
        orm.CalcJobNode,
        with_group="group",
        filters={
            "or": [{"attributes.sealed": False}, {"attributes": {"!has_key": "sealed"}}]
        },
    )
    num_active_slots = qb.count()
    num_available_slots = max_concurrent - num_active_slots
    if num_available_slots < 0:
        click.echo(
            f"Number of active jobs {num_active_slots} exceeds max_concurrent {max_concurrent}"
        )
        return
    elif num_available_slots == 0:
        click.echo(f"No slots available")
        return
    else:
        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={"label": {"==": group_label}}, tag="group")
        qb.append(
            orm.CalcJobNode,
            with_group="group",
            filters={"attributes.sealed": True},
        )
        already_done = qb.count()
        print("Max concurrent :", max_concurrent)
        print("Active slots   :", num_active_slots)
        print("Available slots:", num_available_slots)
        print("Already done   :", already_done)
        print()

        print("Submitting...")
        # ShellJob.input.nodes.bxsf
        qb = orm.QueryBuilder()
        qb.append(orm.Group, filters={"label": {"==": group_label}}, tag="group")
        # qb.append(orm.CalcJobNode, with_group="group", project=["extras.bxsf_pk"])
        qb.append(orm.CalcJobNode, with_group="group", project=["extras.wjl_pk"])
        qb_parent = orm.QueryBuilder()
        qb_parent.append(
            orm.Group, filters={"label": {"==": parent_group_label}}, tag="parent_group"
        )
        qb_parent.append(
            orm.CalcJobNode,
            with_group="parent_group",
            filters={
                "and": [
                    {"attributes.process_state": {"==": "finished"}},
                    {"id": {"!in": [_[0] for _ in qb.all()]}},
                ]
            },
            tag="wjlcalc",
        )
        # qb_parent.append(
        #     orm.RemoteData,
        #     with_incoming="wjlcalc",
        #     filters={
        #         "and": [
        #             {"extras": {"!has_key": "stash_mode"}},
        #             {"id": {"!in": [_[0] for _ in qb.all()]}},
        #         ]
        #     },
        # )
        qb_parent.limit(num_available_slots)
        to_submit = qb_parent.all()
        group = orm.load_group(group_label)
        for parent in to_submit:
            parent = parent[0]
            stash_folder = parent.outputs.remote_stash
            parent_folder = unstash(stash_folder, code.computer)
            w90_calc = parent.inputs.parent_folder.creator.inputs.stash_data.creator
            w90_params = w90_calc.inputs.parameters.get_dict()
            structure = w90_calc.inputs.structure
            num_electrons = get_num_electrons(structure, w90_params)
            bxsf = parent_folder
            if run:
                calc = submit_job(bxsf, num_electrons, code)
                print(f"WannierjlCalculation {parent.uuid} --> ShellJob {calc.uuid}")
                calc.base.extras.set("wjl_pk", parent.pk)
                calc.base.extras.set("bxsf_uuid", bxsf.uuid)
                calc.base.extras.set("bxsf_pk", bxsf.pk)
                group.add_nodes(calc)
            else:
                print(f"WannierjlCalculation {parent.uuid} --> ...")
        if run:
            print(f"Added {len(to_submit)} jobs to the group {group.label}\n\n")


if __name__ == "__main__":
    # cmd_group()
    from aiida_fermisurface.calculations.stash import unstash

    parent_group_label = "workchain/PBEsol/wannier/lumi/final/bxsf/cubic_HT_test"
    group = orm.load_group(
        "workchain/PBEsol/wannier/lumi/final/bxsf/cubic_HT_test/fermi_from_bxsf"
    )
    code = orm.load_code("compute-Fermi@localhost-slurm")

    parent = orm.load_group(parent_group_label).nodes[0]
    stash_folder = parent.outputs.remote_stash
    parent_folder = unstash(stash_folder, code.computer)
    w90_calc = parent.inputs.parent_folder.creator.inputs.stash_data.creator
    w90_params = w90_calc.inputs.parameters.get_dict()
    structure = w90_calc.inputs.structure
    num_electrons = get_num_electrons(structure, w90_params)
    bxsf = parent_folder

    calc = submit_job(bxsf, num_electrons, code)
    print(f"WannierjlCalculation {parent.uuid} --> ShellJob {calc.uuid}")
    calc.base.extras.set("wjl_pk", parent.pk)
    calc.base.extras.set("bxsf_uuid", bxsf.uuid)
    calc.base.extras.set("bxsf_pk", bxsf.pk)
    group.add_nodes(calc)
    print(f"Added job to the group {group.label}")
