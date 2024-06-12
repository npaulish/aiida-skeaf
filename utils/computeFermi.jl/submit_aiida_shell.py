#!/usr/bin/env python
"""Python script to run compute_fermi.jl using aiida-shell."""
import json
import re
from time import sleep

from aiida_shell import launch_shell_job
from tqdm import tqdm

from aiida import load_profile
from aiida.orm import load_code, load_group

load_profile("main")

RUN_FROM_BXSF = True


def parse_output(self, dirpath):  # pylint: disable=unused-argument
    """Parse the output of compute_fermi.jl and return the Fermi energy."""
    from aiida.orm import Dict, Float, Int

    parameters = {}
    n_kpoints = None
    re_num_kpoints = re.compile(
        r"εF at iteration\s*([0-9]+)\s*:\s*([+-]?(?:[0-9]*[.])?[0-9]+)\s*eV, n_kpoints =\s*([0-9]+), ΔεF =\s*([+-]?(?:[0-9]*[.])?[0-9]+)e*([+-]?(?:[0-9]*))\s*eV"  # pylint: disable=line-too-long
    )
    regexs = {
        "fermi_energy": re.compile(
            r"Fermi Energy after interpolation:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "vbm": re.compile(
            r"Valence band maximum on the interpolated k-point grid:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "cbm": re.compile(
            r"Conduction band minimum on the interpolated k-point grid:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "band_gap_eV": re.compile(
            r"Band gap on the interpolated k-point grid:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "start_time": re.compile(r"Started at \s*(.+)"),
        "end_time": re.compile(r"Finished at \s*(.+)"),
        "fermi_energy_from_bxsf": re.compile(
            r"Fermi Energy from bxsf:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "search_window_width": re.compile(
            r"Search window width (if < 0.5 eV, then 0.5 eV will be used):\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "max_band_velocity": re.compile(
            r"Max band velocity in the search window:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "min_band_velocity": re.compile(
            r"Min band velocity in the search window:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "num_bands": re.compile(r"Number of bands:\s*([0-9]+)"),
        "kpoint_mesh": re.compile(r"Grid shape:\s*(.+)"),
    }

    output = (dirpath / "stdout").read_text().strip()
    for line in output.split("\n"):
        for key, reg in regexs.items():
            match = reg.match(line.strip())
            if match:
                parameters[key] = match.group(1)
                regexs.pop(key, None)
                break

        match = re_num_kpoints.match(line.strip())
        if match:
            n_kpoints = match.group(3)
    if "fermi_energy" not in parameters:
        raise ValueError("Could not parse Fermi energy from the output.")
    if n_kpoints is None:
        raise ValueError("Could not parse number of kpoints from the output.")

    parameters["num_kpoints"] = int(n_kpoints)
    parameters["fermi_energy"] = float(parameters["fermi_energy"])
    parameters["vbm"] = float(parameters["vbm"])
    parameters["cbm"] = float(parameters["cbm"])
    parameters["band_gap_eV"] = float(parameters["band_gap_eV"])
    parameters["fermi_energy_from_bxsf"] = float(parameters["fermi_energy_from_bxsf"])
    parameters["search_window_width"] = float(parameters["search_window_width"])
    parameters["max_band_velocity"] = float(parameters["max_band_velocity"])
    parameters["min_band_velocity"] = float(parameters["min_band_velocity"])
    parameters["kpoint_mesh"] = [int(_) for _ in parameters["kpoint_mesh"].split("x")]

    return {
        "output_parameters": Dict(parameters),
        "num_kpoints": Int(n_kpoints),
        "fermi_energy": Float(parameters["fermi_energy"]),
        "vbm": Float(parameters["vbm"]),
        "cbm": Float(parameters["cbm"]),
        "band_gap_eV": Float(parameters["band_gap_eV"]),
    }


def submit_job(
    tb_model, num_electrons, bxsf=None
):  # pylint: disable=redefined-outer-name, missing-function-docstring
    if RUN_FROM_BXSF:
        code = load_code("compute-fermi-from-bxsf-jl@prnmarvelcompute5-hq")
    else:
        code = load_code("compute_fermi-jl@prnmarvelcompute5-hq")

    tbdat_filename = "wan_tb_cube.7z"  # pylint: disable=redefined-outer-name

    if RUN_FROM_BXSF:
        bxsf_filename = "bxsf.7z"
        _, calc = launch_shell_job(  # pylint: disable=redefined-outer-name
            command=code,
            arguments=[
                tbdat_filename,
                bxsf_filename,
                f"-n {num_electrons}",
            ],  # "2>/dev/null"
            submit=True,
            parser=parse_output,
            nodes={
                "tb_model": tb_model,
                "bxsf": bxsf,
            },
            metadata={
                "options": {
                    "redirect_stderr": True,
                    "use_symlinks": True,
                    "resources": {"num_cores": 4, "num_mpiprocs": 4},
                }
            },
        )
    else:
        _, calc = launch_shell_job(
            command=code,
            arguments=[tbdat_filename, f"-n {num_electrons}"],  # "2>/dev/null"
            submit=True,
            parser=parse_output,
            nodes={
                "tb_model": tb_model,
            },
            metadata={
                "options": {
                    "redirect_stderr": True,
                    "use_symlinks": True,
                    "resources": {"num_cores": 1, "num_mpiprocs": 1},
                }
            },
        )
    calc.base.extras.set("tb_model", tb_model.uuid)
    if RUN_FROM_BXSF:
        calc.base.extras.set("bxsf", bxsf.uuid)
    # print(f"Submitted job {calc.pk}")
    return calc


if __name__ == "__main__":
    parent_group = load_group(
        "workchain/PBEsol/wannier/lumi/final/bxsf/test/kpoint_distance/0_04/wan2skeaf"
    )
    group = load_group(
        "workchain/PBEsol/wannier/lumi/final/bxsf/test/compute_fermi_from_bxsf"
    )

    with open(  # pylint: disable=unspecified-encoding
        "/home/aiida/envs/aiida-fermisurf/data/kpoint_distance_test/test_set_metals.json"
    ) as f:
        test_set_metals = json.load(f)
    tbs_test = test_set_metals.keys()
    parent_nodes = [
        n
        for n in tqdm(parent_group.nodes)
        if n.inputs.bxsf.creator.inputs.stash_data.creator.inputs.parent_folder.uuid[
            0:8
        ]
        in tbs_test
    ]

    already_done = [n.extras["tb_model"] for n in group.nodes if n.is_finished_ok]

    tbdat_filename = "wan_tb_cube.7z"  # pylint: disable=redefined-outer-name
    # nodes = [n for n in parent_group.nodes]

    for n in tqdm(parent_nodes):
        tb_model = (
            n.inputs.bxsf.creator.inputs.stash_data.creator.inputs.parent_folder
        )  # pylint: disable=redefined-outer-name
        if tb_model.uuid in already_done:
            continue
        num_electrons = n.inputs.parameters.get_dict()[
            "num_electrons"
        ]  # pylint: disable=redefined-outer-name
        bxsf = n.inputs.bxsf  # pylint: disable=redefined-outer-name
        calc = submit_job(
            tb_model, num_electrons, bxsf
        )  # pylint: disable=redefined-outer-name
        group.add_nodes(calc)
        calc.base.extras.set("group_label", group.label)
        # print(f"Added calculation {calc.pk} to group {group.label}")
        sleep(2)
