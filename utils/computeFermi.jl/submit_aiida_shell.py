#!/usr/bin/env python
"""Python script to run compute_fermi.jl using aiida-shell."""
import re
from time import sleep

from aiida_shell import launch_shell_job
from tqdm import tqdm

from aiida import load_profile
from aiida.orm import load_code, load_group

load_profile("main")


def parse_output(self, dirpath):  # pylint: disable=unused-argument
    """Parse the output of compute_fermi.jl and return the Fermi energy."""
    from aiida.orm import Float, Int

    fermi = None
    n_kpoints = None
    re_Fermi = re.compile(
        r"Fermi Energy after interpolation:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
    )
    re_num_kpoints = re.compile(
        r"εF at iteration\s*([0-9]+)\s*:\s*([+-]?(?:[0-9]*[.])?[0-9]+)\s*eV, n_kpoints =\s*([0-9]+), ΔεF =\s*([+-]?(?:[0-9]*[.])?[0-9]+)e*([+-]?(?:[0-9]*))\s*eV"  # pylint: disable=line-too-long
    )
    output = (dirpath / "stdout").read_text().strip()
    for line in output.split("\n"):
        match = re_Fermi.match(line.strip())
        if match:
            fermi = match.group(1)
            break
        match = re_num_kpoints.match(line.strip())
        if match:
            n_kpoints = match.group(3)
    if fermi is None:
        raise ValueError("Could not parse Fermi energy from the output.")
    if n_kpoints is None:
        raise ValueError("Could not parse number of kpoints from the output.")
    return {"fermi_energy": Float(fermi), "num_kpoints": Int(n_kpoints)}


if __name__ == "__main__":
    code = load_code("compute_fermi-jl@prnmarvelcompute5-hq")
    parent_group = load_group(
        "workchain/PBEsol/wannier/lumi/final/bxsf/test/kpoint_distance/0_02/wan2skeafjl"
    )
    group = load_group("workchain/PBEsol/wannier/lumi/final/bxsf/test/adaptive_mesh")
    tbdat_filename = "wan_tb_cube.7z"
    already_done = [n.extras["parent_folder"] for n in group.nodes if n.is_finished_ok]
    # nodes = [n for n in parent_group.nodes]
    for n in tqdm(parent_group.nodes):
        parent_folder = (
            n.inputs.bxsf.creator.inputs.stash_data.creator.inputs.parent_folder
        )
        if parent_folder.uuid in already_done:
            continue
        num_electrons = n.inputs.parameters.get_dict()["num_electrons"]
        _, calc = launch_shell_job(
            command=code,
            arguments=[tbdat_filename, f"-n {num_electrons}"],  # "2>/dev/null"
            submit=True,
            parser=parse_output,
            nodes={
                "remote_data": parent_folder,
            },
            metadata={
                "options": {
                    "redirect_stderr": True,
                    "use_symlinks": True,
                    "resources": {"num_cores": 1, "num_mpiprocs": 1},
                }
            },
        )
        calc.base.extras.set("parent_folder", parent_folder.uuid)
        # print(f"Submitted job {calc.pk}")
        group.add_nodes(calc)
        calc.base.extras.set("group_label", group.label)
        # print(f"Added calculation {calc.pk} to group {group.label}")
        sleep(2)
