"""Python script to run compute_fermi.jl using aiida-shell."""
import json
import re

from aiida_shell import launch_shell_job
from tqdm import tqdm

from aiida.orm import RemoteData, load_group


def parse_output(output: str):
    """Parse the output of compute_fermi.jl and return the Fermi energy."""
    fermi = None
    n_kpoints = None
    re_Fermi = re.compile(
        r"Fermi Energy after interpolation:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
    )
    re_num_kpoints = re.compile(
        r"εF at iteration\s*([0-9]+)\s*:\s*([+-]?(?:[0-9]*[.])?[0-9]+)\s*eV, n_kpoints =\s*([0-9]+), ΔεF =\s*([+-]?(?:[0-9]*[.])?[0-9]+)e*([+-]?(?:[0-9]*))\s*eV"  # pylint: disable=line-too-long
    )
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
    return float(fermi), int(n_kpoints)


def compute_fermi(path: str, remote_folder: RemoteData, filename: str, num_e: int):
    """Run compute_fermi.jl and return the Fermi energy and number of kpoints."""
    results, node = launch_shell_job(
        path,
        arguments=[filename, f"-n {num_e}"],  # "2>/dev/null"
        nodes={
            "remote_data": remote_folder,
        },
    )
    output = results["stdout"].get_content()
    fermi, n_kpoints = parse_output(output)
    try:
        node.outputs.remote_folder._clean()  # pylint: disable=protected-access
    except (OSError, KeyError):
        pass
    return fermi, n_kpoints


if __name__ == "__main__":
    script_path = "/home/aiida/envs/aiida-fermisurf/code/aiida-skeaf/utils/computeFermi.jl/compute_fermi.jl"  # pylint: disable=redefined-outer-name
    group = load_group(
        "workchain/PBEsol/wannier/lumi/final/bxsf/test/kpoint_distance/0_02/wan2skeafjl"
    )
    tbdat_filename = "wan_tb_cube.7z"
    data = {}
    for n in tqdm(group.nodes):
        parent_folder = (
            n.inputs.bxsf.creator.inputs.stash_data.creator.inputs.parent_folder
        )
        num_electrons = n.inputs.parameters.get_dict()["num_electrons"]
        fermi_energy, num_kpoints = compute_fermi(
            script_path, parent_folder, tbdat_filename, num_electrons
        )
        parent_folder_uuid = parent_folder.uuid[0:8]
        data[parent_folder_uuid] = {
            "fermi_energy": fermi_energy,
            "num_kpoints": num_kpoints,
        }
    with open(
        "/home/aiida/envs/aiida-fermisurf/data/kpoint_distance_test/fermi_energy_kpoints_adaptive_mesh.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(data, f, indent=4)
