""" Tests for calculations."""
from aiida import orm
from aiida.common import datastructures
from aiida.plugins import CalculationFactory

# from . import TEST_DIR


def test_create_bxsf_from_wannier90(aiida_localhost, tmp_path):
    """Test running a ``create_bxsf_from_wannier90``."""

    # Prepare input
    bxsf_filename = "aiida.bxsf"
    bxsf_file = tmp_path / bxsf_filename
    bxsf_file.write_text("w90 bxsf")
    remote_folder = orm.RemoteData(remote_path=str(tmp_path), computer=aiida_localhost)
    assert remote_folder.get_remote_path() == str(tmp_path)
    assert remote_folder.listdir() == [
        bxsf_filename,
    ]

    create_bxsf_from_wannier90 = CalculationFactory("skeaf.create_bxsf_from_wannier90")
    result = create_bxsf_from_wannier90(remote_folder)

    assert isinstance(result, orm.RemoteData)
    assert result.computer.label == aiida_localhost.label
    assert result.get_remote_path() == str(bxsf_file)


def test_create_bxsf_from_file(aiida_localhost, tmp_path):
    """Test running a ``create_bxsf_from_file``."""

    # Prepare input
    bxsf_filename = "aiida.bxsf"
    bxsf_file = tmp_path / bxsf_filename
    bxsf_file.write_text("w90 bxsf")
    remote_path = orm.Str(str(bxsf_file))
    computer = orm.Str(aiida_localhost.label)

    create_bxsf_from_file = CalculationFactory("skeaf.create_bxsf_from_file")
    result = create_bxsf_from_file(remote_path=remote_path, computer=computer)

    assert isinstance(result, orm.RemoteData)
    assert result.computer.label == aiida_localhost.label
    assert result.get_remote_path() == str(bxsf_file)


def test_skeaf(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    skeaf_code,
    generate_bxsf_remotedata,
    generate_input_parameters,
    generate_calc_job,
    fixture_sandbox,
    file_regression,
):
    """Test running a calculation
    note this does not test that the expected outputs are created of output parsing"""

    # Prepare input parameters
    parameters = generate_input_parameters()

    bxsf = generate_bxsf_remotedata

    # set up calculation
    inputs = {
        "code": skeaf_code,
        "parameters": parameters,
        "bxsf": bxsf,
        "metadata": {
            "options": {
                "max_wallclock_seconds": 30,
                "withmpi": False,
            },
        },
    }

    entry_point_name = "skeaf.skeaf"
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    cmdline_params = ["-rdcfg"]
    retrieve_list = [
        "results_short.out",
        "results_orbitoutlines_invAng.out",
        "results_freqvsangle.out",
    ]

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert sorted(calc_info.codes_info[0].cmdline_params) == cmdline_params
    assert sorted(calc_info.retrieve_list) == sorted(retrieve_list)

    with fixture_sandbox.open("config.in") as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    assert sorted(fixture_sandbox.get_content_list()) == sorted(["config.in"])
    file_regression.check(input_written, encoding="utf-8", extension=".in")
