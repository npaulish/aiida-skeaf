"""
Calculations provided by aiida_skeaf.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""
import pathlib

from voluptuous import Any, Optional, Required, Schema  # pylint: disable=unused-import

from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob


class Wan2skeafCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapping the ``wan2skeaf.py``.
    """

    _DEFAULT_INPUT_BXSF = "input.bxsf"
    _DEFAULT_OUTPUT_FILE = "wan2skeaf.out"
    _DEFAULT_OUTPUT_BXSF = "output"

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["parser_name"].default = "skeaf.wan2skeaf"

        # new ports
        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls._DEFAULT_OUTPUT_FILE,
        )
        spec.input(
            "parameters",
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            help="Input parameters for wan2skeaf.py",
        )
        spec.input(
            "bxsf",
            valid_type=orm.RemoteData,
            help="Input BXSF file.",
        )
        spec.input(
            "bxsf_filename",
            valid_type=orm.Str,
            serializer=orm.to_aiida_type,
            help="Input BXSF filename of the RemoteData.",
        )
        spec.input(
            "settings.autolink_bxsf_filename",
            valid_type=orm.Str,
            default=lambda: orm.Str(cls._DEFAULT_INPUT_BXSF),
            serializer=orm.to_aiida_type,
            help=(
                "Automatically create a symlink from the `inputs.bxsf_filename` in the "
                "`bxsf` RemoteData to a file with this name, which is used "
                "as the input file for `wan2skeaf.py`. By default it is `input.bxsf`, "
                "If different from that, you must make sure there is an `input.bxsf` "
                "before running `wan2skeaf.py`."
            ),
        )
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Output parameters.",
        )
        spec.output_namespace(
            "output_bxsf",
            dynamic=True,
            valid_type=orm.RemoteData,
            help="Output bxsf for each band.",
        )

        spec.exit_code(
            300,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Calculation did not produce all expected output files.",
        )

        spec.exit_code(
            301,
            "ERROR_PARSING_OUTPUT",
            message="Parsing output failed.",
        )

        spec.exit_code(
            302,
            "ERROR_MISSING_INPUT_FILE",
            message="Input file is missing.",
        )

        spec.exit_code(
            303,
            "ERROR_JOB_NOT_FINISHED",
            message="Calculation did not finish correctly.",
        )

    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        codeinfo = datastructures.CodeInfo()

        # validate input parameters
        parameters = InputParameters(self.inputs.parameters.get_dict()).get_dict()
        cmdline_params = [
            "-n",
            parameters["num_electrons"],
            "-b",
            parameters["band_index"],
            "-o",
            self._DEFAULT_OUTPUT_BXSF,
        ]
        if "num_spin" in parameters:
            cmdline_params += ["--num_spin", parameters["num_spin"]]
        if "smearing_type" in parameters and "smearing_value" in parameters:
            cmdline_params += ["-s", parameters["smearing_type"]]
            cmdline_params += ["-w", parameters["smearing_value"]]

        cmdline_params.append(self.inputs.bxsf_filename.value)
        #
        codeinfo.cmdline_params = cmdline_params
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.metadata.options.output_filename
        # codeinfo.withmpi = self.inputs.metadata.options.withmpi
        codeinfo.withmpi = False

        # Prepare a `CalcInfo` to be returned to the engine
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        # calcinfo.local_copy_list = [
        #     (
        #         self.inputs.file1.uuid,
        #         self.inputs.file1.filename,
        #         self.inputs.file1.filename,
        #     ),
        # ]

        # symlink the input bxsf
        remote_path = self.inputs.bxsf.get_remote_path()
        # if input bxsf RemoteData is generated by Wannier90Calculation,
        # it should contain a aiida.bxsf file
        bxsf_file = self.inputs.bxsf_filename.value
        calcinfo.remote_symlink_list = [
            (
                self.inputs.bxsf.computer.uuid,
                str(pathlib.Path(remote_path) / bxsf_file),
                self.inputs.settings.autolink_bxsf_filename.value,
            ),
        ]

        calcinfo.retrieve_list = [
            self.metadata.options.output_filename,
        ]

        return calcinfo


input_parameters = {
    Required("num_electrons"): int,
    Optional("num_spin"): int,
    Optional("band_index", default=-1): int,
    Optional("smearing_type"): str,
    Optional("smearing_value"): float,
}


class InputParameters:  # pylint: disable=too-many-ancestors
    """
    Command line options for diff.

    This class represents a python dictionary used to
    pass command line options to the executable.
    """

    # "voluptuous" schema to add automatic validation
    schema = Schema(input_parameters)

    # pylint: disable=redefined-builtin
    def __init__(self, dict, /):
        """
        Constructor for the data class

        Usage: ``DiffParameters(dict{'ignore-case': True})``

        :param parameters_dict: dictionary with commandline parameters
        :param type parameters_dict: dict

        """
        self.dict = self.validate(dict)

    def validate(self, parameters_dict):
        """Validate command line options.

        Uses the voluptuous package for validation. Find out about allowed keys using::

            print(DiffParameters).schema.schema

        :param parameters_dict: dictionary with commandline parameters
        :param type parameters_dict: dict
        :returns: validated dictionary
        """
        return self.schema(parameters_dict)

    def get_dict(self) -> dict:
        """Return validated dict."""
        return self.dict
