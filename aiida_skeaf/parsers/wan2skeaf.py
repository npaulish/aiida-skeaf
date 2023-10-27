"""
Parsers provided by aiida_skeaf.

Register parsers via the "aiida.parsers" entry point in setup.json.
"""
import pathlib
import re
import typing as ty

from aiida import orm
from aiida.common import exceptions
from aiida.engine import ExitCode
from aiida.parsers.parser import Parser

from aiida_skeaf.calculations.wan2skeaf import Wan2skeafCalculation


class BXSFFileNotFoundError(
    Exception
):  # Should be inherited from aiida.common.exceptions.NotExistent?
    """Raised when BXSF file is not found."""


class JobNotFinishedError(
    Exception
):  # Should be inherited from aiida.common.exceptions?
    """Raised when wan2skeaf job is not finished and end timestamp is not in the output."""


class Wan2skeafParser(Parser):
    """
    Parser class for parsing output of ``wan2skeaf.py``.
    """

    def __init__(self, node):
        """
        Initialize Parser instance

        Checks that the ProcessNode being passed was produced by a SkeafCalculation.

        :param node: ProcessNode of calculation
        :param type node: :class:`aiida.orm.ProcessNode`
        """
        super().__init__(node)
        if not issubclass(node.process_class, Wan2skeafCalculation):
            raise exceptions.ParsingError("Can only parse Wan2skeafCalculation")

    def parse(self, **kwargs):
        """
        Parse outputs, store results in database.

        :returns: an exit code, if parsing fails (or nothing if parsing succeeds)
        """
        output_filename = self.node.get_option("output_filename")

        # Check that folder content is as expected
        files_retrieved = self.retrieved.list_object_names()
        files_expected = [
            Wan2skeafCalculation._DEFAULT_OUTPUT_FILE,  # pylint: disable=protected-access
        ]
        # Note: set(A) <= set(B) checks whether A is a subset of B
        if not set(files_expected) <= set(files_retrieved):
            self.logger.error(
                f"Found files '{files_retrieved}', expected to find '{files_expected}'"
            )
            return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

        # parse `wan2skeaf.out`
        self.logger.info(f"Parsing '{output_filename}'")
        with self.retrieved.open(output_filename, "r") as handle:
            try:
                output_node = parse_wan2skeaf_out(handle.readlines())
            except BXSFFileNotFoundError as exc:
                self.logger.error(f"File not found: {exc}")
                return self.exit_codes.ERROR_MISSING_INPUT_FILE
            except JobNotFinishedError as exc:
                self.logger.error(f"Calculation not finished: {exc}")
                return self.exit_codes.ERROR_JOB_NOT_FINISHED
            except KeyError as exc:
                self.logger.error(f"Failed to parse '{output_filename}': {exc}")
                return self.exit_codes.ERROR_PARSING_OUTPUT

        self.out("output_parameters", output_node)

        band_indexes_in_bxsf = output_node.get_dict().get("band_indexes_in_bxsf")

        # attach RemoteData for extracted bxsf
        self.logger.info("Attaching extracted bxsf files")
        self.attach_bxsf_files(band_indexes_in_bxsf)

        return ExitCode(0)

    def attach_bxsf_files(  # pylint: disable=inconsistent-return-statements
        self, band_indexes_in_bxsf
    ):
        """Attach RemoteData for extracted bxsf."""

        input_params = self.node.inputs["parameters"].get_dict()
        input_band_index = input_params.get("band_index", -1)

        if input_band_index == -1:
            indexes = band_indexes_in_bxsf
        else:
            indexes = [input_band_index]

        remote_folder = self.node.outputs.remote_folder
        remote_folder_path = pathlib.Path(remote_folder.get_remote_path())
        remote_files = remote_folder.listdir()
        # bxsf_filename = Wan2skeafCalculation._DEFAULT_OUTPUT_BXSF.replace(  # pylint: disable=protected-access
        #     ".bxsf", "_band_{:d}.bxsf"
        # )
        bxsf_filename = (
            Wan2skeafCalculation._DEFAULT_OUTPUT_BXSF
            + "_band_{:d}.bxsf"  # pylint: disable=protected-access
        )

        for idx in indexes:
            filename = bxsf_filename.format(idx)

            if filename not in remote_files:
                self.logger.error(
                    f"Found files '{remote_files}' in remote_folder, expected to find '{filename}'"
                )
                return self.exit_codes.ERROR_MISSING_OUTPUT_FILES

            remote_path = str(remote_folder_path / filename)
            remote = orm.RemoteData(
                remote_path=remote_path,
                computer=remote_folder.computer,
            )
            self.out(f"output_bxsf.band{idx}", remote)

        return


def parse_wan2skeaf_out(filecontent: ty.List[str]) -> orm.Dict:
    """Parse `wan2skeaf.out`."""
    parameters = {
        "fermi_energy_unit": "eV",
    }

    regexs = {
        "input_file_not_found": re.compile(r"ERROR: Input file\s*(.+) does not exist."),
        "timestamp_started": re.compile(r"Started on\s*(.+)"),
        "fermi_energy_in_bxsf": re.compile(
            r"Fermi Energy from file:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "fermi_energy_computed": re.compile(
            r"Computed Fermi energy:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
        ),
        "fermi_energy_computed_eV": re.compile(
            r"Computed Fermi energy in eV:\s*([+-]?(?:[0-9]*[.])?[0-9]+)"
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
        "band_indexes_in_bxsf": re.compile(r"Bands in bxsf:\s*(.+)"),
        "timestamp_end": re.compile(r"Job done at\s*(.+)"),
    }
    re_band_minmax = re.compile(
        r"Min and max of band\s*([0-9]*)\s*:\s*([+-]?(?:[0-9]*[.])?[0-9]+)\s+([+-]?(?:[0-9]*[.])?[0-9]+)"
    )
    band_minmax = {}

    for line in filecontent:
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
    if "timestamp_end" not in parameters:
        raise JobNotFinishedError("Job not finished!")

    parameters["kpoint_mesh"] = [int(_) for _ in parameters["kpoint_mesh"].split("x")]
    parameters["band_indexes_in_bxsf"] = [
        int(_) for _ in parameters["band_indexes_in_bxsf"].split()
    ]
    parameters["fermi_energy_in_bxsf"] = float(parameters["fermi_energy_in_bxsf"])
    parameters["fermi_energy_computed"] = float(parameters["fermi_energy_computed"])
    parameters["fermi_energy_computed_eV"] = float(
        parameters["fermi_energy_computed_eV"]
    )
    parameters["fermi_energy_unit"] = parameters["fermi_energy_unit"]
    parameters["closest_eigenvalue_below_fermi"] = float(
        parameters["closest_eigenvalue_below_fermi"]
    )
    parameters["closest_eigenvalue_above_fermi"] = float(
        parameters["closest_eigenvalue_above_fermi"]
    )
    # make sure the order is the same as parameters["band_indexes_in_bxsf"]
    parameters["band_min"] = [
        band_minmax[_][0] for _ in parameters["band_indexes_in_bxsf"]
    ]
    parameters["band_max"] = [
        band_minmax[_][1] for _ in parameters["band_indexes_in_bxsf"]
    ]

    return orm.Dict(dict=parameters)
