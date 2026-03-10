"""
Calculations provided by aiida_skeaf.

Register calculations via the "aiida.calculations" entry point in setup.json.
"""
from aiida import orm
from aiida.common import datastructures
from aiida.engine import CalcJob


class SkeafCalculation(CalcJob):
    """
    AiiDA calculation plugin wrapping the SKEAF executable.

    Simple AiiDA plugin wrapper for 'diffing' two files.
    """

    _DEFAULT_INPUT_FILE = "config.in"
    _DEFAULT_INPUT_BXSF = "skeaf.bxsf"
    _DEFAULT_OUTPUT_FILE = "results_short.out"

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        super().define(spec)

        # set default values for AiiDA options
        spec.inputs["metadata"]["options"]["resources"].default = {
            "num_machines": 1,
            "num_mpiprocs_per_machine": 1,
        }
        spec.inputs["metadata"]["options"]["parser_name"].default = "skeaf.skeaf"

        # new ports
        spec.input(
            "metadata.options.input_filename",
            valid_type=str,
            default=cls._DEFAULT_INPUT_FILE,
        )
        spec.input(
            "metadata.options.output_filename",
            valid_type=str,
            default=cls._DEFAULT_OUTPUT_FILE,
        )
        spec.input(
            "parameters",
            valid_type=orm.Dict,
            serializer=orm.to_aiida_type,
            help="Input parameters for SKEAF",
        )
        spec.input(
            "bxsf",
            valid_type=orm.RemoteData,
            help="Input BXSF file.",
        )
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Output parameters.",
        )
        spec.output(
            "frequency",
            valid_type=orm.ArrayData,
            help="Output Frequency arrays.",
        )

        spec.exit_code(
            300,
            "ERROR_MISSING_OUTPUT_FILES",
            message="Calculation did not produce all expected output files.",
        )

    def prepare_for_submission(self, folder):
        """
        Create input files.

        :param folder: an `aiida.common.folders.Folder` where the plugin should temporarily place all files
            needed by the calculation.
        :return: `aiida.common.datastructures.CalcInfo` instance
        """
        codeinfo = datastructures.CodeInfo()
        # auto read `config.in`
        codeinfo.cmdline_params = ["-rdcfg"]
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.stdout_name = self.metadata.options.output_filename

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

        # symlink the input bxsf to skeaf.bxsf
        calcinfo.remote_symlink_list = [
            (
                self.inputs.bxsf.computer.uuid,
                self.inputs.bxsf.get_remote_path(),
                self._DEFAULT_INPUT_BXSF,
            ),
        ]

        with folder.open(self.metadata.options.input_filename, "w") as handle:
            input_filecontent = self.generate_input_filecontent()
            handle.write(input_filecontent)

        calcinfo.retrieve_list = [
            self.metadata.options.output_filename,
            "results_orbitoutlines_invAng.out",
            "results_freqvsangle.out",
        ]

        return calcinfo

    def generate_input_filecontent(self) -> str:
        """Generate the raw input file for skeaf executable."""
        from aiida_skeaf.data import InputParameters, SkeafParameters

        # validate input parameters
        params = InputParameters(self.inputs.parameters.get_dict()).get_dict()

        # add filename
        params["filename"] = self._DEFAULT_INPUT_BXSF

        angle_iso_convention = params.pop("angle_iso_convention")
        if angle_iso_convention:
            #
            theta = params["theta"]
            phi = params["phi"]
            params["theta"] = phi
            params["phi"] = theta
            #
            theta = params["starting_theta"]
            phi = params["starting_phi"]
            params["starting_theta"] = phi
            params["starting_phi"] = theta
            #
            theta = params["ending_theta"]
            phi = params["ending_phi"]
            params["ending_theta"] = phi
            params["ending_phi"] = theta

        # unit conversion, constants from QE/Modules/Constants.f90
        ELECTRONVOLT_SI = 1.602176634e-19
        HARTREE_SI = 4.3597447222071e-18
        RYDBERG_SI = HARTREE_SI / 2.0

        convert_fermi_energy = params.pop("convert_fermi_energy_eV_to_Ry")
        if convert_fermi_energy:
            params["fermi_energy"] = params["fermi_energy"] * (
                ELECTRONVOLT_SI / RYDBERG_SI
            )

        # generate the raw input for skeaf
        params = SkeafParameters(params).generate()

        return params
