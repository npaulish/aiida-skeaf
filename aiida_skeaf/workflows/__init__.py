#!/usr/bin/env python
"""Workchain to run `Wan2skeafCalculation` and `SkeafCalculation`.

Automatically calculate frequencies for a wannier90 generated multi-band BXSF file.
"""
import pathlib
import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ProcessBuilder, ToContext, WorkChain, append_

from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.protocols.utils import ProtocolMixin

from aiida_skeaf.calculations import SkeafCalculation, Wan2skeafCalculation
from aiida_skeaf.utils.str import removeprefix

__all__ = ["validate_inputs", "SkeafWorkChain"]


def validate_inputs(  # pylint: disable=inconsistent-return-statements
    inputs: AttributeDict, ctx=None
):  # pylint: disable=unused-argument
    """Validate the inputs of the entire input namespace."""
    # pylint: disable=no-member

    if "fermi_energy" in inputs["skeaf"]["parameters"]:
        if "fermi_energy" not in inputs["wan2skeaf"]["parameters"]:
            inputs["wan2skeaf"]["parameters"]["fermi_energy"] = inputs["skeaf"][
                "parameters"
            ]["fermi_energy"]
        else:
            if (
                inputs["wan2skeaf"]["parameters"]["fermi_energy"]
                != inputs["skeaf"]["parameters"]["fermi_energy"]
            ):
                return SkeafWorkChain.exit_codes.ERROR_INVALID_INPUT_FERMI.message
    elif "fermi_energy" in inputs["wan2skeaf"]["parameters"]:
        return SkeafWorkChain.exit_codes.ERROR_INVALID_INPUT_FERMI.message


class SkeafWorkChain(ProtocolMixin, WorkChain):
    """Workchain to run ``Wan2skeafCalculation`` and ``SkeafCalculation``.

    Given a wannier90 generated ``RemoteData`` containing a bxsf file,
    run skeaf on all the bands in the bxsf.
    """

    @classmethod
    def define(cls, spec) -> None:
        """Define the process spec."""

        super().define(spec)

        spec.input(
            "bxsf",
            valid_type=orm.RemoteData,
            help="Input BXSF file or the output remote_folder of Wannier90Calculation.",
        )
        spec.expose_inputs(
            Wan2skeafCalculation,
            namespace="wan2skeaf",
            exclude=("bxsf",),
            namespace_options={"help": "Inputs for the `Wan2skeafCalculation`."},
        )
        spec.expose_inputs(
            SkeafCalculation,
            namespace="skeaf",
            exclude=("bxsf",),
            namespace_options={"help": "Inputs for the `Wan2skeafCalculation`."},
        )
        spec.input(
            "clean_workdir",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            serializer=orm.to_aiida_type,
            help="If `True`, work directories of all called calculation jobs will be cleaned at the end of execution.",
        )

        spec.inputs.validator = validate_inputs

        spec.expose_outputs(Wan2skeafCalculation, namespace="wan2skeaf")
        spec.output_namespace(
            "skeaf",
            dynamic=True,
            help="Output SkeafCalculation for each band.",
        )

        spec.outline(
            cls.setup,
            cls.run_wan2skeaf,
            cls.inspect_wan2skeaf,
            cls.run_skeaf_many,
            cls.inspect_skeaf_many,
            cls.results,
        )

        spec.exit_code(
            401,
            "ERROR_SUB_PROCESS_FAILED_WAN2SKEAF",
            message="Unrecoverable error when running wan2skeaf.",
        )
        spec.exit_code(
            402,
            "ERROR_SUB_PROCESS_FAILED_SKEAF",
            message="Unrecoverable error when running skeaf.",
        )
        spec.exit_code(
            500,
            "ERROR_INVALID_INPUT_FERMI",
            message="Invalid input parameters. Fermi energy is not consistent between skeaf and wan2skeaf.",
        )

    @classmethod
    def get_protocol_filepath(cls) -> pathlib.Path:
        """Return the ``pathlib.Path`` to the ``.yaml`` file that defines the protocols."""
        from importlib_resources import files

        from . import protocols

        return files(protocols) / "skeaf.yaml"

    @classmethod
    def get_builder_from_protocol(  # pylint: disable=too-many-statements, too-many-arguments
        cls,
        codes: ty.Dict[str, ty.Union[orm.Code, str, int]],
        *,
        bxsf: orm.RemoteData,
        num_electrons: int,
        protocol: str = None,
        overrides: dict = None,
    ) -> ProcessBuilder:
        """Return a builder prepopulated with inputs selected according to the chosen protocol.

        :param codes: [description]
        :type codes: typing.Dict[str, typing.Union[aiida.orm.Code, str, int]]
        :param bxsf: [description]
        :type bxsf: aiida.orm.RemoteData
        :param protocol: [description], defaults to None
        :type protocol: str, optional
        :param overrides: [description], defaults to None
        :type overrides: dict, optional
        :return: [description]
        :rtype: aiida.engine.ProcessBuilder
        """
        from aiida_wannier90_workflows.utils.workflows.builder.submit import (
            recursive_merge_builder,
        )

        required_codes = ["wan2skeaf", "skeaf"]
        if not all(_ in codes for _ in required_codes):
            raise ValueError(f"`codes` must contain {required_codes}")

        for key, code in codes.items():
            if not isinstance(code, orm.Code):
                codes[key] = orm.load_code(code)

        type_check(bxsf, orm.RemoteData)

        inputs = cls.get_protocol_inputs(protocol, overrides)

        inputs["wan2skeaf"]["code"] = codes["wan2skeaf"]
        inputs["skeaf"]["code"] = codes["skeaf"]
        inputs["bxsf"] = bxsf

        wan2skeaf_parameters = inputs["wan2skeaf"]["parameters"]
        wan2skeaf_parameters["num_electrons"] = num_electrons
        inputs["wan2skeaf"]["parameters"] = orm.Dict(dict=wan2skeaf_parameters)

        builder = cls.get_builder()
        builder = recursive_merge_builder(builder, inputs)

        return builder

    def setup(self) -> None:
        """Create context variables."""

        # A dict for remaning bxsf RemoteData to run SkeafCalculation, e.g.
        # {'band49': RemoteData, 'band48': RemoteData}
        self.ctx.bxsf_to_run = {}
        # A list of band that have finished, e.g.
        # ['band48', 'band49']
        self.ctx.calc_skeaf_band_index = []
        # the skeaf calculations
        self.ctx.calc_skeaf = []

    def run_wan2skeaf(self):
        """Run the `Wan2skeafCalculation`."""
        inputs = AttributeDict(
            self.exposed_inputs(Wan2skeafCalculation, namespace="wan2skeaf")
        )

        inputs.bxsf = self.inputs.bxsf
        inputs.metadata.call_link_label = "wan2skeaf"

        inputs = prepare_process_inputs(Wan2skeafCalculation, inputs)
        running = self.submit(Wan2skeafCalculation, **inputs)
        self.report(f"launching {running.process_label}<{running.pk}>")

        return ToContext(calc_wan2skeaf=running)

    def inspect_wan2skeaf(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `Wan2skeafCalculation` successfully finished."""
        calc = self.ctx.calc_wan2skeaf

        if not calc.is_finished_ok:
            self.report(
                f"{calc.process_label} failed with exit status {calc.exit_status}"
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WAN2SKEAF

        self.ctx.bxsf_to_run = dict(calc.outputs.output_bxsf)

    def run_skeaf_many(self):  # pylint: disable=inconsistent-return-statements
        """Run multiple ``SkeafCalculation`` for all bxsf ``RemoteData``."""
        inputs = AttributeDict(self.exposed_inputs(SkeafCalculation, namespace="skeaf"))

        parameters = inputs.parameters.get_dict()
        w2s_output_params = self.ctx.calc_wan2skeaf.outputs[
            "output_parameters"
        ].get_dict()
        if "fermi_energy" not in parameters:
            parameters["fermi_energy"] = w2s_output_params["fermi_energy_computed"]
            inputs.parameters = orm.Dict(dict=parameters)

        fermi_energy = parameters["fermi_energy"]

        # Launch many SkeafCalculation in parallel
        # Run skeaf with band index as order, from min to max.
        bxsf_to_run = self.ctx.bxsf_to_run
        bxsf_to_run = dict(sorted(bxsf_to_run.items(), key=lambda _: _[0]))

        for band_idx, bxsf in bxsf_to_run.items():
            # Only calculate bands acrossing Fermi
            # idx = int(band_idx.removeprefix("band"))
            idx = int(removeprefix(band_idx, "band"))
            idx = w2s_output_params["band_indexes_in_bxsf"].index(idx)
            band_min = w2s_output_params["band_min"][idx]
            band_max = w2s_output_params["band_max"][idx]
            if band_min > fermi_energy or band_max < fermi_energy:
                continue

            inputs.bxsf = bxsf
            inputs.metadata.call_link_label = f"skeaf_{band_idx}"

            inputs = prepare_process_inputs(SkeafCalculation, inputs)
            future = self.submit(SkeafCalculation, **inputs)
            self.to_context(calc_skeaf=append_(future))
            # I need to store the corresponding band index, otherwise I don't
            # know the band index of the SkeafCalculation in self.ctx.calc_skeaf
            self.ctx.calc_skeaf_band_index.append(band_idx)

            self.report(f"launching {future.process_label}<{future.pk}> for {band_idx}")

    def inspect_skeaf_many(self):  # pylint: disable=inconsistent-return-statements
        """Verify that the `SkeafCalculation` successfully finished."""
        for calc in self.ctx.calc_skeaf:
            if not calc.is_finished_ok:
                self.report(
                    f"{calc.process_label} failed with exit status {calc.exit_status}"
                )
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SKEAF

    def results(self):
        """Attach the relevant output nodes."""

        self.out_many(
            self.exposed_outputs(
                self.ctx.calc_wan2skeaf,
                Wan2skeafCalculation,
                namespace="wan2skeaf",
            )
        )

        # essentially a dot `.`
        namespace_separator = self.spec().namespace_separator

        for i, band in enumerate(self.ctx.calc_skeaf_band_index):
            calc = self.ctx.calc_skeaf[i]
            outputs = calc.outputs._construct_attribute_dict(  # pylint: disable=protected-access
                incoming=False
            )
            self.out(f"skeaf{namespace_separator}{band}", outputs)

        self.report(f"{self.get_name()} successfully completed")

    def on_terminated(self):
        """Clean the working directories of all child calculations if `clean_workdir=True` in the inputs."""
        super().on_terminated()

        if not self.inputs.clean_workdir:
            self.report("remote folders will not be cleaned")
            return

        cleaned_calcs = []

        for called_descendant in self.node.called_descendants:
            if isinstance(called_descendant, orm.CalcJobNode):
                try:
                    called_descendant.outputs.remote_folder._clean()  # pylint: disable=protected-access
                    cleaned_calcs.append(called_descendant.pk)
                except (OSError, KeyError):
                    pass

        if cleaned_calcs:
            self.report(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )
