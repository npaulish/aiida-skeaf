#!/usr/bin/env -S julia --project=/home/paulis_n/Software/aiida-fermisurface/code/aiida-skeaf/utils/wan2skeaf.jl
#
# To use this script, one need to frist instantiate the julia environment, by running
#   julia --project=.
#   ]  # enter pkg mode
#   instantiate
#
# Note that replace the abspath after `--project` in the shebang by the dir path
# where you put this script.
#
# Example usage: ./wan2skeaf.jl wjl.bxsf -n 17
using Pkg
Pkg.instantiate()

using Printf
using Dates
using WannierIO
using Wannier
using Comonicon
using p7zip_jll

"""
Prepare bxsf files for skeaf.

The script
- computes Fermi energy from the input bxsf
- split the input bxsf into several bxsf files, each contains only one band

# Args

- `bxsf`: input bxsf that may contain multiple bands

# Options

- `-n, --num_electrons`: number of electrons
- `-b, --band_index`: band index, default is -1 (all bands)
- `-o, --out_filename`: output filename prefix
- `-s, --smearing_type`: smearing type, default is "none" (corresponding to `NoneSmearing()`, no smearing), other options are "fermi-dirac" or "fd", "marzari-vanderbilt" or "cold"
- `-w, --width_smearing`: smearing width, default is 0.0
- `-p, --prefactor`: occupation prefactor, 2 for non SOC, 1 for SOC, default is 2
- `-t, --tol_n_electrons`: tolerance for number of electrons, default is 1e-6
- `-f, --fermi_energy`: custom Fermi energy, default is none
"""
@main function main(
    bxsf::String;
    num_electrons::Int,
    band_index::Int=-1,
    out_filename::String="skeaf",
    smearing_type::String="none",
    width_smearing::Float64=0.0,
    prefactor::Int=2,
    tol_n_electrons::Float64=1e-6,
    fermi_energy::String="none"
    )
    println("Started on ", Dates.now())
    if !isfile(bxsf)
        println("ERROR: input file $bxsf does not exist.")
        exit(2)
    end
    if endswith(bxsf, ".7z")
        # -y: assume yes, will overwrite existing unzipped files
        ret = run(`$(p7zip()) x -y $bxsf`)
        success(ret) || error("error while unzipping file: $bxsf")
        bxsf_files = filter(x -> endswith(x, ".bxsf"), readdir("."))
        if length(bxsf_files) != 1
            error("expecting only one .bxsf file in the 7z file, but got $(length(bxsf_files))")
        end
        bxsf_filename = bxsf_files[1]
        dst_filename = "input.bxsf"
        mv(bxsf_filename, dst_filename) # not sure this is needed! skeaf reads the output file?
        bxsf = dst_filename
    end
    println("Number of electrons: ", num_electrons)
    bxsf = WannierIO.read_bxsf(bxsf)
    @printf("Fermi Energy from file: %.8f\n", bxsf.fermi_energy)

    nbands, nkx, nky, nkz = size(bxsf.E)
    println("Number of bands: ", nbands)
    println("Grid shape: $nkx x $nky x $nkz")

    kBT = width_smearing
    if smearing_type == "none"
        smearing = Wannier.NoneSmearing()
    elseif smearing_type == "fermi-dirac" || smearing_type == "fd"
        smearing = Wannier.FermiDiracSmearing()
    elseif smearing_type == "marzari-vanderbilt" || smearing_type == "cold"
        smearing = Wannier.ColdSmearing()
    else
        error("unknown smearing type: $smearing_type")
    end

    println("Smearing type: ", smearing_type)
    println("Smearing width: ", width_smearing)
    println("Occupation prefactor: ", prefactor)
    println("Initial tolerance for number of electrons (default 1e-6): ", tol_n_electrons)
    parsed_fermi_energy = fermi_energy == "none" ? nothing : tryparse(Float64, fermi_energy)
    if !isnothing(parsed_fermi_energy)
        println("Custom Fermi energy  will be used to select the bands that are written to separate bxsfs: ", parsed_fermi_energy)
    end

    # some times, w/o smearing, the number of electrons cannot be integrated to
    # the exact number of electrons, since we only have discrete eigenvalues.
    # To bypass such error, one can only decrease the check on number of electrons.
    # However, in general we should use a (very) small smearing so it could almost
    # always possible to integrate to the exact integer for number of electrons.
    # tol_n_electrons = 1e-3
    # this is the default
    # tol_n_electrons = 1e-6

    # note that according to bxsf specification, the eigenvalues are defined
    # on "general grid" instead of "periodic grid", i.e., the right BZ edges are
    # repetitions of the left edges. We remove the redundance here, otherwise,
    # the computed Fermi is wrong.

    eigenvalues = [bxsf.E[:, i, j, k] for i in 1:(nkx-1) for j in 1:(nky-1) for k in 1:(nkz-1)]

    # unit conversion, constants from QE/Modules/Constants.f90
    ELECTRONVOLT_SI  = 1.602176634E-19
    HARTREE_SI       = 4.3597447222071E-18
    RYDBERG_SI       = HARTREE_SI/2.0
    BOHR_TO_ANG   = 0.529177210903

    εF_bxsf = 0.0
    tol_n_electrons_upperbound = 1e-3
    if tol_n_electrons > tol_n_electrons_upperbound
        println("Warning: tolerance for number of electrons is large, > 1e-3, which may lead to inaccurate Fermi energy.")
        tol_n_electrons_upperbound = tol_n_electrons
    end
    tol_n_e_curr = tol_n_electrons

    # try to compute Fermi energy with increasing tolerance for number of electrons
    while tol_n_e_curr <= tol_n_electrons_upperbound
        println("Current tolerance for number of electrons: ", tol_n_e_curr)
        try
            εF_bxsf = Wannier.compute_fermi_energy(eigenvalues, num_electrons, kBT, smearing; tol_n_electrons=tol_n_e_curr, prefactor=prefactor)
            break
        catch e
            println("Error: ", e.msg)
            if startswith(e.msg, "Failed to find Fermi energy within tolerance")
                println("   Increasing tolerance for number of electrons by a factor of 2...")
                tol_n_e_curr = tol_n_e_curr*2
                continue
            end
            exit(3)
        end
    end
    if tol_n_e_curr > tol_n_electrons_upperbound
        println("Error: tolerance for number of electrons exceeded the tol_n_electrons_upperbound. Exiting...")
        exit(3)
    end
    @printf("Computed Fermi energy: %.8f\n", εF_bxsf)
    @printf("Fermi energy unit: eV\n")
    @printf("Final tolerance for number of electrons: %.8f\n", tol_n_e_curr)
    E_bxsf = reduce(vcat, eigenvalues)
    ε_bxsf_below = maximum(E_bxsf[E_bxsf .< εF_bxsf])
    ε_bxsf_above = minimum(E_bxsf[E_bxsf .> εF_bxsf])
    @printf("Closest eigenvalue below Fermi energy: %.8f\n", ε_bxsf_below)
    @printf("Closest eigenvalue above Fermi energy: %.8f\n", ε_bxsf_above)

    @printf("Computed Fermi energy in Ry: %.8f\n", εF_bxsf*(ELECTRONVOLT_SI/RYDBERG_SI))
    println("Constants used for the conversion (from QE/Modules/Constants.f90): ")
    println("  ELECTRONVOLT_SI: ", ELECTRONVOLT_SI)
    println("  RYDBERG_SI: ", RYDBERG_SI)
    println("  BOHR_TO_ANG: ", BOHR_TO_ANG)

    # write each band into one bxsf file
    if band_index < 0
        band_range = 1:nbands
    else
        band_range = [band_index]
    end
    println("Bands in bxsf: ", join([string(_) for _ in band_range], " "))
    bands_crossing_fermi = zeros(Int,0)
    for ib in band_range
        # here I am still using the Fermi energy from input bxsf, i.e., QE scf Fermi
        outfile = out_filename * "_band_$(ib).bxsf"
        band_min = minimum(bxsf.E[ib:ib, :, :, :])
        band_max = maximum(bxsf.E[ib:ib, :, :, :])
        println("Min and max of band $ib : $band_min $band_max")

        # Check if the Fermi energy (could be custom!) is between the band_min and band_max
        # only then write the file
        ϵF = isnothing(parsed_fermi_energy) ? εF_bxsf : parsed_fermi_energy
        if (ϵF >= band_min && ϵF <= band_max)
            push!(bands_crossing_fermi, ib)
            E_band_Ry = bxsf.E[ib:ib, :, :, :].*(ELECTRONVOLT_SI/RYDBERG_SI)
            E_fermi_Ry = bxsf.fermi_energy*(ELECTRONVOLT_SI/RYDBERG_SI)
            span_vectors_bohr = bxsf.span_vectors.*BOHR_TO_ANG/2/pi
            # what about the origin? It has to be zero (Gamma point) for bxsf so I don't change it here
            WannierIO.write_bxsf(outfile, E_fermi_Ry, bxsf.origin, span_vectors_bohr, E_band_Ry)
        end
    end
    println("Bands crossing Fermi energy: ", join([string(_) for _ in bands_crossing_fermi], " "))
    println("Job done at ", Dates.now())
end
