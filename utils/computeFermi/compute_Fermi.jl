#!/usr/bin/env -S julia --project=/home/aiida/envs/aiida-fermisurf/code/aiida-skeaf/utils/wan2skeaf.jl
#
# To use this script, one need to frist instantiate the julia environment, by running
#   julia --project=.
#   ]  # enter pkg mode
#   instantiate
#
# Note that replace the abspath after `--project` in the shebang by the dir path
# where you put this script.
#
# Example usage: ./compute_Fermi wjl.bxsf -n 17

using Pkg: instantiate
instantiate()

using Printf
using Dates
using WannierIO
using Wannier
using Comonicon: @main
using p7zip_jll

"""
Calculate Fermi energy based on the bxsf file and the number of electrons.

The script
- computes Fermi energy from the input bxsf
- finds the energies of the highest occupied and the lowest unoccupied states

# Args

- `bxsf`: input bxsf

# Options

- `-n, --num_electrons`: number of electrons
- `-s, --smearing_type`: smearing type, default is "none" (corresponding to `NoneSmearing()`, no smearing), other options are "fermi-dirac" or "fd", "marzari-vanderbilt" or "cold"
- `-w, --width_smearing`: smearing width, default is 0.0
- `-p, --prefactor`: occupation prefactor, 2 for non SOC, 1 for SOC, default is 2
- `-t, --tol_n_electrons`: tolerance for number of electrons, default is 1e-6
"""
@main function mycmd(
    bxsf::String;
    num_electrons::Int,
    smearing_type::String="none",
    width_smearing::Float64=0.0,
    prefactor::Int=2,
    tol_n_electrons::Float64=1e-6,
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
        bxsf = bxsf_files[1]
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
    # note that according to bxsf specification, the eigenvalues are defined
    # on "general grid" instead of "periodic grid", i.e., the right BZ edges are
    # repetitions of the left edges. We remove the redundance here, otherwise,
    # the computed Fermi is wrong.

    eigenvalues = [bxsf.E[:, i, j, k] for i in 1:(nkx-1) for j in 1:(nky-1) for k in 1:(nkz-1)]
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
    @printf("Fermi energy unit: same as in the bxsf file\n")
    @printf("Final tolerance for number of electrons: %.8f\n", tol_n_e_curr)
    E_bxsf = reduce(vcat, eigenvalues)
    ε_bxsf_below = maximum(E_bxsf[E_bxsf .< εF_bxsf])
    ε_bxsf_above = minimum(E_bxsf[E_bxsf .> εF_bxsf])
    @printf("Closest eigenvalue below Fermi energy: %.8f\n", ε_bxsf_below)
    @printf("Closest eigenvalue above Fermi energy: %.8f\n", ε_bxsf_above)
    band_range = 1:nbands
    println("Bands in bxsf: ", join([string(_) for _ in band_range], " "))
    bands_crossing_fermi = zeros(Int,0)
    for ib in band_range
        band_min = minimum(bxsf.E[ib:ib, :, :, :])
        band_max = maximum(bxsf.E[ib:ib, :, :, :])
        println("Min and max of band $ib : $band_min $band_max")
        # Check if the Fermi energy is between the band_min and band_max
        if (εF_bxsf >= band_min && εF_bxsf <= band_max)
            push!(bands_crossing_fermi, ib)
        end
    end
    println("Bands crossing Fermi energy: ", join([string(_) for _ in bands_crossing_fermi], " "))
    println("Job done at ", Dates.now())
end
