#!/usr/bin/env -S julia -t 1 --project=/home/aiida/envs/aiida-fermisurf/code/aiida-skeaf/utils/computeFermi.jl
#
# To use this script, one need to frist instantiate the julia environment, by running
#   julia --project=.
#   ]  # enter pkg mode
#   instantiate
#
# Note that replace the abspath after `--project` in the shebang by the dir path
# where you put this script.
#
# Example usage: ./compute_fermi.jl remote_path -p seedname -n 18
# WARNING: number of threads hard-coded!!!

using Pkg
Pkg.instantiate()

using Printf
using Dates
using AtomsIO
using LinearAlgebra
using Wannier
using Comonicon
using p7zip_jll

"""
Calculate Fermi energy from tb.dat using adaptive mesh algorithm.

The script
- computes Fermi energy from the input seedname_tb.dat file

# Args

- `filename`: name of the archive with the tight-binding model

# Options

- `-n, --num_electrons`: number of electrons
- `-p, --prefix_tb`: seedname for the seedname_tb.dat and seedname_wsvec.dat file, default is `aiida`
- `-s, --smearing_type`: smearing type, default is `ColdSmearing()`
- `-w, --width_smearing`: smearing width, default is 0.01

"""
@main function main(filename::String; num_electrons::Int, prefix_tb::String="aiida", smearing_type::String="cold", width_smearing::Float64=0.01)

    println("Started at ", Dates.now())
    if !isfile(filename)
        println("ERROR: Input file $filename does not exist.")
        exit(2)
    end
    if endswith(filename, ".7z")
        # -y: assume yes, will overwrite existing unzipped files
        ret = run(`$(p7zip()) x -y $filename`)
        success(ret) || error("Error while unzipping file: $filename")
    end

    tb = read_w90_tb(prefix_tb)
    interp = HamiltonianInterpolator(tb.hamiltonian)

    latt = real_lattice(interp)
    rlatt = reciprocal_lattice(latt)

    kdistance = 0.08
    kgrid = round.(Int, [norm(b) for b in eachcol(rlatt)] ./ kdistance)

    kbT = width_smearing
    if smearing_type == "none"
        smearing = Wannier.NoneSmearing()
    elseif smearing_type == "fermi-dirac" || smearing_type == "fd"
        smearing = Wannier.FermiDiracSmearing()
    elseif smearing_type == "marzari-vanderbilt" || smearing_type == "cold"
        smearing = Wannier.ColdSmearing()
    else
        error("Unknown smearing type: $smearing_type")
    end

    kpoints = get_kpoints(kgrid)
    eigenvalues = interp(kpoints)[1]
    adpt_kgrid = Wannier.AdaptiveKgrid(kpoints, eigenvalues)
    if smearing_type == "none"
        εF = Wannier.compute_fermi_energy!(adpt_kgrid, interp, num_electrons, kbT, smearing, tol_n_electrons=1e-4, tol_εF=5e-3)
    else
        εF = Wannier.compute_fermi_energy!(adpt_kgrid, interp, num_electrons, kbT, smearing, tol_εF=5e-3)
    end
    @printf("Fermi Energy after interpolation: %.8f\n", εF)
    vbm = Wannier.find_vbm(adpt_kgrid_bxsf.vals, εF)[1]
    cbm = Wannier.find_cbm(adpt_kgrid_bxsf.vals, εF)[1]
    @printf("Valence band maximum on the interpolated k-point grid: %.8f\n", vbm)
    @printf("Conduction band minimum on the interpolated k-point grid: %.8f\n", cbm)
    gap = cbm - vbm
    @printf("Band gap on the interpolated k-point grid: %.8f\n", gap)
    println("Finished at ", Dates.now())
end
