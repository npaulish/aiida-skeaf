#!/usr/bin/env -S julia --project=/home/paulis_n/project/aiida_skeaf_dev/git/aiida-skeaf/utils/computeFermi
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
    println("Number of electrons: ", num_electrons)
    println("Smearing type: ", smearing_type)
    println("Smearing width: ", width_smearing)
    println("Occupation prefactor: ", prefactor)
    println("Initial tolerance for number of electrons (default 1e-6): ", tol_n_electrons)
end
