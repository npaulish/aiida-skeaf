#!/usr/bin/env python3
"""
Prepare bxsf files for skeaf.

Python equivalent of wan2skeaf.jl. Reads a (possibly multi-band) bxsf file,
computes the Fermi energy, and writes one bxsf per band crossing the Fermi level.
Output files use Rydberg energies and 1/Bohr k-vectors as required by skeaf.

Example:
    ./wan2skeaf.py aiida.bxsf -n 17
"""

import argparse
from datetime import datetime
import os
import subprocess
import sys

import numpy as np
from scipy import optimize
from scipy.special import erfc

# Unit conversion constants (from QE/Modules/Constants.f90)
ELECTRONVOLT_SI = 1.602176634e-19
HARTREE_SI = 4.3597447222071e-18
RYDBERG_SI = HARTREE_SI / 2.0
BOHR_TO_ANG = 0.529177210903
EV_TO_RY = ELECTRONVOLT_SI / RYDBERG_SI


# ---------------------------------------------------------------------------
# Smearing and Fermi-energy solver
# ---------------------------------------------------------------------------


def _fermi_dirac(x):
    """Fermi-Dirac occupation f(x) = 1/(1+exp(x)), numerically stable."""
    return np.where(x > 500, 0.0, 1.0 / (1.0 + np.exp(np.clip(x, -500, 500))))


def _cold_smearing(x):
    """Marzari-Vanderbilt (cold) smearing occupation."""
    y = x / np.sqrt(2) + 1.0 / np.sqrt(2)
    return 0.5 * erfc(y) + np.exp(-np.clip(y * y, 0, 700)) / np.sqrt(2 * np.pi)


def compute_fermi_energy(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    E_trimmed, num_electrons, kBT, smearing_type, prefactor=2, tol_n_electrons=1e-6
):
    """Find the Fermi energy from a trimmed eigenvalue grid.

    Parameters
    ----------
    E_trimmed : ndarray, shape (nbands, nkx-1, nky-1, nkz-1)
        Eigenvalues without periodic-boundary duplicates, in eV.
    num_electrons : int
        Target number of electrons.
    kBT : float
        Smearing width in eV.
    smearing_type : str
        "none", "fermi-dirac" / "fd", or "marzari-vanderbilt" / "cold".
    prefactor : int
        Occupation prefactor: 2 for non-SOC, 1 for SOC.
    tol_n_electrons : float
        Tolerance on |N_computed - N_target|.

    Returns
    -------
    float
        Fermi energy in eV.

    Raises
    ------
    ValueError
        If convergence fails within the given tolerance (caller may relax it).
    """
    n_kpoints = E_trimmed[0].size
    eps = E_trimmed.ravel()

    if smearing_type == "none" or kBT == 0.0:
        # Step-function case: find the gap between consecutive distinct eigenvalue blocks.
        # Using sorted eigenvalues directly can place epF AT a degenerate eigenvalue;
        # instead we work with unique values so epF ends up in the middle of the real gap.
        #
        # cum_occ[i] = occupation when epF is just above unique_vals[i], i.e. for
        # epF ∈ (unique_vals[i], unique_vals[i+1]) the occupation equals cum_occ[i].
        unique_vals, counts = np.unique(eps, return_counts=True)
        cum_occ = prefactor * np.cumsum(counts) / n_kpoints

        # k = first index where cum_occ[k] >= num_electrons - tol, meaning the gap
        # (unique_vals[k], unique_vals[k+1]) is the one where occupation ≈ num_electrons.
        k = int(np.searchsorted(cum_occ, num_electrons - tol_n_electrons, side="left"))

        if k >= len(unique_vals) - 1:
            raise ValueError(
                f"Failed to find Fermi energy within tolerance: cannot bracket "
                f"(k={k}, n_unique={len(unique_vals)})"
            )

        # Occupation in the gap (unique_vals[k], unique_vals[k+1])
        n_in_gap = cum_occ[k]
        if abs(n_in_gap - num_electrons) > tol_n_electrons:
            raise ValueError(
                f"Failed to find Fermi energy within tolerance: "
                f"computed={n_in_gap:.6f}, target={num_electrons}"
            )
        epF = (unique_vals[k] + unique_vals[k + 1]) / 2.0
        return epF

    # Smeared case: root-find N(εF) - N_target = 0 via brentq.
    def n_diff(epF):
        x = (eps - epF) / kBT
        if smearing_type in ("fermi-dirac", "fd"):
            occ = _fermi_dirac(x)
        elif smearing_type in ("marzari-vanderbilt", "cold"):
            occ = _cold_smearing(x)
        else:
            raise ValueError(f"Unknown smearing type: {smearing_type}")
        return prefactor * np.sum(occ) / n_kpoints - num_electrons

    margin = max(20.0 * kBT, 1.0)
    epF = optimize.brentq(
        n_diff, eps.min() - margin, eps.max() + margin, xtol=1e-10, maxiter=1000
    )
    n_computed = n_diff(epF) + num_electrons
    if abs(n_computed - num_electrons) > tol_n_electrons:
        raise ValueError(
            f"Failed to find Fermi energy within tolerance: "
            f"computed={n_computed:.6f}, target={num_electrons}"
        )
    return epF


# ---------------------------------------------------------------------------
# bxsf I/O  (prefer fermisurface_utils; inline fallback avoids hard dependency)
# ---------------------------------------------------------------------------


def _read_bxsf_inline(filename):  # pylint: disable=too-many-locals
    """Minimal bxsf reader (fallback when fermisurface_utils is unavailable)."""
    fermi_energy = None
    E = None
    origin = None
    span_vectors = None
    with open(filename, encoding="utf-8") as fh:
        it = iter(fh)
        for line in it:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "BEGIN_INFO" in line:
                info_line = next(it).strip()
                while not info_line or info_line.startswith("#"):
                    info_line = next(it).strip()
                fermi_energy = float(info_line.split(":")[1])
                next(it)  # END_INFO
            elif "BEGIN_BLOCK_BANDGRID_3D" in line:
                next(it)  # comment
                next(it)  # BEGIN_BANDGRID_3D_...
                n_bands = int(next(it).strip())
                nkx, nky, nkz = map(int, next(it).strip().split())
                origin = np.array(list(map(float, next(it).strip().split())))
                span_vectors = np.zeros((3, 3))
                for i in range(3):
                    span_vectors[:, i] = list(map(float, next(it).strip().split()))
                E = np.zeros((n_bands, nkx, nky, nkz))
                for ib in range(n_bands):
                    next(it)  # BAND: N
                    vals = []
                    while len(vals) < nkx * nky * nkz:
                        vals.extend(map(float, next(it).strip().split()))
                    E[ib] = np.array(vals[: nkx * nky * nkz]).reshape((nkx, nky, nkz))
                break
    return fermi_energy, origin, span_vectors, E


def _write_bxsf_inline(filename, fermi_energy, origin, span_vectors, E):
    """Minimal bxsf writer (fallback when fermisurface_utils is unavailable)."""
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write("BEGIN_INFO\n")
        fh.write(f"  Fermi Energy: {fermi_energy:21.16f}\n")
        fh.write("END_INFO\n\n")
        fh.write("BEGIN_BLOCK_BANDGRID_3D\nwan2skeaf_py\nBEGIN_BANDGRID_3D_fermi\n")
        n_bands, nx, ny, nz = E.shape
        fh.write(f"{n_bands}\n{nx} {ny} {nz}\n")
        fh.write(f"{origin[0]:12.7f} {origin[1]:12.7f} {origin[2]:12.7f}\n")
        for i in range(3):
            fh.write(
                f"{span_vectors[0, i]:12.7f} {span_vectors[1, i]:12.7f} {span_vectors[2, i]:12.7f}\n"
            )
        for ib in range(n_bands):
            fh.write(f"BAND: {ib + 1}\n")
            ncol = 0
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        fh.write(f" {E[ib, i, j, k]:16.8e}")
                        ncol += 1
                        if ncol == 6:
                            fh.write("\n")
                            ncol = 0
            if ncol != 0:
                fh.write("\n")
        fh.write("END_BANDGRID_3D\nEND_BLOCK_BANDGRID_3D\n")


def read_bxsf(filename):
    """Read a bxsf file; returns (fermi_energy_eV, origin, span_vectors, E)."""
    try:
        from fermisurface_utils.bxsf import read_bxsf as _fs_read

        fe, origin, sv, _X, _Y, _Z, E = _fs_read(filename)
        return fe, origin, sv, E
    except ImportError:
        return _read_bxsf_inline(filename)


def write_bxsf(filename, fermi_energy, origin, span_vectors, E):
    """Write a bxsf file."""
    try:
        from fermisurface_utils.bxsf import write_bxsf as _fs_write

        _fs_write(filename, fermi_energy, origin, span_vectors, E)
        return
    except ImportError:
        _write_bxsf_inline(filename, fermi_energy, origin, span_vectors, E)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():  # pylint: disable=too-many-locals,too-many-statements
    """CLI entry point: parse arguments and run wan2skeaf."""
    parser = argparse.ArgumentParser(
        description="Prepare bxsf files for skeaf.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("bxsf", help="Input bxsf file (may be .7z compressed)")
    parser.add_argument(
        "-n", "--num_electrons", type=int, required=True, help="Number of electrons"
    )
    parser.add_argument(
        "-b",
        "--band_index",
        type=int,
        default=-1,
        help="Band index to process (-1 for all bands)",
    )
    parser.add_argument(
        "-o", "--out_filename", default="skeaf", help="Output filename prefix"
    )
    parser.add_argument(
        "-s",
        "--smearing_type",
        default="none",
        help="Smearing type: none | fermi-dirac | fd | marzari-vanderbilt | cold",
    )
    parser.add_argument(
        "-w", "--width_smearing", type=float, default=0.0, help="Smearing width in eV"
    )
    parser.add_argument(
        "-p",
        "--prefactor",
        type=int,
        default=2,
        help="Occupation prefactor (2 non-SOC, 1 SOC)",
    )
    parser.add_argument(
        "-t",
        "--tol_n_electrons",
        type=float,
        default=1e-6,
        help="Tolerance for number of electrons",
    )
    parser.add_argument(
        "-f",
        "--fermi_energy",
        default="none",
        help="Custom Fermi energy in eV for band selection (none = use computed)",
    )
    args = parser.parse_args()

    print("Started on", datetime.now())

    bxsf_path = args.bxsf
    if not os.path.isfile(bxsf_path):
        print(f"ERROR: input file {bxsf_path} does not exist.")
        sys.exit(2)

    if bxsf_path.endswith(".7z"):
        ret = subprocess.run(["7z", "x", "-y", bxsf_path], check=False)
        if ret.returncode != 0:
            print(f"ERROR: failed to extract {bxsf_path}")
            sys.exit(2)
        bxsf_files = [f for f in os.listdir(".") if f.endswith(".bxsf")]
        if len(bxsf_files) != 1:
            print(
                f"ERROR: expected 1 .bxsf file after extraction, got {len(bxsf_files)}"
            )
            sys.exit(2)
        dst = "input.bxsf"
        os.rename(bxsf_files[0], dst)
        bxsf_path = dst

    print("Number of electrons:", args.num_electrons)
    fermi_energy_file, origin, span_vectors, E = read_bxsf(bxsf_path)
    print(f"Fermi Energy from file: {fermi_energy_file:.8f}")

    nbands, nkx, nky, nkz = E.shape
    print(f"Number of bands: {nbands}")
    print(f"Grid shape: {nkx} x {nky} x {nkz}")

    kBT = args.width_smearing
    smearing_type = args.smearing_type
    print(f"Smearing type: {smearing_type}")
    print(f"Smearing width: {kBT}")
    print(f"Occupation prefactor: {args.prefactor}")
    print(
        f"Initial tolerance for number of electrons (default 1e-6): {args.tol_n_electrons}"
    )

    parsed_fermi_energy = (
        None if args.fermi_energy == "none" else float(args.fermi_energy)
    )
    if parsed_fermi_energy is not None:
        print(
            f"Custom Fermi energy will be used to select bands: {parsed_fermi_energy}"
        )

    # Compute Fermi energy on the trimmed grid (no periodic-boundary duplicates),
    # with automatic tolerance relaxation matching the Julia script.
    E_trimmed = E[:, :-1, :-1, :-1]
    tol_upper = max(1e-3, args.tol_n_electrons)
    tol_curr = args.tol_n_electrons
    epF = None

    while tol_curr <= tol_upper:
        print(f"Current tolerance for number of electrons: {tol_curr}")
        try:
            epF = compute_fermi_energy(
                E_trimmed,
                args.num_electrons,
                kBT,
                smearing_type,
                prefactor=args.prefactor,
                tol_n_electrons=tol_curr,
            )
            break
        except ValueError as exc:
            msg = str(exc)
            print(f"Error: {msg}")
            if "Failed to find Fermi energy within tolerance" in msg:
                print(
                    "   Increasing tolerance for number of electrons by a factor of 2..."
                )
                tol_curr *= 2
                continue
            sys.exit(3)

    if epF is None:
        print(
            "Error: tolerance for number of electrons exceeded tol_n_electrons_upperbound. Exiting..."
        )
        sys.exit(3)

    print(f"Computed Fermi energy: {epF:.8f}")
    print("Fermi energy unit: eV")
    print(f"Final tolerance for number of electrons: {tol_curr:.8f}")

    E_flat = E_trimmed.ravel()
    print(f"Closest eigenvalue below Fermi energy: {np.max(E_flat[E_flat < epF]):.8f}")
    print(f"Closest eigenvalue above Fermi energy: {np.min(E_flat[E_flat > epF]):.8f}")
    print(f"Computed Fermi energy in Ry: {epF * EV_TO_RY:.8f}")
    print("Constants used for the conversion (from QE/Modules/Constants.f90):")
    print(f"  ELECTRONVOLT_SI: {ELECTRONVOLT_SI}")
    print(f"  RYDBERG_SI: {RYDBERG_SI}")
    print(f"  BOHR_TO_ANG: {BOHR_TO_ANG}")

    # Convert span vectors: Å⁻¹ (2π included, Wannier90 convention) → 1/Bohr (skeaf)
    span_vectors_bohr = span_vectors * BOHR_TO_ANG / (2 * np.pi)

    band_range = range(1, nbands + 1) if args.band_index < 0 else [args.band_index]
    print("Bands in bxsf:", " ".join(str(b) for b in band_range))

    bands_crossing = []
    eps_F_select = parsed_fermi_energy if parsed_fermi_energy is not None else epF
    for ib in band_range:
        idx = ib - 1
        band_min = float(E[idx].min())
        band_max = float(E[idx].max())
        print(f"Min and max of band {ib} : {band_min} {band_max}")

        if band_min <= eps_F_select <= band_max:
            bands_crossing.append(ib)
            outfile = f"{args.out_filename}_band_{ib}.bxsf"
            write_bxsf(
                outfile,
                eps_F_select * EV_TO_RY,
                origin,
                span_vectors_bohr,
                E[idx : idx + 1] * EV_TO_RY,
            )

    print("Bands crossing Fermi energy:", " ".join(str(b) for b in bands_crossing))
    print("Job done at", datetime.now())


if __name__ == "__main__":
    main()
