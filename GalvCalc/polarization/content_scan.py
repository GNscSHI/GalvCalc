"""
Alloy-content scans of corrosion descriptors.

This module computes how the corrosion current and potential of an alloy
change with the content of an alloying element. The volume fraction of the
intermetallic second phase is obtained from the weight percent of the
solute, and the corrosion point is then evaluated for each content level
with the Butler-Volmer kinetics fitted in the GalvCalc manuscript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pymatgen.core import Composition

# Physical constants used by the fitted kinetics.
FARADAY = 96485.0  # C/mol
GAS_CONSTANT = 8.314  # J/(mol K)
TEMPERATURE = 300.0  # K
SCE_OFFSET = 0.241  # V, SHE -> SCE shift

# Default transfer coefficients fitted in the manuscript.
ANODE_ALPHA = 0.55       # Mg dissolution anode (n_total = 2)
MG_CATHODE_ALPHA = 0.77  # hydrogen evolution on the Mg matrix
SECOND_PHASE_ALPHA = 0.6  # hydrogen evolution on a second-phase cathode


# ---------------------------------------------------------------------- #
# Volume fraction from weight percent
# ---------------------------------------------------------------------- #
def wt_to_vol_fraction(
    wt_solute: float,
    phase_formula: str,
    rho_matrix: float,
    rho_phase: float,
    matrix_element: str = "Mg",
    solute_element: Optional[str] = None,
    rho_solute: Optional[float] = None,
) -> float:
    """
    Volume fraction of an intermetallic phase from the solute weight percent.

    The stoichiometry of the phase is taken from ``phase_formula`` (e.g.
    ``"Mg3Nd"`` contains 3 Mg and 1 Nd per formula unit). The matrix is
    consumed according to this stoichiometry; any excess solute is assumed
    to remain as the pure element.

    Parameters
    ----------
    wt_solute : float
        Solute content in weight percent (0-100).
    phase_formula : str
        Formula of the intermetallic phase, e.g. ``"Mg3Nd"``.
    rho_matrix : float
        Density of the matrix (g/cm^3).
    rho_phase : float
        Density of the intermetallic phase (g/cm^3).
    matrix_element : str
        Element symbol of the matrix, e.g. ``"Mg"``.
    solute_element : str, optional
        Element symbol of the solute. When omitted it is inferred from
        ``phase_formula`` (the element that is not the matrix element).
    rho_solute : float, optional
        Density of the pure solute (g/cm^3), used when excess solute
        remains. Defaults to the density of the matrix.

    Returns
    -------
    float
        Volume fraction of the intermetallic phase.
    """
    if wt_solute <= 0:
        return 0.0

    phase = Composition(phase_formula)
    if solute_element is None:
        solute_element = next(
            (el.symbol for el in phase.elements if el.symbol != matrix_element),
            matrix_element,
        )

    m_matrix_elem = float(Composition(matrix_element).weight)  # g/mol
    m_solute_elem = float(Composition(solute_element).weight)  # g/mol
    m_phase = float(phase.weight)  # g/mol

    n_matrix_per_phase = phase.get_atomic_fraction(matrix_element) * phase.num_atoms
    n_solute_per_phase = phase.get_atomic_fraction(solute_element) * phase.num_atoms

    # Basis: 100 g of alloy.
    mass_solute = wt_solute
    mass_matrix = 100.0 - wt_solute

    n_solute = mass_solute / m_solute_elem
    n_matrix = mass_matrix / m_matrix_elem

    # The phase amount is limited by the deficient reactant.
    n_phase = min(n_solute / n_solute_per_phase, n_matrix / n_matrix_per_phase)

    mass_phase = n_phase * m_phase
    mass_remaining_matrix = mass_matrix - n_phase * n_matrix_per_phase * m_matrix_elem
    mass_remaining_solute = max(0.0, mass_solute - n_phase * n_solute_per_phase * m_solute_elem)

    vol_phase = mass_phase / rho_phase
    vol_matrix = mass_remaining_matrix / rho_matrix
    vol_solute = mass_remaining_solute / (rho_solute if rho_solute else rho_matrix)

    total_volume = float(vol_phase + vol_matrix + vol_solute)
    return float(vol_phase) / total_volume if total_volume > 0 else 0.0


def mg3nd_vol_fraction(nd_wt_percent: float,
                       rho_mg: float = 1.8,
                       rho_nd: float = 7.01) -> float:
    """Volume fraction of Mg3Nd in Mg-Nd alloys (Nd in weight percent)."""
    return wt_to_vol_fraction(
        nd_wt_percent,
        phase_formula="Mg3Nd",
        rho_matrix=rho_mg,
        rho_phase=3.58,
        matrix_element="Mg",
        solute_element="Nd",
        rho_solute=rho_nd,
    )


# ---------------------------------------------------------------------- #
# Corrosion point versus anode area ratio
# ---------------------------------------------------------------------- #
@dataclass
class CorrosionScanResult:
    """Corrosion descriptors for a sweep of anode area ratios.

    Attributes
    ----------
    anode_ratios : np.ndarray
        Anode area ratios (fraction of the matrix that is anodically active).
    corrosion_currents : np.ndarray
        Corrosion current density in A/cm^2 (absolute values).
    corrosion_potentials : np.ndarray
        Corrosion potential (V) on the requested reference-electrode scale.
    """

    anode_ratios: np.ndarray
    corrosion_currents: np.ndarray
    corrosion_potentials: np.ndarray


def _bv_anode(potential: np.ndarray, i0: float, e_eq: float, alpha: float,
              area: float) -> np.ndarray:
    """Mg dissolution anode: n_total = 2, exponent alpha * n_total."""
    n = 1
    return (
        area
        * i0
        * (
            np.exp(alpha * 2 * n * FARADAY * (potential - e_eq) / (GAS_CONSTANT * TEMPERATURE))
            - np.exp(-(1 - alpha) * 2 * n * FARADAY * (potential - e_eq) / (GAS_CONSTANT * TEMPERATURE))
        )
    )


def _bv_cathode(potential: np.ndarray, i0: float, e_eq: float, alpha: float,
                area: float) -> np.ndarray:
    """Cathodic hydrogen evolution: one-electron Butler-Volmer kinetics."""
    n = 1
    return (
        area
        * i0
        * (
            np.exp(alpha * n * FARADAY * (potential - e_eq) / (GAS_CONSTANT * TEMPERATURE))
            - np.exp(-(1 - alpha) * n * FARADAY * (potential - e_eq) / (GAS_CONSTANT * TEMPERATURE))
        )
    )


def corrosion_vs_anode_ratio(
    anode_i0: float,
    anode_equilibrium_potential: float,
    cathode_i0s: Sequence[float],
    cathode_equilibrium_potential: float,
    area_fractions: Sequence[float],
    anode_alpha: float = ANODE_ALPHA,
    cathode_alphas: Optional[Sequence[float]] = None,
    reference_electrode: str = "SCE",
    n_ratios: int = 1001,
    n_potentials: int = 10000,
) -> CorrosionScanResult:
    """
    Compute the corrosion point for a sweep of anode area ratios.

    Parameters
    ----------
    anode_i0 : float
        Exchange current density of the Mg dissolution anode (A/cm^2).
    anode_equilibrium_potential : float
        Equilibrium potential of the anode (V vs SHE).
    cathode_i0s : sequence of float
        Exchange current densities of the cathodic reactions (A/cm^2). The
        first entry is the hydrogen-evolution cathode on the matrix, the
        optional second entry is a second-phase cathode.
    cathode_equilibrium_potential : float
        Equilibrium potential of the cathodic reactions (V vs SHE).
    area_fractions : sequence of float
        Total area fractions ``[matrix, second_phase]`` (sum = 1).
    anode_alpha : float
        Transfer coefficient of the anode.
    cathode_alphas : sequence of float, optional
        Transfer coefficients of the cathodic reactions; defaults to
        ``[MG_CATHODE_ALPHA, SECOND_PHASE_ALPHA]``.
    reference_electrode : str
        Reference-electrode scale of the returned potentials, ``"SCE"`` or
        ``"SHE"``.
    n_ratios : int
        Number of anode area-ratio points.
    n_potentials : int
        Number of potential points used to refine each corrosion point.

    Returns
    -------
    CorrosionScanResult
        Corrosion current and potential for every anode area ratio.
    """
    if len(area_fractions) != len(cathode_i0s):
        raise ValueError("area_fractions and cathode_i0s must have the same length.")
    if cathode_alphas is None:
        cathode_alphas = [MG_CATHODE_ALPHA] + [SECOND_PHASE_ALPHA] * (len(cathode_i0s) - 1)
    if len(cathode_alphas) != len(cathode_i0s):
        raise ValueError("cathode_alphas and cathode_i0s must have the same length.")

    # The fitted kinetics use potentials on the SCE scale.
    offset = 0.0 if reference_electrode.upper() == "SHE" else SCE_OFFSET
    e_a = anode_equilibrium_potential - offset
    e_c = cathode_equilibrium_potential - offset

    anode_ratios = np.linspace(0.0, 1.0, n_ratios)
    i_corr_list = []
    e_corr_list = []

    for ra in anode_ratios:
        e_corr, i_corr = _corrosion_point_for_ratio(
            ra, anode_i0, e_a, cathode_i0s, e_c, cathode_alphas,
            area_fractions, n_potentials,
        )
        i_corr_list.append(abs(i_corr))
        e_corr_list.append(e_corr)

    potentials = np.array(e_corr_list) + offset
    return CorrosionScanResult(
        anode_ratios=anode_ratios,
        corrosion_currents=np.array(i_corr_list),
        corrosion_potentials=potentials,
    )


def _corrosion_point_for_ratio(ra: float, anode_i0: float, e_a: float,
                               cathode_i0s: Sequence[float], e_c: float,
                               cathode_alphas: Sequence[float],
                               area_fractions: Sequence[float],
                               n_potentials: int) -> Tuple[float, float]:
    """Locate E_corr / i_corr for a single anode area ratio (SCE scale).

    The mixed potential is found by minimizing |i_anode + i_cathode| on a
    dense potential grid between the anode and cathode equilibrium
    potentials.
    """
    matrix_area, *rest = area_fractions
    second_phase_area = rest[0] if rest else 0.0

    potential_grid = np.linspace(e_a, e_c, n_potentials)

    anode_current = _bv_anode(potential_grid, anode_i0, e_a, ANODE_ALPHA, matrix_area * ra)
    cathode_current = _bv_cathode(
        potential_grid, cathode_i0s[0], e_c, cathode_alphas[0], matrix_area * (1 - ra)
    )
    if second_phase_area > 0 and len(cathode_i0s) > 1:
        cathode_current += _bv_cathode(
            potential_grid, cathode_i0s[1], e_c, cathode_alphas[1], second_phase_area
        )

    total_current = anode_current + cathode_current
    min_idx = int(np.argmin(np.abs(total_current)))
    return float(potential_grid[min_idx]), float(anode_current[min_idx])


def max_corrosion_in_domain(
    result: CorrosionScanResult,
    domain: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float, float]:
    """
    Return the corrosion point with the largest current inside a Ra domain.

    Parameters
    ----------
    result : CorrosionScanResult
        Sweep produced by :func:`corrosion_vs_anode_ratio`.
    domain : tuple
        Allowed anode area-ratio window ``[lo, hi]``.

    Returns
    -------
    tuple
        ``(ra_at_max, max_i_corr, e_corr_at_max)``.
    """
    lo, hi = domain
    mask = (result.anode_ratios >= lo) & (result.anode_ratios <= hi)
    if not np.any(mask):
        raise ValueError("The Ra domain does not overlap the scanned ratios.")

    currents = result.corrosion_currents[mask]
    idx = int(np.argmax(currents))
    ra_values = result.anode_ratios[mask]
    return float(ra_values[idx]), float(currents[idx]), float(result.corrosion_potentials[mask][idx])


# ---------------------------------------------------------------------- #
# Content sweeps
# ---------------------------------------------------------------------- #
def scan_corrosion_vs_content(
    content_levels: Sequence[float],
    params_fn: Callable[[float], Dict[str, Any]],
    domain: Tuple[float, float] = (0.0, 1.0),
    **scan_kwargs: Any,
) -> pd.DataFrame:
    """
    Scan corrosion descriptors over alloying-element content levels.

    Parameters
    ----------
    content_levels : sequence of float
        Solute contents (e.g. weight percent) to evaluate.
    params_fn : callable
        ``params_fn(content) -> dict`` returning the kinetic parameters of
        that content level: ``anode_i0``, ``anode_equilibrium_potential``,
        ``cathode_i0s``, ``cathode_equilibrium_potential`` and
        ``area_fractions`` (all potentials vs SHE).
    domain : tuple
        Anode area-ratio window in which the maximum corrosion current is
        sought.
    **scan_kwargs
        Additional keyword arguments forwarded to
        :func:`corrosion_vs_anode_ratio`.

    Returns
    -------
    pandas.DataFrame
        One row per content level with columns ``content``, ``vol_fraction``
        (when supplied by ``params_fn``), ``max_log10_i_corr``,
        ``Ra_at_max`` and ``E_corr_at_max``.
    """
    rows: List[Dict[str, Any]] = []

    for content in content_levels:
        params = params_fn(content)
        result = corrosion_vs_anode_ratio(
            anode_i0=params["anode_i0"],
            anode_equilibrium_potential=params["anode_equilibrium_potential"],
            cathode_i0s=params["cathode_i0s"],
            cathode_equilibrium_potential=params["cathode_equilibrium_potential"],
            area_fractions=params["area_fractions"],
            **scan_kwargs,
        )
        ra_at_max, i_corr_max, e_corr_at_max = max_corrosion_in_domain(result, domain)

        row: Dict[str, Any] = {
            "content": content,
            "max_log10_i_corr": np.log10(i_corr_max),
            "Ra_at_max": ra_at_max,
            "E_corr_at_max": e_corr_at_max,
        }
        if "vol_fraction" in params:
            row["vol_fraction"] = params["vol_fraction"]
        rows.append(row)

    return pd.DataFrame(rows)


def mg3nd_kinetics(nd_wt_percent: float, vol_fraction: float) -> Dict[str, Any]:
    """
    Kinetic parameters of the Mg-Nd example at a given Nd content.

    This reproduces the fitted parameters used in the manuscript example:
    the anode exchange current drops from 6e-23 A/cm^2 (pure Mg) to
    3e-23 A/cm^2 once the Mg3Nd phase appears, and the equilibrium
    potential shifts linearly with the solute remaining in the alpha phase.
    """
    solute_in_alpha = min(nd_wt_percent, 0.0)
    anode_i0 = 6e-23 if nd_wt_percent <= 0 else 3e-23
    anode_equilibrium_potential = -2.37 + solute_in_alpha * (0.0020) / 10.0

    return {
        "anode_i0": anode_i0,
        "anode_equilibrium_potential": anode_equilibrium_potential,
        "cathode_i0s": [10 ** -8.1, 10 ** -8.732],
        "cathode_equilibrium_potential": -0.61,
        "area_fractions": [1.0 - vol_fraction, vol_fraction],
        "vol_fraction": vol_fraction,
    }
