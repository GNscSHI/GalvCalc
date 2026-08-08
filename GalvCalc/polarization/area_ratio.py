"""
Anode/cathode area-ratio analysis.

This module studies how the corrosion current and potential of a galvanic
couple depend on the anodically active area fraction, which is a key
geometric descriptor for Mg-based alloys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


@dataclass
class AreaRatioParameters:
    """Electrochemical parameters used in the area-ratio analysis.

    Attributes
    ----------
    anode_exchange_current : float
        Exchange current density of the anodic reaction (A/cm^2).
    anode_equilibrium_potential : float
        Equilibrium potential of the anodic reaction (V vs SHE).
    cathode_exchange_currents : List[float]
        Exchange current densities of the cathodic reactions; the first
        entry is the hydrogen-evolution cathode on the matrix, the optional
        second entry is a second-phase cathode.
    cathode_equilibrium_potential : float
        Equilibrium potential of the cathodic reactions (V vs SHE).
    area_fractions : List[float]
        Total area fractions ``[matrix, second_phase]`` (sum = 1).
    anode_alpha : float
        Transfer coefficient of the anodic reaction.
    cathode_alpha : float
        Transfer coefficient of the hydrogen-evolution cathode.
    second_phase_alpha : float
        Transfer coefficient of the second-phase cathode.
    """

    anode_exchange_current: float
    anode_equilibrium_potential: float
    cathode_exchange_currents: List[float]
    cathode_equilibrium_potential: float
    area_fractions: List[float]
    anode_alpha: float = 0.55
    cathode_alpha: float = 0.77
    second_phase_alpha: float = 0.5


class AreaRatioAnalyzer:
    """Analyze the effect of the anode area ratio on corrosion kinetics.

    Parameters
    ----------
    reference_electrode : str
        Reference-electrode scale, ``"SCE"`` or ``"SHE"``.
    temperature : float
        Temperature in Kelvin.
    """

    def __init__(self, reference_electrode: str = "SCE", temperature: float = 300.0):
        self.reference_electrode = reference_electrode.upper()
        self.temperature = temperature
        self.F = 96485  # Faraday constant (C/mol)
        self.R = 8.314  # gas constant (J/(mol K))
        self._setup_fonts()

    def _setup_fonts(self) -> None:
        """Configure publication-style fonts."""
        self.label_font = {"family": "Times New Roman", "weight": "bold", "size": 20}
        self.ticks_font = fm.FontProperties(family="Times New Roman", size=14)

    def _convert_potential(self, potential: float) -> float:
        """Convert a potential from the SHE scale to the reference electrode."""
        if self.reference_electrode == "SCE":
            return potential - 0.241  # SHE to SCE
        if self.reference_electrode == "SHE":
            return potential
        raise ValueError("reference_electrode must be 'SCE' or 'SHE'")

    # ------------------------------------------------------------------ #
    # Butler-Volmer kinetics (same empirical form as curve_plot.py)
    # ------------------------------------------------------------------ #
    def _butler_volmer_anode(self, potential: np.ndarray, params: AreaRatioParameters,
                             anode_area_ratio: float) -> np.ndarray:
        """Anodic Butler-Volmer current density.

        Original Mg manuscript form: n = 2 with an anodic exponent
        ``alpha_a * n`` and no prefactor.
        """
        n = 2  # number of transferred electrons
        alpha_a = params.anode_alpha
        E_eq = self._convert_potential(params.anode_equilibrium_potential)

        current = (
            params.area_fractions[0]
            * anode_area_ratio
            * params.anode_exchange_current
            * (
                np.exp(alpha_a * n * self.F * (potential - E_eq) / (self.R * self.temperature))
                - np.exp(-(1 - alpha_a) * n * self.F * (potential - E_eq) / (self.R * self.temperature))
            )
        )
        return current

    def _butler_volmer_cathode(self, potential: np.ndarray, params: AreaRatioParameters,
                               exchange_current: float, alpha: float, area_ratio: float) -> np.ndarray:
        """Cathodic Butler-Volmer current density.

        One-electron HER in the original Mg manuscript form (no prefactor).
        """
        n = 1  # number of transferred electrons
        E_eq = self._convert_potential(params.cathode_equilibrium_potential)

        current = (
            area_ratio
            * exchange_current
            * (
                np.exp(alpha * n * self.F * (potential - E_eq) / (self.R * self.temperature))
                - np.exp(-(1 - alpha) * n * self.F * (potential - E_eq) / (self.R * self.temperature))
            )
        )
        return current

    def _find_corrosion_point(self, potential_range: np.ndarray, anode_current: np.ndarray,
                              cathode_current: np.ndarray, params: AreaRatioParameters,
                              anode_area_ratio: float) -> Tuple[float, float]:
        """Locate the corrosion potential and current for a given area ratio."""
        total_current = anode_current + cathode_current
        min_idx = np.argmin(np.abs(total_current))

        delta = abs(total_current)[min_idx]
        E_corr_ret = potential_range[min_idx]
        I_corr_ret = anode_current[min_idx]

        # Refine the corrosion point on a dense potential grid.
        for E_corr in np.linspace(params.anode_equilibrium_potential,
                                  params.cathode_equilibrium_potential, 10000):
            delta_new = self._butler_volmer_anode(np.array([E_corr]), params, anode_area_ratio)[0]

            # Hydrogen-evolution cathode on the matrix.
            delta_new += self._butler_volmer_cathode(
                np.array([E_corr]), params,
                params.cathode_exchange_currents[0], params.cathode_alpha,
                params.area_fractions[0] * (1 - anode_area_ratio)
            )[0]

            # Second-phase cathode, when present.
            if len(params.cathode_exchange_currents) > 1:
                delta_new += self._butler_volmer_cathode(
                    np.array([E_corr]), params,
                    params.cathode_exchange_currents[1], params.second_phase_alpha,
                    params.area_fractions[1]
                )[0]

            delta_new = abs(delta_new)
            if delta_new < delta:
                delta = delta_new
                E_corr_ret = E_corr
                I_corr_ret = self._butler_volmer_anode(np.array([E_corr]), params, anode_area_ratio)[0]

        return E_corr_ret, I_corr_ret

    def analyze_area_ratio(self, params: AreaRatioParameters,
                           anode_ratio_range: Tuple[float, float] = (0.01, 1.0),
                           n_ratios: int = 100,
                           potential_range: Tuple[float, float] = (-3.0, 0.0),
                           n_potentials: int = 500) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze the effect of the anode area ratio on the corrosion current.

        Parameters
        ----------
        params : AreaRatioParameters
            Electrochemical parameters.
        anode_ratio_range : tuple
            Anode area-ratio window [min, max].
        n_ratios : int
            Number of area-ratio points.
        potential_range : tuple
            Potential window (V).
        n_potentials : int
            Number of potential points.

        Returns
        -------
        anode_ratios : np.ndarray
            Anode area-ratio array.
        corrosion_currents : np.ndarray
            Corrosion current density array (A/cm^2).
        corrosion_potentials : np.ndarray
            Corrosion potential array (V).
        """
        anode_ratios = np.linspace(anode_ratio_range[0], anode_ratio_range[1], n_ratios)
        potential_array = np.linspace(potential_range[0], potential_range[1], n_potentials)

        corrosion_currents = []
        corrosion_potentials = []

        for ra in tqdm(anode_ratios, desc="Scanning anode area ratio"):
            anode_current = self._butler_volmer_anode(potential_array, params, ra)

            cathode_current = np.zeros_like(potential_array)

            # Hydrogen-evolution cathode on the matrix.
            cathode_current += self._butler_volmer_cathode(
                potential_array, params,
                params.cathode_exchange_currents[0], params.cathode_alpha,
                params.area_fractions[0] * (1 - ra)
            )

            # Second-phase cathode, when present.
            if len(params.cathode_exchange_currents) > 1:
                cathode_current += self._butler_volmer_cathode(
                    potential_array, params,
                    params.cathode_exchange_currents[1], params.second_phase_alpha,
                    params.area_fractions[1]
                )

            E_corr, I_corr = self._find_corrosion_point(
                potential_array, anode_current, cathode_current, params, ra
            )

            corrosion_potentials.append(E_corr)
            corrosion_currents.append(abs(I_corr))

        return anode_ratios, np.array(corrosion_currents), np.array(corrosion_potentials)

    def find_optimal_area_ratio(self, anode_ratios: np.ndarray,
                                corrosion_currents: np.ndarray) -> Tuple[float, float]:
        """
        Find the anode area ratio that maximizes the corrosion current.

        Returns
        -------
        optimal_ratio : float
            Optimal anode area ratio.
        max_current : float
            Maximum corrosion current density (A/cm^2).
        """
        max_idx = np.argmax(corrosion_currents)
        return float(anode_ratios[max_idx]), float(corrosion_currents[max_idx])

    def plot_area_ratio_analysis(self, params: AreaRatioParameters,
                                 figsize: Tuple[float, float] = (10, 6),
                                 title: Optional[str] = None):
        """
        Plot the corrosion current and potential versus the anode area ratio.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        optimal_ratio : float
            Optimal anode area ratio.
        max_current : float
            Maximum corrosion current density (A/cm^2).
        """
        anode_ratios, corrosion_currents, corrosion_potentials = self.analyze_area_ratio(params)
        optimal_ratio, max_current = self.find_optimal_area_ratio(anode_ratios, corrosion_currents)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        ax1.plot(anode_ratios * 100, np.log10(corrosion_currents), "b-", linewidth=2.5)
        ax1.axvline(optimal_ratio * 100, color="red", linestyle="--", alpha=0.7,
                    label=f"Optimal: {optimal_ratio * 100:.1f}%")
        ax1.scatter(optimal_ratio * 100, np.log10(max_current), color="red", s=80, zorder=5)

        ax1.set_xlabel("Anode Area Ratio (%)", self.label_font)
        ax1.set_ylabel("log [i$_{corr}$ (A/cm$^2$)]", self.label_font)
        ax1.legend(prop={"family": "Times New Roman", "size": 12})

        ax2.plot(anode_ratios * 100, corrosion_potentials, "g-", linewidth=2.5)
        ax2.axvline(optimal_ratio * 100, color="red", linestyle="--", alpha=0.7,
                    label=f"Optimal: {optimal_ratio * 100:.1f}%")
        ax2.scatter(optimal_ratio * 100, corrosion_potentials[np.argmax(corrosion_currents)],
                    color="red", s=80, zorder=5)

        ax2.set_xlabel("Anode Area Ratio (%)", self.label_font)
        ax2.set_ylabel(f"E$_{{corr}}$ (V vs. {self.reference_electrode})", self.label_font)
        ax2.legend(prop={"family": "Times New Roman", "size": 12})

        for ax in [ax1, ax2]:
            ax.tick_params(axis="both", which="major", labelsize=12)
            for tick in ax.get_xticklabels():
                tick.set_fontproperties(self.ticks_font)
            for tick in ax.get_yticklabels():
                tick.set_fontproperties(self.ticks_font)

        if title:
            plt.suptitle(title, fontsize=16, fontfamily="Times New Roman", fontweight="bold")

        plt.tight_layout()

        print(f"\nOptimal anode area ratio: {optimal_ratio * 100:.2f}%")
        print(f"Maximum corrosion current: {max_current:.2e} A/cm^2")
        print(f"Corrosion potential at the optimal ratio: "
              f"{corrosion_potentials[np.argmax(corrosion_currents)]:.3f} V")

        return fig, optimal_ratio, max_current


# ---------------------------------------------------------------------- #
# Convenience wrappers
# ---------------------------------------------------------------------- #
def analyze_optimal_area_ratio(anode_exchange_current: float, anode_equilibrium_potential: float,
                               cathode_exchange_currents: List[float], cathode_equilibrium_potential: float,
                               area_fractions: List[float], **kwargs):
    """Quickly analyze the optimal anode area ratio for the given kinetics."""
    params = AreaRatioParameters(
        anode_exchange_current=anode_exchange_current,
        anode_equilibrium_potential=anode_equilibrium_potential,
        cathode_exchange_currents=cathode_exchange_currents,
        cathode_equilibrium_potential=cathode_equilibrium_potential,
        area_fractions=area_fractions,
    )

    analyzer = AreaRatioAnalyzer(**kwargs)
    return analyzer.plot_area_ratio_analysis(params)


def create_example_parameters() -> AreaRatioParameters:
    """Create example parameters for a pure-Mg system.

    The Mg dissolution anode uses i0 = 10^-22.2 A/cm^2 and alpha = 0.55,
    while the hydrogen-evolution cathode on the Mg matrix uses
    i0 = 10^-8.078 A/cm^2 (see the Mg-phase table in ``curve_plot.py``).
    """
    return AreaRatioParameters(
        anode_exchange_current=10 ** -22.2,
        anode_equilibrium_potential=-2.37,  # vs SHE
        cathode_exchange_currents=[10 ** -8.078243758202708],  # Mg HER cathode
        cathode_equilibrium_potential=-0.65,  # vs SHE (pH 11)
        area_fractions=[1, 0],  # [matrix area, second-phase area]
    )


if __name__ == "__main__":
    params = create_example_parameters()
    analyzer = AreaRatioAnalyzer(reference_electrode="SCE")
    fig, optimal_ratio, max_current = analyzer.plot_area_ratio_analysis(
        params,
        title="Area Ratio Optimization for Mg-based Alloy",
    )
    plt.show()
