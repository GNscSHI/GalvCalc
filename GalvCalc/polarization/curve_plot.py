"""
Polarization-curve plotting and kinetic parameterization.

This module standardizes the computation and plotting of corrosion
polarization curves from Butler-Volmer kinetics of coupled anodic
dissolution and cathodic hydrogen evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ElectrodeParameters:
    """Kinetic parameters of a single electrode reaction.

    Attributes
    ----------
    exchange_current : float
        Exchange current density (A/cm^2).
    equilibrium_potential : float
        Equilibrium potential (V vs SHE).
    alpha : float
        Transfer coefficient.
    name : str
        Plain-text electrode identifier used in data and prints.
    label : str
        Mathtext label used in legends (falls back to ``name``).
    kinetic_form : str
        Empirical Butler-Volmer parameterization:

        * ``"fe"`` (default): the calibrated form used for the Fe demo. The
          anode uses n = 1 with an exponent ``(alpha_a + 1) * n`` and a
          global prefactor of 2; the cathode uses n = 1 with an exponent
          ``alpha_c * n`` and a global prefactor of 2.
        * ``"mg"``: the original manuscript form for Mg systems. The anode
          uses n = 2 with an exponent ``alpha_a * n`` (no prefactor); the
          cathode uses n = 1 with an exponent ``alpha_c * n`` (no prefactor).
    """

    exchange_current: float
    equilibrium_potential: float
    alpha: float
    name: str = ""
    label: str = ""
    kinetic_form: str = "fe"


@dataclass
class Composition:
    """A multi-electrode corrosion system.

    Attributes
    ----------
    name : str
        Plain-text composition identifier used in data and prints.
    label : str
        Mathtext label used in legends (falls back to ``name``).
    anode : ElectrodeParameters
        Anodic dissolution electrode.
    cathodes : List[ElectrodeParameters]
        Cathodic reactions (HER, second phases, ...).
    area_ratios : List[float]
        Area fractions in the order [anode, cathode1, cathode2, ...].
        The values must sum to 1.
    """

    name: str
    anode: ElectrodeParameters
    cathodes: List[ElectrodeParameters]
    area_ratios: List[float]
    label: str = ""


class PolarizationCurvePlotter:
    """Plot polarization curves and locate corrosion points.

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

    # ------------------------------------------------------------------ #
    # Plot styling helpers
    # ------------------------------------------------------------------ #
    def _setup_fonts(self) -> None:
        """Configure publication-style fonts."""
        self.label_font = {"family": "Times New Roman", "weight": "light", "size": 28}
        self.legend_font = {"family": "Times New Roman", "weight": "light", "size": 16}
        self.ticks_font = fm.FontProperties(family="Times New Roman", size=20, weight="light")

    def _convert_potential(self, potential: float) -> float:
        """Convert a potential from the SHE scale to the reference electrode."""
        if self.reference_electrode == "SCE":
            return potential - 0.241  # SHE to SCE
        if self.reference_electrode == "SHE":
            return potential
        raise ValueError("reference_electrode must be 'SCE' or 'SHE'")

    # ------------------------------------------------------------------ #
    # Butler-Volmer kinetics
    # ------------------------------------------------------------------ #
    def _butler_volmer_anode(self, potential: np.ndarray, electrode: ElectrodeParameters,
                             area_ratio: float) -> np.ndarray:
        """Anodic Butler-Volmer current density.

        Two empirical parameterizations are supported (see
        :class:`ElectrodeParameters`):

        * ``"mg"``: original manuscript form, n = 2 with an anodic exponent
          ``alpha_a * n`` and no prefactor.
        * ``"fe"``: calibrated Fe demo form, n = 1 with an anodic exponent
          ``(alpha_a + 1) * n`` and a global prefactor of 2.
        """
        alpha_a = electrode.alpha
        E_eq = self._convert_potential(electrode.equilibrium_potential)

        if electrode.kinetic_form == "mg":
            n = 2  # number of transferred electrons
            exponent = alpha_a * n
            prefactor = 1.0
        elif electrode.kinetic_form == "fe":
            n = 1  # number of transferred electrons
            exponent = (alpha_a + 1) * n
            prefactor = 2.0
        else:
            raise ValueError(
                f"Unknown kinetic_form {electrode.kinetic_form!r}; use 'mg' or 'fe'."
            )

        current = (
            prefactor
            * area_ratio
            * electrode.exchange_current
            * (
                np.exp(exponent * self.F * (potential - E_eq) / (self.R * self.temperature))
                - np.exp(-(1 - alpha_a) * n * self.F * (potential - E_eq) / (self.R * self.temperature))
            )
        )
        return current

    def _butler_volmer_cathode(self, potential: np.ndarray, electrode: ElectrodeParameters,
                               area_ratio: float) -> np.ndarray:
        """Cathodic Butler-Volmer current density.

        The hydrogen-evolution reaction is treated as a one-electron process
        (n = 1). The ``"mg"`` form uses no prefactor (original manuscript
        form); the ``"fe"`` form adds a global prefactor of 2 (calibrated Fe
        demo form).
        """
        n = 1  # number of transferred electrons
        alpha_c = electrode.alpha
        E_eq = self._convert_potential(electrode.equilibrium_potential)

        if electrode.kinetic_form == "mg":
            prefactor = 1.0
        elif electrode.kinetic_form == "fe":
            prefactor = 2.0
        else:
            raise ValueError(
                f"Unknown kinetic_form {electrode.kinetic_form!r}; use 'mg' or 'fe'."
            )

        current = (
            prefactor
            * area_ratio
            * electrode.exchange_current
            * (
                np.exp(alpha_c * n * self.F * (potential - E_eq) / (self.R * self.temperature))
                - np.exp(-(1 - alpha_c) * n * self.F * (potential - E_eq) / (self.R * self.temperature))
            )
        )
        return current

    def _adaptive_ticks(self, data_range: Tuple[float, float], n_ticks: int = 6) -> np.ndarray:
        """Generate "nice" (1/2/5 x 10^k) tick positions for a data range."""
        min_val, max_val = data_range
        if max_val < min_val:
            min_val, max_val = max_val, min_val
        span = max_val - min_val
        if span <= 0:
            span = 1.0

        rough_step = span / (n_ticks - 1)
        magnitude = 10 ** np.floor(np.log10(rough_step))
        for mult in (1, 2, 5, 10):
            step = mult * magnitude
            if step >= rough_step - 1e-12:
                break

        start = np.floor(min_val / step) * step
        end = np.ceil(max_val / step) * step
        ticks = np.arange(start, end + step * 0.5, step)
        # Snap floating-point residuals (e.g. -2.22e-16 instead of 0) by
        # rounding every tick to the nearest multiple of the step, and
        # normalize -0.0 (which would format as "-0") to plain 0.0.
        ticks = np.round(ticks / step) * step
        return np.where(ticks == 0.0, 0.0, ticks)

    @staticmethod
    def _log10_ticks(data_range: Tuple[float, float]):
        """Integer power-of-ten tick positions for the log-current axis.

        Returns ``(tick_positions, tick_labels)`` where every label is a
        ``10^n`` mathtext string, or ``None`` when the range contains fewer
        than two integer exponents (the caller then falls back to adaptive
        numeric ticks).
        """
        min_val, max_val = data_range
        if max_val < min_val:
            min_val, max_val = max_val, min_val
        start = int(np.ceil(min_val))
        end = int(np.floor(max_val))
        span = end - start
        if span < 2:
            return None
        if span <= 6:
            step = 1
        elif span <= 12:
            step = 2
        else:
            step = 5
        exponents = np.arange(start, end + 1, step)
        labels = [rf"$\mathregular{{10^{{{int(exp)}}}}}$" for exp in exponents]
        return exponents.astype(float), labels

    @staticmethod
    def _safe_log10(values: np.ndarray, floor: float = -40.0) -> np.ndarray:
        """Log10 of absolute current densities with pathological values masked.

        Currents at or below ``10**floor`` A/cm^2 (numerical zeros near the
        mixed potential) are replaced by NaN so they never participate in the
        axis autoscaling and cannot blow up the log-current window.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            log_values = np.log10(np.abs(values))
        return np.where(np.isfinite(log_values) & (log_values > floor), log_values, np.nan)

    def _setup_figure(self, figsize: Tuple[float, float] = (12, 8)) -> None:
        """Configure the figure axes, labels and spines (no limits yet).

        Axis limits and numeric ticks are applied by :meth:`_finalize_axes`
        after the data has been plotted, so the log-current window always
        follows the actual curves.
        """
        plt.rcdefaults()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.figure(figsize=figsize)
        plt.rcParams["figure.dpi"] = 128

        plt.ylabel(f"E (V vs. {self.reference_electrode})", self.label_font)
        plt.xlabel(r"i $\mathregular{(A\cdot cm^{-2})}$", self.label_font, labelpad=4)

        plt.yticks(fontproperties=self.ticks_font)
        plt.xticks(fontproperties=self.ticks_font)

        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

        ax.minorticks_on()
        plt.tick_params(which="major", size=9, width=1.5)
        plt.tick_params(which="minor", size=5, width=1.5)

    def _finalize_axes(self, xlim: Optional[Tuple[float, float]] = None,
                       ylim: Optional[Tuple[float, float]] = None,
                       log_centers: Optional[Sequence[float]] = None,
                       potential_centers: Optional[Sequence[float]] = None,
                       window_pad: float = 3.0,
                       potential_pad: float = 0.5,
                       x_ticks: Optional[Sequence[float]] = None,
                       y_ticks: Optional[Sequence[float]] = None) -> None:
        """Apply axis limits and log-current ticks after plotting.

        Parameters
        ----------
        xlim, ylim : tuple, optional
            Explicit axis limits. When omitted, the corresponding window is
            centered on the corrosion point(s) (``log_centers`` for the
            log-current axis, ``potential_centers`` for the potential axis)
            and padded by ``window_pad`` decades / ``potential_pad`` volts on
            each side. This keeps the mixed-potential region filling the axes
            instead of being crushed into a corner by the extreme branch
            currents, which can span dozens of orders of magnitude.
        window_pad : float
            Half-width (in decades) of the log-current window.
        potential_pad : float
            Half-width (in volts) of the potential window.
        x_ticks, y_ticks : sequence, optional
            Explicit tick positions. The x axis is labelled with ``10^n``
            mathtext; when ``x_ticks`` is omitted, integer power-of-ten
            positions inside the current window are chosen automatically.
        """
        ax = plt.gca()
        if xlim is None and log_centers is not None:
            centers = np.asarray(list(log_centers), dtype=float)
            xlim = (float(centers.min()) - window_pad, float(centers.max()) + window_pad)
        if ylim is None and potential_centers is not None:
            centers = np.asarray(list(potential_centers), dtype=float)
            ylim = (float(centers.min()) - potential_pad, float(centers.max()) + potential_pad)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # ---- log-current (x) axis: integer powers of ten ----
        if x_ticks is None:
            log_ticks = self._log10_ticks(ax.get_xlim())
        else:
            log_ticks = (np.asarray(x_ticks, dtype=float),
                         [rf"$\mathregular{{10^{{{int(t)}}}}}$" for t in x_ticks])
        if log_ticks is None:
            x_ticks_used = self._adaptive_ticks(ax.get_xlim(), n_ticks=7)
            x_labels = [f"{tick:g}" for tick in x_ticks_used]
        else:
            x_ticks_used, x_labels = log_ticks
        if xlim is None:
            ax.set_xlim(x_ticks_used[0], x_ticks_used[-1])
        ax.set_xticks(x_ticks_used)
        ax.set_xticklabels(x_labels, fontproperties=self.ticks_font)

        # ---- potential (y) axis ----
        if y_ticks is None:
            y_ticks_used = self._adaptive_ticks(ax.get_ylim(), n_ticks=7)
        else:
            y_ticks_used = np.asarray(y_ticks, dtype=float)
        if ylim is None:
            ax.set_ylim(y_ticks_used[0], y_ticks_used[-1])
        ax.set_yticks(y_ticks_used)
        ax.set_yticklabels([f"{tick:g}" for tick in y_ticks_used], fontproperties=self.ticks_font)

    # ------------------------------------------------------------------ #
    # Corrosion point
    # ------------------------------------------------------------------ #
    def _calculate_corrosion_point(self, composition: Composition,
                                   potential_range: np.ndarray) -> Tuple[float, float]:
        """Locate the mixed potential (E_corr) and anodic current (i_corr)."""
        anode_current = self._butler_volmer_anode(potential_range, composition.anode, composition.area_ratios[0])
        cathode_current = np.zeros_like(potential_range)

        for i, cathode in enumerate(composition.cathodes):
            cathode_current += self._butler_volmer_cathode(potential_range, cathode, composition.area_ratios[i + 1])

        total_current = anode_current + cathode_current
        min_idx = np.argmin(np.abs(total_current))

        return potential_range[min_idx], anode_current[min_idx]

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #
    def plot_single_composition(self, composition: Composition,
                                potential_range: Tuple[float, float] = (-3, 1),
                                n_points: int = 500,
                                show_corrosion_point: bool = True,
                                xlim: Optional[Tuple[float, float]] = None,
                                ylim: Optional[Tuple[float, float]] = None,
                                figsize: Tuple[float, float] = (10, 8),
                                title: Optional[str] = None) -> plt.Figure:
        """
        Plot the polarization curve of a single composition.

        Parameters
        ----------
        composition : Composition
            Electrode system to plot.
        potential_range : tuple
            Potential window (V) on the reference-electrode scale.
        n_points : int
            Number of potential grid points.
        show_corrosion_point : bool
            Whether to mark E_corr / i_corr on the plot.
        xlim, ylim : tuple, optional
            Axis limits for the log-current / potential axes.
        figsize : tuple
            Figure size.
        title : str, optional
            Figure title.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        U = np.linspace(potential_range[0], potential_range[1], n_points)

        anode_current = self._butler_volmer_anode(U, composition.anode, composition.area_ratios[0])
        cathode_currents = []
        total_cathode_current = np.zeros_like(U)

        for i, cathode in enumerate(composition.cathodes):
            cath_current = self._butler_volmer_cathode(U, cathode, composition.area_ratios[i + 1])
            cathode_currents.append(cath_current)
            total_cathode_current += cath_current

        total_current = anode_current + total_cathode_current

        anode_log = self._safe_log10(anode_current)
        cathode_logs = [self._safe_log10(cath_current) for cath_current in cathode_currents]
        total_log = self._safe_log10(total_current)

        self._setup_figure(figsize=figsize)

        plt.plot(anode_log, U,
                 label=f"{composition.anode.label or composition.anode.name} (Anode)", linewidth=2, linestyle="--")

        for cathode, cath_log in zip(composition.cathodes, cathode_logs):
            plt.plot(cath_log, U,
                     label=f"{cathode.label or cathode.name} (Cathode)", linewidth=2, linestyle="--")

        plt.plot(total_log, U,
                 label="Total Current", color="black", linewidth=3)

        log_centers = None
        potential_centers = None
        if show_corrosion_point:
            E_corr, I_corr = self._calculate_corrosion_point(composition, U)
            log_i_corr = float(np.log10(np.abs(I_corr)))
            log_centers = [log_i_corr]
            potential_centers = [float(E_corr)]
            plt.scatter(log_i_corr, E_corr, color="red", s=100, zorder=5)
            plt.text(log_i_corr + 0.3, E_corr + 0.1,
                     f"$E_{{corr}}$ = {E_corr:.3f} V\n$i_{{corr}}$ = {np.abs(I_corr):.2e} A/cm$^{{2}}$",
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        self._finalize_axes(xlim=xlim, ylim=ylim,
                            log_centers=log_centers, potential_centers=potential_centers)
        plt.legend(prop={"family": "Times New Roman", "size": 16, "weight": "bold"})
        if title:
            plt.title(title, fontsize=20, fontfamily="Times New Roman", fontweight="bold")

        plt.tight_layout()
        return plt.gcf()

    def plot_multiple_compositions(self, compositions: List[Composition],
                                   potential_range: Tuple[float, float] = (-2.5, 0),
                                   n_points: int = 500,
                                   show_corrosion_points: bool = True,
                                   xlim: Optional[Tuple[float, float]] = None,
                                   ylim: Optional[Tuple[float, float]] = None,
                                   figsize: Tuple[float, float] = (12, 8),
                                   title: Optional[str] = None) -> plt.Figure:
        """
        Plot polarization curves of several compositions for comparison.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        U = np.linspace(potential_range[0], potential_range[1], n_points)
        self._setup_figure(figsize=figsize)

        colors = ["#989EDC", "#989EDC", "#36718A", "#96BAC9", "#38917F",
                  "#A2D3A3", "#C49505", "#E1CA82", "#DA7271", "#F0C7C6"]
        linestyles = ["-", "--", "-.", ":"]

        corrosion_data = []
        log_centers: List[float] = []
        potential_centers: List[float] = []

        for i, composition in enumerate(compositions):
            anode_current = self._butler_volmer_anode(U, composition.anode, composition.area_ratios[0])
            total_cathode_current = np.zeros_like(U)

            for j, cathode in enumerate(composition.cathodes):
                total_cathode_current += self._butler_volmer_cathode(U, cathode, composition.area_ratios[j + 1])

            total_current = anode_current + total_cathode_current
            total_log = self._safe_log10(total_current)

            color = colors[i % len(colors)]
            linestyle = linestyles[i % len(linestyles)]

            plt.plot(total_log, U,
                     label=composition.label or composition.name, color=color, linewidth=3, linestyle=linestyle)

            if show_corrosion_points:
                E_corr, I_corr = self._calculate_corrosion_point(composition, U)
                corrosion_data.append((E_corr, I_corr, composition.name, color))
                log_centers.append(float(np.log10(np.abs(I_corr))))
                potential_centers.append(float(E_corr))

        if show_corrosion_points:
            for E_corr, I_corr, name, color in corrosion_data:
                plt.scatter(np.log10(np.abs(I_corr)), E_corr, color=color, s=150,
                            marker="o", edgecolors="black", zorder=5)
                print(f"{name}: E_corr = {E_corr:.3f} V, i_corr = {np.abs(I_corr):.2e} A/cm^2")

        self._finalize_axes(xlim=xlim, ylim=ylim,
                            log_centers=log_centers or None,
                            potential_centers=potential_centers or None)
        plt.legend(prop={"family": "Times New Roman", "size": 16, "weight": "bold"})
        if title:
            plt.title(title, fontsize=20, fontfamily="Times New Roman", fontweight="bold")

        plt.tight_layout()
        return plt.gcf()

    def plot_multi_anode_single_cathode(self,
                                        composition_name: str,
                                        anodes: List[ElectrodeParameters],
                                        cathode: ElectrodeParameters,
                                        area_ratios: List[float],
                                        potential_range: Tuple[float, float] = (-3, 1),
                                        n_points: int = 2000,
                                        show_corrosion_point: bool = True,
                                        xlim: Optional[Tuple[float, float]] = None,
                                        ylim: Optional[Tuple[float, float]] = None,
                                        figsize: Tuple[float, float] = (10, 8),
                                        title: Optional[str] = None,
                                        anode_labels: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot the polarization curve of multiple anodes against one cathode.

        Parameters
        ----------
        composition_name : str
            Label of the alloy system.
        anodes : List[ElectrodeParameters]
            Anodic electrodes.
        cathode : ElectrodeParameters
            Single cathodic electrode.
        area_ratios : List[float]
            Area fractions ``[anode1, anode2, ..., cathode]``; must sum to 1.
        potential_range : tuple
            Potential window (V).
        n_points : int
            Number of potential grid points.
        show_corrosion_point : bool
            Whether to mark the corrosion point with a marker. The axis
            window is always centered on the corrosion point so the
            mixed-potential region fills the axes.
        xlim, ylim : tuple, optional
            Axis limits.
        figsize : tuple
            Figure size.
        title : str, optional
            Figure title.
        anode_labels : list, optional
            Custom labels for the anodes.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        if len(area_ratios) != len(anodes) + 1:
            raise ValueError(
                f"area_ratios must contain {len(anodes) + 1} entries "
                f"(anodes + cathode), got {len(area_ratios)}."
            )

        U = np.linspace(potential_range[0], potential_range[1], n_points)

        anode_currents = []
        for i, anode in enumerate(anodes):
            anode_currents.append(self._butler_volmer_anode(U, anode, area_ratios[i]))

        cathode_current = self._butler_volmer_cathode(U, cathode, area_ratios[-1])

        self._setup_figure(figsize=figsize)

        if anode_labels is None:
            anode_labels = [anode.label or anode.name for anode in anodes]

        colors = ["#A47DC0", "#989EDC", "#36718A", "#96BAC9", "#38917F",
                  "#A2D3A3", "#C49505", "#E1CA82", "#DA7271", "#F0C7C6"]

        cathode_log = self._safe_log10(cathode_current)
        anode_logs = [self._safe_log10(an_current) for an_current in anode_currents]

        plt.plot(cathode_log, U,
                 label=cathode.label or cathode.name, linewidth=2, linestyle="dotted", color="k")

        for i, (anode, an_log, label) in enumerate(zip(anodes, anode_logs, anode_labels)):
            if i == 0:
                plt.plot(an_log, U,
                         label=label, linewidth=2, linestyle="-", color="k")
            else:
                plt.plot(an_log, U,
                         label=label, linewidth=2, linestyle="-.", color=colors[i - 1])

        # Always center the axes on the corrosion point (keeps the
        # mixed-potential region from being crushed into a corner by the
        # extreme branch currents), but only draw a marker on request.
        E_corr, I_corr = self._calculate_corrosion_point_multi_anode(
            anodes, cathode, area_ratios, U
        )
        log_centers = [float(np.log10(np.abs(I_corr)))]
        potential_centers = [float(E_corr)]
        if show_corrosion_point:
            plt.scatter(log_centers[0], E_corr, color="red", s=100,
                        marker="o", edgecolors="black", zorder=5)

        self._finalize_axes(xlim=xlim, ylim=ylim,
                            log_centers=log_centers, potential_centers=potential_centers)
        plt.legend(prop={"family": "Times New Roman", "size": 16, "weight": "bold"})

        if title:
            plt.title(title, fontsize=20, fontfamily="Times New Roman", fontweight="bold")

        plt.tight_layout()
        return plt.gcf()

    def _calculate_corrosion_point_multi_anode(self,
                                               anodes: List[ElectrodeParameters],
                                               cathode: ElectrodeParameters,
                                               area_ratios: List[float],
                                               potential_range: np.ndarray) -> Tuple[float, float]:
        """Locate the corrosion point of a multi-anode system."""
        total_anode_current = np.zeros_like(potential_range)
        for i, anode in enumerate(anodes):
            total_anode_current += self._butler_volmer_anode(potential_range, anode, area_ratios[i])

        cathode_current = self._butler_volmer_cathode(potential_range, cathode, area_ratios[-1])

        total_current = total_anode_current + cathode_current
        min_idx = np.argmin(np.abs(total_current))

        return potential_range[min_idx], total_anode_current[min_idx]

    @staticmethod
    def create_multi_anode_composition(name: str,
                                       anodes: List[ElectrodeParameters],
                                       cathodes: List[ElectrodeParameters],
                                       area_ratios: List[float]) -> Composition:
        """
        Build a Composition object from multiple anodes (kept for
        compatibility with the legacy single-anode representation).

        The first anode becomes ``Composition.anode``; the remaining anodes
        are stored in the ``additional_anodes`` attribute.
        """
        if len(anodes) == 0:
            raise ValueError("At least one anode is required.")

        composition = Composition(
            name=name,
            anode=anodes[0],
            cathodes=cathodes,
            area_ratios=area_ratios,
        )
        composition.additional_anodes = anodes[1:] if len(anodes) > 1 else []
        return composition


# ---------------------------------------------------------------------- #
# Predefined Mg-based compositions
# ---------------------------------------------------------------------- #
# Exchange current densities (A/cm^2) of the cathodic reactions on the
# Mg matrix and on common Mg second phases. The manuscript reports these
# as "10e-X", which means i0 = 10^(-X).
MG_CATHODE_CURRENT_LOG10 = {
    "Mg": 8.078243758202708,
    "LaMg12": 8.247522161899012,
    "Mg2Al3": 5.667729095161651,
    "MgZn2": 5.706912185308812,
    "CaMg2": 11.405152326499127,
    "Y5Mg24": 7.740551780321441,
    "NdMg3": 9.454721303657884,
    "Mg2Si": 8.792102970038897,
    "CeMg12": 8.029754467287972,
    "Mg17Al12": 10.95328903331877,
}


def _cathode(phase: str, name: Optional[str] = None, label: Optional[str] = None,
             equilibrium_potential: float = -0.65, alpha: float = 0.5) -> ElectrodeParameters:
    """Build a cathodic electrode from the predefined Mg-phase table."""
    log10_i0 = MG_CATHODE_CURRENT_LOG10[phase]
    return ElectrodeParameters(
        exchange_current=10 ** -log10_i0,
        equilibrium_potential=equilibrium_potential,  # vs SHE (pH 11)
        alpha=alpha,
        name=name or phase,
        label=label or name or phase,
        kinetic_form="mg",  # original manuscript form (no prefactor)
    )


def create_mg_based_compositions() -> List[Composition]:
    """
    Create example Mg-based compositions parameterized with the
    DFT-derived exchange currents used in the GalvCalc manuscript.

    The Mg dissolution anode uses i0 = 10^-22.2 A/cm^2, alpha_a = 0.55 and
    n = 2. The pure-Mg anode/HER cathode area ratio is [0.158, 0.842].
    Area ratios of the second-phase alloys are illustrative examples and
    should be adjusted for a specific system.
    """
    mg_anode = ElectrodeParameters(
        exchange_current=10 ** -22.2,
        equilibrium_potential=-2.37,  # vs SHE
        alpha=0.55,
        name="Mg/Mg2+",
        label=r"Mg/Mg$^{2+}$",
        kinetic_form="mg",  # n = 2, exponent alpha_a * n, no prefactor
    )

    # Cathodic hydrogen evolution on the Mg matrix.
    h_evolution = _cathode("Mg", name="H+/H2 (Mg)", label=r"H$^{+}$/H$_{2}$ (Mg)", alpha=0.77)

    # Common Mg second phases acting as cathodes.
    mg17al12_cathode = _cathode("Mg17Al12", name="Mg17Al12", label=r"Mg$_{17}$Al$_{12}$")
    mg2al3_cathode = _cathode("Mg2Al3", name="Mg2Al3", label=r"Mg$_{2}$Al$_{3}$")
    mgzn2_cathode = _cathode("MgZn2", name="MgZn2", label=r"MgZn$_{2}$")
    lamg12_cathode = _cathode("LaMg12", name="LaMg12", label=r"LaMg$_{12}$")
    camg2_cathode = _cathode("CaMg2", name="CaMg2", label=r"CaMg$_{2}$")
    y5mg24_cathode = _cathode("Y5Mg24", name="Y5Mg24", label=r"Y$_{5}$Mg$_{24}$")
    ndmg3_cathode = _cathode("NdMg3", name="NdMg3", label=r"NdMg$_{3}$")
    mg2si_cathode = _cathode("Mg2Si", name="Mg2Si", label=r"Mg$_{2}$Si")
    cemg12_cathode = _cathode("CeMg12", name="CeMg12", label=r"CeMg$_{12}$")

    return [
        Composition(
            name="Pure Mg",
            anode=mg_anode,
            cathodes=[h_evolution],
            area_ratios=[0.158, 0.842],
        ),
        Composition(
            name="Mg-Al (Mg17Al12)", label=r"Mg-Al (Mg$_{17}$Al$_{12}$)",
            anode=mg_anode,
            cathodes=[h_evolution, mg17al12_cathode],
            area_ratios=[0.6, 0.3, 0.1],
        ),
        Composition(
            name="Mg-Al (Mg2Al3)", label=r"Mg-Al (Mg$_{2}$Al$_{3}$)",
            anode=mg_anode,
            cathodes=[h_evolution, mg2al3_cathode],
            area_ratios=[0.6, 0.3, 0.1],
        ),
        Composition(
            name="Mg-Zn (MgZn2)", label=r"Mg-Zn (MgZn$_{2}$)",
            anode=mg_anode,
            cathodes=[h_evolution, mgzn2_cathode],
            area_ratios=[0.7, 0.2, 0.1],
        ),
        Composition(
            name="Mg-La (LaMg12)", label=r"Mg-La (LaMg$_{12}$)",
            anode=mg_anode,
            cathodes=[h_evolution, lamg12_cathode],
            area_ratios=[0.65, 0.25, 0.1],
        ),
        Composition(
            name="Mg-Ca (CaMg2)", label=r"Mg-Ca (CaMg$_{2}$)",
            anode=mg_anode,
            cathodes=[h_evolution, camg2_cathode],
            area_ratios=[0.6, 0.3, 0.1],
        ),
        Composition(
            name="Mg-Y (Y5Mg24)", label=r"Mg-Y (Y$_{5}$Mg$_{24}$)",
            anode=mg_anode,
            cathodes=[h_evolution, y5mg24_cathode],
            area_ratios=[0.6, 0.3, 0.1],
        ),
        Composition(
            name="Mg-Nd (NdMg3)", label=r"Mg-Nd (NdMg$_{3}$)",
            anode=mg_anode,
            cathodes=[h_evolution, ndmg3_cathode],
            area_ratios=[0.6, 0.3, 0.1],
        ),
        Composition(
            name="Mg-Si (Mg2Si)", label=r"Mg-Si (Mg$_{2}$Si)",
            anode=mg_anode,
            cathodes=[h_evolution, mg2si_cathode],
            area_ratios=[0.6, 0.3, 0.1],
        ),
        Composition(
            name="Mg-Ce (CeMg12)", label=r"Mg-Ce (CeMg$_{12}$)",
            anode=mg_anode,
            cathodes=[h_evolution, cemg12_cathode],
            area_ratios=[0.6, 0.3, 0.1],
        ),
    ]


# ---------------------------------------------------------------------- #
# Manuscript-style Mg second-phase comparison
# ---------------------------------------------------------------------- #
# Parameters of the paper's multi-second-phase example. Exchange currents
# are reported in the manuscript as "10e-X", i.e. i0 = 10^(-X). Area ratios
# follow the order [alpha-Mg anode, alpha-Mg HER cathode, second-phase
# cathode] and sum to one for every composition. Potentials are given on
# the SCE scale used by the manuscript (pH ~10.3, E_c0 = -0.61 V).

MG_SECOND_PHASE_ANODE_ALPHA = 0.55    # fixed anodic transfer coefficient
MG_HER_ALPHA = 0.77                   # HER transfer coefficient on Mg
MG_HER_I0 = 10 ** -8.1                # exchange current tagging the Mg HER cathode
MG_HER_EQUILIBRIUM_POTENTIAL = -0.61  # V vs. SCE (pH ~10.3)


def mg_second_phase_example() -> List[Composition]:
    """Nine Mg/second-phase compositions of the manuscript example.

    Every composition shares the Mg dissolution anode (i0 = 6e-23 A/cm^2,
    alpha_a = 0.55, n = 2) and the hydrogen-evolution cathode on the alpha
    matrix (i0 = 10^-8.1 A/cm^2, alpha_c = 0.77); a phase-specific cathode
    with its own exchange current and transfer coefficient is added on top.
    Area ratios are taken from the paper's example.
    """
    anode = ElectrodeParameters(
        exchange_current=6e-23,
        equilibrium_potential=-2.37,  # V vs. SCE (paper scale)
        alpha=MG_SECOND_PHASE_ANODE_ALPHA,
        name="Mg/Mg2+",
        label=r"Mg/Mg$^{2+}$",
        kinetic_form="mg",  # n = 2, exponent alpha_a * n
    )
    her = ElectrodeParameters(
        exchange_current=MG_HER_I0,
        equilibrium_potential=MG_HER_EQUILIBRIUM_POTENTIAL,
        alpha=MG_HER_ALPHA,
        name="H+/H2 (Mg)",
        label=r"H$^{+}$/H$_{2}$ (Mg)",
        kinetic_form="mg",
    )

    def _phase(phase, label, log10_i0, alpha, a_anode, a_her, a_phase):
        second = ElectrodeParameters(
            exchange_current=10 ** -log10_i0,
            equilibrium_potential=MG_HER_EQUILIBRIUM_POTENTIAL,
            alpha=alpha,
            name=phase,
            label=label,
            kinetic_form="mg",
        )
        return Composition(
            name=phase,
            label=label,
            anode=anode,
            cathodes=[her, second],
            area_ratios=[a_anode, a_her, a_phase],
        )

    return [
        _phase("Mg17Al12", r"$\mathregular{\alpha_{Mg}+Mg_{17}Al_{12}}$",
               10.95328903331877, 0.351, 0.99 * 0.996, 0.99 * (1 - 0.996), 0.01),
        _phase("CeMg12", r"$\mathregular{\alpha_{Mg}+Mg_{12}Ce}$",
               8.029754467287972, 0.705, 0.99 * 0.177, 0.99 * (1 - 0.177), 0.01),
        _phase("Mg2Si", r"$\mathregular{\alpha_{Mg}+Mg_{2}Si}$",
               8.792102970038897, 0.653, 0.99 * 0.179, 0.99 * (1 - 0.179), 0.01),
        _phase("NdMg3", r"$\mathregular{\alpha_{Mg}+Mg_{3}Nd}$",
               9.454721303657884, 0.635, 0.99 * 0.17, 0.99 * (1 - 0.17), 0.01),
        _phase("Y5Mg24", r"$\mathregular{\alpha_{Mg}+Mg_{24}Y_{5}}$",
               7.740551780321441, 0.663, 0.99 * 0.295, 0.99 * (1 - 0.295), 0.01),
        _phase("CaMg2", r"$\mathregular{\alpha_{Mg}+Mg_{2}Ca}$",
               11.405152326499127, 0.483, 0.99 * 0.167, 0.99 * (1 - 0.167), 0.01),
        _phase("MgZn2", r"$\mathregular{\alpha_{Mg}+MgZn_{2}}$",
               5.706912185308812, 0.496, 0.99 * 0.992, 0.99 * (1 - 0.992), 0.01),
        _phase("Mg2Al3", r"$\mathregular{\alpha_{Mg}+Mg_{2}Al_{3}}$",
               5.667729095161651, 0.684, 0.99 * 0.999, 0.99 * (1 - 0.999), 0.01),
        _phase("LaMg12", r"$\mathregular{\alpha_{Mg}+Mg_{12}La}$",
               8.247522161899012, 0.668, 0.99 * 0.194, 0.99 * (1 - 0.194), 0.01),
    ]


def plot_mg_second_phases(
    compositions: Optional[List[Composition]] = None,
    figsize: Tuple[float, float] = (9.5, 7.6),
    xlim: Tuple[float, float] = (-8.0, 0.0),
    ylim: Tuple[float, float] = (-2.8, -1.0),
    include_mg: bool = True,
    n_points: int = 3000,
    n_potentials: int = 50000,
    show_corrosion_points: bool = False,
    title: Optional[str] = None,
) -> plt.Figure:
    """Reproduce the manuscript's multi-second-phase polarization figure.

    The anodic dissolution of the alpha-Mg matrix (n = 2, alpha_a = 0.55)
    is balanced against the hydrogen-evolution cathode on the matrix and a
    phase-specific second-phase cathode. The kinetic equations and the
    parameter values replicate the paper's ``diff_wt`` example.

    Parameters
    ----------
    compositions : list of Composition, optional
        Second-phase systems; defaults to :func:`mg_second_phase_example`.
    figsize : tuple
        Figure size in inches.
    xlim, ylim : tuple
        Log-current and potential windows.
    include_mg : bool
        Overlay the pure-Mg reference curve.
    n_points : int
        Number of potential points used to draw the curves.
    n_potentials : int
        Number of grid points used to locate each corrosion point.
    show_corrosion_points : bool
        Mark the located corrosion points on the figure.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure (``fig.corrosion_points`` lists the located
        ``{name, E_corr, i_corr}`` entries).
    """
    if compositions is None:
        compositions = mg_second_phase_example()

    plt.rcdefaults()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["figure.dpi"] = 300
    plt.figure(figsize=figsize)

    label_font = {"family": "Times New Roman", "weight": "light", "size": 24}
    legend_font = {"family": "Times New Roman", "weight": "light", "size": 18}
    ticks_font = fm.FontProperties(family="Times New Roman", size=18, weight="light")

    plt.ylabel("E (V vs. SCE)", label_font, labelpad=5)
    plt.xlabel(r"i $\mathregular{(A\cdot cm^{-2})}$", label_font, labelpad=5)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.minorticks_on()
    plt.tick_params(which="major", size=9, width=1.5)
    plt.tick_params(which="minor", size=5, width=1.5)

    F = 96485.0
    R = 8.314
    T = 300.0
    n = 1
    alpha_a = MG_SECOND_PHASE_ANODE_ALPHA
    alpha_c00 = MG_HER_ALPHA
    E_c0 = MG_HER_EQUILIBRIUM_POTENTIAL

    U = np.linspace(-3.0, 0.0, n_points)  # paper's potential sweep window

    # ---- pure-Mg reference curve ----
    if include_mg:
        I_a00 = 6e-23
        E_a00 = -2.37
        I_c00 = MG_HER_I0
        E_c00 = MG_HER_EQUILIBRIUM_POTENTIAL
        mg_curve = np.abs(
            0.158 * I_a00 * np.exp(2 * alpha_a * n * F * (U - E_a00) / (R * T))
            - 0.842 * I_c00 * np.exp(-(1 - alpha_c00) * n * F * (U - E_c00) / (R * T))
        )
        plt.plot(np.log10(mg_curve), U,
                 label=r"$\mathregular{\alpha_{Mg}}$", lw=2, color="k")

    colors = ["#A47DC0", "#989EDC", "#36718A", "#96BAC9", "#38917F",
              "#A2D3A3", "#C49505", "#E1CA82", "#DA7271", "#F0C7C6"]

    def _branch_alpha(cathode: ElectrodeParameters) -> float:
        """Return the transfer coefficient of a cathodic branch.

        The hydrogen-evolution cathode on the alpha-Mg matrix is tagged by
        its exchange current (10^-8.1 A/cm^2) and always uses the fixed
        matrix alpha (0.77); phase-specific cathodes use their own alpha.
        """
        if np.isclose(cathode.exchange_current, MG_HER_I0, rtol=0, atol=1e-12):
            return alpha_c00
        return cathode.alpha

    corrosion_points = []
    for w, comp in enumerate(compositions):
        area = comp.area_ratios
        I_a0 = comp.anode.exchange_current
        E_a0 = comp.anode.equilibrium_potential

        # ---- full branches (used for the corrosion-point search) ----
        Ia = area[0] * I_a0 * (
            np.exp(2 * alpha_a * n * F * (U - E_a0) / (R * T))
            - np.exp(-2 * (1 - alpha_a) * n * F * (U - E_a0) / (R * T))
        )
        Ic = np.zeros_like(U)
        for i, cathode in enumerate(comp.cathodes):
            alpha_c = _branch_alpha(cathode)
            Ic += area[i + 1] * cathode.exchange_current * (
                np.exp(alpha_c * n * F * (U - E_c0) / (R * T))
                - np.exp(-(1 - alpha_c) * n * F * (U - E_c0) / (R * T))
            )

        # ---- plotted net current (paper's forward-anode/backward-cathode form) ----
        net = area[0] * I_a0 * np.exp(2 * alpha_a * n * F * (U - E_a0) / (R * T))
        for i, cathode in enumerate(comp.cathodes):
            alpha_c = _branch_alpha(cathode)
            net -= area[i + 1] * cathode.exchange_current * np.exp(
                -(1 - alpha_c) * n * F * (U - E_c0) / (R * T)
            )
        net = np.abs(net)

        color = colors[w % len(colors)]
        plt.plot(np.log10(net), U,
                 label=comp.label or comp.name, lw=2, c=color)

        # ---- corrosion point: minimum |Ia + Ic| on a dense grid ----
        E_grid = np.linspace(E_a0, E_c0, n_potentials)
        Ia_grid = area[0] * I_a0 * (
            np.exp(2 * alpha_a * n * F * (E_grid - E_a0) / (R * T))
            - np.exp(-2 * (1 - alpha_a) * n * F * (E_grid - E_a0) / (R * T))
        )
        Ic_grid = np.zeros_like(E_grid)
        for i, cathode in enumerate(comp.cathodes):
            alpha_c = _branch_alpha(cathode)
            Ic_grid += area[i + 1] * cathode.exchange_current * (
                np.exp(alpha_c * n * F * (E_grid - E_c0) / (R * T))
                - np.exp(-(1 - alpha_c) * n * F * (E_grid - E_c0) / (R * T))
            )
        total_grid = np.abs(Ia_grid + Ic_grid)
        idx = int(np.argmin(total_grid))
        E_corr = float(E_grid[idx])
        I_corr = float(Ia_grid[idx])
        corrosion_points.append(
            {"name": comp.name, "E_corr": E_corr, "i_corr": I_corr}
        )
        print(f"{comp.name}: E_corr = {E_corr:.3f} V, "
              f"i_corr = {I_corr:.2e} A/cm^2 "
              f"(log10 = {np.log10(I_corr):.3f})")
        if show_corrosion_points:
            plt.scatter(np.log10(I_corr), E_corr, s=40, c=color,
                        edgecolors="black", linewidths=0.6, zorder=5)

    # ---- axes ----
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    exponents = np.arange(int(np.ceil(xlim[0])), int(np.floor(xlim[1])) + 1, 2)
    x_labels = [rf"$\mathregular{{10^{{{int(exp)}}}}}$" for exp in exponents]
    plt.xticks(exponents, x_labels, fontproperties=ticks_font)
    plt.yticks([-2.8, -2.2, -1.6, -1.0], fontproperties=ticks_font)

    plt.legend(frameon=False, loc="lower left", prop=legend_font)
    if title:
        plt.title(title, fontsize=20)
    plt.tight_layout()

    fig = plt.gcf()
    fig.corrosion_points = corrosion_points
    return fig


# ---------------------------------------------------------------------- #
# Convenience wrappers
# ---------------------------------------------------------------------- #
def plot_single_polarization(composition: Composition, reference_electrode: str = "SCE",
                             **kwargs) -> plt.Figure:
    """Quickly plot the polarization curve of a single composition."""
    plotter = PolarizationCurvePlotter(reference_electrode=reference_electrode)
    return plotter.plot_single_composition(composition, **kwargs)


def plot_comparison_polarization(compositions: List[Composition], reference_electrode: str = "SCE",
                                 **kwargs) -> plt.Figure:
    """Quickly plot and compare several compositions."""
    plotter = PolarizationCurvePlotter(reference_electrode=reference_electrode)
    return plotter.plot_multiple_compositions(compositions, **kwargs)


if __name__ == "__main__":
    compositions = create_mg_based_compositions()

    fig1 = plot_single_polarization(
        compositions[0],
        reference_electrode="SCE",
        xlim=(-6, 2),
        ylim=(-3, 0),
    )
    plt.show()

    fig2 = plot_comparison_polarization(
        compositions,
        reference_electrode="SCE",
        xlim=(-8, 5),
        ylim=(-2.7, -1.5),
    )
    plt.show()
