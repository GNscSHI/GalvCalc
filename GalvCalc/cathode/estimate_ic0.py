"""
Estimate facet-weighted exchange-current densities (i_c0) for cathodic
hydrogen evolution.

This module implements Wulff-shape weighting of facet-resolved
electrochemical descriptors (surface energies, work functions and hydrogen
adsorption free energies) following the GalvCalc manuscript.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.constants as con
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Structure

# Reference work function (eV) used to align hydrogen adsorption free
# energies across facets with different work functions.
REFERENCE_WORK_FUNCTION = 3.614

# Dimensionless prefactor of the BEP-type exchange-current expression.
I0_PREFACTOR = 10**10.5

# Site-record keys that may carry the adsorption free energy. "Gads" is the
# native key of the current pipeline; "Eads" is kept for legacy notebooks.
_ADSORPTION_ENERGY_KEYS = ("Gads", "Eads")


def ic0(delta_G: Optional[float] = None, pH: float = 11.0, wf: float = REFERENCE_WORK_FUNCTION) -> float:
    """
    Estimate the exchange current density of hydrogen evolution from a
    hydrogen adsorption free energy using a BEP-type relation.

    Parameters
    ----------
    delta_G : float
        Hydrogen adsorption free energy (eV) of the active site. The value
        is aligned to the facet work function using REFERENCE_WORK_FUNCTION
        as the reference level.
    pH : float
        Solution pH. Kept in the signature for consistency with the
        electrochemical conventions of the manuscript; the Nernst-type pH
        dependence enters through the equilibrium potential (handled by the
        polarization module) rather than through i0 itself.
    wf : float
        Work function of the facet (eV).

    Returns
    -------
    float
        Exchange current density i0 (A/cm^2).
    """
    if delta_G is None:
        raise ValueError("delta_G must be provided (in eV).")

    e = con.e
    k = con.k
    h = con.h
    T = 298.15
    kT = k * T / e

    # Align the adsorption free energy to the facet work function.
    delta_G = delta_G - (wf - REFERENCE_WORK_FUNCTION)

    if delta_G < 0:
        # Strong-binding branch of the BEP relation.
        Ea = -1.19 * delta_G + 0.58
        i0 = e * I0_PREFACTOR * (k * T / h) / (1 + math.exp(-delta_G / kT)) * math.exp(-Ea / kT)
    else:
        # Weak-binding branch of the BEP relation.
        Ea = 0.51 * delta_G + 0.72
        i0 = (
            e
            * I0_PREFACTOR
            * (k * T / h)
            / (1 + math.exp(-delta_G / kT))
            * math.exp(-delta_G / kT)
            * math.exp(-Ea / kT)
        )
    return i0


def ic0_mg(
    delta_H: float = 0.0,
    delta_G: Optional[float] = None,
    correction: float = 0.19,
    wf: float = REFERENCE_WORK_FUNCTION,
) -> Tuple[float, float]:
    """
    Backward-compatible wrapper around :func:`ic0`.

    The legacy interface accepts adsorption enthalpies (``delta_H``) together
    with an entropy-type correction and returns both i0 and the
    work-function-aligned adsorption free energy.

    Parameters
    ----------
    delta_H : float
        Hydrogen adsorption enthalpy (eV).
    delta_G : float, optional
        Hydrogen adsorption free energy (eV). When given, it takes
        precedence over ``delta_H``.
    correction : float
        Free-energy correction applied to ``delta_H`` (eV).
    wf : float
        Work function of the facet (eV).

    Returns
    -------
    tuple
        ``(i0, delta_G_aligned)`` where ``delta_G_aligned`` is the adsorption
        free energy after the work-function alignment.
    """
    if delta_G is None:
        delta_G = delta_H + correction if delta_H else 0.0
    i0 = ic0(delta_G=delta_G, wf=wf)
    delta_G_aligned = delta_G - (wf - REFERENCE_WORK_FUNCTION)
    return i0, delta_G_aligned


class facet_dependant_property:
    """
    Compute Wulff-shape weighted electrochemical properties of a crystal
    from facet-resolved surface data.

    Two surface-data formats are supported:

    * Current format: a list or dict of records with ``Formula``, ``Facet``
      (e.g. ``'111_1'``), ``Surface_energy``, ``Work_function`` and
      ``Material_id`` fields.
    * Legacy format: a pandas DataFrame produced by
      :class:`GalvCalc.cathode.surfaces.SurfaceProperties` (columns
      ``formula``, ``miller_index``, ``termination_index``,
      ``surface_energy_J_m2`` and ``work_function_eV``).

    When a Materials Project ID is available (or a bulk ``Structure`` is
    supplied directly), the Wulff shape is constructed from the crystal
    lattice and the facet-resolved surface energies.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Parameters
        ----------
        api_key : str, optional
            Materials Project API key. When omitted, the default
            ``MPRester`` credential lookup is used (``MP_API_KEY``
            environment variable or ``~/.pmgrc.yaml``).
        """
        self.api_key = api_key
        self.compound: Optional[str] = None
        self._bulk: Optional[Structure] = None
        self._mpr: Any = None

    # ------------------------------------------------------------------ #
    # Materials Project helper
    # ------------------------------------------------------------------ #
    def _get_mpr(self) -> Any:
        """Return a lazily-initialized Materials Project client."""
        if self._mpr is None:
            try:
                from mp_api.client import MPRester
            except ImportError as exc:
                raise ImportError(
                    "mp-api is required to fetch structures from the Materials "
                    "Project. Install it with `pip install mp-api`."
                ) from exc
            self._mpr = MPRester(self.api_key)
        return self._mpr

    def _get_structure(self, bulk_or_mpid: Union[str, Structure]) -> Structure:
        """Return a Structure from a Materials Project ID or an already-built
        structure object."""
        if isinstance(bulk_or_mpid, Structure):
            return bulk_or_mpid
        return self._get_mpr().get_structure_by_material_id(bulk_or_mpid)

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    def _resolve_compound(self, compound: Union[str, Structure]) -> str:
        """Normalize the compound argument (formula string or bulk structure).

        A bulk structure supplied by the caller is kept so that Wulff
        construction does not require a Materials Project lookup.
        """
        if isinstance(compound, Structure):
            self.compound = compound.reduced_formula
            self._bulk = compound
        else:
            self.compound = str(compound)
        return self.compound

    @staticmethod
    def _parse_miller_index(index: str) -> Tuple[int, int, int]:
        """Parse a compact Miller string such as ``'1-11'`` into ``(1, -1, 1)``."""
        numbers: List[int] = []
        sign = 1
        for char in index:
            if char == "-":
                sign = -1
            else:
                numbers.append(sign * int(char))
                sign = 1
        return tuple(numbers)

    @staticmethod
    def _miller_to_string(miller: Sequence[int]) -> str:
        """Convert a Miller tuple into a compact string key, e.g.
        ``(1, -1, 1) -> '1-11'``."""
        return "".join(str(i) if i >= 0 else f"-{-i}" for i in miller)

    @staticmethod
    def _site_energy(site: Dict[str, Any]) -> Optional[float]:
        """Extract the adsorption free energy from a site record.

        Supports both the native ``Gads`` key and the legacy ``Eads`` key,
        as well as the legacy ``{filename: {'Eads': ...}}`` nesting.
        """
        for key in _ADSORPTION_ENERGY_KEYS:
            value = site.get(key)
            if value is not None:
                return float(value)
        for value in site.values():
            if isinstance(value, dict):
                for key in _ADSORPTION_ENERGY_KEYS:
                    if value.get(key) is not None:
                        return float(value[key])
        return None

    def _facet_index_and_number(self, facet: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """Extract (miller_index_key, termination_number) from a record."""
        facet_str = facet.get("Facet")
        if facet_str is not None:
            parts = str(facet_str).split("_")
            return parts[0], parts[1] if len(parts) == 2 else ""
        miller = facet.get("miller_index")
        if miller is None:
            return None, ""
        if isinstance(miller, str):
            miller_tuple = tuple(int(x) for x in miller.strip("()").split(","))
        else:
            miller_tuple = tuple(int(x) for x in miller)
        index_key = self._miller_to_string(miller_tuple)
        return index_key, str(facet.get("termination_index", ""))

    def _structure_source(self, mp_id: Optional[str]) -> Union[str, Structure]:
        """Return the structure source for Wulff construction.

        A directly supplied bulk structure is preferred (offline-friendly);
        otherwise the Materials Project ID is used.
        """
        if self._bulk is not None:
            return self._bulk
        if mp_id is not None:
            return mp_id
        raise ValueError(
            "Neither a Materials Project ID nor a bulk structure is available "
            f"for {self.compound}. Provide a 'Material_id' in the facet records "
            "or pass a bulk structure as the 'compound' argument."
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def list2dict(
        self,
        facet_list: Any,
        compound: Optional[Union[str, Structure]] = None,
    ) -> Tuple[Dict[str, Tuple[float, float, str]], Optional[str]]:
        """
        Build a dictionary that keeps, for every Miller index, the
        termination with the lowest surface energy.

        Parameters
        ----------
        facet_list : list, dict or pandas.DataFrame
            Facet-resolved surface records.
        compound : str or Structure, optional
            Compound formula (or bulk structure). Defaults to
            ``self.compound`` when omitted.

        Returns
        -------
        sur_dict : dict
            Mapping ``'111' -> (surface_energy, work_function, termination)``.
        mp_id : str or None
            Materials Project ID, when present in the records.
        """
        if compound is not None:
            self._resolve_compound(compound)
        if self.compound is None:
            raise ValueError("A compound formula or bulk structure must be provided.")

        sur_dict: Dict[str, Tuple[float, float, str]] = {}
        mp_id: Optional[str] = None

        if isinstance(facet_list, dict):
            facet_iter = facet_list.values()
        elif isinstance(facet_list, pd.DataFrame):
            facet_iter = facet_list.to_dict("records")
        else:
            facet_iter = facet_list

        for facet in facet_iter:
            formula = facet.get("Formula", facet.get("formula"))
            if formula != self.compound:
                continue

            index, number = self._facet_index_and_number(facet)
            if index is None:
                continue

            se = facet.get("Surface_energy", facet.get("surface_energy_J_m2"))
            wf = facet.get("Work_function", facet.get("work_function_eV"))
            if se is None or wf is None:
                continue
            se = float(se)
            wf = float(wf)

            if (index not in sur_dict) or (se < sur_dict[index][0]):
                sur_dict[index] = (se, wf, number)

            mp_id = facet.get("Material_id", mp_id)

        if not sur_dict:
            raise ValueError(f"No matching compound {self.compound} found in the facet list.")
        return sur_dict, mp_id

    def wulff_weights(
        self,
        bulk_or_mpid: Union[str, Structure],
        miller_index: List[Tuple[int, int, int]],
        surface_energy: Sequence[float],
    ) -> List[float]:
        """
        Construct the Wulff shape and return the area weight (in percent) of
        every facet.

        Parameters
        ----------
        bulk_or_mpid : str or Structure
            Materials Project ID of the compound or a pymatgen structure.
        miller_index : list of tuple
            Miller indices, e.g. ``[(1, 0, 0), (1, 1, 0), (1, 1, 1)]``.
        surface_energy : list of float
            Surface energies in any consistent unit (only the relative values
            matter for the Wulff weights), e.g. ``[885.469, 797.495, 738.735]``.

        Returns
        -------
        list of float
            Area weight of each facet (percent, sum = 100).
        """
        struct = self._get_structure(bulk_or_mpid)
        ws = WulffShape(struct.lattice, miller_index, surface_energy)
        total_area = sum(float(entry.split(":")[-1]) for entry in ws.miller_area)
        weights = [float(entry.split(":")[-1]) / total_area * 100 for entry in ws.miller_area]
        return weights

    def weighted_ic(
        self,
        sur_list: Any,
        prop_dict: Dict[str, Any],
        compound: Union[str, Structure],
        correction: float = 0,
        pH: float = 11,
        ads_criteria: str = "lowest",
    ) -> float:
        """
        Calculate the Wulff-shape weighted exchange current density.

        Parameters
        ----------
        sur_list : list, dict or pandas.DataFrame
            Facet-resolved surface records (see :meth:`list2dict`).
        prop_dict : dict
            Adsorption data of the form::

                {compound: {miller_index_key: {termination: {site: {"Gads": float}}}}}

            The legacy ``"Eads"`` key is also accepted.
        compound : str or Structure
            Compound formula or bulk structure. When a bulk structure is
            given, the Wulff shape is built from it directly and no
            Materials Project lookup is required.
        correction : float
            Extra free-energy correction (eV) added to every adsorption
            energy (default 0).
        pH : float
            Solution pH.
        ads_criteria : str
            Site-selection rule:

            * ``'lowest'``: most negative adsorption free energy.
            * ``'near_zero'``: site with the largest exchange current density.
            * ``'average'``: mean i0 over all sites of the most stable
              termination.

        Returns
        -------
        float
            Wulff-shape weighted exchange current density (A/cm^2).
        """
        self._resolve_compound(compound)
        sur_dict, mp_id = self.list2dict(sur_list)

        print(f"* {self.compound}:")

        # Compact Miller strings -> tuples, e.g. '111' -> (1, 1, 1).
        miller_index_string = list(sur_dict.keys())
        miller_index = [self._parse_miller_index(index) for index in miller_index_string]

        surface_energy = [values[0] for values in sur_dict.values()]
        work_function = [values[1] for values in sur_dict.values()]
        numbers = [values[2] for values in sur_dict.values()]

        stable_idx = int(np.argmin(surface_energy))
        stable_surface = miller_index_string[stable_idx]
        stable_number = numbers[stable_idx]
        print(
            f"Among Miller indices {miller_index_string}, {stable_surface}_{stable_number} "
            "has the lowest surface energy."
        )

        source = self._structure_source(mp_id)
        weights = self.wulff_weights(source, miller_index, surface_energy)
        print(f"and the weights on each Miller index: {weights}")

        wf_lookup = {index: values[1] for index, values in sur_dict.items()}
        index_order = list(sur_dict.keys())

        if self.compound not in prop_dict:
            raise KeyError(f"No adsorption data found for {self.compound} in prop_dict.")

        surface_property: List[float] = []
        miller_index_string_prop: List[str] = []

        for index, facets in prop_dict[self.compound].items():
            wf = wf_lookup[index]

            for number, facet in facets.items():
                # Only the most stable termination of the facet contributes.
                if number != numbers[index_order.index(index)]:
                    continue

                if ads_criteria == "lowest":
                    energies = [e for e in (self._site_energy(s) for s in facet.values()) if e is not None]
                    if not energies:
                        raise ValueError(f"No adsorption energy found on facet {index}_{number}.")
                    Gads = min(energies)
                    i_c = ic0(delta_G=Gads + correction, pH=pH, wf=wf)

                elif ads_criteria == "near_zero":
                    i0_max = -np.inf
                    Gads = None
                    for site in facet.values():
                        energy = self._site_energy(site)
                        if energy is None:
                            continue
                        i0 = ic0(delta_G=energy + correction, pH=pH, wf=wf)
                        if i0 > i0_max:
                            i0_max = i0
                            Gads = energy
                    if not np.isfinite(i0_max):
                        raise ValueError(f"No adsorption energy found on facet {index}_{number}.")
                    i_c = i0_max

                elif ads_criteria == "average":
                    i_list = []
                    for site in facet.values():
                        energy = self._site_energy(site)
                        if energy is not None:
                            i_list.append(ic0(delta_G=energy + correction, pH=pH, wf=wf))
                    if not i_list:
                        raise ValueError(f"No adsorption energy found on facet {index}_{number}.")
                    i_c = float(np.mean(i_list))

                else:
                    raise ValueError(f"Unknown ads_criteria: {ads_criteria}")

                surface_property.append(i_c)
                miller_index_string_prop.append(index)
                print(f"{index}: WF={wf:.3f} eV, Gads={Gads:.3f} eV, i_c={i_c:.3e}")

        missing = [index for index in miller_index_string if index not in miller_index_string_prop]
        if missing:
            raise ValueError(
                f"No adsorption data found for facets {missing} of {self.compound}; "
                "the Wulff weights cannot be aligned."
            )

        # Align the facet properties with the Wulff weights.
        order = [miller_index_string_prop.index(index) for index in miller_index_string]
        surface_property = np.array(surface_property)[order]
        weighted_value = np.array(weights) * np.array(surface_property) * 0.01
        result = float(weighted_value.sum())
        print(f"Weighted i_c for {self.compound}: {result:.3e}")
        return result

    def weighted_wf(self, sur_list: Any, compound: Union[str, Structure]) -> Tuple[float, List[float]]:
        """
        Calculate the Wulff-shape weighted work function.

        Parameters
        ----------
        sur_list : list, dict or pandas.DataFrame
            Facet-resolved surface records.
        compound : str or Structure
            Compound formula or bulk structure.

        Returns
        -------
        tuple
            ``(weighted_wf, work_functions)`` where ``weighted_wf`` is the
            area-weighted work function (eV) and ``work_functions`` is the
            list of facet work functions (eV).
        """
        self._resolve_compound(compound)
        sur_dict, mp_id = self.list2dict(sur_list)

        print(f"* {self.compound}:")

        miller_index_string = list(sur_dict.keys())
        miller_index = [self._parse_miller_index(index) for index in miller_index_string]
        surface_energy = [values[0] for values in sur_dict.values()]
        work_function = [values[1] for values in sur_dict.values()]

        if any(se < 0 for se in surface_energy):
            raise ValueError("Surface energy list contains negative values.")

        source = self._structure_source(mp_id)
        struct = self._get_structure(source)
        ws = WulffShape(struct.lattice, miller_index, surface_energy)
        total_area = sum(float(entry.split(":")[-1]) for entry in ws.miller_area)
        weights = [float(entry.split(":")[-1]) / total_area * 100 for entry in ws.miller_area]

        weighted_wf = sum(np.array(weights) * np.array(work_function) * 0.01)
        return float(weighted_wf), work_function


def one_facet_property(
    facet_data: pd.DataFrame,
    ads_criteria: str = "near_zero",
    systems: str = "Mg",
) -> Tuple[float, float]:
    """
    Calculate the electrochemical hydrogen-evolution descriptor of a single
    facet from its adsorption-site data.

    Parameters
    ----------
    facet_data : pandas.DataFrame
        DataFrame with per-site adsorption data. Expected columns:
        ``Eads`` (eV) and ``work_function`` (eV).
    ads_criteria : str
        ``'near_zero'`` selects the site with the largest exchange current
        density; ``'lowest'`` selects the site with the most negative
        adsorption free energy.
    systems : str
        Material system. Only ``'Mg'`` is currently implemented.

    Returns
    -------
    tuple
        ``(log10(i0), delta_G)`` of the selected site.
    """
    if systems != "Mg":
        raise NotImplementedError("Only the 'Mg' system is currently implemented.")

    i0_list = []
    Gads_list = []

    for _, row in facet_data.iterrows():
        eads = row["Eads"]
        wf = row.get("work_function", REFERENCE_WORK_FUNCTION)
        i0, delta_G = ic0_mg(delta_H=eads, wf=wf)
        i0_list.append(i0)
        Gads_list.append(delta_G)

    if ads_criteria == "near_zero":
        i0_max = np.array(i0_list).max()
        max_index = np.array(i0_list).argmax()
        return float(np.log10(i0_max)), Gads_list[max_index]

    if ads_criteria == "lowest":
        Gads_min = np.array(Gads_list).min()
        min_index = np.array(Gads_list).argmin()
        return float(np.log10(i0_list[min_index])), Gads_min

    raise ValueError(f"Unknown ads_criteria: {ads_criteria}")
