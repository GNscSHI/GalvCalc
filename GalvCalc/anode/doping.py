"""Substitutional surface doping for anode slabs.

The doping workflow is built on top of pymatgen's defect-analysis framework:

- ``pymatgen.analysis.defects.core.Substitution`` performs the actual
  substitutional doping (a dedicated ``pymatgen-analysis-defects`` package is
  required for pymatgen releases >= 2024.10).
- ``pymatgen.symmetry.analyzer.SpacegroupAnalyzer`` is used to select a
  symmetry-equivalent host site inside the requested layer and to report the
  site multiplicity.

Dopants are placed in the *second* layer below the surface by default, which
keeps the outermost surface plane intact.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pymatgen.core import Element, PeriodicSite
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tqdm.auto import tqdm

from GalvCalc.core.structures import Surface

try:  # pymatgen >= 2024.10 with the pymatgen-analysis-defects extension
    from pymatgen.analysis.defects.core import Substitution

    _DEFECTS_AVAILABLE = True
except ImportError:  # legacy pymatgen fallback (kept for development only)
    Substitution = None  # type: ignore[assignment]
    _DEFECTS_AVAILABLE = False

_FALLBACK_WARNED = False


@dataclass
class DopingSite:
    """Information about a single substitutional doping event.

    Attributes:
        host_element: Element symbol of the substituted host atom.
        dopant_element: Element symbol of the inserted dopant.
        site_index: Index of the dopant site in the doped surface.
        fractional_coords: Fractional coordinates of the doping site.
        cartesian_coords: Cartesian coordinates of the doping site.
        depth: Distance from the doping site to the nearest surface plane (A).
        layer: Layer of the doping site (positive: counted from the top
            surface, 1 = outermost; negative: counted from the bottom).
        multiplicity: Symmetry multiplicity of the doping site.
    """

    host_element: str
    dopant_element: str
    site_index: int
    fractional_coords: Tuple[float, float, float]
    cartesian_coords: Tuple[float, float, float]
    depth: float
    layer: int = 2
    multiplicity: int = 1


class SurfaceDopingManager:
    """
    Manager for substitutional doping of a single base surface.

    Doping sites are selected inside a requested slab layer (the second layer
    below the surface by default) using ``SpacegroupAnalyzer`` for symmetry
    information, and the substitution itself is carried out with pymatgen's
    ``Substitution`` defect object.
    """

    def __init__(
        self,
        base_surface: Surface,
        base_name: Optional[str] = None,
        symprec: float = 0.01,
        angle_tolerance: float = 5.0,
    ):
        """
        Initialize doping manager for a single base surface.

        Args:
            base_surface: Base Surface object.
            base_name: Custom base name (if None, uses base_surface.full_name).
            symprec: Symmetry precision (A) passed to pymatgen's
                ``SpacegroupAnalyzer``.
            angle_tolerance: Angle tolerance (degrees) passed to pymatgen's
                ``SpacegroupAnalyzer``.
        """
        self.base_surface = base_surface
        self.symprec = symprec
        self.angle_tolerance = angle_tolerance

        # Get basic information from base surface
        if base_name is None:
            if hasattr(base_surface, "full_name") and base_surface.full_name:
                self.base_name = base_surface.full_name
            else:
                # Generate default name
                formula = base_surface.bulk_structure.reduced_formula
                miller_str = "".join(str(x) for x in base_surface.miller_index)
                self.base_name = f"{formula}_{miller_str}"
        else:
            self.base_name = base_name

        self.formula = base_surface.bulk_structure.reduced_formula
        self.miller_index = base_surface.miller_index
        self.all_surfaces = [base_surface]
        self.doping_info: Dict[str, DopingSite] = {}
        self.surface_properties: Dict[str, dict] = {self.base_name: {}}

        self._properties_df: Optional[pd.DataFrame] = None

        base_surface.full_name = self.base_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def dope_surface(
        self,
        dopant_element: str,
        host_element: Optional[str] = None,
        layer: int = 2,
        save_to_file: bool = True,
        output_dir: Union[str, Path] = "doped_surfaces",
    ) -> Surface:
        """
        Perform single-atom substitutional doping on the base surface.

        Args:
            dopant_element: Dopant element symbol.
            host_element: Host element to replace (defaults to the first
                element of the base formula).
            layer: Slab layer containing the doping site. Positive values are
                counted from the top surface (1 = outermost layer, 2 = second
                layer); negative values are counted from the bottom surface
                (-1 = bottommost layer). Defaults to 2 (second layer).
            save_to_file: Whether to save the doped structure to a file.
            output_dir: Output directory for structure files.

        Returns:
            Doped surface structure.
        """
        if host_element is None:
            host_element = self.formula.split()[0]

        print(f"Doping {self.base_name}: {host_element} -> {dopant_element} (layer {layer})")

        global _FALLBACK_WARNED
        if not _DEFECTS_AVAILABLE and not _FALLBACK_WARNED:
            _FALLBACK_WARNED = True
            warnings.warn(
                "pymatgen.analysis.defects is not available; falling back to "
                "direct site replacement. Install pymatgen>=2024.10 together "
                "with the 'pymatgen-analysis-defects' package to enable the "
                "full defect-based workflow.",
                stacklevel=2,
            )

        surface = self.base_surface.copy()
        selected_atom, multiplicity = self._select_doping_site(
            surface, host_element, layer
        )

        original_frac = tuple(float(x) for x in surface[selected_atom].frac_coords)
        cartesian_coords = tuple(float(x) for x in surface[selected_atom].coords)
        depth = self._calculate_surface_depth(surface, selected_atom)

        doped_surface = self._apply_substitution(
            surface, selected_atom, dopant_element
        )
        dopant_index = self._locate_dopant(
            doped_surface, dopant_element, original_frac
        )

        doped_name = f"{self.base_name}_{dopant_element}"
        doped_surface.full_name = doped_name

        # Create doping record
        doping_site = DopingSite(
            host_element=host_element,
            dopant_element=dopant_element,
            site_index=dopant_index,
            fractional_coords=original_frac,
            cartesian_coords=cartesian_coords,
            depth=depth,
            layer=layer,
            multiplicity=multiplicity,
        )

        self.all_surfaces.append(doped_surface)
        self.doping_info[doped_name] = doping_site
        self.surface_properties[doped_name] = {}

        if save_to_file:
            self._save_doped_structure(doped_surface, output_dir)

        print(f"Created doped surface: {doped_name}")
        print(
            f"  Doping site: atom {dopant_index}, layer {layer}, "
            f"depth: {depth:.3f} A, multiplicity: {multiplicity}"
        )

        return doped_surface

    def batch_dope(
        self,
        dopant_elements: List[str],
        host_element: Optional[str] = None,
        layer: int = 2,
        save_to_file: bool = True,
        output_dir: Union[str, Path] = "doped_surfaces",
    ) -> List[Surface]:
        """
        Batch doping with multiple elements.

        Args:
            dopant_elements: List of dopant element symbols.
            host_element: Host element to replace (defaults to the first
                element of the base formula).
            layer: Slab layer containing the doping site (see
                ``dope_surface``). Defaults to 2 (second layer).
            save_to_file: Whether to save the doped structures to files.
            output_dir: Output directory for structure files.

        Returns:
            List of created doped surfaces.
        """
        if not dopant_elements:
            print("No dopant elements provided")
            return []

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(f"Batch doping with elements: {', '.join(dopant_elements)}")

        doped_surfaces = []
        for dopant in tqdm(dopant_elements, desc="Doping"):
            try:
                surface = self.dope_surface(
                    dopant_element=dopant,
                    host_element=host_element,
                    layer=layer,
                    save_to_file=save_to_file,
                    output_dir=output_dir,
                )
                doped_surfaces.append(surface)
            except Exception as e:  # noqa: BLE001 - one bad element must not abort the batch
                print(f"Failed to dope with {dopant}: {str(e)[:80]}")

        print(f"Generated {len(doped_surfaces)} doped surfaces")
        return doped_surfaces

    def set_property(self, surface_name: str, property_name: str, value: float):
        """
        Set surface property (work function, vacancy energy, etc.).

        Args:
            surface_name: Name of the surface.
            property_name: Property name (e.g., 'work_function').
            value: Property value.
        """
        if surface_name not in self.surface_properties:
            raise ValueError(f"Surface {surface_name} not found in manager")

        self.surface_properties[surface_name][property_name] = value
        self._properties_df = None

    def get_property(self, surface_name: str, property_name: str) -> Optional[float]:
        """
        Get surface property value.

        Args:
            surface_name: Name of the surface.
            property_name: Property name.

        Returns:
            Property value or None if not found.
        """
        if surface_name not in self.surface_properties:
            return None
        return self.surface_properties[surface_name].get(property_name)

    def calculate_electrochemical_properties(
        self,
        E00: float,
        ia00: float,
        temperature: float = 298.15,
        use_work_function: bool = True,
        use_vacancy_energy: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate electrochemical properties for all surfaces.

        Based on the pure surface properties (E00, ia00) and differences
        in work function and vacancy formation energy.

        Equations:
        E0 = E00 + (work_function - work_function_pure)
        ia0 = ia00 * exp(-(vacancy_energy - vacancy_energy_pure) / kT)

        Args:
            E00: Equilibrium potential for pure surface (V).
            ia00: Anodic exchange current density for pure surface (A/cm^2).
            temperature: Temperature in Kelvin (default: 298.15 K).
            use_work_function: Whether to use work function differences.
            use_vacancy_energy: Whether to use vacancy energy differences.

        Returns:
            DataFrame with calculated electrochemical properties.
        """
        kB = 8.617333262145e-5  # Boltzmann constant in eV/K
        kT = kB * temperature

        work_function_pure = self.get_property(self.base_name, "work_function")
        vacancy_energy_pure = self.get_property(self.base_name, "vacancy_energy")

        if use_work_function and work_function_pure is None:
            print("Warning: Pure surface work_function not set")
            use_work_function = False

        if use_vacancy_energy and vacancy_energy_pure is None:
            print("Warning: Pure surface vacancy_energy not set")
            use_vacancy_energy = False

        results = []
        for surface in self.all_surfaces:
            surface_name = surface.full_name
            work_function = self.get_property(surface_name, "work_function")
            vacancy_energy = self.get_property(surface_name, "vacancy_energy")

            result = {
                "surface_name": surface_name,
                "is_pure": surface_name == self.base_name,
                "E00_reference": E00,
                "ia00_reference": ia00,
            }

            # Calculate equilibrium potential E0
            if (
                use_work_function
                and work_function is not None
                and work_function_pure is not None
            ):
                delta_work_function = work_function - work_function_pure
                result["work_function"] = work_function
                result["delta_work_function"] = delta_work_function
                result["E0_calculated"] = E00 + delta_work_function
            else:
                result["E0_calculated"] = E00
                if work_function is not None:
                    result["work_function"] = work_function

            # Calculate exchange current density ia0
            if (
                use_vacancy_energy
                and vacancy_energy is not None
                and vacancy_energy_pure is not None
            ):
                delta_vacancy_energy = vacancy_energy - vacancy_energy_pure
                if delta_vacancy_energy != 0:
                    ia0 = ia00 * np.exp(-delta_vacancy_energy / kT)
                else:
                    ia0 = ia00
                result["vacancy_energy"] = vacancy_energy
                result["delta_vacancy_energy"] = delta_vacancy_energy
                result["ia0_calculated"] = ia0
            else:
                result["ia0_calculated"] = ia00
                if vacancy_energy is not None:
                    result["vacancy_energy"] = vacancy_energy

            # Add doping information if available
            if surface_name != self.base_name:
                doping_info = self.doping_info.get(surface_name)
                if doping_info:
                    result["dopant_element"] = doping_info.dopant_element
                    result["host_element"] = doping_info.host_element
                    result["doping_depth"] = doping_info.depth
                    result["doping_layer"] = doping_info.layer

            results.append(result)

        return pd.DataFrame(results)

    def get_surface(self, surface_name: str) -> Optional[Surface]:
        """
        Get surface by name.

        Args:
            surface_name: Name of the surface.

        Returns:
            Surface object or None if not found.
        """
        for surface in self.all_surfaces:
            if surface.full_name == surface_name:
                return surface
        return None

    def get_properties_dataframe(self) -> pd.DataFrame:
        """
        Get DataFrame with all surface properties.

        Returns:
            DataFrame with columns: surface_name, property, value.
        """
        if self._properties_df is not None:
            return self._properties_df

        rows = []
        for surface_name, props in self.surface_properties.items():
            for prop_name, value in props.items():
                rows.append(
                    {"surface_name": surface_name, "property": prop_name, "value": value}
                )

        self._properties_df = pd.DataFrame(rows)
        return self._properties_df

    def get_doping_info(self) -> Dict[str, DopingSite]:
        """Return the doping records keyed by doped surface name."""
        return dict(self.doping_info)

    def get_summary(self) -> dict:
        """
        Get a summary of the manager state.

        Returns:
            Dictionary with manager summary information.
        """
        dopant_elements = set()
        for doping_info in self.doping_info.values():
            dopant_elements.add(doping_info.dopant_element)

        property_types = set()
        for props in self.surface_properties.values():
            property_types.update(props.keys())

        return {
            "base_surface": self.base_name,
            "formula": self.formula,
            "miller_index": self.miller_index,
            "total_surfaces": len(self.all_surfaces),
            "pure_surfaces": 1,
            "doped_surfaces": len(self.all_surfaces) - 1,
            "dopant_elements": sorted(dopant_elements),
            "property_types": sorted(property_types),
            "surface_names": [s.full_name for s in self.all_surfaces],
        }

    def print_summary(self):
        """Print summary of all surfaces."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print(f"SURFACE DOPING MANAGER SUMMARY: {self.base_name}")
        print("=" * 60)
        print(f"Base surface: {summary['base_surface']}")
        print(f"Formula: {summary['formula']}")
        print(f"Miller index: {summary['miller_index']}")
        print(f"Total surfaces: {summary['total_surfaces']}")
        print(f"Pure surfaces: {summary['pure_surfaces']}")
        print(f"Doped surfaces: {summary['doped_surfaces']}")

        if summary["dopant_elements"]:
            print(f"Dopant elements: {', '.join(summary['dopant_elements'])}")

        if summary["property_types"]:
            print(f"Properties tracked: {', '.join(summary['property_types'])}")

        print("\nAll surfaces:")
        for name in summary["surface_names"]:
            is_pure = name == self.base_name
            prefix = "  - " if is_pure else "  * "
            print(f"{prefix}{name}")

        print("=" * 60)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_layer_indices(self, surface: Surface) -> Dict[int, List[int]]:
        """
        Group slab sites into layers along the surface normal (z axis).

        Layers are numbered from the top: layer 1 is the outermost (topmost)
        layer, layer 2 the second layer, etc. A Cartesian tolerance of 0.3 A
        is used to cluster atoms that belong to the same plane.

        Args:
            surface: Surface structure.

        Returns:
            Mapping ``{layer_number: [site indices]}``.
        """
        z_coords = np.array([site.coords[2] for site in surface])
        order = np.argsort(z_coords)

        groups: List[List[int]] = []
        for idx in order:
            if groups and z_coords[idx] - z_coords[groups[-1][-1]] <= 0.3:
                groups[-1].append(int(idx))
            else:
                groups.append([int(idx)])

        # groups are ordered from bottom (z min) to top (z max)
        n_layers = len(groups)
        return {n_layers - i: groups[i] for i in range(n_layers)}

    def _select_doping_site(
        self, surface: Surface, host_element: str, layer: int
    ) -> Tuple[int, int]:
        """
        Select the doping site inside the requested layer.

        Among the symmetry-inequivalent host candidates, the site with the
        highest symmetry multiplicity is preferred (deterministic tie-break by
        site index).

        Args:
            surface: Surface structure.
            host_element: Host element symbol to replace.
            layer: Layer number (positive from top, negative from bottom).

        Returns:
            Tuple ``(site_index, multiplicity)``.
        """
        layer_indices = self._get_layer_indices(surface)
        n_layers = max(layer_indices)

        if layer == 0:
            raise ValueError(
                "layer must be a non-zero integer (positive: from top, negative: from bottom)"
            )

        target = layer if layer > 0 else n_layers + 1 + layer
        if target < 1 or target > n_layers:
            raise ValueError(
                f"Layer {layer} is out of range: surface has {n_layers} layers"
            )

        candidates = [
            i for i in layer_indices[target] if surface[i].species_string == host_element
        ]
        if not candidates:
            raise ValueError(
                f"No {host_element} atoms found in layer {layer} "
                f"(surface has {n_layers} layers)"
            )

        # Rank candidates by symmetry multiplicity using SpacegroupAnalyzer.
        try:
            analyzer = SpacegroupAnalyzer(
                surface, symprec=self.symprec, angle_tolerance=self.angle_tolerance
            )
            symmetrized = analyzer.get_symmetrized_structure()

            best_index, best_mult = None, -1
            for idx in candidates:
                equivalent = symmetrized.find_equivalent_sites(surface[idx])
                mult = len(equivalent)
                if mult > best_mult or (mult == best_mult and (best_index is None or idx < best_index)):
                    best_index, best_mult = idx, mult
            return best_index, best_mult
        except Exception:  # noqa: BLE001 - symmetry analysis is best-effort
            # Deterministic fallback: pick the candidate closest to the layer centre.
            z_values = [surface[i].coords[2] for i in candidates]
            z_median = float(np.median(z_values))
            chosen = min(candidates, key=lambda i: abs(surface[i].coords[2] - z_median))
            return chosen, 1

    def _apply_substitution(
        self, surface: Surface, site_index: int, dopant_element: str
    ) -> Surface:
        """
        Replace the host atom at ``site_index`` with the dopant.

        Uses ``pymatgen.analysis.defects.core.Substitution`` when available and
        falls back to a direct ``Structure.replace`` otherwise.

        Args:
            surface: Surface structure to dope (modified in place on fallback).
            site_index: Index of the host site.
            dopant_element: Dopant element symbol.

        Returns:
            Doped surface structure.
        """
        if _DEFECTS_AVAILABLE:
            dopant_site = PeriodicSite(
                Element(dopant_element),
                surface[site_index].coords,
                lattice=surface.lattice,
                coords_are_cartesian=True,
            )
            substitution = Substitution(
                structure=surface,
                site=dopant_site,
                symprec=self.symprec,
                angle_tolerance=self.angle_tolerance,
            )
            substituted = substitution.defect_structure

            # Rebuild a Surface that preserves the original slab metadata.
            return Surface(
                lattice=substituted.lattice,
                species=substituted.species,
                coords=substituted.cart_coords,
                miller_index=surface.miller_index,
                oriented_unit_cell=surface.oriented_unit_cell,
                shift=surface.shift,
                scale_factor=surface.scale_factor,
                bulk_structure=surface.bulk_structure,
                reorient_lattice=surface.reorient_lattice,
                coords_are_cartesian=True,
                site_properties=surface.site_properties,
                energy=surface.energy,
            )

        surface.replace(site_index, dopant_element)
        return surface

    @staticmethod
    def _locate_dopant(
        surface: Surface, dopant_element: str, original_frac: Tuple[float, float, float]
    ) -> int:
        """
        Find the index of the dopant site in a doped surface.

        The dopant normally keeps the index of the substituted host atom;
        this helper also scans the structure for robustness.

        Args:
            surface: Doped surface structure.
            dopant_element: Dopant element symbol.
            original_frac: Fractional coordinates of the original host site.

        Returns:
            Site index of the dopant.
        """
        for i, site in enumerate(surface):
            if site.species_string == dopant_element and np.allclose(
                site.frac_coords, original_frac, atol=1e-3
            ):
                return i
        for i, site in enumerate(surface):
            if site.species_string == dopant_element:
                return i
        raise ValueError(f"Dopant {dopant_element} not found in doped surface")

    def _calculate_surface_depth(self, surface: Surface, atom_index: int) -> float:
        """
        Calculate distance (A) from an atom to the nearest surface plane.

        Args:
            surface: Surface structure.
            atom_index: Atom index.

        Returns:
            Distance to the nearest surface plane in Angstrom.
        """
        z_coords = np.array([site.coords[2] for site in surface])
        z = surface[atom_index].coords[2]
        return float(min(abs(z - z_coords.max()), abs(z - z_coords.min())))

    def _save_doped_structure(
        self, surface: Surface, output_dir: Union[str, Path]
    ):
        """
        Save a doped structure to a POSCAR file.

        Args:
            surface: Doped surface structure.
            output_dir: Output directory.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        structure_file = output_path / f"{surface.full_name}.vasp"
        try:
            surface.to(fmt="poscar", filename=str(structure_file))
        except Exception:  # noqa: BLE001 - best-effort file export
            with open(structure_file, "w", encoding="utf-8") as f:
                f.write(str(surface))

    def __repr__(self):
        summary = self.get_summary()
        return (
            f"SurfaceDopingManager(base={self.base_name}, "
            f"surfaces={summary['total_surfaces']}, "
            f"doped={summary['doped_surfaces']})"
        )
