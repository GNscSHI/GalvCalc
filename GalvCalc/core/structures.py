from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml
from pymatgen.core import Molecule, Structure
from pymatgen.core.surface import Slab, SlabGenerator
from scipy import constants as con

R = con.R


class Bulk(Structure):
    """Class for handling cathode bulk materials with electrochemical functionality"""

    def __init__(
            self,
            lattice=None,
            species=None,
            coords=None,
            charge: Optional[float] = None,
            validate_proximity: bool = False,
            to_unit_cell: bool = False,
            coords_are_cartesian: bool = False,
            site_properties: Optional[Dict] = None,
            labels: Optional[List[str]] = None,
            properties: Optional[Dict] = None,
    ):
        """
        Initialize cathode material structure from file or formula

        Args:
            lattice: Lattice parameters (if not using a file)
            species: Atomic species (if not using a file)
            coords: Atomic coordinates (if not using a file)
            charge: Net charge of the structure
            validate_proximity: Check for minimum distance between atoms
            to_unit_cell: Fold coordinates into unit cell
            coords_are_cartesian: Whether coordinates are Cartesian
            site_properties: Additional site properties
            labels: Labels for sites
            properties: Additional properties
        """

        # Initialize parent class
        super().__init__(
            lattice=lattice,
            species=species,
            coords=coords,
            charge=charge or 0.0,
            validate_proximity=validate_proximity,
            to_unit_cell=to_unit_cell,
            coords_are_cartesian=coords_are_cartesian,
            site_properties=site_properties,
            labels=labels,
            properties=properties,
        )

    @staticmethod
    def get_equilibrium_potential(
            ions: Union[str, List[str]],
            ion_numbers: Optional[List[float]] = None,
            energy_formation: float = 0,
            T: float = 298.15,
            ion_con: float = 1,
            valence: Optional[List[int]] = None,
            config_path: str = "./ion_energy.yaml"
    ) -> float:
        """
        Calculate the equilibrium potential of a compound from the chemical
        potentials of its constituent ions.

        Parameters
        ----------
        ions : str or list of str
            Ion species, e.g. ``"Mg[2+], Ge[2+]"`` or ``["Mg[2+]", "Ge[2+]"]``.
        ion_numbers : list of float, optional
            Stoichiometric number of each ion species in the compound.
            Required (no interactive prompt is used by the library).
        energy_formation : float, optional
            Formation energy of the compound in eV (default: 0).
        T : float, optional
            Temperature in Kelvin (default: 298.15).
        ion_con : float, optional
            Ion concentration (activity) used in the Nernst term.
        valence : list of int, optional
            Valence state of each ion. When omitted (or ``None`` for an entry),
            the valence is parsed automatically from the ion string, e.g.
            ``Mg[2+]`` -> 2.
        config_path : str, optional
            Ion-energy YAML file name inside ``GalvCalc/core``.

        Returns
        -------
        float
            Equilibrium potential in V.
        """
        # Load ion energies
        script_dir = Path(__file__).parent
        config_file = script_dir / config_path

        if not config_file.exists():
            raise FileNotFoundError(f"Ion energy configuration file not found: {config_path}")

        with open(config_file, "r", encoding="utf-8") as f:
            ion_data = yaml.safe_load(f)

        # Parse ions
        if isinstance(ions, str):
            ions = [ion.strip() for ion in re.split(r",\s*", ions) if ion.strip()]

        if not ions:
            raise ValueError("At least one ion species must be provided")

        if ion_numbers is None:
            raise ValueError(
                "ion_numbers must be provided, e.g. ion_numbers=[1, 2]. "
                "Interactive input is not supported in library mode."
            )

        if len(ion_numbers) != len(ions):
            raise ValueError("Number of ion counts must match number of ion species")

        G_ions = []
        valences = []

        for i, ion in enumerate(ions):
            if ion not in ion_data:
                raise ValueError(
                    f"Ion {ion} not found in configuration file. Please add it to {config_path}"
                )
            G_ions.append(ion_data[ion]["Energy"])
            print(f"The energy of {ion} is {ion_data[ion]['Energy']} eV, "
                  f"Source: {ion_data[ion]['Source']}")

            # Extract valence state: explicit entry wins, otherwise parse "[n+]"
            if valence is not None and i < len(valence) and valence[i]:
                valences.append(int(valence[i]))
            else:
                match = re.search(r"\[(\d+)\+?\]", ion)
                if match:
                    valences.append(int(match.group(1)))
                else:
                    raise ValueError(
                        f"Could not parse valence state from ion: {ion}. "
                        "Please provide the valence list explicitly."
                    )

        ion_numbers = np.asarray(ion_numbers, dtype=float)
        valences = np.asarray(valences, dtype=float)

        # Chemical potentials (Nernst term converted to eV)
        mu = np.asarray(G_ions) + R * T * np.log(ion_con) / 96485

        # Equilibrium potential
        E = np.sum(mu * ion_numbers) - energy_formation
        E = E / np.sum(valences * ion_numbers)

        print(f"Equilibrium potential at {T} K: {E:.3f} V")
        return float(E)

    def basic_properties(self) -> Dict:
        """Calculate basic electrochemical properties"""
        volume = self.volume
        density = self.density
        formula = self.composition.reduced_formula

        return {
            "formula": formula,
            "volume": volume,
            "density": density,
            "elements": list(self.composition.as_dict().keys()),
        }

    def as_structure(self) -> Structure:
        """
        Convert to standard pymatgen Structure object

        Returns:
            Standard Structure object
        """
        return Structure(
            lattice=self.lattice,
            species=self.species,
            coords=self.frac_coords,
            charge=self._charge,
            validate_proximity=False,
            to_unit_cell=False,
            coords_are_cartesian=False,
            site_properties=self.site_properties,
            labels=self.labels,
            properties=self.properties
        )


class Surface(Slab):
    """Class for handling material surfaces with electrochemical functionality, inheriting from pymatgen's Slab"""

    def __init__(
            self,
            lattice,
            species,
            coords,
            miller_index: Tuple[int, int, int],
            oriented_unit_cell: Structure,
            shift: float,
            scale_factor,
            bulk_structure: Structure = None,
            reorient_lattice: bool = True,
            validate_proximity: bool = False,
            to_unit_cell: bool = False,
            reconstruction: str = None,
            coords_are_cartesian: bool = False,
            site_properties: dict = None,
            energy: float = None,
            **kwargs
    ):
        """
        Initialize Surface from pymatgen Slab parameters

        Args:
            All parameters from pymatgen Slab plus:
            bulk_structure: Bulk structure object for surface energy calculations
        """
        # Initialize the parent Slab class
        super().__init__(
            lattice=lattice,
            species=species,
            coords=coords,
            miller_index=miller_index,
            oriented_unit_cell=oriented_unit_cell,
            shift=shift,
            scale_factor=scale_factor,
            reorient_lattice=reorient_lattice,
            validate_proximity=validate_proximity,
            to_unit_cell=to_unit_cell,
            reconstruction=reconstruction,
            coords_are_cartesian=coords_are_cartesian,
            site_properties=site_properties,
            energy=energy
        )

        self.bulk_structure = bulk_structure
        self._full_name = None

    @property
    def full_name(self) -> str:
        """Human-readable surface name, e.g. ``Mg_001``.

        Auto-generated from the reduced formula and Miller index when it has
        not been set explicitly.
        """
        if self._full_name is None:
            formula = self.composition.reduced_formula
            miller = "".join(str(x) for x in self.miller_index)
            self._full_name = f"{formula}_{miller}"
        return self._full_name

    @full_name.setter
    def full_name(self, value: Optional[str]) -> None:
        self._full_name = value

    @classmethod
    def from_bulk(
            cls,
            bulk_structure: Structure,
            miller_index: Tuple[int, int, int],
            min_slab_size: float = 10.0,
            min_vacuum_size: float = 15.0,
            **kwargs
    ):
        """
        Alternative constructor: create Surface from bulk structure

        Args:
            bulk_structure: Bulk structure object
            miller_index: Miller indices for surface orientation
            min_slab_size: Minimum slab size in Angstroms
            min_vacuum_size: Minimum vacuum size in Angstroms
        """
        if hasattr(bulk_structure, 'as_structure'):
            bulk_for_slab = bulk_structure.as_structure()
        else:
            bulk_for_slab = bulk_structure

        slab_generator = SlabGenerator(
            bulk_for_slab,
            miller_index,
            min_slab_size,
            min_vacuum_size,
            **kwargs
        )

        slabs = slab_generator.get_slabs(symmetrize=True)
        if not slabs:
            raise ValueError("No slabs generated for the given Miller index")

        # Get the most stable slab
        most_stable_slab = min(slabs, key=lambda x: x.get_sorted_structure().density)

        # Create Surface instance from the slab
        return cls(
            lattice=most_stable_slab.lattice,
            species=most_stable_slab.species,
            coords=most_stable_slab.cart_coords,
            miller_index=most_stable_slab.miller_index,
            oriented_unit_cell=most_stable_slab.oriented_unit_cell,
            shift=most_stable_slab.shift,
            scale_factor=most_stable_slab.scale_factor,
            bulk_structure=bulk_structure,
            reorient_lattice=most_stable_slab.reorient_lattice,
            coords_are_cartesian=True,  # Using cartesian coordinates
            site_properties=most_stable_slab.site_properties,
            energy=most_stable_slab.energy
        )

    @classmethod
    def from_slab(cls, slab: Slab, bulk_structure: Structure = None):
        """
        Create Surface from existing Slab object

        Args:
            slab: pymatgen Slab object
            bulk_structure: Corresponding bulk structure
        """
        return cls(
            lattice=slab.lattice,
            species=slab.species,
            coords=slab.cart_coords,
            miller_index=slab.miller_index,
            oriented_unit_cell=slab.oriented_unit_cell,
            shift=slab.shift,
            scale_factor=slab.scale_factor,
            bulk_structure=bulk_structure,
            reorient_lattice=slab.reorient_lattice,
            coords_are_cartesian=True,
            site_properties=slab.site_properties,
            energy=slab.energy
        )

    def calculate_surface_energy(self, slab_energy: float, bulk_energy_per_atom: float) -> float:
        """
        Calculate surface energy

        Args:
            slab_energy: Total energy of the slab
            bulk_energy_per_atom: Total energy per atom of bulk

        Returns:
            Surface energy in eV/A^2
        """
        if self.bulk_structure is None:
            raise ValueError("Bulk structure is required for surface energy calculation")

        n_atoms_slab = len(self)
        n_atoms_bulk = len(self.bulk_structure)

        # Calculate surface area
        area = self.surface_area

        # For symmetric slabs, divide by 2
        surface_energy = (slab_energy - bulk_energy_per_atom * n_atoms_slab) / (2 * area)
        return surface_energy

    def calculate_adsorption_energy(
            self,
            slab_energy: float,
            adsorbate_energy: float,
            adsorbed_system_energy: float
    ) -> float:
        """
        Calculate adsorption energy

        Args:
            slab_energy: Energy of clean slab
            adsorbate_energy: Energy of isolated adsorbate
            adsorbed_system_energy: Energy of slab with adsorbate

        Returns:
            Adsorption energy in eV
        """
        return adsorbed_system_energy - slab_energy - adsorbate_energy

    @property
    def surface_area(self) -> float:
        """Calculate surface area"""
        # For orthogonal cells
        a, b, c = self.lattice.abc
        alpha, beta, gamma = self.lattice.angles
        # Simple approximation for orthogonal surfaces
        if abs(alpha - 90) < 1 and abs(beta - 90) < 1 and abs(gamma - 90) < 1:
            return a * b
        else:
            # For non-orthogonal cells, use cross product
            matrix = self.lattice.matrix
            return np.linalg.norm(np.cross(matrix[0], matrix[1]))

    def visualize_3d(
            self,
            show_unit_cell: bool = True,
            show_bonds: bool = False,
            show_polyhedron: bool = False,  # disabled by default (was unstable in older pymatgen)
            **kwargs
    ) -> None:
        """
        Visualize the surface structure using VTK

        Args:
            show_unit_cell: Whether to show the unit cell
            show_bonds: Whether to show bonds between atoms
            show_polyhedron: Whether to show coordination polyhedra
            **kwargs: Additional parameters passed to StructureVis
        """
        try:
            # build a temporary structure for visualization
            vis_structure = Structure(
                lattice=self.lattice,
                species=self.species,
                coords=self.cart_coords,
                coords_are_cartesian=True,
                site_properties=self.site_properties
            )

            from pymatgen.vis.structure_vtk import StructureVis

            # pass only the essential parameters
            vis_kwargs = {
                'show_unit_cell': show_unit_cell,
                'show_bonds': show_bonds,
                'show_polyhedron': show_polyhedron,
            }

            # forward only the valid parameters present in kwargs
            valid_params = ['element_color_mapping', 'poly_radii_tol_factor', 'excluded_bonding_elements']
            for param in valid_params:
                if param in kwargs:
                    vis_kwargs[param] = kwargs[param]

            vis = StructureVis(**vis_kwargs)
            vis.set_structure(vis_structure)
            vis.show()

        except ImportError:
            print("VTK visualization not available. Please install vtk package.")
        except Exception as e:
            print(f"VTK visualization error: {e}")
            # print a detailed traceback
            import traceback
            traceback.print_exc()

            # print structure information as a fallback
            print("\nStructure information:")
            print(f"Miller index: {self.miller_index}")
            print(f"Composition: {self.composition}")
            print(f"Number of atoms: {len(self)}")
            print(f"Lattice parameters: {self.lattice.parameters}")

    def visualize_adsorption_sites(
            self,
            highlight_index: int = None,
            show_all_sites: bool = True,
            highlight_color: str = 'red',
            normal_color: str = 'black',
            marker_size: int = 20,
            adsorption_sites: dict = None,
            draw_unit_cell: bool = True,
            save_path: Optional[str] = None,
            **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Visualize adsorption sites on the surface.

        Args:
            highlight_index: Index of the adsorption site to highlight (1-based)
            show_all_sites: Whether to show all adsorption sites
            highlight_color: Color for highlighted site
            normal_color: Color for other sites
            marker_size: Size of site markers
            adsorption_sites: Pre-computed adsorption sites (optional)
            draw_unit_cell: Whether to draw unit cell
            save_path: Optional path to save the figure (e.g. a PNG file)
            **kwargs: Additional parameters for AdsorbateSiteFinder

        Returns:
            (fig, ax) tuple of the created matplotlib figure and axes.
        """
        try:
            from pymatgen.analysis.adsorption import AdsorbateSiteFinder, plot_slab
            import matplotlib.pyplot as plt

            # locate adsorption sites
            if adsorption_sites is None:
                asf = AdsorbateSiteFinder(self, **kwargs)
                adsorption_sites = asf.find_adsorption_sites()

            # create the figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # draw the slab
            plot_slab(self, ax, adsorption_sites=adsorption_sites, draw_unit_cell=draw_unit_cell, **kwargs)

            # draw the adsorption sites
            if 'all' in adsorption_sites and show_all_sites:
                for i, site in enumerate(adsorption_sites['all']):
                    if highlight_index is not None and (i + 1) == highlight_index:
                        # highlight the requested site
                        ax.scatter(site[0], site[1],
                                   c=highlight_color, marker='X',
                                   s=marker_size * 2, zorder=1000,
                                   label=f'Site {highlight_index}')
                        ax.legend()
                    else:
                        # regular sites
                        ax.scatter(site[0], site[1],
                                   c=normal_color, marker='x',
                                   s=marker_size, zorder=1000,
                                   alpha=1)

            # add legend and title
            ax.set_title(f'Adsorption Sites on {self.miller_index} Surface\n'
                         f'Composition: {self.composition.reduced_formula}',
                         fontsize=14, fontweight='bold')

            plt.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.show()

            return fig, ax

        except ImportError as e:
            print(f"Required packages not available: {e}")
            print("Please install pymatgen.analysis.adsorption module")
        except Exception as e:
            print(f"Error visualizing adsorption sites: {e}")
            import traceback
            traceback.print_exc()

    def get_adsorption_sites(self, **kwargs) -> dict:
        """
        Get adsorption sites for the surface

        Args:
            **kwargs: Parameters for AdsorbateSiteFinder

        Returns:
            Dictionary of adsorption sites
        """
        try:
            from pymatgen.analysis.adsorption import AdsorbateSiteFinder

            asf = AdsorbateSiteFinder(self, **kwargs)
            adsorption_sites = asf.find_adsorption_sites()

            print(f"Found {len(adsorption_sites.get('all', []))} adsorption sites")
            print("Site types available:", list(adsorption_sites.keys()))

            return adsorption_sites

        except ImportError:
            print("pymatgen.analysis.adsorption module not available")
            return {}
        except Exception as e:
            print(f"Error finding adsorption sites: {e}")
            return {}

    def generate_adsorption_structures(
            self,
            adsorbate: str,
            site_indices: Union[int, List[int], str] = "all",
            adsorption_sites: dict = None,
            **kwargs
    ) -> Dict[int, Structure]:
        """
        Generate structures with adsorbate at specified sites

        Args:
            adsorbate: Adsorbate molecule (e.g., 'H', 'O', 'OH', 'H2O', 'CO')
            site_indices: Site indices to place adsorbate (1-based)
                         - int: single site
                         - List[int]: multiple sites
                         - "all": all available sites
            adsorption_sites: Pre-computed adsorption sites
            **kwargs: Parameters for adsorbate placement

        Returns:
            Dictionary of {site_index: structure_with_adsorbate}
        """
        try:
            from pymatgen.analysis.adsorption import AdsorbateSiteFinder
            from pymatgen.core.structure import Molecule

            if adsorption_sites is None:
                adsorption_sites = self.get_adsorption_sites()

            # build the adsorbate molecule
            adsorbate = adsorbate.upper()
            if adsorbate == 'H':
                adsorbate_mol = Molecule(['H'], [[0, 0, 0]])
            elif adsorbate == 'O':
                adsorbate_mol = Molecule(['O'], [[0, 0, 0]])
            elif adsorbate == 'OH':
                adsorbate_mol = Molecule(['O', 'H'], [[0, 0, 0], [0, 0, 1.0]])
            elif adsorbate == 'H2O':
                adsorbate_mol = Molecule(['O', 'H', 'H'],
                                         [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
            else:
                raise ValueError(f"Unsupported adsorbate: {adsorbate}")

            # resolve the site indices to process
            if site_indices == "all":
                target_indices = list(range(1, len(adsorption_sites['all']) + 1))
            elif isinstance(site_indices, int):
                target_indices = [site_indices]
            elif isinstance(site_indices, list):
                target_indices = site_indices
            else:
                raise ValueError("site_indices should be int, list, or 'all'")

            # generate the adsorbed structures
            asf = AdsorbateSiteFinder(self)
            adsorbed_structures = {}

            for site_idx in target_indices:
                try:
                    target_site = adsorption_sites['all'][site_idx - 1]
                    adsorbed_structure = asf.add_adsorbate(adsorbate_mol, target_site, **kwargs)
                    adsorbed_structures[site_idx] = adsorbed_structure
                    print(f"Successfully placed {adsorbate} at site {site_idx}")
                except Exception as e:
                    print(f"Failed to place {adsorbate} at site {site_idx}: {e}")

            return adsorbed_structures

        except Exception as e:
            print(f"Error generating adsorption structures: {e}")
            return {}

    def __repr__(self):
        return f"Surface({self.miller_index}, {self.composition.formula}, E={self.energy})"
