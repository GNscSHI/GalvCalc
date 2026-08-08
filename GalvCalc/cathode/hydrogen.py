from typing import List, Dict, Optional, Union, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
import json
from pymatgen.core import Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Molecule
from tqdm.auto import tqdm

from GalvCalc.core.structures import Surface


class AdsorptionManager:
    """
    Class for managing and analyzing adsorption sites on multiple surfaces
    """

    def __init__(self, surfaces: Optional[List[Surface]] = None):
        """
        Initialize AdsorptionManager with a list of surfaces

        Args:
            surfaces: List of Surface objects to manage
        """
        self.surfaces = surfaces or []
        self._adsorption_sites = {}  # Dict: surface_name -> adsorption_sites_dict
        self._adsorbed_structures = {}  # Dict: surface_name -> Dict[site_index, Structure]
        self._adsorption_energies = {}  # Dict: surface_name -> Dict[site_index, energy]
        self._site_properties_df = None
        self._summary_df = None
        self._structured_data = []  # Store data in the requested format

    # ========== CORE METHODS ==========

    def get_adsorption_sites_for_all_surfaces(
            self,
            surface_names: Optional[List[str]] = None,
            **kwargs
    ) -> Dict[str, Dict]:
        """
        Find adsorption sites for all surfaces or specified surfaces

        Args:
            surface_names: List of surface names to process (if None, process all)
            **kwargs: Additional parameters for AdsorbateSiteFinder

        Returns:
            Dictionary mapping surface names to adsorption sites
        """
        surfaces_to_process = []

        if surface_names is None:
            surfaces_to_process = self.surfaces
        else:
            surfaces_to_process = [s for s in self.surfaces
                                   if s.full_name in surface_names]

        if not surfaces_to_process:
            print("No surfaces to process")
            return {}

        print(f"Finding adsorption sites for {len(surfaces_to_process)} surfaces...")

        total_sites = 0
        for surface in tqdm(surfaces_to_process, desc="Surfaces", unit="surface"):
            surface_name = getattr(surface, "full_name", None) or repr(surface)
            try:
                asf = AdsorbateSiteFinder(surface, **kwargs)
                adsorption_sites = asf.find_adsorption_sites()

                self._adsorption_sites[surface_name] = adsorption_sites

                n_sites = len(adsorption_sites.get('all', []))
                total_sites += n_sites

            except Exception as e:
                print(f"  Warning: Failed to find sites for {surface_name}: {str(e)[:50]}")
                self._adsorption_sites[surface_name] = {}

        print(f"Found {total_sites} adsorption sites across {len(surfaces_to_process)} surfaces")
        return self._adsorption_sites

    def generate_adsorbed_structures(
            self,
            adsorbate: str = "H",
            surface_names: Optional[List[str]] = None,
            site_indices: Union[int, List[int], str] = "all",
            **kwargs
    ) -> Dict[str, Dict[int, Structure]]:
        """
        Generate structures with adsorbate at specified sites on multiple surfaces

        Args:
            adsorbate: Adsorbate species (e.g., 'H', 'O', 'OH', 'H2O', 'CO')
            surface_names: List of surface names to process (if None, process all)
            site_indices: Site indices to place adsorbate
                         - int: single site
                         - List[int]: multiple sites
                         - "all": all available sites
            **kwargs: Additional parameters for adsorbate placement

        Returns:
            Dictionary: surface_name -> {site_index: adsorbed_structure}
        """
        # First ensure adsorption sites are found
        if not self._adsorption_sites:
            self.get_adsorption_sites_for_all_surfaces(surface_names)

        surfaces_to_process = []
        if surface_names is None:
            surfaces_to_process = self.surfaces
        else:
            surfaces_to_process = [s for s in self.surfaces
                                   if s.full_name in surface_names]

        if not surfaces_to_process:
            print("No surfaces to process")
            return {}

        # Create adsorbate molecule
        adsorbate_mol = self._create_adsorbate_molecule(adsorbate)

        print(f"Generating adsorbed structures ({adsorbate})...")

        for surface in tqdm(surfaces_to_process, desc="Surfaces", unit="surface"):
            surface_name = surface.full_name

            if surface_name not in self._adsorption_sites:
                continue

            adsorption_sites = self._adsorption_sites[surface_name]

            # Determine site indices to process
            if site_indices == "all":
                target_indices = list(range(1, len(adsorption_sites.get('all', [])) + 1))
            elif isinstance(site_indices, int):
                target_indices = [site_indices]
            elif isinstance(site_indices, list):
                target_indices = site_indices
            else:
                raise ValueError("site_indices should be int, list, or 'all'")

            # Generate adsorbed structures for each site
            asf = AdsorbateSiteFinder(surface)
            surface_structures = {}

            for site_idx in target_indices:
                try:
                    if 'all' not in adsorption_sites or site_idx > len(adsorption_sites['all']):
                        continue

                    target_site = adsorption_sites['all'][site_idx - 1]
                    adsorbed_structure = asf.add_adsorbate(
                        adsorbate_mol,
                        target_site,
                        **kwargs
                    )

                    surface_structures[site_idx] = adsorbed_structure

                except Exception:
                    continue

            self._adsorbed_structures[surface_name] = surface_structures

        # Count total generated structures
        total_structures = sum(len(structures) for structures in self._adsorbed_structures.values())
        print(f"Generated {total_structures} adsorbed structures")

        return self._adsorbed_structures

    def _create_adsorbate_molecule(self, adsorbate: str) -> Molecule:
        """
        Create pymatgen Molecule object for the adsorbate

        Args:
            adsorbate: Adsorbate species

        Returns:
            pymatgen Molecule object
        """
        adsorbate = adsorbate.upper()

        if adsorbate == 'H':
            return Molecule(['H'], [[0, 0, 0]])
        elif adsorbate == 'O':
            return Molecule(['O'], [[0, 0, 0]])
        elif adsorbate == 'OH':
            return Molecule(['O', 'H'], [[0, 0, 0], [0, 0, 1.0]])
        elif adsorbate == 'H2O':
            return Molecule(['O', 'H', 'H'],
                            [[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]])
        elif adsorbate == 'CO':
            return Molecule(['C', 'O'], [[0, 0, 0], [1.128, 0, 0]])
        else:
            raise ValueError(f"Unsupported adsorbate: {adsorbate}")

    def calculate_adsorption_energies(
            self,
            adsorbate: str = "H",
            surface_names: Optional[List[str]] = None,
            **kwargs
    ) -> Dict[str, Dict[int, float]]:
        """
        Calculate adsorption energies for all generated adsorbed structures.

        Args:
            adsorbate: Adsorbate species.
            surface_names: List of surface names to process.

        Returns:
            Dictionary:
                surface_name -> {site_index: adsorption_energy}
        """
        try:
            from ..predictor.Eads.utils.ActiveLearningPred import predict
        except ImportError:
            raise ImportError(
                "ActiveLearningPred.predict not found. "
                "Ensure utils module is available."
            )

        # Ensure structures are generated
        if not self._adsorbed_structures:
            self.generate_adsorbed_structures(adsorbate, surface_names)

        if surface_names is None:
            surfaces_to_process = self.surfaces
        else:
            surfaces_to_process = [
                s for s in self.surfaces
                if s.full_name in surface_names
            ]

        if not surfaces_to_process:
            print("No surfaces to process")
            return {}

        print("Calculating adsorption energies...")

        # ==================================================
        # Collect ALL structures
        # ==================================================
        all_structures = []
        mapping = []  # (surface_name, site_idx)

        for surface in surfaces_to_process:

            surface_name = surface.full_name

            if surface_name not in self._adsorbed_structures:
                continue

            surface_structures = self._adsorbed_structures[surface_name]

            for site_idx, adsorbed_structure in surface_structures.items():
                all_structures.append(adsorbed_structure)
                mapping.append((surface_name, site_idx))

        if len(all_structures) == 0:
            print("No adsorption structures found.")
            return {}

        # ==================================================
        # Batch prediction (ONLY ONCE)
        # ==================================================
        try:
            predictions = predict(all_structures)

        except Exception as e:
            print(f"Prediction failed: {e}")

            self._adsorption_energies = {}

            for surface_name, site_idx in mapping:
                self._adsorption_energies.setdefault(surface_name, {})
                self._adsorption_energies[surface_name][site_idx] = None

            return self._adsorption_energies

        # ==================================================
        # Recover dictionary
        # ==================================================
        self._adsorption_energies = {}

        for (surface_name, site_idx), pred in zip(mapping, predictions):
            self._adsorption_energies.setdefault(surface_name, {})

            self._adsorption_energies[surface_name][site_idx] = float(pred)

        successful = sum(
            1
            for energies in self._adsorption_energies.values()
            for e in energies.values()
            if e is not None
        )

        print(f"Calculated energies for {successful} sites")

        return self._adsorption_energies

    # ========== DATA FORMATTING METHODS ==========

    def get_formatted_data(self) -> Dict:
        """
        Get adsorption data in the specified nested dictionary format

        Returns:
            Dictionary containing formatted adsorption data
        """
        formatted_data = {}

        # Group surfaces by formula
        formula_groups = {}
        for surface in self.surfaces:
            formula = surface.bulk_structure.reduced_formula
            if formula not in formula_groups:
                formula_groups[formula] = []
            formula_groups[formula].append(surface)

        # Build the nested structure
        for formula, surfaces in formula_groups.items():
            formula_dict = {}

            for surface in surfaces:
                surface_name = surface.full_name
                miller_index_str = ''.join(map(str, surface.miller_index))

                # Add surface entry
                if miller_index_str not in formula_dict:
                    formula_dict[miller_index_str] = {}

                # Check if we have adsorption sites for this surface
                if surface_name in self._adsorption_sites:
                    sites_dict = {}

                    # Get all adsorption sites
                    adsorption_sites = self._adsorption_sites[surface_name].get('all', [])

                    for site_idx, site_coords in enumerate(adsorption_sites, 1):
                        site_key = str(site_idx)
                        sites_dict[site_key] = {}

                        # Check if we have adsorbed structures for this site
                        if (surface_name in self._adsorbed_structures and
                                site_idx in self._adsorbed_structures[surface_name]):

                            structure = self._adsorbed_structures[surface_name][site_idx]

                            # Generate filename
                            filename = f"H{site_idx}.vasp"

                            # Get adsorption energy and convert to Python float
                            eads = self._adsorption_energies.get(surface_name, {}).get(site_idx, None)
                            if eads is not None:
                                eads = float(eads) if hasattr(eads, '__float__') else eads

                            sites_dict[site_key][filename] = {
                                "POSCAR": structure.to(fmt="poscar"),
                                "Eads": eads
                            }

                    if sites_dict:
                        formula_dict[miller_index_str] = sites_dict

            if formula_dict:  # Only add if there's data
                formatted_data[formula] = formula_dict

        return formatted_data

    def save_formatted_data(
            self,
            output_file: Union[str, Path] = "adsorption_data.json",
            indent: int = 2
    ) -> Path:
        """
        Save adsorption data in the specified format to JSON file

        Args:
            output_file: Path to output JSON file
            indent: JSON indentation level

        Returns:
            Path to saved file
        """
        output_path = Path(output_file)

        # Get formatted data
        formatted_data = self.get_formatted_data()

        # Convert to serializable format
        formatted_data_serializable = self._convert_to_serializable(formatted_data)

        # Save to JSON
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(formatted_data_serializable, f, indent=indent, ensure_ascii=False)
            print(f"Saved JSON data to: {output_path}")
        except Exception as e:
            print(f"Error saving JSON file: {str(e)[:50]}")

        return output_path

    # ========== FILE SAVING METHODS ==========

    def save_adsorbed_structures_vasp(
            self,
            output_dir: Union[str, Path] = "adsorbed_structures_vasp",
            include_clean_slab: bool = True
    ) -> Dict[str, Any]:
        """
        Save adsorbed structures as VASP POSCAR files in organized directory structure
        and generate the formatted data structure

        Args:
            output_dir: Output directory path
            include_clean_slab: Whether to also save clean slab structure

        Returns:
            Dictionary containing the formatted data structure
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize data structure
        formatted_data = []
        formula_groups = {}

        # Group surfaces by formula
        for surface in self.surfaces:
            formula = surface.bulk_structure.reduced_formula
            if formula not in formula_groups:
                formula_groups[formula] = []
            formula_groups[formula].append(surface)

        print(f"Saving VASP files to: {output_path}")

        saved_files = 0

        # Process each formula group
        for formula, surfaces in formula_groups.items():
            formula_dict = {formula: {}}
            formula_dir = output_path / formula
            formula_dir.mkdir(parents=True, exist_ok=True)

            for surface in tqdm(surfaces, desc=f"{formula}", unit="surface"):
                surface_name = surface.full_name
                miller_index_str = ''.join(map(str, surface.miller_index))

                # Create surface directory
                surface_dir = formula_dir / miller_index_str
                surface_dir.mkdir(parents=True, exist_ok=True)

                # Save clean slab if requested
                if include_clean_slab:
                    clean_slab_file = surface_dir / "clean_slab.vasp"
                    try:
                        surface.to(fmt="poscar", filename=str(clean_slab_file))
                        saved_files += 1
                    except Exception:
                        pass

                # Initialize sites dictionary for this surface
                sites_dict = {}

                # Check if we have adsorbed structures for this surface
                if surface_name in self._adsorbed_structures:
                    surface_structures = self._adsorbed_structures[surface_name]

                    for site_idx, structure in surface_structures.items():
                        # Create filename
                        filename = f"H{site_idx}.vasp"
                        file_path = surface_dir / filename

                        # Save structure as VASP POSCAR
                        try:
                            structure.to(fmt="poscar", filename=str(file_path))
                            saved_files += 1

                            # Get adsorption energy if available
                            eads = self._adsorption_energies.get(surface_name, {}).get(site_idx, None)

                            # Convert float32 to Python float if necessary
                            if eads is not None:
                                eads = float(eads)

                            # Add to sites dictionary
                            sites_dict[str(site_idx)] = {
                                filename: {
                                    "POSCAR": structure.to(fmt="poscar"),
                                    "Eads": eads
                                }
                            }

                        except Exception:
                            continue

                # Add to formula dictionary if we have sites
                if sites_dict:
                    formula_dict[formula][miller_index_str] = sites_dict
                else:
                    # Add empty entry if no adsorbed structures
                    formula_dict[formula][miller_index_str] = {}

            # Add formula dictionary to data if it has any surfaces
            if formula_dict[formula]:
                formatted_data.append(formula_dict)

        print(f"Saved {saved_files} VASP files")
        return formatted_data

    def _convert_to_serializable(self, obj):
        """
        Recursively convert numpy and other non-serializable types to serializable types
        """
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            # Try to convert to string as fallback
            try:
                return str(obj)
            except:
                return repr(obj)

    # ========== COMPLETE WORKFLOW METHODS ==========

    def generate_adsorbed_structures_with_predictions(
            self,
            adsorbate: str = "H",
            surface_names: Optional[List[str]] = None,
            site_indices: Union[int, List[int], str] = "all",
            save_vasp_files: bool = True,
            output_dir: Optional[Union[str, Path]] = None,
            **kwargs
    ) -> Dict[str, Any]:
        """
        Generate adsorbed structures and predict adsorption energies,
        then save in the requested format.

        Args:
            adsorbate: Adsorbate species
            surface_names: List of surface names to process
            site_indices: Site indices to process
            save_vasp_files: Whether to save VASP files
            output_dir: Output directory for VASP files
            **kwargs: Additional parameters for adsorbate placement

        Returns:
            Dictionary containing formatted adsorption data
        """
        try:
            from ..predictor.Eads.utils.ActiveLearningPred import predict
        except ImportError:
            raise ImportError(
                "ActiveLearningPred.predict not found. Ensure utils module is available."
            )

        self.generate_adsorbed_structures(
            adsorbate=adsorbate,
            surface_names=surface_names,
            site_indices=site_indices,
            **kwargs
        )

        if surface_names is None:
            surfaces_to_process = self.surfaces
        else:
            surfaces_to_process = [
                s for s in self.surfaces
                if s.full_name in surface_names
            ]

        print("Calculating adsorption energies...")

        all_structures = []
        mapping = []  # (surface_name, site_idx)

        for surface in surfaces_to_process:

            surface_name = surface.full_name

            if surface_name not in self._adsorbed_structures:
                continue

            for site_idx, adsorbed_structure in self._adsorbed_structures[surface_name].items():
                all_structures.append(adsorbed_structure)
                mapping.append((surface_name, site_idx))

        total_energies = len(all_structures)
        successful_energies = 0

        self._adsorption_energies = {}

        try:

            predictions = predict(all_structures)

            for (surface_name, site_idx), pred in zip(mapping, predictions):
                self._adsorption_energies.setdefault(surface_name, {})
                self._adsorption_energies[surface_name][site_idx] = float(pred)

                successful_energies += 1

        except Exception as e:

            print(f"Prediction failed: {e}")

            for surface_name, site_idx in mapping:
                self._adsorption_energies.setdefault(surface_name, {})
                self._adsorption_energies[surface_name][site_idx] = None

        print(f"Calculated {successful_energies}/{total_energies} adsorption energies")

        formatted_data = None

        if save_vasp_files:

            if output_dir is None:
                output_dir = f"{adsorbate}_adsorption_results"

            formatted_data = self.save_adsorbed_structures_vasp(
                output_dir=output_dir,
                include_clean_slab=True
            )

        self.get_site_properties_dataframe()

        return formatted_data if formatted_data else self.get_formatted_data()

    def H_adsorption_analysis(
            self,
            adsorbate: str = "H",
            site_indices: Union[int, List[int], str] = "all",
            output_dir: Union[str, Path] = "H_adsorption_analysis",
            include_summary: bool = True,
            include_dataframe: bool = True,
            include_visualization: bool = False
    ) -> Dict[str, Any]:
        """
        Complete workflow: generate structures, predict energies, and export all results

        Args:
            adsorbate: Adsorbate species
            site_indices: Site indices to process
            output_dir: Output directory for all files
            include_summary: Whether to generate summary files
            include_dataframe: Whether to save DataFrame as CSV
            include_visualization: Whether to generate visualization files

        Returns:
            Dictionary containing all analysis results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("=" * 50)
        print(f"Adsorption Analysis: {adsorbate}")
        print(f"Surfaces: {len(self.surfaces)}")
        print(f"Output: {output_path}")
        print("=" * 50)

        # Step 1: Get adsorption sites
        print("\n[1/5] Finding adsorption sites...")
        self.get_adsorption_sites_for_all_surfaces()

        # Step 2: Generate structures and predict energies
        print("\n[2/5] Generating structures and predicting energies...")
        formatted_data = self.generate_adsorbed_structures_with_predictions(
            adsorbate=adsorbate,
            site_indices=site_indices,
            save_vasp_files=True,
            output_dir=output_path / "vasp_structures"
        )

        # Step 3: Save formatted data
        print("\n[3/5] Saving formatted data...")
        json_file = output_path / "adsorption_data.json"
        self.save_formatted_data(output_file=json_file)

        # Step 4: Save comprehensive DataFrame
        if include_dataframe:
            print("\n[4/5] Saving analysis data...")
            df = self.get_site_properties_dataframe()
            if df is not None:
                csv_file = output_path / "adsorption_analysis.csv"
                df.to_csv(csv_file, index=False)

                # Save summary statistics
                summary_stats = {
                    "total_surfaces": len(self.surfaces),
                    "total_sites_analyzed": len(df) if df is not None else 0,
                    "adsorbate": adsorbate,
                }

                if 'adsorption_energy' in df.columns:
                    energies = df['adsorption_energy'].dropna()
                    if len(energies) > 0:
                        summary_stats["energy_statistics"] = {
                            "min": float(energies.min()),
                            "max": float(energies.max()),
                            "mean": float(energies.mean()),
                            "std": float(energies.std()),
                        }

                stats_file = output_path / "analysis_summary.json"
                with open(stats_file, 'w') as f:
                    json.dump(summary_stats, f, indent=2)

                print(f"  Saved: {csv_file}")
                print(f"  Saved: {stats_file}")

        # Step 5: Visualizations (if requested)
        if include_visualization:
            print("\n[5/5] Generating visualizations...")
            self._generate_visualizations(output_path)

        print("\n" + "=" * 50)
        print("Analysis complete")
        print(f"Results saved to: {output_path}")
        print("=" * 50)

        return {
            "formatted_data": formatted_data,
            "dataframe": self._site_properties_df,
            "output_directory": output_path,
        }

    # ========== HELPER METHODS ==========

    def get_site_properties_dataframe(self) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with all adsorption site properties

        Returns:
            pandas DataFrame with site properties
        """
        data = []

        print("Creating properties DataFrame...")

        for surface in tqdm(self.surfaces, desc="Processing", unit="surface"):
            surface_name = surface.full_name
            miller_index = surface.miller_index

            # Get adsorption sites for this surface
            if surface_name in self._adsorption_sites:
                adsorption_sites = self._adsorption_sites[surface_name]

                if 'all' in adsorption_sites:
                    for i, site_coords in enumerate(adsorption_sites['all'], 1):
                        # Create row for each site
                        row = {
                            'surface_name': surface_name,
                            'miller_index': miller_index,
                            'site_index': i,
                            'site_coordinates_x': site_coords[0],
                            'site_coordinates_y': site_coords[1],
                            'site_coordinates_z': site_coords[2],
                            'formula': surface.bulk_structure.reduced_formula,
                            'n_atoms': len(surface),
                            'surface_area': getattr(surface, 'surface_area', None),
                            'surface_energy': getattr(surface, 'surface_energy', None),
                        }

                        # Add adsorption energy if calculated
                        if (surface_name in self._adsorption_energies and
                                i in self._adsorption_energies[surface_name]):
                            row['adsorption_energy'] = self._adsorption_energies[surface_name][i]

                        data.append(row)
            else:
                # Add row even if no adsorption sites found
                row = {
                    'surface_name': surface_name,
                    'miller_index': miller_index,
                    'site_index': None,
                    'formula': surface.composition.reduced_formula,
                    'n_atoms': len(surface),
                    'surface_area': getattr(surface, 'surface_area', None),
                    'surface_energy': getattr(surface, 'surface_energy', None),
                }
                data.append(row)

        if data:
            self._site_properties_df = pd.DataFrame(data)
            print(f"Created DataFrame: {len(self._site_properties_df)} rows")

        return self._site_properties_df

    def _generate_visualizations(self, output_path: Path):
        """Generate visualization files using Surface.visualize_adsorption_sites method"""
        viz_dir = output_path / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        print("Generating adsorption site visualizations...")

        processed_count = 0
        for surface in tqdm(self.surfaces, desc="Visualizing", unit="surface"):
            surface_name = surface.full_name

            if surface_name not in self._adsorption_sites:
                continue

            adsorption_sites = self._adsorption_sites[surface_name]

            if 'all' not in adsorption_sites or not adsorption_sites['all']:
                continue

            try:
                colors = self._prepare_site_colors(surface_name, adsorption_sites)
                self._visualize_and_save_surface(
                    surface=surface,
                    surface_name=surface_name,
                    formula=surface.composition.reduced_formula,
                    adsorption_sites=adsorption_sites,
                    site_colors=colors,
                    output_dir=viz_dir
                )
                processed_count += 1

            except Exception:
                continue

        print(f"Generated {processed_count} visualizations")

    def _prepare_site_colors(self, surface_name: str, adsorption_sites: dict) -> List[str]:
        """
        Prepare colors for adsorption sites based on adsorption energies

        Args:
            surface_name: Name of the surface
            adsorption_sites: Dictionary of adsorption sites

        Returns:
            List of color strings for each site
        """
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Get adsorption energies for this surface
        energies = {}
        if surface_name in self._adsorption_energies:
            energies = self._adsorption_energies[surface_name]

        # Get sites
        sites = adsorption_sites.get('all', [])
        n_sites = len(sites)

        if not n_sites:
            return []

        # Default colors (black)
        colors = ['black'] * n_sites

        if energies:
            # Find min and max energies for normalization
            valid_energies = [e for e in energies.values() if e is not None]
            if valid_energies:
                min_energy = min(valid_energies)
                max_energy = max(valid_energies)

                # Use RdYlBu_r colormap
                cmap = cm.RdYlBu_r

                for i in range(1, n_sites + 1):
                    if i in energies and energies[i] is not None:
                        # Normalize energy
                        if max_energy != min_energy:
                            norm_energy = (energies[i] - min_energy) / (max_energy - min_energy)
                        else:
                            norm_energy = 0.5

                        color = cmap(norm_energy)
                        colors[i - 1] = color

        return colors

    def _visualize_and_save_surface(
            self,
            surface,
            surface_name: str,
            formula: str,
            adsorption_sites: dict,
            site_colors: List[str],
            output_dir: Path
    ):
        """
        Use Surface.visualize_adsorption_sites to create and save visualization
        """
        try:
            import matplotlib.pyplot as plt
            from pymatgen.analysis.adsorption import plot_slab

            fig, ax = plt.subplots(figsize=(10, 8))

            plot_slab(
                surface,
                ax,
                adsorption_sites=adsorption_sites,
                draw_unit_cell=True,
            )

            # Remove and replace adsorption site markers
            lines_to_remove = []

            for line in ax.lines:
                if line.get_marker() == 'x':
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    lines_to_remove.append((line, x_data, y_data))

            # Clear all adsorption site lines
            for line, _, _ in lines_to_remove:
                line.remove()

            # Re-draw with colored markers
            site_counter = 0

            for line, x_data, y_data in lines_to_remove:
                for j, (x, y) in enumerate(zip(x_data, y_data)):
                    if site_counter < len(site_colors):
                        color = site_colors[site_counter]
                    else:
                        color = 'black'

                    label = None
                    if surface_name in self._adsorption_energies:
                        energy = self._adsorption_energies[surface_name].get(site_counter + 1, None)
                        if energy is not None:
                            label = f'Site {site_counter + 1}: {energy:.2f} eV'

                    ax.plot(x, y,
                            marker='X',
                            markersize=12,
                            markeredgewidth=1,
                            markeredgecolor='black',
                            markerfacecolor=color,
                            linestyle='',
                            zorder=10000,
                            label=label)

                    site_counter += 1

            ax.set_title(f"Adsorption Sites on {surface_name}", fontsize=14, fontweight='bold', pad=15)

            # Add colorbar if needed
            if any(c != 'black' for c in site_colors):
                self._add_energy_colorbar(ax, surface_name)

            # Add legend for sites with energy labels
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 0:
                # Only show unique labels
                unique = {}
                for h, l in zip(handles, labels):
                    if l and l not in unique:
                        unique[l] = h

                if unique:
                    ax.legend(unique.values(), unique.keys(),
                              loc='upper left', bbox_to_anchor=(-0.32, 1),
                              fontsize=11, framealpha=0.9)

            plt.tight_layout()

            # Save figure
            safe_name = "".join(c for c in surface_name if c.isalnum() or c in ('_', '-')).rstrip()
            plot_file = output_dir / f"{safe_name}_adsorption_sites.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

        except Exception:
            pass

    def _add_energy_colorbar(self, ax, surface_name: str):
        """Add colorbar legend for adsorption energy colors on the right side"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
            from matplotlib.colorbar import ColorbarBase
            from matplotlib.colors import Normalize

            if surface_name not in self._adsorption_energies:
                return

            valid_energies = [e for e in self._adsorption_energies[surface_name].values()
                              if e is not None]
            if not valid_energies:
                return

            min_energy = min(valid_energies)
            max_energy = max(valid_energies)

            cmap = cm.RdYlBu_r
            norm = Normalize(vmin=min_energy, vmax=max_energy)

            cax = ax.inset_axes([1.05, 0.15, 0.05, 0.7])
            cb = ColorbarBase(cax, cmap=cmap, norm=norm,
                              orientation='vertical')

            cb.set_label('Adsorption Energy (eV)', fontsize=16, labelpad=10)
            cb.ax.tick_params(labelsize=12)

        except Exception:
            pass

    # ========== UTILITY METHODS ==========

    @classmethod
    def from_surfaces_list(cls, surfaces: List[Surface]):
        """
        Create AdsorptionManager from list of surfaces

        Args:
            surfaces: List of Surface objects

        Returns:
            AdsorptionManager instance
        """
        return cls(surfaces)

    def add_surface(self, surface: Surface):
        """
        Add a surface to the manager

        Args:
            surface: Surface object to add
        """
        self.surfaces.append(surface)
        self._clear_cached_data()

    def _clear_cached_data(self):
        """Clear cached data when surfaces are modified"""
        self._adsorption_sites.clear()
        self._adsorbed_structures.clear()
        self._adsorption_energies.clear()
        self._site_properties_df = None
        self._summary_df = None

    def get_most_stable_adsorption_sites(
            self,
            n_sites: int = 5,
            surface_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get the most stable adsorption sites across all surfaces

        Args:
            n_sites: Number of top sites to return
            surface_names: List of surface names to consider (if None, all surfaces)

        Returns:
            DataFrame sorted by adsorption energy (most stable first)
        """
        if self._site_properties_df is None:
            self.get_site_properties_dataframe()

        if 'adsorption_energy' not in self._site_properties_df.columns:
            raise ValueError("Adsorption energies not calculated.")

        # Filter by surface names if specified
        df = self._site_properties_df.copy()
        if surface_names is not None:
            df = df[df['surface_name'].isin(surface_names)]

        # Remove rows without adsorption energy
        df = df.dropna(subset=['adsorption_energy'])

        # Sort by adsorption energy (most negative is most stable)
        df_sorted = df.sort_values('adsorption_energy').head(n_sites)

        return df_sorted

    @staticmethod
    def load_adsorption_energies(csv_file):
        """
        Read adsorption energies from csv.

        Parameters
        ----------
        csv_file : str or Path

        Returns
        -------
        dict

        Example
        -------
        {
            "Mg_001_1":{
                1:0.88,
                2:-0.18,
                3:-0.19,
                4:-0.10
            },
            "CaMg2_101_8":{
                1:-0.14,
                2:-0.27,
                5:-0.25,
                ...
            }
        }
        """

        df = pd.read_csv(csv_file)

        adsorption_energies = {}

        for _, row in df.iterrows():

            # surface name
            surface_name = (
                f"{row['formula']}_"
                f"{str(row['miller_index']).zfill(3)}_"
                f"{row['termination']}"
            )

            # H8 -> 8
            site_index = int(str(row["ads_position"]).replace("H", ""))

            eads = float(row["Eads"])

            if surface_name not in adsorption_energies:
                adsorption_energies[surface_name] = {}

            adsorption_energies[surface_name][site_index] = eads

        return adsorption_energies

    def __repr__(self):
        total_sites = sum(len(sites.get('all', [])) for sites in self._adsorption_sites.values())
        return f"AdsorptionManager({len(self.surfaces)} surfaces, {total_sites} adsorption sites)"