from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from pymatgen.core import Structure
from pymatgen.core.surface import Slab, generate_all_slabs
from pymatgen.analysis.wulff import WulffShape
from pymatgen.analysis.surface_analysis import SlabEntry
from pymatgen.entries.computed_entries import ComputedStructureEntry
import warnings
from sympy import Symbol
from tqdm.auto import tqdm  # import tqdm

from GalvCalc.core.utils import MyPlotter
from GalvCalc.core.structures import Surface, Bulk


class SurfaceProperties:
    """
    Class for managing and calculating properties for a batch of surfaces
    """

    def __init__(self, surfaces: Optional[List[Surface]] = None):
        self.bulk_structure = None
        self.surfaces = surfaces or []
        self._properties_df = None
        self._surface_terminations = {}
        self.surface_energies_dict = {}
        self.work_functions_dict = {}
        self.plotter = None

    @classmethod
    def from_bulk_structure(
            cls,
            bulk_structure: Union[Structure, Bulk],
            max_index: int = 2,
            min_slab_size: float = 10.0,
            min_vacuum_size: float = 15.0,
            center_slab: bool = True,
            symmetrize: bool = True,
            max_normal_search: int = 1,
            **kwargs
    ):
        """Create SurfaceProperties from bulk structure using generate_all_slabs"""
        if hasattr(bulk_structure, 'as_structure'):
            bulk_for_slab = bulk_structure.as_structure()
        else:
            bulk_for_slab = bulk_structure

        slabs = generate_all_slabs(
            bulk_for_slab,
            max_index=max_index,
            min_slab_size=min_slab_size,
            min_vacuum_size=min_vacuum_size,
            center_slab=center_slab,
            symmetrize=symmetrize,
            max_normal_search=max_normal_search,
            **kwargs
        )

        surfaces = []
        termination_counts = {}

        print(f"Generating surfaces from bulk structure...")

        # show a progress bar with tqdm
        for slab in tqdm(slabs, desc="Creating surfaces", unit="slab"):
            miller_index = slab.miller_index

            if miller_index not in termination_counts:
                termination_counts[miller_index] = 0
            termination_counts[miller_index] += 1

            surface = Surface.from_slab(slab, bulk_structure=bulk_structure)
            surface.termination_index = termination_counts[miller_index]
            surface.full_name = f"{bulk_structure.composition.reduced_formula}_{''.join(map(str, miller_index))}_{termination_counts[miller_index]}"

            surfaces.append(surface)

        print(f"Generated {len(surfaces)} surfaces with {len(termination_counts)} unique Miller indices")

        instance = cls(surfaces)
        instance._surface_terminations = termination_counts
        instance.bulk_structure = bulk_structure
        return instance

    def create_slab_entries(self, slab_energies: Dict[str, float]) -> List[SlabEntry]:
        """
        Create proper SlabEntry objects for MyPlotter

        Args:
            slab_energies: Dictionary mapping surface names to energies

        Returns:
            List of SlabEntry objects
        """
        slab_entries = []
        successful = 0

        print(f"Creating SlabEntry objects...")
        # show a progress bar with tqdm
        for surface in tqdm(self.surfaces, desc="Creating entries", unit="surface"):
            print(surface.full_name)
            print(surface.miller_index)
            full_name = surface.full_name
            if full_name in slab_energies:
                try:
                    slab_entry = SlabEntry(
                        structure=surface,
                        energy=slab_energies[full_name],
                        miller_index=surface.miller_index,
                        label=full_name,
                    )
                    slab_entries.append(slab_entry)
                    successful += 1
                except Exception as e:
                    print(f"Warning: Failed to create SlabEntry for {full_name}: {str(e)[:50]}...")

        print(f"Successfully created {successful}/{len(self.surfaces)} SlabEntry objects")
        return slab_entries

    def create_surface_energy_plotter(self, slab_energies: Dict[str, float],
                                      bulk_energy: float) -> MyPlotter:
        """
        Create a MyPlotter instance with proper SlabEntry objects

        Args:
            slab_energies: Dictionary of slab energies
            bulk_energy: Total energy of bulk structure

        Returns:
            MyPlotter instance
        """
        slab_entries = self.create_slab_entries(slab_energies)

        if not slab_entries:
            raise ValueError("No valid SlabEntry objects created")

        if self.surfaces and hasattr(self.surfaces[0], 'bulk_structure') and self.surfaces[0].bulk_structure:
            bulk_structure = self.surfaces[0].bulk_structure
        else:
            bulk_structure = self.surfaces[0]

        bulk_entry = ComputedStructureEntry(
            structure=bulk_structure,
            energy=bulk_energy
        )

        plotter = MyPlotter(
            all_slab_entries=slab_entries,
            ucell_entry=bulk_entry
        )
        self.plotter = plotter

        print(f"Surface energy plotter created with {len(slab_entries)} slab entries")
        return plotter

    def calculate_surface_energies_with_plotter(
            self,
            slab_energies: Dict[str, float],
            bulk_energy: float,
            delu_dict: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Calculate surface energies using MyPlotter with chemical potentials

        Args:
            slab_energies: Dictionary mapping surface names to slab energies (eV)
            bulk_energy: Total energy of bulk structure (eV)
            delu_dict: Dictionary of chemical potentials {Symbol: value}

        Returns:
            Dictionary of surface energies (J/m^2)
        """
        try:
            plotter = self.create_surface_energy_plotter(slab_energies, bulk_energy)

            if delu_dict is None:
                delu_dict = self._get_default_chempots()
                print(f"Using default chemical potentials: {delu_dict}")

            surface_energies = {}
            self.entries = plotter.all_slab_entries
            unique_miller_indices = set(plotter.all_slab_entries.keys())

            print(f"Calculating surface energies for {len(unique_miller_indices)} Miller indices...")

            for hkl in tqdm(unique_miller_indices, desc="Calculating energies", unit="index"):
                try:
                    stable_entry, gamma = plotter.get_stable_entry_at_u(
                        miller_index=hkl,
                        delu_dict=delu_dict,
                        delu_default=0
                    )
                    gamma_jm2 = gamma * 1.60218e-19 / 1e-20
                    surface_energies[stable_entry.label] = gamma_jm2
                except Exception as e:
                    print(f"Warning: Failed to calculate energy for {hkl}: {str(e)[:50]}...")

            print(f"Surface energy calculation completed: {len(surface_energies)} surfaces processed")
            return surface_energies

        except Exception as e:
            print(f"Error in surface energy calculation: {str(e)[:100]}")
            raise

    def _get_default_chempots(self) -> Dict[Symbol, float]:
        """
        Get default chemical potentials based on the elements in the system
        """
        if not self.surfaces:
            return {}

        elements = list(self.surfaces[0].composition.as_dict().keys())
        delu_dict = {Symbol(f"delu_{element}"): 0.0 for element in elements}
        return delu_dict

    def get_properties_dataframe(
            self,
            slab_energies: Optional[Dict[str, float]] = None,
            bulk_energy: Optional[float] = None,
            work_functions: Optional[Dict[str, float]] = None,
            delu_dict: Optional[Dict] = None,
            **kwargs
    ) -> pd.DataFrame:
        """
        Generate a pandas DataFrame with all surface properties
        """
        data = []
        surface_energies = {}

        if slab_energies is not None and bulk_energy is not None:
            print(f"Calculating surface properties...")
            surface_energies = self.calculate_surface_energies_with_plotter(
                slab_energies, bulk_energy, delu_dict
            )
            self._update_surface_energies_dict(surface_energies)
            self._update_surface_energies(slab_energies, surface_energies)

        print(f"Compiling properties DataFrame...")

        # show a progress bar with tqdm
        for surface in tqdm(self.surfaces, desc="Processing surfaces", unit="surface"):
            full_name = surface.full_name
            row = {
                'full_name': full_name,
                'miller_index': surface.miller_index,
                'termination_index': surface.termination_index,
                'formula': self.bulk_structure.reduced_formula,
                'n_atoms': len(surface),
                'surface_area': surface.surface_area,
                'shift': surface.shift,
                'scale_factor': surface.scale_factor
            }

            if full_name in surface_energies and surface_energies[full_name] is not None:
                row['surface_energy_J_m2'] = surface_energies[full_name]
                row['surface_energy_eV_A2'] = surface_energies[full_name] * 1e-20 / 1.60218e-19

            if work_functions and full_name in work_functions:
                row['work_function_eV'] = work_functions[full_name]

            data.append(row)

        self._properties_df = pd.DataFrame(data)
        print(
            f"Properties DataFrame created: {len(self._properties_df)} rows, {len(self._properties_df.columns)} columns")
        return self._properties_df

    def _update_surface_energies_dict(self, surface_energies: Dict[str, float], work_functions: Dict[str, float] = None):
        """
        Update surface energy dictionary attribute
        """
        self.surface_energies_dict.clear()
        self.work_functions_dict.clear()
        energy_dict_by_miller = {}
        wf_dict_by_miller = {}

        for surface in self.surfaces:
            full_name = surface.full_name
            if full_name in surface_energies and surface_energies[full_name] is not None:
                miller_index = surface.miller_index
                energy = surface_energies[full_name]

                # keep the termination with the lowest energy for each Miller index
                if (miller_index not in energy_dict_by_miller or
                        energy < energy_dict_by_miller[miller_index]):
                    energy_dict_by_miller[miller_index] = energy
                    if work_functions and full_name in work_functions:
                        wf_dict_by_miller[miller_index] = work_functions[full_name]

        # store the lowest-energy surface of each Miller index in surface_props
        for miller_index, energy in energy_dict_by_miller.items():
            self.surface_energies_dict[miller_index] = energy
            if work_functions and miller_index in wf_dict_by_miller:
                self.work_functions_dict[miller_index] = wf_dict_by_miller[miller_index]

    def _update_surface_energies(self, slab_energies: Dict[str, float], surface_energies: Dict[str, float]):
        """
        Update energy values in surfaces list
        """
        updated_count = 0
        for surface in self.surfaces:
            full_name = surface.full_name
            if full_name in slab_energies:
                surface.energy = slab_energies[full_name]
            if full_name in surface_energies and surface_energies[full_name] is not None:
                surface.surface_energy = surface_energies[full_name]
                updated_count += 1

        if updated_count > 0:
            print(f"Updated surface energies for {updated_count} surfaces")

    def wulff_construct(
            self,
            surface_energies: Dict[Tuple[int, int, int], float],
            work_functions: Optional[Dict[Tuple[int, int, int], float]] = None,
            output_dir: Union[str, Path] = ".",
            save_plot: bool = True,
            save_csv: bool = True,
            show_plot: bool = True,
    ) -> Dict[str, Any]:
        """
        Construct Wulff shape using pymatgen's methods
        """
        try:
            print(f"Constructing Wulff shape...")

            miller_energy_map = {}
            miller_wf_map = {}

            # show a progress bar with tqdm
            for surface in tqdm(self.surfaces, desc="Grouping surfaces", unit="surface"):
                miller_index = surface.miller_index
                if miller_index in surface_energies:
                    if miller_index not in miller_energy_map:
                        miller_energy_map[miller_index] = []
                    energy_ev_a2 = surface_energies[miller_index] * 1e-20 / 1.60218e-19
                    miller_energy_map[miller_index].append(energy_ev_a2)

                    if work_functions and miller_index in work_functions:
                        if miller_index not in miller_wf_map:
                            miller_wf_map[miller_index] = []
                        miller_wf_map[miller_index].append(work_functions[miller_index])

            miller_list = []
            e_surf_list = []
            wf_list = []

            for miller_index, energies in miller_energy_map.items():
                miller_list.append(miller_index)
                e_surf_list.append(np.mean(energies))
                wf_list.append(np.mean(miller_wf_map.get(miller_index, [0.0])))

            if not miller_list:
                raise ValueError("No surface energies provided for Wulff construction")

            if self.surfaces and hasattr(self.surfaces[0], 'bulk_structure') and self.surfaces[0].bulk_structure:
                bulk_lattice = self.surfaces[0].bulk_structure.lattice
            else:
                bulk_lattice = self.surfaces[0].lattice

            ws = WulffShape(bulk_lattice, miller_list, e_surf_list)
            total_area = sum(ws.area_fraction_dict.values())
            area_fractions = [(area / total_area * 100) for area in ws.area_fraction_dict.values()]

            avg_surface_energy = np.sum(np.array(area_fractions) / 100 * np.array(e_surf_list))
            avg_work_function = np.sum(np.array(area_fractions) / 100 * np.array(wf_list)) if work_functions else None

            formula = self.bulk_structure.reduced_formula

            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            if save_csv:
                csv_path = output_path / "wulff_results.csv"
                with open(csv_path, "a", newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile)
                    if not csv_path.exists():
                        writer.writerow(["Formula", "Avg_Surface_Energy (J/m2)", "Avg_Work_Function (eV)"])
                    writer.writerow([formula, avg_surface_energy, avg_work_function if avg_work_function else "N/A"])

                print(f"Wulff results saved to: {csv_path}")

            if save_plot:
                plt.figure(figsize=(10, 8))
                ws.get_plot(direction=(1, 1, 2), show_area=True, alpha=0.9)
                plt.title(f"{formula} Wulff Construction")

                plot_path = output_path / f"wulff_{formula}.png"
                plt.savefig(plot_path, bbox_inches='tight', dpi=300, transparent=True)

                print(f"Wulff plot saved to: {plot_path}")

                if show_plot:
                    plt.show()
                else:
                    plt.close()

            # compact output format
            print(f"\n{formula} Wulff construction summary:")
            print(f"Average surface energy: {avg_surface_energy:.3f} J/m^2")
            if avg_work_function:
                print(f"Average work function: {avg_work_function:.3f} eV")

            print(f"\nSurface area fractions:")
            for hkl, area in ws.area_fraction_dict.items():
                percentage = (area / total_area) * 100
                print(f"  {hkl}: {percentage:5.1f}%")

            results = {
                "formula": formula,
                "wulff_shape": ws,
                "avg_surface_energy": avg_surface_energy,
                "avg_work_function": avg_work_function,
                "area_fractions": ws.area_fraction_dict,
                "miller_list": miller_list,
                "surface_energies": e_surf_list,
                "work_functions": wf_list if work_functions else None
            }

            print(f"Wulff construction completed successfully")
            return results

        except Exception as e:
            print(f"Error in Wulff construction: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return {}

    def get_most_stable_surfaces(self, n: int = 5) -> List[Surface]:
        if self._properties_df is None or 'surface_energy_J_m2' not in self._properties_df.columns:
            raise ValueError("Surface energies not calculated. Call get_properties_dataframe first.")

        sorted_df = self._properties_df.sort_values('surface_energy_J_m2').head(n)
        stable_surfaces = []

        for _, row in sorted_df.iterrows():
            full_name = row['full_name']
            surface = next((s for s in self.surfaces if s.full_name == full_name), None)
            if surface:
                stable_surfaces.append(surface)

        print(f"Identified {len(stable_surfaces)} most stable surfaces")
        return stable_surfaces

    def save_properties(self, filename: Union[str, Path]):
        if self._properties_df is None:
            raise ValueError("No properties calculated. Call get_properties_dataframe first.")

        self._properties_df.to_csv(filename, index=False)
        print(f"Surface properties saved to: {filename}")

    def get_predicted_surface_properties(
            self,
            output_dir: Union[str, Path] = "surfaces_output",
            copy_atom_init: Optional[Union[str, Path]] = Path(
                __file__).parent / "../predictor/surface_properties/atom_init.json"
    ) -> pd.DataFrame:
        """
        Save all surfaces to a folder as CIF files and create a CSV file.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving surface structures to: {output_path}")

        surface_info = []
        surface_mapping = {}
        saved_count = 0

        # show a progress bar with tqdm
        for i, surface in enumerate(tqdm(self.surfaces, desc="Saving CIF files", unit="surface")):
            if hasattr(surface, 'full_name') and surface.full_name:
                filename = surface.full_name
            else:
                miller_str = ''.join(str(x) for x in surface.miller_index)
                filename = f"surface_{i + 1}_{miller_str}"
                surface.full_name = filename

            safe_filename = "".join(c for c in filename if c.isalnum() or c in ('_', '-')).rstrip()
            cif_file = output_path / f"{safe_filename}.cif"

            try:
                surface.to(fmt="cif", filename=str(cif_file))
                surface_info.append((surface.full_name, 0, 0))
                surface_mapping[safe_filename] = i
                saved_count += 1
            except Exception as e:
                print(f"Warning: Failed to save {filename}: {str(e)[:50]}...")
                surface_info.append((surface.full_name, 0, 0))

        print(f"Successfully saved {saved_count}/{len(self.surfaces)} CIF files")

        csv_file = output_path / "id_prop.csv"
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in surface_info:
                    writer.writerow(row)
            print(f"CSV file created: {csv_file}")
        except Exception as e:
            print(f"Failed to create CSV file: {str(e)[:50]}...")

        if copy_atom_init:
            atom_init_path = Path(copy_atom_init)
            if atom_init_path.exists():
                try:
                    dest_path = output_path / "atom_init.json"
                    import shutil
                    shutil.copy2(atom_init_path, dest_path)
                    print(f"Copied atom_init.json to: {dest_path}")
                except Exception as e:
                    print(f"Warning: Failed to copy atom_init.json: {str(e)[:50]}...")
            else:
                print(f"Warning: atom_init.json not found at: {atom_init_path}")

        try:
            from ..predictor import predict_cgcnn

            print(f"\nRunning CGCNN predictions...")
            results = predict_cgcnn(
                cifpath=str(output_path),
                task='regression',
                batch_size=256,
                depth=2,
                save_results=False,
                output_file='surface_props.csv'
            )

            self._create_properties_df_from_predictions(results, surface_mapping)
            print(f"CGCNN predictions completed successfully")

        except ImportError as e:
            print(f"Warning: CGCNN predictor not available")
        except Exception as e:
            print(f"Warning: CGCNN prediction failed: {str(e)[:100]}")

        return self._properties_df

    def _create_properties_df_from_predictions(self, results: Dict, surface_mapping: Dict[str, int]):
        """
        Create properties DataFrame from CGCNN predictions.
        """
        try:
            cif_ids = results.get('cif_ids', [])
            predictions = results.get('predictions', [])

            if not cif_ids or not predictions:
                print("Warning: No predictions returned from CGCNN")
                return

            print(f"Processing {len(predictions)} predictions...")
            data = []
            processed_count = 0

            # show a progress bar with tqdm
            for cif_id, pred_se, pred_wf in tqdm(zip(cif_ids, predictions[0], predictions[1]),
                                                 total=len(cif_ids),
                                                 desc="Processing predictions"):
                if cif_id.endswith('.cif'):
                    cif_id = cif_id[:-4]

                surface_idx = surface_mapping.get(cif_id)

                if surface_idx is not None and surface_idx < len(self.surfaces):
                    surface = self.surfaces[surface_idx]

                    if pred_se and pred_wf:
                        surface_energy = float(pred_se)
                        work_function = float(pred_wf)
                    else:
                        surface_energy = None
                        work_function = None

                    row = {
                        'full_name': surface.full_name,
                        'miller_index': surface.miller_index,
                        'termination_index': getattr(surface, 'termination_index', None),
                        'formula': self.bulk_structure.reduced_formula,
                        'n_atoms': len(surface),
                        'surface_area': getattr(surface, 'surface_area', None),
                        'shift': getattr(surface, 'shift', None),
                        'scale_factor': getattr(surface, 'scale_factor', None),
                        'surface_energy_J_m2': surface_energy / 1e-20 * 1.60218e-19 if surface_energy else None,
                        'surface_energy_eV_A2': surface_energy,
                        'work_function_eV': work_function
                    }

                    data.append(row)
                    processed_count += 1

            if data:
                surface_energies = {}
                work_functions = {}
                for row in data:
                    if row['surface_energy_J_m2']:
                        surface_energies[row['full_name']] = row['surface_energy_J_m2']
                    if row['work_function_eV']:
                        work_functions[row['full_name']] = row['work_function_eV']

                self._update_surface_energies_dict(surface_energies, work_functions)

                self._properties_df = pd.DataFrame(data)
                print(f"Properties DataFrame created: {len(self._properties_df)} surfaces processed")
            else:
                print("Warning: No valid prediction data")

        except Exception as e:
            print(f"Error creating properties DataFrame: {str(e)[:100]}")