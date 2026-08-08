<p align="center">
  <img src="GalvCalc/assets/logo.png" alt="GalvCalc" width="400"/>
</p>

<p align="center">
  <em>From atomic-scale descriptors to macroscopic corrosion polarization curves</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/GalvCalc/"><img alt="PyPI" src="https://img.shields.io/pypi/v/GalvCalc?color=2b6cb0&label=PyPI"></a>
  <a href="https://pypi.org/project/GalvCalc/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/GalvCalc?color=2b6cb0"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/GNscSHI/GalvCalc?color=2b6cb0"></a>
</p>

**GalvCalc** is a computational framework for modeling micro-galvanic
corrosion of alloys with coupled anodic dissolution and hydrogen evolution
kinetics. Starting from a bulk crystal structure, it automates surface-model
generation, estimates electrochemical descriptors (equilibrium potentials,
work functions, surface energies and hydrogen adsorption free energies), and
assembles multi-phase polarization curves — with built-in anode/cathode
area-ratio optimization and alloy-content scans of the corrosion current and
potential.

The package is developed alongside the manuscript
*"GalvCalc: A Framework for Modeling Micro-galvanic Corrosion of Alloys with
Coupled Anodic Dissolution and Hydrogen Evolution Kinetics"*.

---

## Features

| Module | Purpose |
| --- | --- |
| `GalvCalc.core` | Bulk / surface structure classes, equilibrium-potential engine, Wulff-aware plotting helpers |
| `GalvCalc.cathode` | Surface-property manager (surface energies, work functions, Wulff shapes), hydrogen-adsorption analysis, facet-weighted exchange-current estimation |
| `GalvCalc.anode` | Substitutional surface doping of anode materials and electrochemical descriptor tables |
| `GalvCalc.polarization` | Butler–Volmer polarization curves, multi-anode/multi-cathode plots, area-ratio optimization, alloy-content scans |
| `GalvCalc.predictor` | ML predictors (CGCNN / TabPFN) for surface properties and hydrogen adsorption free energies |

**Highlights**

- End-to-end pipeline: bulk crystal → slab generation → Wulff shape →
  facet-weighted descriptors → corrosion polarization curves.
- Nernst-based equilibrium potentials for arbitrary ion combinations, backed
  by a built-in thermodynamic ion-energy database.
- Calibrated Butler–Volmer kinetics for both Mg- and Fe-based systems.
- DFT-grounded second-phase exchange currents for nine common Mg-alloy
  intermetallics (Mg<sub>17</sub>Al<sub>12</sub>, Mg<sub>2</sub>Al<sub>3</sub>, MgZn<sub>2</sub>, LaMg<sub>12</sub>, CaMg<sub>2</sub>, Y<sub>5</sub>Mg<sub>24</sub>, NdMg<sub>3</sub>,
  Mg<sub>2</sub>Si, CeMg<sub>12</sub>).
- CGCNN prediction of surface energies and work functions, and
  TabPFN-based prediction of hydrogen adsorption free energies.

---

## Installation

```bash
pip install GalvCalc
```

Optional extras:

```bash
pip install "GalvCalc[defects]"   # full pymatgen defect-based anode doping
pip install "GalvCalc[ml]"        # ML predictors (CGCNN / TabPFN)
pip install "GalvCalc[examples]"  # run the bundled Jupyter notebooks
```

The `defects` extra requires Python >= 3.10 and enables the substitution-based
doping workflow in `GalvCalc.anode`.

**Requirements:** Python >= 3.9; `numpy`, `scipy`, `pandas`, `matplotlib`,
`pymatgen`, `PyYAML`, `sympy`, `tqdm`, `joblib`, `openpyxl`. The ML extras add
`torch`, `scikit-learn`, `tabpfn`.

---

## Quick start

### 1. Equilibrium potential

```python
from GalvCalc.core.structures import Bulk

Ee = Bulk.get_equilibrium_potential(
    ions="Mg[2+]",
    ion_numbers=[1],
    energy_formation=0.0,
)
print(f"E_eq = {Ee:.3f} V vs. SHE")

# Multi-ion dissolution, e.g. Mg<sub>2</sub>Ge
Ee2 = Bulk.get_equilibrium_potential(
    ions="Mg[2+], Ge[2+]",
    ion_numbers=[2, 1],
    energy_formation=-0.24,
)
```

### 2. Surfaces, Wulff shape and adsorption

```python
from GalvCalc.core.structures import Bulk
from GalvCalc.cathode.surfaces import SurfaceProperties
from GalvCalc.cathode.hydrogen import AdsorptionManager

bulk = Bulk.from_file("Mg2Si.poscar")
surfaces = SurfaceProperties.from_bulk_structure(bulk, max_index=1, min_slab_size=10)

# DFT inputs: slab energies, bulk energy and facet work functions
df = surfaces.get_properties_dataframe(
    slab_energies={"Mg2Si_111_1": -0.62135394, "Mg2Si_110_1": -0.21012105},
    bulk_energy=-0.36261690,
    work_functions={"Mg2Si_111_1": 2.9941, "Mg2Si_110_1": 3.7378},
)

# Wulff construction from facet surface energies
wulff = surfaces.wulff_construct(surfaces.surface_energies_dict)

# Hydrogen adsorption sites and energetics
manager = AdsorptionManager(surfaces.surfaces)
manager.H_adsorption_analysis(adsorbate="H", output_dir="H_adsorption")
df_sites = manager.get_site_properties_dataframe()
```

### 3. Polarization curves and area-ratio optimization

```python
from GalvCalc.polarization import (
    ElectrodeParameters, Composition, plot_single_polarization,
    create_mg_based_compositions, plot_comparison_polarization,
)
from GalvCalc.polarization.area_ratio import (
    AreaRatioAnalyzer, create_example_parameters,
)

# Fe-based system
anode = ElectrodeParameters(4.1e-8, -0.44, 0.5, name="Fe/Fe2+")
cathode = ElectrodeParameters(7.9e-8, -0.059, 0.5, name="H+/H2")
comp = Composition("Fe", anode=anode, cathodes=[cathode], area_ratios=[1, 1])
fig = plot_single_polarization(comp, reference_electrode="SHE")

# Mg-based system: pure Mg matrix + nine intermetallic second phases
comps = create_mg_based_compositions()
fig = plot_comparison_polarization(comps, reference_electrode="SCE")

# Anode/cathode area-ratio optimization
params = create_example_parameters()
fig, optimal_ratio, i_max = AreaRatioAnalyzer().plot_area_ratio_analysis(params)
print(f"optimal anode area ratio = {optimal_ratio:.2%}")
```

The second-phase cathodic exchange currents used by
`create_mg_based_compositions()` are DFT-calibrated for nine Mg
intermetallics; supply your own values to study other phases.

### 4. Alloy-content scan

```python
import numpy as np
from GalvCalc.polarization import (
    mg3nd_vol_fraction, mg3nd_kinetics, scan_corrosion_vs_content,
)

nd_wt = np.linspace(0, 6, 25)  # Nd content (wt%)
df_scan = scan_corrosion_vs_content(
    nd_wt,
    params_fn=lambda w: mg3nd_kinetics(w, mg3nd_vol_fraction(w)),
    domain=(0.0, 1.0),
    n_ratios=51,
    n_potentials=1500,
)
df_scan[["content", "max_log10_i_corr", "E_corr_at_max"]].round(4)
```

### 5. ML prediction

```python
from GalvCalc.predictor import predict_cgcnn, predict
from pymatgen.core import Structure

# Surface energy + work function from a folder of CIF files
res = predict_cgcnn(cifpath="surfaces_output", task="regression")

# Hydrogen adsorption free energy (eV) for pymatgen structures
energies = predict([Structure.from_file("H1.vasp")])
```

### 6. Substitutional doping of anode surfaces

```python
from GalvCalc.core.structures import Bulk, Surface
from GalvCalc.anode import SurfaceDopingManager

bulk = Bulk.from_file("Mg.poscar")
surface = Surface.from_bulk(bulk, miller_index=(0, 0, 1))
manager = SurfaceDopingManager(surface, "Mg_001")

# Substitute surface Mg by Zn, Al and Y (second layer by default)
manager.batch_dope(["Zn", "Al", "Y"])
manager.set_property("Mg_001", "work_function", 3.72)
manager.set_property("Mg_001", "vacancy_energy", 0.84)
manager.set_property("Mg_001_Zn", "work_function", 3.75)
manager.set_property("Mg_001_Zn", "vacancy_energy", 0.79)

df = manager.calculate_electrochemical_properties(E00=-2.37, ia00=1e-5)
print(df[["surface_name", "E0_calculated", "ia0_calculated", "dopant_element"]])
```

Doped structures are exported as POSCAR files into `doped_surfaces/`; the full
doping record (host/dopant, site, layer, symmetry multiplicity) is available
from `manager.doping_info`.

---

## Example notebooks

Runnable notebooks are shipped under `GalvCalc/examples/`:

- `galvcalc_demo.ipynb` — end-to-end walkthrough covering equilibrium
  potentials, slab generation, adsorption-site schematics, Wulff construction,
  facet-weighted exchange currents, anode doping, polarization curves,
  area-ratio optimization and alloy-content scans.

Example structures (`POSCAR`, `*.poscar`, `*.vasp`, `surface/*.vasp`) and the
Mg<sub>2</sub>Si DFT adsorption dataset (`adsorption_analysis.csv`) are bundled in the
same folder.

---

## Command line

```bash
galvcalc --version     # GalvCalc 1.0.0
galvcalc --modules     # list available submodules
```

## License

Distributed under the [MIT License](LICENSE).

## Contact

Gaoning Shi — gaoning_shi@sjtu.edu.cn
