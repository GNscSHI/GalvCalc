<p align="center">
  <img src="GalvCalc/assets/logo.tif" alt="GalvCalc" width="400"/>
</p>

<p align="center">
  <em>From atomic-scale descriptors to macroscopic corrosion polarization curves</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/GalvCalc/"><img alt="PyPI" src="https://img.shields.io/pypi/v/GalvCalc?color=2b6cb0&label=PyPI"></a>
  <a href="https://pypi.org/project/GalvCalc/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/GalvCalc?color=2b6cb0"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/GNscSHI/GalvCalc?color=2b6cb0"></a>
</p>

**GalvCalc** models the micro-galvanic corrosion of alloys with coupled
anodic dissolution and hydrogen-evolution kinetics. Starting from a bulk
crystal structure, it builds the surface models, estimates the
electrochemical descriptors (equilibrium potentials, work functions, surface
energies and hydrogen adsorption energies), and assembles multi-phase
polarization curves — with built-in anode/cathode area-ratio optimization and
alloy-content scans of the corrosion current and potential.

The package is developed alongside the manuscript *"GalvCalc: A Framework for
Modeling Micro-galvanic Corrosion of Alloys with Coupled Anodic Dissolution
and Hydrogen Evolution Kinetics"*.

---

## Features

| Module | Purpose |
| --- | --- |
| `GalvCalc.core` | `Bulk` / `Surface` structure classes, Nernst-equation equilibrium potentials (bundled aqueous-ion thermodynamic data), Wulff-aware plotting helpers |
| `GalvCalc.cathode` | Surface-property manager (surface energies, work functions, Wulff shapes), hydrogen-adsorption analysis, facet-weighted exchange-current estimation |
| `GalvCalc.anode` | Substitutional surface doping of anode materials and electrochemical descriptor tables |
| `GalvCalc.polarization` | Butler–Volmer polarization curves, multi-anode/multi-cathode plots, area-ratio optimization, alloy-content scans |
| `GalvCalc.predictor` | ML predictors (CGCNN / TabPFN) for surface properties and hydrogen adsorption energies |

**Highlights**

- Full pipeline: bulk crystal → slab generation → Wulff shape →
  facet-weighted descriptors → corrosion polarization curves.
- Nernst-equation equilibrium potentials for arbitrary ion combinations (Mg,
  Fe, Zn, Al, ...), backed by a bundled database of aqueous-ion formation
  free energies.
- Calibrated Butler–Volmer kinetics for both Mg- and Fe-based systems.
- DFT-grounded second-phase exchange currents for nine common Mg-alloy
  intermetallics (Mg<sub>17</sub>Al<sub>12</sub>, Mg<sub>2</sub>Al<sub>3</sub>, MgZn<sub>2</sub>, LaMg<sub>12</sub>, CaMg<sub>2</sub>, Y<sub>5</sub>Mg<sub>24</sub>, NdMg<sub>3</sub>, Mg<sub>2</sub>Si, CeMg<sub>12</sub>).
- CGCNN prediction of surface energies and work functions, and
  TabPFN-based prediction of hydrogen adsorption energies.

---

## Installation

```bash
pip install GalvCalc
```

Core dependencies are installed by default. The `ml` extra adds the
pre-trained predictors (CGCNN / TabPFN), and the `defects` extra (Python ≥
3.10) enables the substitution-based doping workflow in `GalvCalc.anode`.

**Requirements:** Python >= 3.9; `numpy`, `scipy`, `pandas`, `matplotlib`,
`pymatgen`, `PyYAML`, `sympy`, `tqdm`, `joblib`, `openpyxl`. The `ml` extra
adds `torch`, `scikit-learn`, `tabpfn`.

---

## Quick start

All structure files used below (`Mg2Si.poscar`, `Mg.poscar`, ...)
and the DFT adsorption dataset (`adsorption_analysis.csv`) ship with the
package under `GalvCalc/examples/`, so the snippets run fully offline from
that folder (as in the bundled notebook).

### 1. Equilibrium potential

```python
from GalvCalc.core.structures import Bulk

# Single-ion dissolution: Mg -> Mg[2+] + 2 e-
Ee = Bulk.get_equilibrium_potential(
    ions="Mg[2+]",
    ion_numbers=[1],
    energy_formation=0.0,
)
print(f"E_eq = {Ee:.3f} V vs. SHE")

# Compound dissolution: MgZn2 -> Mg[2+] + 2 Zn[2+] + 6 e-
Ee2 = Bulk.get_equilibrium_potential(
    ions="Mg[2+], Zn[2+]",
    ion_numbers=[1, 2],
    energy_formation=-0.24,
)
```

### 2. Surfaces and Wulff construction

```python
from GalvCalc.core.structures import Bulk
from GalvCalc.cathode.surfaces import SurfaceProperties

bulk = Bulk.from_file("Mg2Si.poscar")
surfaces = SurfaceProperties.from_bulk_structure(bulk, max_index=1, min_slab_size=10)

# DFT inputs: slab energies, bulk energy and facet work functions
df = surfaces.get_properties_dataframe(
    slab_energies={"Mg2Si_111_1": -0.62135394, "Mg2Si_110_1": -0.21012105},
    bulk_energy=-0.36261690,
    work_functions={"Mg2Si_111_1": 2.9941, "Mg2Si_110_1": 3.7378},
)
print(df)

# Wulff shape from the facet surface energies
wulff = surfaces.wulff_construct(surfaces.surface_energies_dict)
```

### 3. Hydrogen adsorption and exchange-current estimation

The demo DFT adsorption dataset is bundled as
`adsorption_analysis.csv`; the facet-weighted hydrogen-evolution exchange
current follows from the adsorption free energy through the BEP-type relation
of the manuscript:

```python
import pandas as pd
from GalvCalc.cathode import ic0

df_ads = pd.read_csv("adsorption_analysis.csv")
mg2si = df_ads[df_ads["formula"] == "Mg2Si"]
print(mg2si[["miller_index", "termination", "ads_position", "Eads", "workfunction"]].head())

# Mg2Si (110) H6 site: G_ads = 0.028 eV, work function = 3.7378 eV
i0 = ic0(delta_G=0.028, wf=3.7378)
print(f"ic0 = {i0:.3e} A/cm^2")
```

`AdsorptionManager.H_adsorption_analysis` runs the full model-driven workflow
instead: it locates the adsorption sites on every surface, builds the
H-adsorbed slabs, predicts adsorption energies with the bundled TabPFN model
and exports POSCAR files plus CSV/JSON summaries. This path needs the `ml`
extra.

### 4. Polarization curves and area-ratio optimization

```python
from GalvCalc.polarization import (
    ElectrodeParameters, Composition, plot_single_polarization,
    create_mg_based_compositions, plot_comparison_polarization,
)
from GalvCalc.polarization.area_ratio import (
    AreaRatioAnalyzer, create_example_parameters,
)

# Fe-based system: n = 1, exponent (alpha_a + 1) * n, prefactor 2
anode = ElectrodeParameters(4.1e-8, -0.44, 0.5, name="Fe/Fe2+", kinetic_form="fe")
cathode = ElectrodeParameters(7.9e-8, -0.059, 0.5, name="H+/H2", kinetic_form="fe")
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

### 5. Alloy-content scan

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

### 6. ML prediction

```python
from GalvCalc.predictor import predict_cgcnn, predict
from pymatgen.core import Structure

# Surface energy + work function from a folder of CIF files
res = predict_cgcnn(cifpath="surfaces_output", task="regression")

# Hydrogen adsorption energy (eV) for your own H-adsorbed structures
energies = predict([Structure.from_file("H1.vasp")])
```

Both predictors need the `ml` extra; every other workflow above runs without
it.

### 7. Substitutional doping of anode surfaces

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
from `manager.doping_info`. This workflow builds on pymatgen's defect
framework and needs the `defects` extra (Python ≥ 3.10).

---

## Example notebooks

Runnable notebooks are shipped under `GalvCalc/examples/`:

- `galvcalc_demo.ipynb` — a step-by-step walkthrough covering equilibrium
  potentials, slab generation, adsorption-site schematics, Wulff construction,
  facet-weighted exchange currents, anode doping, polarization curves,
  area-ratio optimization and alloy-content scans.

Example structures (`*.poscar`, `*.vasp`, `surface/*.vasp`) and the
demo DFT adsorption dataset (`adsorption_analysis.csv`) are bundled in the
same folder.

---

## License

Distributed under the [MIT License](LICENSE).

## Contact

Gaoning Shi — gaoning_shi@sjtu.edu.cn