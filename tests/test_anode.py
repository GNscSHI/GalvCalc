"""Tests for the anode doping manager (pymatgen defects + symmetry based)."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from GalvCalc.anode import SurfaceDopingManager
from GalvCalc.core.structures import Bulk, Surface

EXAMPLES = Path(__file__).resolve().parents[1] / "GalvCalc" / "examples"


@pytest.fixture(scope="module")
def mg_surface():
    bulk = Bulk.from_file(EXAMPLES / "Mg.poscar")
    return Surface.from_bulk(
        bulk_structure=bulk,
        miller_index=(0, 0, 1),
        min_slab_size=8.0,
        min_vacuum_size=12.0,
        center_slab=True,
    )


def _layer_of(surface, site_index):
    """Return the layer number (1 = topmost) of a site using z-clustering."""
    zs = np.array([site.coords[2] for site in surface])
    order = np.argsort(zs)
    groups = []
    for idx in order:
        if groups and zs[idx] - zs[groups[-1][-1]] <= 0.3:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    for layer_no, group in enumerate(reversed(groups), start=1):
        if site_index in group:
            return layer_no
    raise AssertionError("site not found")


def test_dope_second_layer_by_default(mg_surface):
    manager = SurfaceDopingManager(mg_surface, "Mg_001")
    doped = manager.dope_surface("Zn", save_to_file=False)

    assert doped.full_name == "Mg_001_Zn"
    assert len(doped) == len(mg_surface)
    assert "Zn" in doped.composition

    info = manager.doping_info["Mg_001_Zn"]
    assert info.layer == 2
    assert info.dopant_element == "Zn"
    assert info.multiplicity >= 1
    # The dopant must sit in the second layer, not the outermost one.
    assert _layer_of(doped, info.site_index) == 2
    assert doped[info.site_index].species_string == "Zn"


def test_dope_first_layer(mg_surface):
    manager = SurfaceDopingManager(mg_surface, "Mg_001")
    doped = manager.dope_surface("Zn", layer=1, save_to_file=False)
    info = manager.doping_info["Mg_001_Zn"]
    assert info.layer == 1
    assert _layer_of(doped, info.site_index) == 1


def test_dope_negative_layer_from_bottom(mg_surface):
    manager = SurfaceDopingManager(mg_surface, "Mg_001")
    doped = manager.dope_surface("Zn", layer=-2, save_to_file=False)
    info = manager.doping_info["Mg_001_Zn"]
    assert info.layer == -2
    # The dopant must sit in the second layer counted from the bottom.
    n_layers = max(manager._get_layer_indices(doped).keys())
    assert _layer_of(doped, info.site_index) == n_layers - 1


def test_layer_out_of_range_raises(mg_surface):
    manager = SurfaceDopingManager(mg_surface, "Mg_001")
    with pytest.raises(ValueError, match="out of range"):
        manager.dope_surface("Zn", layer=99, save_to_file=False)
    with pytest.raises(ValueError, match="non-zero"):
        manager.dope_surface("Zn", layer=0, save_to_file=False)


def test_batch_dope(mg_surface):
    manager = SurfaceDopingManager(mg_surface, "Mg_001")
    surfaces = manager.batch_dope(["Zn", "Al", "Y"], save_to_file=False)
    assert [s.full_name for s in surfaces] == ["Mg_001_Zn", "Mg_001_Al", "Mg_001_Y"]
    assert manager.get_summary()["doped_surfaces"] == 3


def test_doping_info_json_serializable(mg_surface):
    manager = SurfaceDopingManager(mg_surface, "Mg_001")
    manager.dope_surface("Zn", save_to_file=False)
    record = asdict(manager.doping_info["Mg_001_Zn"])
    payload = json.dumps(record)
    assert "Mg_001_Zn" not in payload  # sanity: serializes without error


def test_electrochemical_table_reports_layer(mg_surface):
    manager = SurfaceDopingManager(mg_surface, "Mg_001")
    manager.dope_surface("Zn", save_to_file=False)
    manager.set_property("Mg_001", "work_function", 3.72)
    manager.set_property("Mg_001", "vacancy_energy", 0.84)
    manager.set_property("Mg_001_Zn", "work_function", 3.75)
    manager.set_property("Mg_001_Zn", "vacancy_energy", 0.79)

    df = manager.calculate_electrochemical_properties(E00=-2.37, ia00=1e-5)
    row = df[df["surface_name"] == "Mg_001_Zn"].iloc[0]
    assert row["doping_layer"] == 2
    assert row["doping_depth"] > 1.0  # second layer is below the surface
    assert row["E0_calculated"] == pytest.approx(-2.34)
