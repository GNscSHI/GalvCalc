"""Tests for the core structures and equilibrium-potential engine."""

from pathlib import Path

import numpy as np
import pytest

from GalvCalc.core.structures import Bulk, Surface

EXAMPLES = Path(__file__).resolve().parents[1] / "GalvCalc" / "examples"


def test_bulk_from_file():
    bulk = Bulk.from_file(EXAMPLES / "POSCAR")
    props = bulk.basic_properties()
    assert "formula" in props
    assert "volume" in props
    assert props["volume"] > 0


def test_bulk_as_structure():
    bulk = Bulk.from_file(EXAMPLES / "POSCAR")
    struct = bulk.as_structure()
    assert len(struct) == len(bulk)


def test_equilibrium_potential_mg():
    Ee = Bulk.get_equilibrium_potential(ions="Mg[2+]", ion_numbers=[1], energy_formation=0)
    # Mg/Mg2+ standard potential is about -2.37 V vs SHE
    assert -2.5 < Ee < -2.2


def test_equilibrium_potential_list_input():
    Ee = Bulk.get_equilibrium_potential(
        ions=["Mg[2+]", "Zn[2+]"], ion_numbers=[1, 2], energy_formation=-0.24
    )
    assert np.isfinite(Ee)


def test_equilibrium_potential_requires_ion_numbers():
    with pytest.raises(ValueError, match="ion_numbers"):
        Bulk.get_equilibrium_potential(ions="Mg[2+]")


def test_equilibrium_potential_unknown_ion():
    with pytest.raises(ValueError, match="not found"):
        Bulk.get_equilibrium_potential(ions="Xx[99+]", ion_numbers=[1])


def test_surface_from_bulk():
    bulk = Bulk.from_file(EXAMPLES / "Mg.poscar")
    surface = Surface.from_bulk(
        bulk_structure=bulk,
        miller_index=(0, 0, 1),
        min_slab_size=8.0,
        min_vacuum_size=12.0,
        center_slab=True,
    )
    assert surface.surface_area > 0
    assert len(surface) > 0
    assert surface.miller_index == (0, 0, 1)


def test_surface_from_slab():
    from pymatgen.core.surface import SlabGenerator

    bulk = Bulk.from_file(EXAMPLES / "Mg.poscar")
    gen = SlabGenerator(bulk, (1, 0, 0), min_slab_size=8.0, min_vacuum_size=12.0)
    slab = gen.get_slabs()[0]
    surface = Surface.from_slab(slab, bulk_structure=bulk)
    assert surface.bulk_structure is not None
