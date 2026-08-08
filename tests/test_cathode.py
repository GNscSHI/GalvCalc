"""Tests for the cathode module (surface properties and i_c0 estimation)."""

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure

from GalvCalc.cathode import facet_dependant_property, ic0, ic0_mg


def _fcc_bulk() -> Structure:
    """Small FCC structure used as an offline Wulff-shape bulk."""
    lattice = Lattice.cubic(4.05)
    return Structure(
        lattice,
        ["Al"] * 4,
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]],
    )


def _facet_records(compound: str = "Al", mp_id: str = "mp-134"):
    return [
        {"Formula": compound, "Facet": "100_1", "Surface_energy": 0.6,
         "Work_function": 4.2, "Material_id": mp_id},
        {"Formula": compound, "Facet": "111_1", "Surface_energy": 0.4,
         "Work_function": 4.1, "Material_id": mp_id},
    ]


def _adsorption_dict(compound: str = "Al"):
    return {
        compound: {
            "100": {"1": {"s1": {"Gads": -0.3}, "s2": {"Gads": -0.2}}},
            "111": {"1": {"s1": {"Gads": -0.25}}},
        }
    }


def test_ic0_basic():
    i0 = ic0(delta_G=-0.3)
    assert i0 > 0
    assert np.isfinite(i0)


def test_ic0_work_function_shift():
    i0_ref = ic0(delta_G=-0.3, wf=3.614)
    i0_high_wf = ic0(delta_G=-0.3, wf=4.2)
    assert i0_high_wf > 0
    assert i0_high_wf < i0_ref


def test_ic0_requires_delta_g():
    with pytest.raises(ValueError):
        ic0(delta_G=None)


def test_ic0_mg_basic():
    i0, delta_G = ic0_mg(delta_H=-0.3)
    assert i0 > 0
    assert delta_G == pytest.approx(-0.11)  # -0.3 + correction(0.19)


def test_ic0_mg_default_work_function():
    i0, _ = ic0_mg(delta_H=0.0)
    assert i0 > 0


def test_ic0_mg_with_work_function_shift():
    i0_high_wf, _ = ic0_mg(delta_H=-0.3, wf=4.2)
    i0_ref, _ = ic0_mg(delta_H=-0.3, wf=3.614)
    assert i0_high_wf > 0
    assert np.isfinite(i0_high_wf)
    assert i0_high_wf < i0_ref


def test_facet_dependant_weighted_ic_new_format():
    fd = facet_dependant_property()
    val = fd.weighted_ic(_facet_records(), _adsorption_dict(), _fcc_bulk(), ads_criteria="lowest")
    assert val > 0
    assert np.isfinite(val)


def test_facet_dependant_weighted_wf():
    fd = facet_dependant_property()
    wwf, wfs = fd.weighted_wf(_facet_records(), _fcc_bulk())
    assert 4.0 < wwf < 4.3
    assert len(wfs) == 2


def test_facet_dependant_legacy_dataframe():
    df = pd.DataFrame([
        {"full_name": "Al_100_1", "miller_index": (1, 0, 0), "termination_index": 1,
         "formula": "Al", "surface_energy_J_m2": 0.6, "work_function_eV": 4.2},
        {"full_name": "Al_111_1", "miller_index": (1, 1, 1), "termination_index": 1,
         "formula": "Al", "surface_energy_J_m2": 0.4, "work_function_eV": 4.1},
    ])
    formatted = {
        "Al": {
            "100": {"1": {"s1": {"H1.vasp": {"POSCAR": "", "Eads": -0.3}}}},
            "111": {"1": {"s1": {"H1.vasp": {"POSCAR": "", "Eads": -0.25}}}},
        }
    }
    fd = facet_dependant_property()
    val = fd.weighted_ic(df, formatted, _fcc_bulk(), correction=0.19, ads_criteria="near_zero")
    assert val > 0
    assert np.isfinite(val)


def test_facet_dependant_unknown_criteria():
    fd = facet_dependant_property()
    with pytest.raises(ValueError):
        fd.weighted_ic(_facet_records(), _adsorption_dict(), _fcc_bulk(), ads_criteria="bogus")
