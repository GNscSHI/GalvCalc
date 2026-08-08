"""Tests for the polarization-curve, area-ratio and content-scan modules."""

import numpy as np
import pytest

from GalvCalc.polarization import (
    Composition,
    ElectrodeParameters,
    PolarizationCurvePlotter,
    create_mg_based_compositions,
    corrosion_vs_anode_ratio,
    max_corrosion_in_domain,
    mg3nd_vol_fraction,
    plot_single_polarization,
    scan_corrosion_vs_content,
    wt_to_vol_fraction,
)
from GalvCalc.polarization.area_ratio import (
    AreaRatioAnalyzer,
    create_example_parameters,
)


def _fe_composition():
    anode = ElectrodeParameters(4.1e-8, -0.44, 0.5, name="Fe/Fe2+")
    cathode = ElectrodeParameters(7.9e-8, -0.059, 0.5, name="H+/H2")
    return Composition("Fe", anode=anode, cathodes=[cathode], area_ratios=[1, 1])


def test_single_polarization_returns_figure():
    fig = plot_single_polarization(_fe_composition(), reference_electrode="SHE")
    assert fig is not None


def test_corrosion_point_in_range():
    plotter = PolarizationCurvePlotter(reference_electrode="SHE")
    comp = _fe_composition()
    U = np.linspace(-3, 1, 500)
    E_corr, I_corr = plotter._calculate_corrosion_point(comp, U)
    assert -3 <= E_corr <= 1
    assert I_corr > 0


def test_butler_volmer_anode_matches_calibrated_form():
    """The Fe demo form uses n=1, the (alpha+1) exponent and a factor of 2."""
    plotter = PolarizationCurvePlotter(reference_electrode="SHE")
    anode = ElectrodeParameters(1e-8, -0.44, 0.5, name="Fe")
    potential = np.array([-0.5, -0.4, -0.3])

    current = plotter._butler_volmer_anode(potential, anode, 0.6)

    alpha = anode.alpha
    n = 1
    f = plotter.F
    r = plotter.R
    t = plotter.temperature
    eta = potential - anode.equilibrium_potential
    expected = 2 * (0.6 * anode.exchange_current *
                    (np.exp((alpha + 1) * n * f * eta / (r * t)) -
                     np.exp(-(1 - alpha) * n * f * eta / (r * t))))
    assert np.allclose(current, expected)


def test_butler_volmer_cathode_matches_calibrated_form():
    """The Fe demo form uses n=1 and a factor of 2."""
    plotter = PolarizationCurvePlotter(reference_electrode="SHE")
    cathode = ElectrodeParameters(1e-8, -0.059, 0.5, name="HER")
    potential = np.array([-0.5, -0.3, -0.1])

    current = plotter._butler_volmer_cathode(potential, cathode, 0.4)

    alpha = cathode.alpha
    n = 1
    f = plotter.F
    r = plotter.R
    t = plotter.temperature
    eta = potential - cathode.equilibrium_potential
    expected = 2 * (0.4 * cathode.exchange_current *
                    (np.exp(alpha * n * f * eta / (r * t)) -
                     np.exp(-(1 - alpha) * n * f * eta / (r * t))))
    assert np.allclose(current, expected)


def test_butler_volmer_mg_form():
    """The Mg form uses n=2 anode with exponent alpha*n and no prefactor."""
    plotter = PolarizationCurvePlotter(reference_electrode="SHE")
    anode = ElectrodeParameters(1e-22, -2.37, 0.55, name="Mg", kinetic_form="mg")
    cathode = ElectrodeParameters(1e-8, -0.61, 0.77, name="HER", kinetic_form="mg")
    potential = np.array([-2.0, -1.6, -1.2])

    current_a = plotter._butler_volmer_anode(potential, anode, 0.158)

    alpha = anode.alpha
    f = plotter.F
    r = plotter.R
    t = plotter.temperature
    eta = potential - anode.equilibrium_potential
    expected_a = (0.158 * anode.exchange_current *
                  (np.exp(alpha * 2 * f * eta / (r * t)) -
                   np.exp(-(1 - alpha) * 2 * f * eta / (r * t))))
    assert np.allclose(current_a, expected_a)

    current_c = plotter._butler_volmer_cathode(potential, cathode, 0.842)
    eta_c = potential - cathode.equilibrium_potential
    expected_c = (0.842 * cathode.exchange_current *
                  (np.exp(0.77 * f * eta_c / (r * t)) -
                   np.exp(-(1 - 0.77) * f * eta_c / (r * t))))
    assert np.allclose(current_c, expected_c)


def test_create_mg_based_compositions():
    comps = create_mg_based_compositions()
    assert len(comps) >= 3
    for comp in comps:
        assert len(comp.area_ratios) == 1 + len(comp.cathodes)
        assert np.isclose(sum(comp.area_ratios), 1.0, atol=1e-6)

    # Pure Mg uses the corrected anode kinetics and area ratios.
    pure_mg = comps[0]
    assert pure_mg.anode.exchange_current == pytest.approx(10 ** -22.2)
    assert pure_mg.anode.alpha == pytest.approx(0.55)
    assert pure_mg.anode.kinetic_form == "mg"
    assert all(cath.kinetic_form == "mg" for cath in pure_mg.cathodes)
    assert pure_mg.area_ratios == pytest.approx([0.158, 0.842])


def test_multi_anode_composition_helper():
    anode = ElectrodeParameters(1e-10, -2.37, 0.36, name="Mg")
    cathode = ElectrodeParameters(1e-8, -0.413, 0.5, name="HER")
    comp = PolarizationCurvePlotter.create_multi_anode_composition(
        name="Mg-Zn",
        anodes=[anode, anode],
        cathodes=[cathode],
        area_ratios=[0.5, 0.3, 0.2],
    )
    assert comp.name == "Mg-Zn"
    assert comp.additional_anodes


def test_area_ratio_analysis():
    params = create_example_parameters()
    analyzer = AreaRatioAnalyzer(reference_electrode="SCE")
    ratios, currents, potentials = analyzer.analyze_area_ratio(
        params, n_ratios=10, n_potentials=200
    )
    assert len(ratios) == 10
    assert np.all(currents > 0)
    optimal, i_max = analyzer.find_optimal_area_ratio(ratios, currents)
    assert 0 < optimal <= 1
    assert i_max == pytest.approx(np.max(currents))


def test_example_parameters_use_corrected_mg_kinetics():
    params = create_example_parameters()
    assert params.anode_exchange_current == pytest.approx(10 ** -22.2)
    assert params.anode_alpha == pytest.approx(0.55)
    assert params.cathode_exchange_currents[0] == pytest.approx(10 ** -8.078243758202708)


def test_wt_to_vol_fraction():
    # 0 wt% gives no phase; a small content gives a small volume fraction.
    assert wt_to_vol_fraction(0.0, "Mg3Nd", rho_matrix=1.8, rho_phase=3.58) == 0.0
    vf = mg3nd_vol_fraction(1.0)
    assert 0.005 < vf < 0.01


def test_corrosion_vs_anode_ratio():
    result = corrosion_vs_anode_ratio(
        anode_i0=3e-23,
        anode_equilibrium_potential=-2.37,
        cathode_i0s=[10 ** -8.1, 10 ** -8.732],
        cathode_equilibrium_potential=-0.61,
        area_fractions=[0.99, 0.01],
        n_ratios=20,
        n_potentials=500,
    )
    assert len(result.anode_ratios) == 20
    # The corrosion current vanishes at ra=0 (no anode present).
    assert np.all(result.corrosion_currents >= 0)
    assert np.all(result.corrosion_potentials < 0)
    ra, i_max, e_corr = max_corrosion_in_domain(result, (0.0, 1.0))
    assert 0 <= ra <= 1
    assert i_max > 0


def test_scan_corrosion_vs_content():
    import numpy as np

    levels = np.linspace(0, 4, 5)
    df = scan_corrosion_vs_content(
        levels,
        params_fn=lambda w: {
            "anode_i0": 6e-23 if w == 0 else 3e-23,
            "anode_equilibrium_potential": -2.37,
            "cathode_i0s": [10 ** -8.1, 10 ** -8.732],
            "cathode_equilibrium_potential": -0.61,
            "area_fractions": [1.0 - mg3nd_vol_fraction(w), mg3nd_vol_fraction(w)],
            "vol_fraction": mg3nd_vol_fraction(w),
        },
        domain=(0.0, 1.0),
        n_ratios=20,
        n_potentials=500,
    )
    assert len(df) == 5
    assert np.isfinite(df["max_log10_i_corr"]).all()
    assert df["Ra_at_max"].between(0, 1).all()
