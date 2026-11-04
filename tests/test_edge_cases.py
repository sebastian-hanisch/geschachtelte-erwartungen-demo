"""Randfälle: lockere und enge Toleranz, konstante Funktion, kleinste Stichproben, Eingabeprüfung der Spielwiese, unbekannte Namen, äußerste Signalstärken."""

import math

import numpy as np
import pytest

import ger_algorithm as A
import ger_constants as C
import ger_evaluation as ev
from ger_evaluation import Settings
from ger_scenario import Szenario, check_szenario, standard_szenario


def test_sehr_lockere_toleranz_verlangt_trotzdem_mindestens_ebene_eins():
    assert ev.level_for(ev.analyse(Settings(lmax=6)), 10.0) == 1


def test_sehr_enge_toleranz_bleibt_endlich():
    a = ev.analyse(Settings(lmax=6))
    assert 1 <= ev.level_for(a, 1e-9) <= 40 and math.isfinite(ev.cost_row(a, 1e-9)["ml"])


def test_konstante_funktion_gibt_keine_division_durch_null_und_ein_ehrliches_verdikt():
    s = Settings(ausgabe="sprung", lmax=6)
    a = ev.analyse(s, Szenario(0.6, 0.1, 2.5, 1.0))              # die innere Ausgabe ist nie positiv
    assert ev.degenerate(a) and not np.isfinite(a.beta) and a.c == 0.0
    row = ev.cost_row(a, 0.01)
    assert row["nested"] == 0.0 and row["ml"] == 0.0 and row["ratio"] == float("inf")
    for kind, text in (ev.verdict_main(a, s), ev.verdict_levels(a), ev.verdict_cost(a)):
        assert kind == "info" and "nichts zu schätzen" in text and "nan" not in text and "inf" not in text


def test_giles_verfahren_bricht_bei_null_varianz_ab():
    est, cost, L, Nl = A.mlmc("sprung", 0.01, 3, 0.6, 0.1, 2.5, 1e-12, n0=100)
    assert est == 0.0 and L == 2 and cost > 0


def test_eine_aeussere_ziehung_hat_varianz_null_ohne_fehler():
    m, v, vp, kurt, nz, mp = A.level_stats("knick", 3, 1, 5, 0.6, 0.5, 1.3, 0.1655)
    assert v == 0.0 and vp == 0.0 and np.isnan(kurt)


def test_rate_mit_zu_wenigen_punkten_ist_nicht_bestimmbar():
    assert np.isnan(ev.fit_rate([1.0, 0.5, 0.25])) and np.isnan(ev.fit_rate([1.0, 0.5, 0.25, 0.1]))
    assert ev.fit_rate([1.0, 0.5, 0.25, 0.125, 0.0625]) == pytest.approx(1.0)
    assert np.isnan(ev.fit_rate([1.0, 0.5, 0.25, 0.0, 0.0, 0.0]))


def test_verzerrungskonstante_ohne_passende_ebenen_ist_nan():
    assert np.isnan(ev.bias_const([{"l": 0, "mean": 1.0}, {"l": 1, "mean": 0.5}]))


def test_unbekannte_namen_werden_abgelehnt():
    G = [[0.3, 0.1]]
    with pytest.raises(ValueError):
        A.level_pair("knick", 1, 1, 0, 0.6, 0.5, 1.3, 0.15, "unbekannt", values=G)
    with pytest.raises(ValueError):
        A.f_eval("unbekannt", np.ones(2), 0.1)


def test_ein_term_schaetzer_mit_nur_ebene_null():
    mean, var, cost = A.single_term("knick", 1.0, 0, 2000, 1, 0.6, 0.5, 1.3, A.k_default(0.5, 1.3))
    assert cost == 1.0 and mean == pytest.approx(A.e_p0("knick", 0.5, 1.3, A.k_default(0.5, 1.3)), abs=0.01)


def test_spielwiese_eingabepruefung():
    assert check_szenario(0.6, 0.5, 1.3, 1.0) == standard_szenario(0.6)
    for args, frag in (((None, 0.5, 1.3, 1.0), "keine Zahl"), ((float("nan"), 0.5, 1.3, 1.0), "keine Zahl"), ((0.6, 0.5, 1.3, float("inf")), "keine Zahl"), ((0.99, 0.5, 1.3, 1.0), "Signalstärke"),
                       ((0.6, 2.0, 1.3, 1.0), "Streuung"), ((0.6, 0.5, 9.0, 1.0), "Schwelle q"), ((0.6, 0.5, 1.3, 9.0), "Schwellenfaktor")):
        with pytest.raises(ValueError, match=frag):
            check_szenario(*args)


def test_eigenes_modell_traegt_seine_parameter_im_etikett():
    s = Settings(lmax=6)
    a = ev.analyse(s, Szenario(0.6, 0.4, 1.1, 2.0))
    assert "σ = 0,40" in a.label and "q = 1,10" in a.label and "k = 2,00·Mittel" in a.label and a.sigma == 0.4
    assert "σ = " not in ev.analyse(s).label


def test_zahlenformate():
    assert ev.de(1234.5, 1) == "1.234,5" and ev.de(-0.5, 1) == "−0,5"
    assert ev.ke(999.0) == "999" and ev.ke(123456.0) == "1,2e+05" and ev.ke(float("nan")) == "nicht bestimmbar"
    assert ev.eps_text(0.0025) == "0,0025" and ev.eps_text(0.01) == "0,01"
    assert ev.dn(float("inf")) == "unendlich" and ev.dn(float("nan")) == "nicht bestimmbar" and ev.prozent(float("nan")) == "nicht bestimmbar"


def test_kleinste_und_groesste_einstellung_und_aeusserste_signale_rechnen():
    for lmax in (C.LMAX_MIN, C.LMAX_MAX):
        for rho in (C.RHO_MIN, C.RHO_MAX):
            a = ev.analyse(Settings(lmax=lmax, rho=rho))
            assert all(math.isfinite(r["ratio"]) for r in ev.cost_table(a))
    for rho in (C.CUSTOM_RHO_MIN, C.CUSTOM_RHO_MAX):
        a = ev.analyse(Settings(lmax=6), Szenario(rho, 0.5, 1.3, 1.0))
        assert math.isfinite(a.truth) and math.isfinite(a.var_cond)


def test_extrapolation_jenseits_der_gemessenen_ebenen_faellt_hoechstens_ab():
    a = ev.analyse(Settings(lmax=6))
    assert ev.level_variance(a, 6) == a.levels[6]["var"] and 0 < ev.level_variance(a, 10) <= a.levels[6]["var"]
