"""Eigenschaften, die für jede Wahl gelten müssen (nicht nur an einem Zahlenwert): Raten, Kopplung, Verzerrung, Kosten, Reproduzierbarkeit."""

import math

import numpy as np
import pytest

import ger_algorithm as A
import ger_evaluation as ev
from ger_evaluation import Settings


def _a(kind="knick", **kw):
    return ev.analyse(Settings(ausgabe=kind, lmax=7, **kw))


def test_antithetische_rate_je_aeusserer_funktion():
    g, k, s = _a("glatt"), _a("knick"), _a("sprung")
    assert 1.7 < g.beta_by_variant["anti"] < 2.5
    assert 1.05 < k.beta_by_variant["anti"] < 1.8
    assert 0.2 < s.beta_by_variant["anti"] < 0.8


def test_einseitige_und_unabhaengige_kopplung_heben_die_rate_nicht():
    for kind, lo, hi in (("glatt", 0.9, 1.5), ("knick", 0.8, 1.25)):
        a = _a(kind)
        assert lo < a.beta_by_variant["one"] < hi and lo < a.beta_by_variant["indep"] < hi
        assert a.beta_by_variant["anti"] > a.beta_by_variant["one"] + 0.2


def test_antithetisch_hat_die_kleinste_varianz_auf_den_tiefen_ebenen():
    for kind in ("glatt", "knick", "sprung"):
        v = _a(kind).var_by_variant
        assert v["anti"][7] < v["one"][7] < v["indep"][7], kind


def test_verzerrung_halbiert_sich_je_ebene_und_beim_mittelwert_exakt():
    a = _a("glatt")
    for l in (2, 3):
        assert ev.bias_at(a, l) / ev.bias_at(a, l - 1) == pytest.approx(0.5, abs=0.1)
        assert ev.bias_at(a, l) == pytest.approx((a.e0 - a.truth) / 2**l, abs=8e-4)


def test_ebene_waechst_mit_enger_werdender_toleranz():
    a = _a()
    ls = [ev.level_for(a, e) for e in sorted(ev.C.EPS_GRID, reverse=True)]
    assert ls == sorted(ls) and ls[-1] > ls[0]


def test_kosten_wachsen_mit_enger_werdender_toleranz_und_bedingt_ist_am_billigsten():
    for kind in ("glatt", "knick", "sprung"):
        t = ev.cost_table(_a(kind))
        for lo, hi in zip(t[:-1], t[1:]):
            assert hi["nested"] > lo["nested"] and hi["ml"] > lo["ml"] > 0 and hi["cond"] > lo["cond"]
        assert all(r["cond"] < r["ml"] and r["cond"] < r["nested"] for r in t)


def test_multilevel_gewinnt_beim_knick_mit_enger_werdender_toleranz_und_beim_sprung_nicht():
    k, s = ev.cost_table(_a("knick")), ev.cost_table(_a("sprung"))
    assert k[-1]["ratio"] > 5 * k[0]["ratio"] / 2 and k[-1]["ratio"] > 8
    assert max(r["ratio"] for r in s) < 1.25


def test_kostenanteile_summieren_sich_zu_eins():
    for r in ev.cost_table(_a("glatt")):
        assert sum(r["shares"]) == pytest.approx(1.0)


def test_staerkeres_signal_gibt_mehr_gewinn_beim_mittelwert():
    eps = 0.001
    ratios = [ev.cost_row(ev.analyse(Settings(ausgabe="glatt", rho=r, lmax=7)), eps)["ratio"] for r in (0.3, 0.6, 0.9)]
    assert ratios == sorted(ratios) and ratios[2] > 5 * ratios[0]


def test_gleiche_einstellung_gleiche_zahlen():
    s = Settings(lmax=6)
    assert ev.analyse(s) == ev.analyse(s)
    assert ev.analyse(Settings(lmax=6, seed=77)).levels[3]["mean"] != ev.analyse(s).levels[3]["mean"]


def test_gelaufener_multilevel_schaetzer_trifft_die_toleranz():
    s = Settings(eps=0.005, lmax=7)
    a = ev.analyse(s)
    errs = np.array([A.mlmc("knick", s.eps, A.stream_seed(5, r), 0.6, 0.5, 1.3, a.k)[0] - a.truth for r in range(30)])
    assert math.sqrt((errs**2).mean()) < 1.3 * s.eps


def test_ein_term_schaetzer_ist_beim_knick_mit_s_eins_unverzerrt_und_hat_beschraenkte_varianz():
    a = _a("knick")
    mean, var, cost = A.single_term("knick", 1.0, 7, 40000, 9, 0.6, 0.5, 1.3, a.k)
    assert abs(mean - a.truth) < 4 * math.sqrt(var / 40000) + 5e-4
    assert var < 0.6 and cost == pytest.approx(4.0, abs=0.4)


def test_ein_term_varianz_waechst_beim_sprung_fuer_jedes_s():
    a = _a("sprung")
    for s in (1.0, 1.5, 2.0):
        v = [ev.single_term_theory(a, s, lm)[0] for lm in (3, 7)]
        assert v[1] > 3 * v[0], s
