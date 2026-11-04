"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Exakte Zahlen (Quadratur, geschlossene Formeln) mit engen Bändern. Gemessene Zahlen stammen aus festen Strömen (40 000 äußere Ziehungen je Ebene, Ebenen 0 bis 8, Strom 37) und sind reproduzierbar; sie stehen hier mit Bändern,
die das Rauschen (die Ebenenvarianz ist auf etwa 10 bis 40 % genau, beim Sprung wegen seltener Ereignisse schlechter) und kleine Plattformunterschiede abdecken. Verhältnisse werden mal/durch verglichen."""

import numpy as np
import pytest

import ger_algorithm as A
import ger_evaluation as ev
from ger_evaluation import Settings


def _band(wert, soll, mal=1.5):
    return soll / mal <= wert <= soll * mal


def _a(kind, **kw):
    return ev.analyse(Settings(ausgabe=kind, lmax=8, **kw))


# --- exakte Werte ------------------------------------------------------------------------------------------------------------------------------

def test_exakte_werte():
    k = A.k_default(0.5, 1.3)
    assert k == pytest.approx(0.16554, abs=5e-6)
    assert A.truth("glatt", 0.6, 0.5, 1.3, k) == pytest.approx(0.064501, abs=5e-7)
    assert A.truth("knick", 0.6, 0.5, 1.3, k) == pytest.approx(0.067545, abs=5e-7)
    assert A.truth("sprung", 0.6, 0.5, 1.3, k) == pytest.approx(0.339560, abs=5e-7)
    assert A.x_star(k, 0.6, 0.5, 1.3) == pytest.approx(0.4137, abs=5e-5)


def test_erwartung_der_ersten_ebene_und_verzerrungskonstante_beim_mittelwert():
    k = A.k_default(0.5, 1.3)
    assert A.e_p0("glatt", 0.5, 1.3, k) == pytest.approx(0.18836, abs=5e-6)
    assert A.e_p0("knick", 0.5, 1.3, k) == pytest.approx(0.12262, abs=5e-6)
    assert A.e_p0("sprung", 0.5, 1.3, k) == pytest.approx(0.22230, abs=5e-6)
    assert A.e_p0("glatt", 0.5, 1.3, k) - A.truth("glatt", 0.6, 0.5, 1.3, k) == pytest.approx(0.12386, abs=5e-6)


def test_varianz_der_bedingten_funktion():
    k = A.k_default(0.5, 1.3)
    for kind, soll in (("glatt", 0.03261), ("knick", 0.02423), ("sprung", 0.22426)):
        assert A.var_cond(kind, 0.6, 0.5, 1.3, k) == pytest.approx(soll, abs=5e-5)


# --- Raten ---------------------------------------------------------------------------------------------------------------------------------------

BETA = {("glatt", "anti"): (1.7, 2.5), ("knick", "anti"): (1.1, 1.8), ("sprung", "anti"): (0.2, 0.8),
        ("glatt", "one"): (0.9, 1.5), ("knick", "one"): (0.8, 1.25), ("sprung", "one"): (0.2, 0.8),
        ("glatt", "indep"): (0.9, 1.5), ("knick", "indep"): (0.8, 1.25), ("sprung", "indep"): (0.2, 0.8)}


@pytest.mark.parametrize("kind,variant", list(BETA))
def test_rate_der_ebenenvarianz(kind, variant):
    lo, hi = BETA[(kind, variant)]
    b = _a(kind).beta_by_variant[variant]
    assert lo < b < hi, (kind, variant, b)


def test_varianz_auf_ebene_8_je_kopplung():
    soll = {"glatt": (1.83e-6, 5.49e-4, 1.57e-3), "knick": (4.73e-6, 3.82e-4, 1.10e-3), "sprung": (1.62e-2, 3.19e-2, 5.61e-2)}
    for kind, (an, on, ind) in soll.items():
        v = _a(kind).var_by_variant
        assert _band(v["anti"][8], an, 2.5) and _band(v["one"][8], on, 2.0) and _band(v["indep"][8], ind, 2.0), kind
    g, k = _a("glatt").var_by_variant["anti"], _a("knick").var_by_variant["anti"]
    assert _band(g[8] / g[7], 0.26, 1.35) and _band(k[8] / k[7], 0.37, 1.35)


def test_verzerrung_beim_mittelwert_ist_exakt_c_durch_m():
    a = _a("glatt")
    for l in range(1, 7):
        assert ev.bias_at(a, l) == pytest.approx(0.12386 / 2**l, abs=8e-4)
    assert _band(a.c, 0.1239, 1.12)


# --- idealisierte Kosten ------------------------------------------------------------------------------------------------------------------------

RATIO = {"glatt": (0.42, 10.0), "knick": (1.45, 20.6), "sprung": (0.40, 0.82)}


@pytest.mark.parametrize("kind", list(RATIO))
def test_kostenverhaeltnis_geschachtelt_durch_multilevel(kind):
    t = ev.cost_table(_a(kind))
    assert _band(t[0]["ratio"], RATIO[kind][0], 1.6) and _band(t[-1]["ratio"], RATIO[kind][1], 1.6), (kind, t[0]["ratio"], t[-1]["ratio"])


def test_beim_sprung_ist_multilevel_bei_jeder_toleranz_teurer_und_beim_mittelwert_erst_ab_0_0025_billiger():
    s = ev.cost_table(_a("sprung"))
    assert all(r["ratio"] < 1.0 for r in s) and all(0.3 < r["ratio"] < 0.95 for r in s)
    g = ev.cost_table(_a("glatt"))
    assert g[0]["ratio"] < 1.0 and g[1]["ratio"] < 1.0 and g[3]["ratio"] > 1.0 and g[-1]["ratio"] > 5


def test_saettigung_und_wachstum():
    g, k, s = ev.cost_table(_a("glatt")), ev.cost_table(_a("knick")), ev.cost_table(_a("sprung"))
    assert _band(g[0]["eps2_ml"], 5.6, 1.3) and _band(g[-1]["eps2_ml"], 6.4, 1.3) and g[-1]["eps2_ml"] / g[0]["eps2_ml"] < 1.5
    assert _band(k[0]["eps2_ml"], 0.65, 1.4) and _band(k[-1]["eps2_ml"], 1.21, 1.4)
    assert s[-1]["eps2_ml"] / s[0]["eps2_ml"] > 10 and g[-1]["eps2_nested"] / g[0]["eps2_nested"] > 20


def test_bedingt_ist_um_das_vielfache_billiger_als_multilevel():
    for kind, (lo, hi) in (("glatt", (85.4, 98.1)), ("knick", (13.5, 25.1)), ("sprung", (39.5, 620.4))):
        t = ev.cost_table(_a(kind))
        assert _band(t[0]["ml_over_cond"], lo, 1.5) and _band(t[-1]["ml_over_cond"], hi, 1.5), kind


def test_kostenanteile_je_ebene():
    k = ev.cost_table(_a("knick"))[-2]["shares"]
    g = ev.cost_table(_a("glatt"))[-2]["shares"]
    s = ev.cost_table(_a("sprung"))[-2]["shares"]
    assert k[0] > 0.4 and 0.04 < min(k[1:]) and max(k[1:]) < 0.11
    assert g[0] > 0.4 and all(g[i] >= g[i + 1] for i in range(len(g) - 1))
    assert s[-1] > 0.18 and s[0] < 0.07 and all(s[i] <= s[i + 1] for i in range(2, len(s) - 1))


# --- gelaufene Schätzungen ------------------------------------------------------------------------------------------------------------------------

def test_gelaufene_schaetzung_mittelwert_und_knick_bei_enger_toleranz():
    for kind, soll in (("glatt", 1.08), ("knick", 1.85)):
        a = _a(kind)
        ex = ev.run_experiment(Settings(ausgabe=kind, lmax=8, eps=0.0025), a)
        assert _band(ex["ratio"], soll, 2.0) and ex["ml"]["rmse"] / 0.0025 < 1.3, kind
    ex = ev.run_experiment(Settings(ausgabe="knick", lmax=8, eps=0.01), _a("knick"))
    assert _band(ex["ratio"], 0.27, 2.0) and ex["ratio"] < 1.0


def test_gelaufene_schaetzung_beim_sprung_bleibt_teurer():
    ex = ev.run_experiment(Settings(ausgabe="sprung", lmax=8, eps=0.01), _a("sprung"))
    assert _band(ex["ratio"], 0.26, 2.0) and ex["ratio"] < 1.0 and ex["ml"]["rmse"] / 0.01 < 1.3


def test_gelaufener_fehler_ist_im_mittel_nicht_ueber_dem_verzerrungsbudget():
    a = _a("knick")
    errs = np.array([A.mlmc("knick", 0.005, A.stream_seed(5, r), 0.6, 0.5, 1.3, a.k)[0] - a.truth for r in range(30)])
    assert np.sqrt((errs**2).mean()) / 0.005 < 1.3 and abs(errs.mean()) / 0.005 < 0.9


# --- Signal und Schwelle ---------------------------------------------------------------------------------------------------------------------------

def test_signalstaerke_und_schwelle():
    s = Settings(ausgabe="knick", lmax=8)
    ex = ev.signal_experiment(s, _a("knick"))
    r = {x["rho"]: x["ratios"] for x in ex["rho_rows"]}
    assert _band(r[0.3][0], 1.25, 1.6) and _band(r[0.6][0], 6.22, 1.6) and _band(r[0.9][0], 12.0, 1.6) and r[0.3][0] < r[0.6][0] < r[0.9][0]
    kk = {x["factor"]: x["ratios"] for x in ex["k_rows"]}
    assert _band(kk[0.5][0], 11.0, 1.6) and _band(kk[1.0][0], 6.22, 1.6) and _band(kk[2.0][0], 2.76, 1.6) and kk[0.5][0] > kk[1.0][0] > kk[2.0][0]
    g = ev.signal_experiment(Settings(ausgabe="glatt", lmax=8), _a("glatt"))
    rg = {x["rho"]: x["ratios"] for x in g["rho_rows"]}
    assert _band(rg[0.3][0], 0.13, 2.0) and _band(rg[0.9][0], 30.7, 1.6) and rg[0.3][0] < 1.0
    assert len({round(x["ratios"][0], 6) for x in g["k_rows"]}) == 1                     # der Mittelwert hängt nicht von der Schwelle ab
    sp = ev.signal_experiment(Settings(ausgabe="sprung", lmax=8), _a("sprung"))
    assert all(x["ratios"][0] < 1.0 for x in sp["rho_rows"])


def test_ohne_antithetik_und_bei_schwachem_signal_rechnet_multilevel_nicht():
    e = _a("glatt", kopplung="one")
    assert _band(e.beta, 1.18, 1.3) and ev.cost_row(e, 0.0025)["ratio"] < 0.4 and ev.cost_row(e, 0.000625)["ratio"] < 1.0
    sw = ev.analyse(Settings(ausgabe="glatt", rho=0.3, lmax=8))
    assert _band(sw.c, 0.154, 1.3) and ev.cost_row(sw, 0.0025)["ratio"] < 0.2
    st = ev.analyse(Settings(ausgabe="glatt", rho=0.9, lmax=8))
    assert _band(ev.cost_row(st, 0.0025)["ratio"], 8.4, 1.6)


# --- Entzerrter Schätzer ----------------------------------------------------------------------------------------------------------------------------

def _curve(kind, s):
    a = _a(kind)
    return [ev.single_term_theory(a, s, lm)[0] for lm in range(3, 9)]


def test_ein_term_varianz_beim_mittelwert_und_knick():
    g1, k1, k15, k2 = _curve("glatt", 1.0), _curve("knick", 1.0), _curve("knick", 1.5), _curve("knick", 2.0)
    assert _band(g1[0], 1.73, 1.3) and _band(g1[-1], 1.88, 1.3) and g1[-1] < 1.3 * g1[0]
    assert _band(k1[0], 0.28, 1.4) and _band(k1[-1], 0.32, 1.4) and k1[-1] < 1.3 * k1[0]
    assert _band(k15[0], 0.24, 1.4) and _band(k15[-1], 0.39, 1.5) and k15[-1] > 1.3 * k15[0]
    assert _band(k2[-1], 1.41, 2.0) and k2[-1] > 3 * k2[0]
    assert _band(_curve("glatt", 2.0)[-1], 2.68, 1.5)


def test_ein_term_varianz_beim_sprung_waechst_fuer_jedes_s():
    for s, ende in ((1.0, 24.8), (1.5, 199.0), (2.0, 2167.0)):
        v = _curve("sprung", s)
        assert v[-1] > 8 * v[0] and _band(v[-1], ende, 3.0), s


def test_ein_term_lauf_beim_knick():
    ex = ev.single_term_experiment(Settings(ausgabe="knick", lmax=8), _a("knick"))
    assert abs(ex["run"]["z"]) < 3.5 and abs(ex["run"]["mean"] - 0.0655) < 0.004 and _band(ex["run"]["var"], 0.316, 1.4) and _band(ex["theory"]["var"], 0.321, 1.4)
    assert ex["run"]["cost"] == pytest.approx(4.62, abs=0.3) and _band(ex["theory"]["work"], 1.446, 1.4)


# --- Ergänzungen: Zahlen der README, die oben nur als Richtung geprüft sind ------------------------------------------------------------------------------

def test_signalstaerke_zahlen_der_readme():
    g = ev.signal_experiment(Settings(ausgabe="glatt", lmax=8), _a("glatt"))
    assert _band({x["rho"]: x["ratios"] for x in g["rho_rows"]}[0.6][0], 2.57, 1.6)
    sp = ev.signal_experiment(Settings(ausgabe="sprung", lmax=8), _a("sprung"))
    r = {x["rho"]: x["ratios"][0] for x in sp["rho_rows"]}
    assert _band(r[0.3], 0.41, 1.6) and _band(r[0.6], 0.66, 1.6) and _band(r[0.9], 0.87, 1.6) and r[0.3] < r[0.6] < r[0.9]


def test_einseitige_kopplung_und_schwaches_signal_zahlen_der_readme():
    e = _a("glatt", kopplung="one")
    assert _band(ev.cost_row(e, 0.0025)["ratio"], 0.21, 1.8) and _band(ev.cost_row(e, 0.000625)["ratio"], 0.59, 1.8)
    sw = ev.analyse(Settings(ausgabe="glatt", rho=0.3, lmax=8))
    assert _band(ev.cost_row(sw, 0.0025)["ratio"], 0.08, 2.0)


def test_gelaufene_verhaeltnisse_der_readme_bei_mittleren_toleranzen():
    for kind, eps, soll in (("glatt", 0.01, 0.22), ("glatt", 0.005, 0.52), ("knick", 0.005, 0.81), ("sprung", 0.005, 0.36)):
        ex = ev.run_experiment(Settings(ausgabe=kind, lmax=8, eps=eps), _a(kind))
        assert _band(ex["ratio"], soll, 2.0), (kind, eps, ex["ratio"])


def test_gelaufener_sprung_bei_enger_toleranz_bleibt_teurer():
    ex = ev.run_experiment(Settings(ausgabe="sprung", lmax=8, eps=0.0025), _a("sprung"))
    assert _band(ex["ratio"], 0.43, 2.0) and ex["ratio"] < 1.0


def test_ein_term_lauf_beim_sprung_hat_unzuverlaessige_stichprobenvarianz():
    ex = ev.single_term_experiment(Settings(ausgabe="sprung", lmax=8), _a("sprung"))
    assert _band(ex["run"]["var"], 35.6, 2.5) and _band(ex["theory"]["var"], 24.8, 1.5)
