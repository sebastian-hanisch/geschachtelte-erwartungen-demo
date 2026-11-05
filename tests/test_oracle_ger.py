"""Unabhängige Gegenproben (scipy): Quadratur der bedingten Erwartung und der Wahrheit, Kolmogorov–Smirnov, Monte Carlo mit exakter bedingter Erwartung; ORACLE-PLAYBOOK §5."""

import math

import numpy as np
import pytest
from scipy import integrate, stats

import ger_algorithm as A
from conftest import STD

PARAMS = [(0.6, 0.5, 1.3, 1.0), (0.3, 0.5, 1.3, 1.0), (0.9, 0.5, 1.3, 1.0), (0.6, 0.5, 1.3, 0.5), (0.6, 0.5, 1.3, 2.0), (0.7, 0.8, 1.0, 1.0), (0.5, 0.3, 0.8, 1.5)]


def _k(sigma, q, kmul):
    return A.k_default(sigma, q) * kmul


def _g(D, q):
    return max(D - q, 0.0)


@pytest.mark.parametrize("rho,sigma,q,kmul", PARAMS)
def test_bedingte_erwartung_gegen_quadratur_der_inneren_verteilung(rho, sigma, q, kmul):
    for x in (-1.5, 0.0, 0.7, 2.0):
        inner = integrate.quad(lambda y: _g(math.exp(sigma * (rho * x + math.sqrt(1 - rho**2) * y)), q) * stats.norm.pdf(y), -12, 12, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
        assert float(A.mu_cond(x, rho, sigma, q)) == pytest.approx(inner, abs=1e-9)


@pytest.mark.parametrize("sigma,q", [(0.5, 1.3), (0.8, 1.0), (0.3, 0.8)])
def test_schwelle_gegen_quadratur(sigma, q):
    ref = integrate.quad(lambda z: _g(math.exp(sigma * z), q) * stats.norm.pdf(z), -12, 12, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
    assert A.k_default(sigma, q) == pytest.approx(ref, abs=1e-10)


def _quad_truth(kind, rho, sigma, q, k):
    mu = lambda x: float(A.mu_cond(x, rho, sigma, q))
    f = {"glatt": lambda m: m * m, "knick": lambda m: max(m - k, 0.0), "sprung": lambda m: float(m > k)}[kind]
    xs = A.x_star(k, rho, sigma, q)
    pts = sorted({-12.0, xs, 12.0})
    return sum(integrate.quad(lambda x: f(mu(x)) * stats.norm.pdf(x), lo, hi, epsabs=1e-12, epsrel=1e-12, limit=200)[0] for lo, hi in zip(pts[:-1], pts[1:]))


@pytest.mark.parametrize("kind", ["glatt", "knick", "sprung"])
@pytest.mark.parametrize("rho,sigma,q,kmul", PARAMS)
def test_exakter_wert_gegen_quadratur(kind, rho, sigma, q, kmul):
    k = _k(sigma, q, kmul)
    assert A.truth(kind, rho, sigma, q, k) == pytest.approx(_quad_truth(kind, rho, sigma, q, k), abs=2e-9)


@pytest.mark.parametrize("kind", ["glatt", "knick", "sprung"])
@pytest.mark.parametrize("rho,sigma,q,kmul", PARAMS[:5])
def test_varianz_der_bedingten_funktion_gegen_quadratur(kind, rho, sigma, q, kmul):
    k = _k(sigma, q, kmul)
    mu = lambda x: float(A.mu_cond(x, rho, sigma, q))
    f = {"glatt": lambda m: m * m, "knick": lambda m: max(m - k, 0.0), "sprung": lambda m: float(m > k)}[kind]
    xs = A.x_star(k, rho, sigma, q)
    pts = sorted({-12.0, xs, 12.0})
    second = sum(integrate.quad(lambda x: f(mu(x)) ** 2 * stats.norm.pdf(x), lo, hi, epsabs=1e-12, epsrel=1e-12, limit=200)[0] for lo, hi in zip(pts[:-1], pts[1:]))
    assert A.var_cond(kind, rho, sigma, q, k) == pytest.approx(second - _quad_truth(kind, rho, sigma, q, k) ** 2, abs=2e-9)


@pytest.mark.parametrize("kind", ["glatt", "knick", "sprung"])
@pytest.mark.parametrize("rho,sigma,q,kmul", PARAMS[:5] + PARAMS[5:])
def test_e_p0_gegen_quadratur_der_marginalverteilung(kind, rho, sigma, q, kmul):
    k = _k(sigma, q, kmul)
    f = {"glatt": lambda m: m * m, "knick": lambda m: max(m - k, 0.0), "sprung": lambda m: float(m > k)}[kind]
    zq = math.log(q + k) / sigma if kind != "glatt" else math.log(q) / sigma
    pts = sorted({-12.0, zq, math.log(q) / sigma, 12.0})
    ref = sum(integrate.quad(lambda z: f(_g(math.exp(sigma * z), q)) * stats.norm.pdf(z), lo, hi, epsabs=1e-12, epsrel=1e-12, limit=200)[0] for lo, hi in zip(pts[:-1], pts[1:]))
    assert A.e_p0(kind, sigma, q, k) == pytest.approx(ref, abs=2e-9)


def test_normalzahlen_gegen_scipy_kolmogorov_smirnov():
    assert stats.kstest(A.normals(A.stream_seed(7), 20000), "norm").pvalue > 0.001


@pytest.mark.parametrize("kind", ["glatt", "knick", "sprung"])
def test_wahrheit_gegen_monte_carlo_mit_exakter_bedingter_erwartung(kind):
    k = A.k_default(0.5, 1.3)
    X = A.normals(A.stream_seed(4), 400000)
    v = A.f_eval(kind, A.mu_cond(X, 0.6, 0.5, 1.3), k)
    assert abs(v.mean() - A.truth(kind, 0.6, 0.5, 1.3, k)) < 4 * v.std() / math.sqrt(len(v))


def test_beziehung_beim_mittelwert_wahrheit_plus_c_gleicht_e_p0():
    # E g² = E μ² + E Var(g | X): die geschachtelte Verzerrung c/M hat c = E g² − E μ² = e_p0 − truth; E g² unabhängig per Quadratur
    k = A.k_default(0.5, 1.3)
    e_g2 = integrate.quad(lambda z: _g(math.exp(0.5 * z), 1.3) ** 2 * stats.norm.pdf(z), -12, 12, epsabs=1e-13, epsrel=1e-12, limit=200)[0]
    assert A.e_p0("glatt", 0.5, 1.3, k) == pytest.approx(e_g2, abs=1e-10)
    assert A.e_p0("glatt", 0.5, 1.3, k) - A.truth("glatt", 0.6, 0.5, 1.3, k) == pytest.approx(0.12386, abs=2e-5)


@pytest.mark.parametrize("level", [2, 3, 4])
def test_ebenenmittel_beim_mittelwert_gleicht_wahrheit_plus_c_durch_m(level):
    k = A.k_default(0.5, 1.3)
    N = 200000
    st = A.level_stats("glatt", level, N, 11, 0.6, 0.5, 1.3, k)
    exakt = A.truth("glatt", 0.6, 0.5, 1.3, k) + (A.e_p0("glatt", 0.5, 1.3, k) - A.truth("glatt", 0.6, 0.5, 1.3, k)) / 2**level
    assert abs(st[5] - exakt) < 4 * math.sqrt(st[2] / N)


def test_y_l_beim_mittelwert_ist_minus_quadrat_der_haelftendifferenz_je_probe():
    k = A.k_default(0.5, 1.3)
    G = A.inner_values(5, 2000, 16, **STD)
    p, pc = A.level_pair("glatt", 4, 2000, 5, **STD, k=k, values=G)
    a, b = G[:, :8].mean(axis=1), G[:, 8:].mean(axis=1)
    assert np.allclose(p - pc, -((a - b) ** 2) / 4)


@pytest.mark.parametrize("kind", ["glatt", "knick", "sprung"])
def test_teleskopsumme_gleicht_dem_mittel_der_feinen_ebene(kind):
    k = A.k_default(0.5, 1.3)
    L, N = 4, 100000
    stats_l = [A.level_stats(kind, l, N, 21, 0.6, 0.5, 1.3, k) for l in range(L + 1)]
    tele = A.e_p0(kind, 0.5, 1.3, k) + sum(s[0] for s in stats_l[1:])
    se = math.sqrt(sum(s[1] / N for s in stats_l[1:])) + math.sqrt(stats_l[L][2] / N)
    assert abs(tele - stats_l[L][5]) < 4 * se


def test_grobe_stufe_nutzt_dieselben_inneren_werte_vertauschung_aendert_nichts():
    k = A.k_default(0.5, 1.3)
    G = A.inner_values(8, 3000, 8, **STD)
    rng = np.random.default_rng(1)
    G2 = np.concatenate([rng.permuted(G[:, :4], axis=1), rng.permuted(G[:, 4:], axis=1)], axis=1)       # innerhalb der Hälften mischen
    G3 = np.concatenate([G[:, 4:], G[:, :4]], axis=1)                                                    # Hälften tauschen
    base = A.level_pair("knick", 3, 3000, 0, **STD, k=k, values=G)
    for other in (G2, G3):
        p, pc = A.level_pair("knick", 3, 3000, 0, **STD, k=k, values=other)
        assert np.allclose(p, base[0]) and np.allclose(pc, base[1])


def test_zufaellige_stoerung_der_zuteilung_ist_nie_billiger():
    rng = np.random.default_rng(1)
    vl, cl, eps = [0.12, 0.001, 0.0008, 0.0004], [1.0, 2.0, 4.0, 8.0], 0.01
    ns, cost = A.optimal_allocation(vl, cl, eps)
    for _ in range(200):
        trial = np.array(ns) * rng.uniform(0.5, 2.0, len(ns))
        var = sum(v / n for v, n in zip(vl, trial))
        trial = trial * var / (eps**2 / 2)
        assert sum(n * c for n, c in zip(trial, cl)) >= cost * (1 - 1e-9)


def _sprung_verzerrung_mal_m(level, n_total, chunk):
    """Verzerrung E 1[Mittel von M > k] − P(X > x*) mal M, mit gemeinsamen äußeren Ziehungen (Differenz klein, daher genau); Kontrollrechnung, nicht Teil der Demo."""
    rho, sigma, q = 0.6, 0.5, 1.3
    k = A.k_default(sigma, q)
    xs = A.x_star(k, rho, sigma, q)
    M = 2**level
    s, done, c = 0.0, 0, 0
    while done < n_total:
        n = min(chunk, n_total - done)
        seed = A.stream_seed(2024, level, c)
        X = A.normals(A.stream_seed(seed, 1), n)
        G = A.inner_values(seed, n, M, rho, sigma, q)
        s += float(((G.mean(axis=1) > k).astype(float) - (X > xs).astype(float)).sum())
        done += n
        c += 1
    return s / n_total * M


def test_sprung_verzerrung_ist_kein_c_durch_m_und_die_konstante_der_demo_ist_zu_gross():
    """README (Sprung): Verzerrung mal M fällt von 0,118 (M = 1) auf etwa 0,06 (M = 8) und 0,03 (M = 32); die Konstante c der Demo (Median über die Ebenen 2 bis 5) ist rund 0,1,
    also bei großem M ein Mehrfaches der wahren Verzerrung mal M."""
    import ger_evaluation as ev
    b0 = _sprung_verzerrung_mal_m(0, 1_000_000, 500_000)
    b3 = _sprung_verzerrung_mal_m(3, 1_000_000, 250_000)
    b5 = _sprung_verzerrung_mal_m(5, 600_000, 100_000)
    assert b0 == pytest.approx(-0.118, abs=0.01) and b3 == pytest.approx(-0.060, abs=0.02)
    assert abs(b3) < 0.7 * abs(b0)
    assert abs(b5) < 0.07
    a = ev.analyse(ev.Settings(ausgabe="sprung", lmax=8))
    assert a.c == pytest.approx(0.105, abs=0.03)
    assert a.c > 2.0 * abs(b5)
