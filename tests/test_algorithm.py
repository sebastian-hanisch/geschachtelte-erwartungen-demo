"""Handrechnung der Einheiten (DEMO-PLAYBOOK §1b): jede Zahl hier ist von Hand nachgerechnet, nicht aus dem Code gelesen."""

import math

import numpy as np
import pytest

import ger_algorithm as A
from conftest import STD

# Mini-Instanz: zwei innere Werte g = (0,3; 0,1) für eine äußere Ziehung, Schwelle k = 0,15.
G2 = [[0.3, 0.1]]
K = 0.15


def _y(kind, variant="anti", values=G2):
    p, pc = A.level_pair(kind, 1, 1, 0, 0.6, 0.5, 1.3, K, variant, values=values)
    return float(p[0]), float(pc[0])


def test_mittelwert_von_hand():
    # fein f(0,2) = 0,04; grob ½ (0,09 + 0,01) = 0,05; Y_1 = −0,01 = −(a − b)²/4
    p, pc = _y("glatt")
    assert (p, pc) == pytest.approx((0.04, 0.05)) and p - pc == pytest.approx(-(0.3 - 0.1) ** 2 / 4)


def test_knick_von_hand():
    # fein (0,2 − 0,15)⁺ = 0,05; grob ½ (0,15 + 0) = 0,075; Y_1 = −0,025
    assert _y("knick") == pytest.approx((0.05, 0.075))


def test_sprung_von_hand():
    # fein 1[0,2 > 0,15] = 1; grob ½ (1 + 0) = 0,5; Y_1 = 0,5
    assert _y("sprung") == pytest.approx((1.0, 0.5))


def test_einseitige_kopplung_von_hand():
    # grob nur die erste Hälfte (0,3): Mittelwert 0,09, Knick 0,15, Sprung 1
    assert _y("glatt", "one")[1] == pytest.approx(0.09)
    assert _y("knick", "one")[1] == pytest.approx(0.15)
    assert _y("sprung", "one")[1] == pytest.approx(1.0)


def test_ebene_null_hat_keine_grobe_stufe():
    p, pc = A.level_pair("glatt", 0, 1, 0, 0.6, 0.5, 1.3, K, values=[[0.3]])
    assert float(p[0]) == pytest.approx(0.09) and float(pc[0]) == 0.0


def test_antithetisch_ist_symmetrisch_in_den_haelften():
    a = _y("knick", values=[[0.3, 0.1]])
    b = _y("knick", values=[[0.1, 0.3]])
    assert a == pytest.approx(b)


def test_kosten_je_probe():
    assert [A.cost_per_sample(l) for l in range(4)] == [1.0, 2.0, 4.0, 8.0]


def test_optimale_zuteilung_von_hand():
    # V = (1, ½, ¼), C = (1, 2, 4): √(V C) = (1, 1, 1), Summe 3; N_l = 2 ε⁻² · 3 · √(V/C) = ε⁻² (6; 3; 1,5); Kosten 18 ε⁻² = 2 · 3² ε⁻²
    ns, cost = A.optimal_allocation([1.0, 0.5, 0.25], [1.0, 2.0, 4.0], 1.0)
    assert ns == pytest.approx([6.0, 3.0, 1.5]) and cost == pytest.approx(18.0)
    assert A.optimal_allocation([1.0, 0.5, 0.25], [1.0, 2.0, 4.0], 0.1)[1] == pytest.approx(1800.0)


def test_schwelle_von_hand():
    # E g(D) = e^{σ²/2} Φ(σ − a) − q Φ(−a) mit a = ln 1,3/0,5 = 0,5247
    a = math.log(1.3) / 0.5
    assert A.k_default(0.5, 1.3) == pytest.approx(math.exp(0.125) * A.Phi(0.5 - a) - 1.3 * A.Phi(-a))
    assert A.k_default(0.5, 1.3) == pytest.approx(0.16554, abs=1e-5)


def test_e_p0_sprung_von_hand():
    # g > k ⇔ D > q + k = 1,4655: P = Φ(−ln 1,4655/0,5) = Φ(−0,7645)
    b = math.log(1.3 + A.k_default(0.5, 1.3)) / 0.5
    assert b == pytest.approx(0.7645, abs=2e-4)
    assert A.e_p0("sprung", 0.5, 1.3, A.k_default(0.5, 1.3)) == pytest.approx(A.Phi(-b)) and A.Phi(-b) == pytest.approx(0.2223, abs=1e-4)


def test_e_p0_knick_von_hand():
    # E[(g − k)⁺] = E[(D − 1,4655)⁺] = e^{0,125} Φ(0,5 − 0,7645) − 1,4655 Φ(−0,7645)
    k = A.k_default(0.5, 1.3)
    b = math.log(1.3 + k) / 0.5
    assert A.e_p0("knick", 0.5, 1.3, k) == pytest.approx(math.exp(0.125) * A.Phi(0.5 - b) - (1.3 + k) * A.Phi(-b))
    assert A.e_p0("knick", 0.5, 1.3, k) == pytest.approx(0.12262, abs=1e-5)


def test_bedingte_erwartung_von_hand_bei_x_null():
    # x = 0: Lage 0, Streuung s = 0,5·√(1 − 0,36) = 0,4; d2 = −ln 1,3/0,4 = −0,6559, d1 = −0,2559; μ = e^{0,08} Φ(−0,2559) − 1,3 Φ(−0,6559) ≈ 1,0833·0,3990 − 1,3·0,2559
    assert float(A.mu_cond(0.0, 0.6, 0.5, 1.3)) == pytest.approx(1.0833 * 0.3990 - 1.3 * 0.2559, abs=5e-4)
    assert float(A.mu_cond(0.0, 0.6, 0.5, 1.3)) == pytest.approx(0.0995, abs=1e-3)


def test_bedingte_erwartung_waechst_mit_dem_signal():
    m = A.mu_cond(np.array([-1.0, 0.0, 1.0]), 0.6, 0.5, 1.3)
    assert m[0] < m[1] < m[2]


def test_ein_term_wahrscheinlichkeiten_von_hand():
    p = A.single_term_probs(1.0, 2)           # (1, ½, ¼)/(7/4)
    assert p == pytest.approx([4 / 7, 2 / 7, 1 / 7]) and p.sum() == pytest.approx(1.0)


def test_strom_ist_reproduzierbar_und_verschieden():
    assert A.stream_seed(3, 4) == A.stream_seed(3, 4) and A.stream_seed(3, 4) != A.stream_seed(4, 3)
    u = A.uniforms(A.stream_seed(1), 10000)
    assert u.min() > 0 and u.max() < 1
    assert np.array_equal(A.normals(5, 100), A.normals(5, 100)) and not np.array_equal(A.normals(5, 100), A.normals(6, 100))


def test_normalzahlen_haben_mittel_null_und_varianz_eins():
    z = A.normals(A.stream_seed(2), 200000)
    assert abs(z.mean()) < 0.01 and z.var() == pytest.approx(1.0, abs=0.01)


def test_innere_werte_sind_nichtnegativ_und_haben_die_form_n_mal_m():
    G = A.inner_values(3, 50, 8, **STD)
    assert G.shape == (50, 8) and (G >= 0).all()
