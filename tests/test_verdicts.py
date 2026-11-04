"""Verdikte (reine Funktionen in ger_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import pytest

import ger_evaluation as ev

S = ev.Settings(eps=0.005, lmax=4)
VA = (0.125, 1e-3, 2.5e-4, 6.25e-5, 1.5625e-5)      # antithetisch: V_l = 1e-3 · 4^(−(l−1)), β = 2
VO = (0.125, 0.05, 0.025, 0.0125, 0.00625)          # einseitig: β = 1
VI = (0.125, 0.1, 0.1, 0.1, 0.1)                    # unabhängig: konstant


def _levels(vars_):
    return tuple({"l": l, "mean": 0.0 if l == 0 else -0.1 / 2**l, "var": v, "var_p": 0.03, "kurt": 20.0, "nonzero": 0.3, "mean_p": 0.06} for l, v in enumerate(vars_))


def _a(**kw) -> ev.Analysis:
    basis = ev.Analysis(label="Mittelwert: f(m) = m², Kopplung antithetisch", kind="glatt", variant="anti", rho=0.6, sigma=0.5, q=1.3, k=0.1655, truth=0.0645, var_cond=0.03, lmax=4, n=40000, e0=0.1884,
                        levels=_levels(VA), var_by_variant={"anti": VA, "one": VO, "indep": VI}, c=0.1, beta=2.0, beta_by_variant={"anti": 2.0, "one": 1.0, "indep": 0.0})
    return dataclasses.replace(basis, **kw)


def test_kostenfaelle_der_rate():
    assert "β > γ" in ev.regime(2.0) and "sättigen" in ev.regime(2.0)
    assert "β ≈ γ" in ev.regime(1.0) and "β < γ" in ev.regime(0.5) and ev.regime(float("nan")) == "nicht bestimmbar"
    assert "β > γ" in ev.regime(1.26) and "β ≈ γ" in ev.regime(1.25) and "β ≈ γ" in ev.regime(0.8) and "β < γ" in ev.regime(0.79)


def test_ebene_und_kosten_von_hand():
    a = _a()
    # c = 0,1: kleinste Ebene mit 0,1/2^L ≤ 0,005/√2 = 0,003536 ⇒ 2^L ≥ 28,3 ⇒ L = 5 (M = 32)
    assert ev.level_for(a, 0.005) == 5
    row = ev.cost_row(a, 0.005)
    vl = list(VA) + [VA[4] / 4]                          # Ebene 5: Fortsetzung mit β = 2
    cl = [1, 2, 4, 8, 16, 32]
    s = sum((v * c) ** 0.5 for v, c in zip(vl, cl))
    assert row["L"] == 5 and row["ml"] == pytest.approx(2 * s * s / 0.005**2)
    assert row["nested"] == pytest.approx(2 * 0.03 / 0.005**2 * 32)
    assert row["cond"] == pytest.approx(2 * 0.03 / 0.005**2)
    assert row["ratio"] == pytest.approx(row["nested"] / row["ml"]) and sum(row["shares"]) == pytest.approx(1.0)


def test_hauptverdikt_schwellen_und_zahlen():
    a = _a()
    kind, text = ev.verdict_main(a, S)
    assert kind == "success" and "mindestens die Hälfte" in text
    for frag in ("Mittelwert: f(m) = m², Kopplung antithetisch, ρ = 0,60, Toleranz ε = 0,005", "M = 32 innere Ziehungen (Ebene 5; Verzerrung ≈ 0,1000/M", "wäre die innere Erwartung bekannt, genügten 2.400",
                 "β ≈ 2,00 (β > γ", "auf Ebene 4 ist V_l antithetisch 1,6e-05, einseitig 6,3e-03, unabhängig 1,0e-01"):
        assert frag in text, frag
    # Multilevel teurer als geschachtelt
    cheap = dataclasses.replace(a, levels=tuple({**r, "var_p": 1e-9} for r in a.levels))
    kind, text = ev.verdict_main(cheap, S)
    assert kind == "warning" and "teurer als einfaches Schachteln" in text
    # kleiner Vorteil
    klein = dataclasses.replace(a, levels=_levels(VO))
    assert ev.verdict_main(klein, S)[0] in ("info", "warning")


def test_ebenenverdikt():
    kind, text = ev.verdict_levels(_a())
    assert kind == "info"
    for frag in ("antithetisch 2,00, einseitig 1,00, unabhängig 0,00", "aus den Ebenen 3 bis 4", "≈ 0,1000/M", "Größte Kurtosis der Differenz (gewählte Kopplung): 20", "40.000 äußeren Ziehungen"):
        assert frag in text, frag


def test_kostenverdikt():
    a = _a()
    kind, text = ev.verdict_cost(a)
    assert kind == "info"
    assert "Geschachtelt/Multilevel bei ε = 0,01:" in text and "bei ε = 0,0003125:" in text and "ε²·Kosten: Multilevel" in text and "„bedingt“" in text
    assert "bei allen gezeigten Toleranzen billiger" in text or "ab ε = " in text
    teuer = dataclasses.replace(a, levels=tuple({**r, "var_p": 1e-9} for r in a.levels))
    assert "noch nicht billiger" in ev.verdict_cost(teuer)[1]


def test_gelaufenes_verdikt():
    ex = {"eps": 0.005, "reps": 30, "pilot": 1000, "L_std": 5, "M_std": 32, "truth": 0.0675,
          "ml": {"rmse": 0.0043, "cost": 84365.0, "L_mean": 5.1, "L_min": 5, "L_max": 6, "over": 0.23}, "nested": {"rmse": 0.0049, "cost": 68352.0, "over": 0.2}, "ratio": 0.81}
    kind, text = ev.verdict_run(ex)
    assert kind == "warning"
    for frag in ("Toleranz ε = 0,005, 30 Wiederholungen", "Pilot 1.000 je Ebene", "RMSE/ε = 0,86", "84.365 inneren Auswertungen", "Ebenen 5 bis 6, Mittel 5,1", "23 % der Läufe", "(M = 32) RMSE/ε = 0,98", "68.352", "geschachtelt/Multilevel: 0,81"):
        assert frag in text, frag
    assert ev.verdict_run({**ex, "ratio": 1.7})[0] == "info"


def test_signalverdikt():
    rows = [{"rho": r, "beta": 2.0, "c": 0.1, "ratios": [x, 2 * x]} for r, x in ((0.3, 0.14), (0.6, 2.3), (0.9, 31.0))]
    krows = [{"factor": f, "beta": 1.3, "c": 0.1, "ratios": [x, 2 * x]} for f, x in ((0.5, 10.1), (1.0, 6.0), (2.0, 2.6))]
    kind, text = ev.verdict_signal({"rho_rows": rows, "k_rows": krows, "eps": (0.001, 0.00025), "rho": 0.6, "kind": "glatt"})
    assert kind == "info"
    for frag in ("bei ε = 0,001 (und 0,00025)", "ρ = 0,3: 0,14 (0,28)", "ρ = 0,6: 2,30 (4,60)", "ρ = 0,9: 31,00 (62,00)", "k = 0,5 / 1,0 / 2,0 mal Mittel (bei ρ = 0,6): 10,10 / 6,00 / 2,60"):
        assert frag in text, frag


def test_ein_term_verdikt():
    def cv(vs):
        return [{"lmax": 3 + i, "var": v, "cost": 2.0, "work": 2 * v} for i, v in enumerate(vs)]
    ex = {"curves": {1.0: cv([0.28, 0.32]), 1.5: cv([0.24, 0.42]), 2.0: cv([0.25, 0.98])}, "s_run": 1.0, "run": {"mean": 0.0663, "var": 0.30, "cost": 4.06, "n": 20000, "z": -0.3},
          "theory": {"var": 0.32, "cost": 4.0, "work": 1.27}, "truth": 0.0675, "lmax": 4, "eps": 0.005, "eps2_ml": 0.78, "eps2_nested": 1.71}
    kind, text = ev.verdict_single(ex)
    assert kind == "info"
    for frag in ("s = 1,0: 0,28 (Abschneideebene 3) → 0,32 (4)", "s = 2,0: 0,25 (Abschneideebene 3) → 0,98 (4)", "Lauf mit s = 1,0", "Mittel 0,0663 gegen exakt 0,0675 (−0,3 Standardfehler)", "Stichprobenvarianz 0,30", "erwartet 0,32",
                 "1,27/ε² gegen Multilevel 0,78/ε² (geschachtelt 1,71/ε²"):
        assert frag in text, frag
    wachsend = {**ex, "curves": {**ex["curves"], 1.0: cv([0.28, 0.9])}}
    kind, text = ev.verdict_single(wachsend)
    assert kind == "warning" and "taugt bei diesem s nicht" in text
