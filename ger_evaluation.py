"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Gemessen wird, was Multilevel gegenüber geschachteltem Monte Carlo spart. Kosten zählen **innere Auswertungen** (nicht Laufzeit); Ziel ist ein mittlerer quadratischer Fehler ≤ ε²
(Verzerrung² ≤ ε²/2 und Varianz ≤ ε²/2). Die exakten Werte sind geschlossen bekannt (Gauß–Hermite, Gauß–Legendre, Φ); die Schätzer kennen sie nicht.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import ger_algorithm as A
import ger_constants as C
from ger_scenario import Szenario, standard_szenario


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def dn(x: float, nd: int = 2) -> str:
    """Wie `de`, aber „nicht bestimmbar“ für NaN und „unendlich“ für ∞."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if np.isinf(x):
        return "unendlich"
    return de(x, nd)


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


def ke(x: float) -> str:
    """Kosten in inneren Auswertungen: ganze Zahl bis 99 999, darüber wissenschaftlich."""
    if np.isnan(x):
        return "nicht bestimmbar"
    return de(x, 0) if abs(x) < 1e5 else de_e(x)


def prozent(x: float, nd: int = 0) -> str:
    return "nicht bestimmbar" if np.isnan(x) else f"{de(x * 100, nd)} %"


def eps_text(eps: float) -> str:
    return f"{eps:g}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    ausgabe: str = C.DEFAULT_AUSGABE
    kopplung: str = C.DEFAULT_KOPPLUNG
    rho: float = C.DEFAULT_RHO
    eps: float = C.DEFAULT_EPS
    lmax: int = C.DEFAULT_LMAX
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    label: str
    kind: str
    variant: str                          # gewählte Kopplung (bestimmt Kosten und Läufe)
    rho: float
    sigma: float
    q: float
    k: float
    truth: float
    var_cond: float                       # Var f(μ(X)): Varianz je äußerer Ziehung, wenn die innere Erwartung bekannt ist
    lmax: int
    n: int                                # äußere Ziehungen je Ebene
    e0: float                             # exakter Erwartungswert von P_0
    levels: tuple                         # je Ebene {l, mean, var, var_p, kurt, nonzero, mean_p} (gewählte Kopplung)
    var_by_variant: dict                  # Kopplung -> V_l je Ebene
    c: float                              # Verzerrungskonstante: |E[P_L] − exakt| ≈ c · 2^(−L)
    beta: float                           # Rate der Ebenenvarianz der gewählten Kopplung
    beta_by_variant: dict


def label_of(settings: Settings, sc: Szenario | None = None) -> str:
    extra = ""
    if sc is not None and (sc.sigma != C.SIGMA or sc.q != C.Q or sc.kmul != 1.0):
        extra = f", σ = {de(sc.sigma, 2)}, q = {de(sc.q, 2)}, k = {de(sc.kmul, 2)}·Mittel"
    return f"{C.AUSGABEN[settings.ausgabe]}, Kopplung {C.KOPPLUNG_KURZ[settings.kopplung]}{extra}"


# --- Rate und Verzerrungskonstante ---------------------------------------------------------------------------------------------------------

def fit_rate(variances, first: int = C.FIT_FROM) -> float:
    """β aus der Regressionsgeraden von log₂ V_l gegen l über die Ebenen ab `first` (nur positive Varianzen); NaN, wenn weniger als zwei Punkte."""
    pts = [(l, math.log2(v)) for l, v in enumerate(variances) if l >= first and v > 0]
    if len(pts) < 2:
        return float("nan")
    return float(-np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0])


def bias_const(levels, first: int = C.BIAS_FROM, last: int = C.BIAS_TO) -> float:
    """c aus dem Median von |E[Y_l]|·2^l über die Ebenen first … last (Verzerrung ≈ c/M)."""
    vals = [abs(r["mean"]) * 2.0 ** r["l"] for r in levels if first <= r["l"] <= last]
    return float(np.median(vals)) if vals else float("nan")


def bias_at(a: Analysis, level: int) -> float:
    """E[P_l] − exakt aus E[P_0] (geschlossen) plus den gemessenen Ebenenmitteln."""
    return a.e0 + sum(r["mean"] for r in a.levels[1 : level + 1]) - a.truth


# --- Messung ----------------------------------------------------------------------------------------------------------------------------------

def analyse(settings: Settings, sc: Szenario | None = None) -> Analysis:
    """Ebenenstatistik der Ebenen 0 bis lmax für alle drei Kopplungen; N_LEVEL äußere Ziehungen je Ebene."""
    sc = sc if sc is not None else standard_szenario(settings.rho)
    k = A.k_default(sc.sigma, sc.q) * sc.kmul
    seed = A.stream_seed(settings.seed, 1)
    by_variant, rows_by = {}, {}
    for variant in C.KOPPLUNGEN:
        rows = []
        for l in range(settings.lmax + 1):
            if l == 0 and variant != "anti":
                rows.append(rows_by["anti"][0])
                continue
            m, v, vp, kurt, nz, mp = A.level_stats(settings.ausgabe, l, C.N_LEVEL, seed, sc.rho, sc.sigma, sc.q, k, variant)
            rows.append({"l": l, "mean": m, "var": v, "var_p": vp, "kurt": kurt, "nonzero": nz, "mean_p": mp})
        rows_by[variant] = rows
        by_variant[variant] = tuple(r["var"] for r in rows)
        if variant == "anti":
            rows_by["anti"] = rows
    levels = rows_by[settings.kopplung]
    betas = {v: fit_rate(by_variant[v]) for v in by_variant}
    return Analysis(label_of(settings, sc), settings.ausgabe, settings.kopplung, sc.rho, sc.sigma, sc.q, k, A.truth(settings.ausgabe, sc.rho, sc.sigma, sc.q, k),
                    A.var_cond(settings.ausgabe, sc.rho, sc.sigma, sc.q, k), settings.lmax, C.N_LEVEL, A.e_p0(settings.ausgabe, sc.sigma, sc.q, k), tuple(levels), by_variant,
                    bias_const(levels), betas[settings.kopplung], betas)


# --- Kosten (idealisiert) -------------------------------------------------------------------------------------------------------------------

def level_for(a: Analysis, eps: float) -> int:
    """Kleinste Ebene L ≥ 1 mit c·2^(−L) ≤ ε/√2 (Verzerrung² ≤ ε²/2)."""
    if not a.c > 0:
        return 1
    return max(1, min(40, math.ceil(math.log2(a.c / (eps / math.sqrt(2))))))


def level_variance(a: Analysis, l: int) -> float:
    """V_l gemessen bis zur größten Ebene, darüber aus der Rate β fortgesetzt (β nicht kleiner als 0)."""
    if l <= a.lmax:
        return a.levels[l]["var"]
    beta = a.beta if np.isfinite(a.beta) else 1.0
    return a.levels[a.lmax]["var"] * 2.0 ** (-max(beta, 0.0) * (l - a.lmax))


def costs_for_level(a: Analysis, eps: float, L: int) -> dict:
    """Idealisierte Kosten in inneren Auswertungen bei bekannten V_l und bekannter Verzerrung: geschachtelt 2 Var(P_L) ε⁻² 2^L; Multilevel 2 ε⁻² (Σ √(V_l C_l))²;
    bedingt (innere Erwartung bekannt) 2 Var f(μ(X)) ε⁻²."""
    vl = [level_variance(a, l) for l in range(L + 1)]
    cl = [A.cost_per_sample(l) for l in range(L + 1)]
    s = sum(math.sqrt(v * c) for v, c in zip(vl, cl))
    var_p = a.levels[min(L, a.lmax)]["var_p"]
    nested = 2.0 * var_p / eps**2 * 2.0**L
    ml = A.optimal_allocation(vl, cl, eps)[1]
    cond = 2.0 * a.var_cond / eps**2
    shares = [math.sqrt(v * c) / s if s > 0 else float("nan") for v, c in zip(vl, cl)]
    return {"eps": eps, "L": L, "nested": nested, "ml": ml, "cond": cond, "ratio": nested / ml if ml > 0 else float("inf"), "ml_over_cond": ml / cond if cond > 0 else float("inf"),
            "eps2_nested": eps**2 * nested, "eps2_ml": eps**2 * ml, "shares": shares}


def cost_row(a: Analysis, eps: float) -> dict:
    return costs_for_level(a, eps, level_for(a, eps))


def cost_table(a: Analysis, grid=None) -> list[dict]:
    return [cost_row(a, e) for e in (grid or C.EPS_GRID)]


# --- Experiment 3: gelaufene Schätzungen ------------------------------------------------------------------------------------------------------

def run_experiment(settings: Settings, a: Analysis | None = None, sc: Szenario | None = None) -> dict:
    """Giles' adaptives Verfahren gegen geschachteltes Monte Carlo (mit Orakelwissen über M und Varianz) bei der Toleranz ε, RUN_REPS Wiederholungen mit festen Strömen."""
    sc = sc if sc is not None else standard_szenario(settings.rho)
    a = a if a is not None else analyse(settings, sc)
    eps, truth = settings.eps, a.truth
    L_std = level_for(a, eps)
    var_p = a.levels[min(L_std, a.lmax)]["var_p"]
    err_ml, cost_ml, levels_ml, err_n, cost_n = [], [], [], [], []
    for r in range(C.RUN_REPS):
        est, cost, L, _ = A.mlmc(settings.ausgabe, eps, A.stream_seed(settings.seed, 3, r, int(round(eps * 1e6))), sc.rho, sc.sigma, sc.q, a.k, settings.kopplung, n0=C.PILOT)
        err_ml.append(est - truth), cost_ml.append(cost), levels_ml.append(L)
        est2, cost2 = A.naive_run(settings.ausgabe, eps, L_std, var_p, A.stream_seed(settings.seed, 4, r, int(round(eps * 1e6))), sc.rho, sc.sigma, sc.q, a.k)
        err_n.append(est2 - truth), cost_n.append(cost2)
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
    return {"eps": eps, "truth": truth, "reps": C.RUN_REPS, "pilot": C.PILOT, "L_std": L_std, "M_std": 2**L_std,
            "ml": {"rmse": rm(err_ml), "cost": float(np.mean(cost_ml)), "L_mean": float(np.mean(levels_ml)), "L_min": int(min(levels_ml)), "L_max": int(max(levels_ml)), "over": float(np.mean(np.abs(err_ml) > eps))},
            "nested": {"rmse": rm(err_n), "cost": float(np.mean(cost_n)), "over": float(np.mean(np.abs(err_n) > eps))},
            "ratio": float(np.mean(cost_n) / np.mean(cost_ml))}


# --- Experiment 4: Signal und Schwelle ----------------------------------------------------------------------------------------------------------

def signal_experiment(settings: Settings, a: Analysis | None = None) -> dict:
    """Idealisiertes Kostenverhältnis geschachtelt/Multilevel bei ε = 0,001 und 0,00025 für alle Signalstärken ρ und für die Schwelle k = 0,5 / 1 / 2 mal Mittel (bei der gewählten ρ)."""
    rho_rows, k_rows = [], []
    for rho in C.RHO_CHOICES:
        b = analyse(Settings(settings.ausgabe, settings.kopplung, rho, settings.eps, settings.lmax, settings.seed))
        rho_rows.append({"rho": rho, "beta": b.beta, "c": b.c, "ratios": [cost_row(b, e)["ratio"] for e in C.EPS_SIGNAL]})
    for f in C.K_FACTORS:
        b = analyse(settings, Szenario(settings.rho, C.SIGMA, C.Q, f))
        k_rows.append({"factor": f, "beta": b.beta, "c": b.c, "ratios": [cost_row(b, e)["ratio"] for e in C.EPS_SIGNAL]})
    return {"rho_rows": rho_rows, "k_rows": k_rows, "eps": C.EPS_SIGNAL, "rho": settings.rho, "kind": settings.ausgabe}


# --- Experiment 5: entzerrter Ein-Term-Schätzer -------------------------------------------------------------------------------------------------

def single_term_theory(a: Analysis, s: float, lmax: int) -> tuple[float, float, float]:
    """(Varianz, Kosten je Probe, Varianz·Kosten) des Ein-Term-Schätzers mit p_l ∝ 2^(−s l), abgeschnitten bei lmax, aus den gemessenen zweiten Momenten."""
    p = A.single_term_probs(s, lmax)
    m2 = [r["var"] + r["mean"] ** 2 for r in a.levels[: lmax + 1]]
    mean = sum(r["mean"] for r in a.levels[: lmax + 1])
    var = sum(m / pl for m, pl in zip(m2, p)) - mean**2
    cost = sum(pl * A.cost_per_sample(l) for l, pl in enumerate(p))
    return float(var), float(cost), float(var * cost)


def single_term_experiment(settings: Settings, a: Analysis | None = None, sc: Szenario | None = None) -> dict:
    sc = sc if sc is not None else standard_szenario(settings.rho)
    a = a if a is not None else analyse(settings, sc)
    curves = {s: [{"lmax": lm, **dict(zip(("var", "cost", "work"), single_term_theory(a, s, lm)))} for lm in range(3, a.lmax + 1)] for s in C.SINGLE_S}
    s_run = 1.0
    mean, var, cost = A.single_term(settings.ausgabe, s_run, a.lmax, C.N_SINGLE, A.stream_seed(settings.seed, 7), sc.rho, sc.sigma, sc.q, a.k, settings.kopplung)
    theo = single_term_theory(a, s_run, a.lmax)
    ml = cost_row(a, settings.eps)
    return {"curves": curves, "s_run": s_run, "run": {"mean": mean, "var": var, "cost": cost, "n": C.N_SINGLE, "z": (mean - a.truth) / math.sqrt(var / C.N_SINGLE) if var > 0 else float("nan")},
            "theory": {"var": theo[0], "cost": theo[1], "work": theo[2]}, "truth": a.truth, "lmax": a.lmax, "eps": settings.eps, "eps2_ml": ml["eps2_ml"], "eps2_nested": ml["eps2_nested"]}


# --- Verdikte -----------------------------------------------------------------------------------------------------------------------------

def _rate_text(beta: float) -> str:
    return "nicht bestimmbar" if not np.isfinite(beta) else de(beta, 2)


def regime(beta: float) -> str:
    """Welcher der drei Kostenfälle (Kostenrate je Ebene γ = 1)? Schwellen mit Spielraum für das Rauschen der Regression."""
    if not np.isfinite(beta):
        return "nicht bestimmbar"
    if beta > 1.25:
        return "β > γ: die Kosten sättigen bei ε²·Kosten (Ordnung ε⁻²)"
    if beta >= 0.8:
        return "β ≈ γ: ε²·Kosten wächst langsam (Logarithmen)"
    return "β < γ: die Kosten liegen zwischen ε⁻² und geschachteltem Monte Carlo"


def degenerate(a: Analysis) -> bool:
    """Die äußere Funktion ist auf allen gemessenen Ebenen konstant (etwa eine Schwelle, die nie überschritten wird): es gibt nichts zu schätzen."""
    return all(r["var_p"] == 0 for r in a.levels)


NICHTS_ZU_SCHAETZEN = "Die äußere Funktion ist auf allen gemessenen Ebenen konstant (keine Streuung, etwa weil die Schwelle nie überschritten wird): es gibt nichts zu schätzen und kein Kostenverhältnis. Schwelle oder Signal ändern."


def verdict_main(a: Analysis, settings: Settings) -> tuple[str, str]:
    """Kernfrage: Lohnt Multilevel bei dieser Wahl, und wie viel?"""
    if degenerate(a):
        return "info", f"{a.label}: " + NICHTS_ZU_SCHAETZEN
    row = cost_row(a, settings.eps)
    last = a.lmax
    v = a.var_by_variant
    head = (f"{a.label}, ρ = {de(a.rho, 2)}, Toleranz ε = {eps_text(settings.eps)}: geschachteltes Monte Carlo braucht M = {2 ** row['L']} innere Ziehungen (Ebene {row['L']}; Verzerrung ≈ {de(a.c, 4)}/M ≤ ε/√2) und kostet idealisiert "
            f"{ke(row['nested'])} innere Auswertungen, Multilevel {ke(row['ml'])}: geschachtelt/Multilevel = {dn(row['ratio'], 2)}; wäre die innere Erwartung bekannt, genügten {ke(row['cond'])}. "
            f"Die Ebenenvarianz fällt mit der Rate β ≈ {_rate_text(a.beta)} ({regime(a.beta)}); auf Ebene {last} ist V_l antithetisch {de_e(v['anti'][last])}, einseitig {de_e(v['one'][last])}, unabhängig {de_e(v['indep'][last])}.")
    if row["ratio"] < 1.0:
        return "warning", head + " Multilevel ist hier selbst im Idealfall teurer als einfaches Schachteln."
    if row["ratio"] >= 2.0:
        return "success", head + " Multilevel spart hier mindestens die Hälfte der Kosten."
    return "info", head + " Der Vorteil ist hier klein."


def verdict_levels(a: Analysis) -> tuple[str, str]:
    if degenerate(a):
        return "info", NICHTS_ZU_SCHAETZEN
    last = a.lmax
    b = a.beta_by_variant
    kurt = max((r["kurt"] for r in a.levels[1:] if np.isfinite(r["kurt"])), default=float("nan"))
    return "info", (f"Rate β der Ebenenvarianz (aus den Ebenen {C.FIT_FROM} bis {last}): antithetisch {_rate_text(b['anti'])}, einseitig {_rate_text(b['one'])}, unabhängig {_rate_text(b['indep'])}. "
                    f"Die Verzerrung fällt je Ebene etwa auf die Hälfte (≈ {de(a.c, 4)}/M; beim Mittelwert exakt, beim Sprung im Rauschen kaum messbar). Größte Kurtosis der Differenz (gewählte Kopplung): {dn(kurt, 0)} (je größer, desto unsicherer die Schätzung von V_l aus {de(a.n, 0)} äußeren Ziehungen).")


def verdict_cost(a: Analysis) -> tuple[str, str]:
    if degenerate(a):
        return "info", NICHTS_ZU_SCHAETZEN
    rows = cost_table(a)
    first, lastr = rows[0], rows[-1]
    tail = []                                          # zusammenhängend von der engsten Toleranz her, in der Multilevel billiger ist
    for r in reversed(rows):
        if r["ratio"] >= 1.0:
            tail.append(r)
        else:
            break
    if len(tail) == len(rows):
        wechsel = "Multilevel ist bei allen gezeigten Toleranzen billiger"
    elif tail:
        wechsel = f"Multilevel ist ab ε = {eps_text(tail[-1]['eps'])} und enger billiger"
    else:
        wechsel = "Multilevel ist bei der engsten Toleranz noch nicht billiger"
    return "info", (f"Geschachtelt/Multilevel bei ε = {eps_text(first['eps'])}: {dn(first['ratio'], 2)}, bei ε = {eps_text(lastr['eps'])}: {dn(lastr['ratio'], 2)}; {wechsel}. "
                    f"ε²·Kosten: Multilevel {de(first['eps2_ml'], 2)} → {de(lastr['eps2_ml'], 2)}, geschachtelt {de(first['eps2_nested'], 2)} → {de(lastr['eps2_nested'], 2)}. "
                    f"Mit bekannter innerer Erwartung („bedingt“) wäre Multilevel bei ε = {eps_text(lastr['eps'])} noch das {dn(lastr['ml_over_cond'], 1)}-Fache.")


def verdict_run(ex: dict) -> tuple[str, str]:
    ml, nv = ex["ml"], ex["nested"]
    text = (f"Toleranz ε = {eps_text(ex['eps'])}, {ex['reps']} Wiederholungen: Multilevel (Giles, Pilot {de(ex['pilot'], 0)} je Ebene) erreicht RMSE/ε = {de(ml['rmse'] / ex['eps'], 2)} bei mittleren Kosten {ke(ml['cost'])} inneren Auswertungen "
            f"(Ebenen {ml['L_min']} bis {ml['L_max']}, Mittel {de(ml['L_mean'], 1)}; Fehler über ε in {prozent(ml['over'], 0)} der Läufe); geschachteltes Monte Carlo mit Orakelwissen (M = {ex['M_std']}) RMSE/ε = {de(nv['rmse'] / ex['eps'], 2)}, Kosten {ke(nv['cost'])}. "
            f"Kostenverhältnis geschachtelt/Multilevel: {dn(ex['ratio'], 2)}.")
    return ("warning" if ex["ratio"] < 1.0 else "info"), text


def verdict_signal(ex: dict) -> tuple[str, str]:
    e1, e2 = ex["eps"]
    rr = {r["rho"]: r for r in ex["rho_rows"]}
    lo, hi = C.RHO_CHOICES[0], C.RHO_CHOICES[-1]
    kr = {r["factor"]: r for r in ex["k_rows"]}
    text = (f"Geschachtelt/Multilevel bei ε = {eps_text(e1)} (und {eps_text(e2)}): Signalstärke ρ = {de(lo, 1)}: {dn(rr[lo]['ratios'][0], 2)} ({dn(rr[lo]['ratios'][1], 2)}), ρ = {de(C.RHO_CHOICES[1], 1)}: {dn(rr[C.RHO_CHOICES[1]]['ratios'][0], 2)} ({dn(rr[C.RHO_CHOICES[1]]['ratios'][1], 2)}), "
            f"ρ = {de(hi, 1)}: {dn(rr[hi]['ratios'][0], 2)} ({dn(rr[hi]['ratios'][1], 2)}); Schwelle k = {de(C.K_FACTORS[0], 1)} / {de(C.K_FACTORS[1], 1)} / {de(C.K_FACTORS[2], 1)} mal Mittel (bei ρ = {de(ex['rho'], 1)}): "
            f"{dn(kr[C.K_FACTORS[0]]['ratios'][0], 2)} / {dn(kr[C.K_FACTORS[1]]['ratios'][0], 2)} / {dn(kr[C.K_FACTORS[2]]['ratios'][0], 2)}.")
    return "info", text


def verdict_single(ex: dict) -> tuple[str, str]:
    lm = ex["lmax"]
    parts = []
    for s in C.SINGLE_S:
        cv = ex["curves"][s]
        parts.append(f"s = {de(s, 1)}: {de(cv[0]['var'], 2)} (Abschneideebene {cv[0]['lmax']}) → {de(cv[-1]['var'], 2)} ({cv[-1]['lmax']})")
    run, th = ex["run"], ex["theory"]
    text = (f"Varianz des Ein-Term-Schätzers bei wachsender Abschneideebene: " + "; ".join(parts) + ". "
            f"Lauf mit s = {de(ex['s_run'], 1)}, Abschneideebene {lm}, {de(run['n'], 0)} Proben: Mittel {de(run['mean'], 4)} gegen exakt {de(ex['truth'], 4)} ({dn(run['z'], 1)} Standardfehler), Stichprobenvarianz {de(run['var'], 2)} "
            f"(aus den Ebenenmomenten erwartet {de(th['var'], 2)}), Kosten je Probe {de(run['cost'], 2)} innere Auswertungen. Varianz·Kosten/ε² = {de(th['work'], 2)}/ε² gegen Multilevel {de(ex['eps2_ml'], 2)}/ε² "
            f"(geschachtelt {de(ex['eps2_nested'], 2)}/ε²; Toleranz ε = {eps_text(ex['eps'])}).")
    cv = ex["curves"][ex["s_run"]]
    growing = cv[-1]["var"] > 1.5 * cv[0]["var"]
    if growing:
        text += " Die Varianz wächst mit der Abschneideebene: der Schätzer taugt bei diesem s nicht, und auch die Stichprobenvarianz im Lauf ist wegen der schweren Ränder unzuverlässig."
    return ("warning" if growing else "info"), text
