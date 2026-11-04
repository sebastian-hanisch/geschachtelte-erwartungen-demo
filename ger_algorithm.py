"""Kern der Demo (nur numpy und math).

Gesucht ist E_X[ f( E_Y[ g(X, Y) | X ] ) ] mit unabhängigen Standardnormalen X (außen) und Y (innen):
  Nachfrage    D = exp(σ (ρ X + √(1 − ρ²) Y)),  innere Ausgabe g(D) = (D − q)⁺ (Fehlbestand); die bedingte Erwartung μ(X) = E[g | X] ist geschlossen bekannt (Black-Scholes-Form)
  äußere Stufe f(m) = m² (glatt), (m − k)⁺ (Knick), 1[m > k] (Sprung); Schwelle k = kmul · E g(D)
Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom    stream_seed, uniforms, normals                    SplitMix64, zählerbasiert; Box–Muller
  Wahrheit        mu_cond, k_default, f_eval, truth, var_cond, e_p0 geschlossene Formeln und Quadratur (numpy-Knoten)
  Ebenen          inner_values, level_pair                          M = 2^l innere Ziehungen; grobe Stufe antithetisch (beide Hälften), einseitig (erste Hälfte) oder unabhängig
  Schätzer        level_stats, mlmc, naive_run, single_term         Ebenenstatistik, Giles' adaptives Verfahren, geschachteltes Monte Carlo, entzerrter Ein-Term-Schätzer
Kosten zählen innere Auswertungen (2^l je äußere Ziehung).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1
SIGMA, Q = 0.5, 1.3


# --- Zufallsstrom ---------------------------------------------------------------------------------------------------------------------------

def mix_int(z):
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z):
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys):
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed, count):
    """count gleichverteilte Zahlen in (0, 1), nie 0 oder 1."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))


def normals(seed, count):
    """count Standardnormalzahlen (Box–Muller)."""
    n2 = (count + 1) // 2
    u = uniforms(seed, 2 * n2)
    r = np.sqrt(-2.0 * np.log(u[:n2]))
    z = np.concatenate([r * np.cos(2 * np.pi * u[n2:]), r * np.sin(2 * np.pi * u[n2:])])
    return z[:count]


# --- Normalverteilung, bedingte Erwartung und Wahrheit --------------------------------------------------------------------------------------

def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


_erf = np.vectorize(math.erf, otypes=[float])


def Phi_v(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, dtype=float) / math.sqrt(2.0)))


def mu_cond(x, rho: float, sigma: float, q: float):
    """μ(x) = E[(D − q)⁺ | X = x]: Black-Scholes-Form mit Lage σρx und Streuung σ√(1 − ρ²); x darf ein Feld sein."""
    m = sigma * rho * np.asarray(x, dtype=float)
    s = sigma * math.sqrt(1 - rho**2)
    d2 = (m - math.log(q)) / s
    d1 = d2 + s
    return np.exp(m + s * s / 2) * Phi_v(d1) - q * Phi_v(d2)


def k_default(sigma: float, q: float) -> float:
    """E g(D) = E (D − q)⁺ für D ~ log-normal(0, σ²) (unabhängig von ρ)."""
    a = math.log(q) / sigma
    return math.exp(sigma**2 / 2) * Phi(sigma - a) - q * Phi(-a)


def f_eval(kind: str, m, k: float):
    m = np.asarray(m)
    if kind == "glatt":
        return m * m
    if kind == "knick":
        return np.maximum(m - k, 0.0)
    if kind == "sprung":
        return (m > k).astype(float)
    raise ValueError(kind)


def x_star(k: float, rho: float, sigma: float, q: float) -> float:
    """Der Punkt mit μ(x*) = k (μ wächst in x); Bisektion. Liegt k außerhalb des Wertebereichs, geht x* an den Rand ±40."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(mu_cond(mid, rho, sigma, q)) < k:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _quad_legendre(fun, a: float, b: float, n: int = 600) -> float:
    x, w = np.polynomial.legendre.leggauss(n)
    xx = 0.5 * (b - a) * x + 0.5 * (b + a)
    return float(0.5 * (b - a) * (fun(xx) * w).sum())


def _pdf(x):
    return np.exp(-0.5 * np.asarray(x) ** 2) / math.sqrt(2 * math.pi)


def truth(kind: str, rho: float, sigma: float, q: float, k: float) -> float:
    """Exakter Wert E_X f(μ(X)): Gauß–Hermite (glatt), Gauß–Legendre ab der Knickstelle (Knick), Φ (Sprung)."""
    if kind == "glatt":
        x, w = np.polynomial.hermite_e.hermegauss(200)
        return float((mu_cond(x, rho, sigma, q) ** 2 * w).sum() / w.sum())
    xs = x_star(k, rho, sigma, q)
    if kind == "sprung":
        return 1.0 - Phi(xs)
    return _quad_legendre(lambda xx: (mu_cond(xx, rho, sigma, q) - k) * _pdf(xx), xs, xs + 14.0)


def var_cond(kind: str, rho: float, sigma: float, q: float, k: float) -> float:
    """Var f(μ(X)): die Varianz je äußerer Ziehung, wenn die innere Erwartung geschlossen bekannt ist („bedingt“)."""
    if kind == "glatt":
        x, w = np.polynomial.hermite_e.hermegauss(200)
        m = mu_cond(x, rho, sigma, q)
        return float((m**4 * w).sum() / w.sum() - ((m**2 * w).sum() / w.sum()) ** 2)
    xs = x_star(k, rho, sigma, q)
    t = truth(kind, rho, sigma, q, k)
    if kind == "sprung":
        return t * (1.0 - t)
    second = _quad_legendre(lambda xx: (mu_cond(xx, rho, sigma, q) - k) ** 2 * _pdf(xx), xs, xs + 14.0)
    return second - t * t


def e_p0(kind: str, sigma: float, q: float, k: float) -> float:
    """E[f(g(D))] (eine innere Ziehung, M = 1) in geschlossener Form; D ist marginal log-normal(0, σ²)."""
    if kind == "glatt":
        a = math.log(q) / sigma
        return math.exp(2 * sigma**2) * Phi(2 * sigma - a) - 2 * q * math.exp(sigma**2 / 2) * Phi(sigma - a) + q * q * Phi(-a)
    b = math.log(q + k) / sigma
    if kind == "knick":
        return math.exp(sigma**2 / 2) * Phi(sigma - b) - (q + k) * Phi(-b)
    return Phi(-b)


# --- Ebenen ---------------------------------------------------------------------------------------------------------------------------------

def inner_values(seed: int, N: int, M: int, rho: float, sigma: float, q: float, xseed: int | None = None):
    """g-Werte (N × M): äußere Ziehung je Zeile, M innere je Zeile. `xseed` setzt den Strom der äußeren Ziehungen (Voreinstellung: aus `seed`)."""
    X = normals(stream_seed(seed if xseed is None else xseed, 1), N)
    Z = normals(stream_seed(seed, 2), N * M).reshape(N, M)
    D = np.exp(sigma * (rho * X[:, None] + math.sqrt(1 - rho**2) * Z))
    return np.maximum(D - q, 0.0)


def level_pair(kind: str, level: int, N: int, seed: int, rho: float, sigma: float, q: float, k: float, variant: str = "anti", values=None):
    """(P_l, P_{l−1}) für N äußere Ziehungen mit M = 2^l inneren; Ebene 0: P_{−1} = 0.

    Fein: f(Mittel aller M Werte). Grob (Hälften h = M/2): antithetisch ½ (f(Mittel der ersten Hälfte) + f(Mittel der zweiten)), einseitig f(Mittel der ersten Hälfte),
    unabhängig f(Mittel von h frischen inneren Ziehungen zur selben äußeren). `values` ersetzt die g-Werte (N × M) für Tests mit festen Zahlen."""
    M = 2**level
    G = inner_values(seed, N, M, rho, sigma, q) if values is None else np.asarray(values, dtype=float)
    pf = f_eval(kind, G.mean(axis=1), k)
    if level == 0:
        return pf, np.zeros(N)
    h = M // 2
    if variant == "anti":
        pc = 0.5 * (f_eval(kind, G[:, :h].mean(axis=1), k) + f_eval(kind, G[:, h:].mean(axis=1), k))
    elif variant == "one":
        pc = f_eval(kind, G[:, :h].mean(axis=1), k)
    elif variant == "indep":
        G2 = inner_values(stream_seed(seed, 77), N, h, rho, sigma, q, xseed=seed)          # dieselben äußeren Ziehungen, frische innere
        pc = f_eval(kind, G2.mean(axis=1), k)
    else:
        raise ValueError(variant)
    return pf, pc


def cost_per_sample(level: int) -> float:
    """Innere Auswertungen je äußere Ziehung: 2^l (die antithetische grobe Stufe nutzt dieselben Werte)."""
    return float(2**level)


@lru_cache(maxsize=4096)
def level_stats(kind: str, level: int, N: int, seed: int, rho: float, sigma: float, q: float, k: float, variant: str = "anti") -> tuple:
    """(E[Y_l], Var Y_l, Var P_l, Kurtosis von Y_l, Anteil Y_l ≠ 0, E[P_l]) aus N äußeren Ziehungen; Y_l = P_l − P_{l−1}."""
    p, pc = level_pair(kind, level, N, seed, rho, sigma, q, k, variant)
    y = p - pc
    v = float(y.var())
    kurt = float(((y - y.mean()) ** 4).mean() / v**2) if v > 0 else float("nan")
    return float(y.mean()), v, float(p.var()), kurt, float((y != 0).mean()), float(p.mean())


# --- Schätzer -------------------------------------------------------------------------------------------------------------------------------

def optimal_allocation(vl, cl, eps: float) -> tuple[list, float]:
    """Optimale Zahl äußerer Ziehungen je Ebene für Gesamtvarianz ≤ ε²/2: N_l = 2 ε⁻² √(V_l/C_l) Σ_k √(V_k C_k) und die Kosten 2 ε⁻² (Σ √(V_l C_l))²."""
    tot = sum(math.sqrt(v * c) for v, c in zip(vl, cl))
    ns = [2.0 * eps**-2 * math.sqrt(v / c) * tot if c > 0 else 0.0 for v, c in zip(vl, cl)]
    return ns, 2.0 * tot * tot / eps**2


def level_sums(kind: str, level: int, n: int, seed: int, rho: float, sigma: float, q: float, k: float, variant: str = "anti") -> tuple[float, float]:
    """(Σ Y, Σ Y²) über n frische äußere Ziehungen der Ebene l (in Blöcken, damit der Speicher klein bleibt)."""
    s1 = s2 = 0.0
    done, c = 0, 0
    chunk = max(1000, 4_000_000 // 2**level)
    while done < n:
        m = min(chunk, n - done)
        p, pc = level_pair(kind, level, m, stream_seed(seed, level, 900 + c), rho, sigma, q, k, variant)
        y = p - pc
        s1 += float(y.sum())
        s2 += float((y * y).sum())
        done += m
        c += 1
    return s1, s2


def mlmc(kind: str, eps: float, seed: int, rho: float, sigma: float, q: float, k: float, variant: str = "anti", n0: int = 1000, lmax: int = 11) -> tuple[float, float, int, list]:
    """Giles' adaptives Multilevel-Verfahren (wie in Stück 12; Verfeinerung M = 2, schwache Ordnung α = 1 fest): Pilot n0 je Ebene 0 bis 2, optimale Zuteilung, Nachziehen,
    Abbruchtest max(|Ȳ_{L−1}|/2, |Ȳ_L|) ≤ ε/√2, sonst eine Ebene dazu. Rückgabe: (Schätzwert, Kosten in inneren Auswertungen, Ebenenzahl L, Ziehungen je Ebene)."""
    L = 2
    Nl, S1, S2 = [0] * (L + 1), [0.0] * (L + 1), [0.0] * (L + 1)
    dN = [n0] * (L + 1)
    calls = 0
    while True:
        for l in range(L + 1):
            if dN[l] > 0:
                s1, s2 = level_sums(kind, l, dN[l], stream_seed(seed, calls), rho, sigma, q, k, variant)
                calls += 1
                Nl[l] += dN[l]
                S1[l] += s1
                S2[l] += s2
        ml = [S1[l] / Nl[l] for l in range(L + 1)]
        Vl = [max(S2[l] / Nl[l] - ml[l] ** 2, 0.0) for l in range(L + 1)]
        Cl = [cost_per_sample(l) for l in range(L + 1)]
        Ns = [math.ceil(n) for n in optimal_allocation(Vl, Cl, eps)[0]]
        dN = [max(0, Ns[l] - Nl[l]) for l in range(L + 1)]
        if all(dN[l] <= 0.01 * Nl[l] for l in range(L + 1)):
            rem = max(abs(ml[L - 1]) / 2.0, abs(ml[L]))
            if rem > eps / math.sqrt(2) and L < lmax:
                L += 1
                Nl.append(0), S1.append(0.0), S2.append(0.0)
                dN = [0] * L + [n0]
                continue
            break
    cost = sum(Nl[l] * cost_per_sample(l) for l in range(L + 1))
    return float(sum(ml)), float(cost), L, Nl


def naive_run(kind: str, eps: float, level: int, var_p: float, seed: int, rho: float, sigma: float, q: float, k: float) -> tuple[float, float]:
    """Geschachteltes Monte Carlo mit M = 2^level inneren und N = ⌈2 Var(P)/ε²⌉ äußeren Ziehungen (Orakelwissen: M und Varianz sind gegeben). Rückgabe (Schätzwert, Kosten N·M)."""
    M = 2**level
    N = max(2, math.ceil(2 * var_p / eps**2))
    s, done, c = 0.0, 0, 0
    chunk = max(1000, 4_000_000 // M)
    while done < N:
        m = min(chunk, N - done)
        G = inner_values(stream_seed(seed, 800 + c), m, M, rho, sigma, q)
        s += float(f_eval(kind, G.mean(axis=1), k).sum())
        done += m
        c += 1
    return s / N, float(N * M)


def single_term_probs(s: float, lmax: int) -> np.ndarray:
    p = np.array([2.0 ** (-s * l) for l in range(lmax + 1)])
    return p / p.sum()


def single_term(kind: str, s: float, lmax: int, n: int, seed: int, rho: float, sigma: float, q: float, k: float, variant: str = "anti") -> tuple[float, float, float]:
    """Entzerrter Ein-Term-Schätzer Z = Y_L / p_L mit zufälliger Ebene L ~ p_l ∝ 2^(−s l), l ≤ lmax. Rückgabe (Mittel, Stichprobenvarianz, mittlere Kosten je Probe).
    Erwartungswert: E[P_lmax] (Abschneideverzerrung nur durch lmax)."""
    p = single_term_probs(s, lmax)
    u = uniforms(stream_seed(seed, 55), n)
    levels = np.searchsorted(np.cumsum(p), u, side="right").clip(0, lmax)
    zs, cost = [], 0.0
    for l in range(lmax + 1):
        nl = int((levels == l).sum())
        if nl == 0:
            continue
        pf, pc = level_pair(kind, l, nl, stream_seed(seed, 56, l), rho, sigma, q, k, variant)
        zs.append((pf - pc) / p[l])
        cost += nl * cost_per_sample(l)
    z = np.concatenate(zs)
    return float(z.mean()), float(z.var(ddof=1)), cost / n
