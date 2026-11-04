"""Das Szenario: Signalstärke ρ, Streuung σ, Schwelle q der inneren Ausgabe und die Schwelle k der äußeren Funktion als Vielfaches des Mittels E g(D)."""

from __future__ import annotations

from dataclasses import dataclass

import ger_constants as C


@dataclass(frozen=True)
class Szenario:
    rho: float
    sigma: float
    q: float
    kmul: float = 1.0


def standard_szenario(rho: float) -> Szenario:
    return Szenario(rho, C.SIGMA, C.Q, 1.0)


def _bad(x) -> bool:
    return x is None or x != x or x in (float("inf"), float("-inf"))


def check_szenario(rho, sigma, q, kmul) -> Szenario:
    """Prüft die Eingaben der Spielwiese; ValueError mit lesbarer Meldung."""
    for name, x in (("Signalstärke ρ", rho), ("Streuung σ", sigma), ("Schwelle q", q), ("Schwellenfaktor k", kmul)):
        if _bad(x):
            raise ValueError(f"{name} ist keine Zahl.")
    for name, x, lo, hi in (("Die Signalstärke ρ", rho, C.CUSTOM_RHO_MIN, C.CUSTOM_RHO_MAX), ("Die Streuung σ", sigma, C.CUSTOM_SIGMA_MIN, C.CUSTOM_SIGMA_MAX),
                            ("Die Schwelle q", q, C.CUSTOM_Q_MIN, C.CUSTOM_Q_MAX), ("Der Schwellenfaktor k", kmul, C.CUSTOM_K_MIN, C.CUSTOM_K_MAX)):
        if not lo <= x <= hi:
            raise ValueError(f"{name} muss zwischen {lo} und {hi} liegen.".replace(".", ","))
    return Szenario(float(rho), float(sigma), float(q), float(kmul))
