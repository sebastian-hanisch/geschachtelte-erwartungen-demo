"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

from ger_evaluation import Analysis, Settings, cost_table, de, dn, eps_text, ke, verdict_main

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "sigma", "μ": "mu", "ρ": "rho", "α": "alpha", "β": "beta", "γ": "gamma", "θ": "theta", "τ": "tau", "ε": "eps", "∞": "unendlich", "Σ": "Summe", "δ": "delta", "Δ": "Delta",
            "½": "1/2", "̂": "", "ᵢ": "_i", "∫": "Integral", "⁺": "+", "′": "'", "⁻": "-", "¹": "1", "≠": "!=", "∝": "~"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Geschachtelte Erwartungen: was der Erwartungswert über einen Erwartungswert kostet"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Nachfrage D = exp(sigma (rho X + Wurzel(1 - rho^2) Y)) mit X (aussen) und Y (innen) standardnormal; sigma = {de(a.sigma, 2)}, rho = {de(a.rho, 2)}, innere Ausgabe (D - q)+ mit q = {de(a.q, 2)}, "
                            f"äußere Funktion {a.label}, Schwelle k = {de(a.k, 4)}. Ebenen 0 bis {a.lmax}, {de(a.n, 0)} äußere Ziehungen je Ebene, Strom {settings.seed}. Die exakte Größe ist {de(a.truth, 5)}. "
                            f"Kosten in inneren Auswertungen, idealisiert (bekannte Ebenenvarianzen und Verzerrung)."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    heads = ("Toleranz", "Ebene L", "geschachtelt", "Multilevel", "bedingt", "Verhältnis")
    widths = (24, 20, 38, 38, 38, 28)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(heads, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for r in cost_table(a):
        for text, w in zip((eps_text(r["eps"]), str(r["L"]), ke(r["nested"]), ke(r["ml"]), ke(r["cond"]), dn(r["ratio"], 2)), widths):
            pdf.cell(w, 6, _s(text), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(verdict_main(a, settings)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: geschachtelte-erwartungen-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
