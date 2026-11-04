"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import ger_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(exponentformat="power")
    return lock_axes(fig)


def _eps_axis(fig, xs):
    fig.update_xaxes(title_text="Toleranz ε", type="log", autorange="reversed", tickvals=xs, ticktext=[f"{x:g}".replace(".", ",") for x in xs])


def build_levels(a):
    """Varianz der Ebenendifferenz gegen die Ebene (logarithmisch) für alle drei Kopplungen, dazu |E[Y_l]| der gewählten und Steigungen −1 und −2 als Hilfslinien."""
    ls = [r["l"] for r in a.levels]
    dash = {"anti": "solid", "one": "dash", "indep": "dot"}
    fig = go.Figure()
    for v in C.KOPPLUNGEN:
        ys = [x if x > 0 else None for x in a.var_by_variant[v]]
        fig.add_trace(go.Scatter(x=ls, y=ys, mode="lines+markers", name=f"V_l {C.KOPPLUNG_KURZ[v]}", line=dict(color=C.COLORS[v], dash=dash[v])))
    fig.add_trace(go.Scatter(x=ls[1:], y=[abs(r["mean"]) if r["mean"] != 0 else None for r in a.levels[1:]], mode="lines+markers", name="|E[Y_l]|", line=dict(color=C.COLORS["bias"], dash="dashdot")))
    base = a.var_by_variant["anti"][1] if a.var_by_variant["anti"][1] > 0 else 1.0
    for slope, name in ((1, "Steigung −1"), (2, "Steigung −2")):
        fig.add_trace(go.Scatter(x=ls[1:], y=[base * 2.0 ** (-slope * (l - 1)) for l in ls[1:]], mode="lines", name=name, line=dict(color=C.COLORS["ref"], dash="dot", width=1)))
    fig.update_xaxes(title_text="Ebene l (2^l innere Ziehungen)", tickvals=ls)
    fig.update_yaxes(title_text="Varianz bzw. Mittel der Differenz", type="log")
    return _base(fig, 400)


def build_cost(rows):
    """Idealisierte Kosten (innere Auswertungen) gegen die Toleranz ε, doppelt logarithmisch: geschachtelt, Multilevel, bedingt (innere Erwartung bekannt)."""
    fig = go.Figure()
    xs = [r["eps"] for r in rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["nested"] for r in rows], mode="lines+markers", name="geschachtelt", line=dict(color=C.COLORS["nested"])))
    fig.add_trace(go.Scatter(x=xs, y=[r["ml"] for r in rows], mode="lines+markers", name="Multilevel", line=dict(color=C.COLORS["ml"])))
    fig.add_trace(go.Scatter(x=xs, y=[r["cond"] for r in rows], mode="lines+markers", name="bedingt (innere Erwartung bekannt)", line=dict(color=C.COLORS["cond"], dash="dash")))
    _eps_axis(fig, xs)
    fig.update_yaxes(title_text="Kosten (innere Auswertungen)", type="log")
    return _base(fig, 340)


def build_cost_norm(rows):
    """ε² · Kosten gegen ε: bei Sättigung (Multilevel mit β > γ) waagerecht, geschachtelt wächst mit 1/ε."""
    fig = go.Figure()
    xs = [r["eps"] for r in rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["eps2_nested"] for r in rows], mode="lines+markers", name="geschachtelt", line=dict(color=C.COLORS["nested"])))
    fig.add_trace(go.Scatter(x=xs, y=[r["eps2_ml"] for r in rows], mode="lines+markers", name="Multilevel", line=dict(color=C.COLORS["ml"])))
    fig.add_trace(go.Scatter(x=xs, y=[r["eps"] ** 2 * r["cond"] for r in rows], mode="lines+markers", name="bedingt", line=dict(color=C.COLORS["cond"], dash="dash")))
    _eps_axis(fig, xs)
    fig.update_yaxes(title_text="ε² · Kosten", type="log")
    return _base(fig, 340)


def build_shares(row):
    """Kostenanteil je Ebene des Multilevel-Schätzers bei der gewählten Toleranz."""
    fig = go.Figure(go.Bar(x=list(range(len(row["shares"]))), y=[100 * s for s in row["shares"]], marker_color=C.COLORS["ml"], name="Kostenanteil"))
    fig.update_xaxes(title_text="Ebene l", tickvals=list(range(len(row["shares"]))))
    fig.update_yaxes(title_text="Anteil an den Kosten (%)", range=[0, 100])
    fig.update_layout(showlegend=False)
    return _base(fig, 260)


def build_run(ex):
    """Mittlere Kosten der gelaufenen Schätzungen: geschachtelt (Orakel) gegen Multilevel."""
    fig = go.Figure(go.Bar(x=["geschachtelt (Orakelwissen)", "Multilevel (Giles)"], y=[ex["nested"]["cost"], ex["ml"]["cost"]], marker_color=[C.COLORS["nested"], C.COLORS["ml"]],
                           text=[f"RMSE/ε = {ex['nested']['rmse'] / ex['eps']:.2f}".replace(".", ","), f"RMSE/ε = {ex['ml']['rmse'] / ex['eps']:.2f}".replace(".", ",")], textposition="outside"))
    fig.update_yaxes(title_text="mittlere Kosten (innere Auswertungen)", type="log")
    fig.update_layout(showlegend=False)
    return _base(fig, 320)


def build_signal(ex):
    """Kostenverhältnis geschachtelt/Multilevel (idealisiert) gegen die Signalstärke ρ und gegen die Schwelle k, bei zwei Toleranzen; 1 ist Gleichstand."""
    fig = go.Figure()
    xs = [f"ρ = {r['rho']:g}".replace(".", ",") for r in ex["rho_rows"]] + [f"k = {r['factor']:g}·Mittel".replace(".", ",") for r in ex["k_rows"]]
    rows = ex["rho_rows"] + ex["k_rows"]
    palette = ["#2F6B65", "#4c78a8"]
    for i, eps in enumerate(ex["eps"]):
        fig.add_trace(go.Bar(x=xs, y=[max(r["ratios"][i], 1e-3) for r in rows], name=f"ε = {eps:g}".replace(".", ","), marker_color=palette[i]))
    fig.add_hline(y=1.0, line=dict(color=C.COLORS["ref"], dash="dot"), annotation_text="Gleichstand", annotation_position="top left")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="geschachtelt / Multilevel", type="log")
    return _base(fig, 340)


def build_single(ex):
    """Varianz des entzerrten Ein-Term-Schätzers gegen die Abschneideebene für drei Abfallexponenten."""
    fig = go.Figure()
    palette = {1.0: C.COLORS["anti"], 1.5: C.COLORS["cond"], 2.0: C.COLORS["one"]}
    for s, cv in ex["curves"].items():
        fig.add_trace(go.Scatter(x=[r["lmax"] for r in cv], y=[r["var"] if np.isfinite(r["var"]) and r["var"] > 0 else None for r in cv], mode="lines+markers",
                                 name=f"p_l ∝ 2^(−{s:g} l)".replace(".", ","), line=dict(color=palette[s])))
    fig.update_xaxes(title_text="Abschneideebene", tickvals=[r["lmax"] for r in next(iter(ex["curves"].values()))])
    fig.update_yaxes(title_text="Varianz je Probe", type="log")
    return _base(fig, 340)
