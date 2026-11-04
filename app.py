"""Geschachtelte Erwartungen: was der Erwartungswert über einen Erwartungswert kostet - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 13 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Eine äußere Erwartung über eine nichtlineare Funktion einer inneren bedingten Erwartung lässt sich nicht mit einer Ziehung je Stufe schätzen.
Geschachteltes Monte Carlo kostet wegen der Verzerrung der inneren Mittelung ε⁻³. Multilevel mit M_l = 2^l inneren Ziehungen und antithetischer Kopplung senkt das bei glatter oder geknickter äußerer
Funktion auf etwa ε⁻², beim Sprung nicht; bei lockerer Toleranz und schwachem Signal gewinnt das einfache Schachteln, und ist die innere Erwartung bekannt, ist beides Verschwendung.

Lauffähig mit: streamlit run app.py
"""

import dataclasses

import streamlit as st

import ger_constants as C
import ger_evaluation as ev
from ger_evaluation import Settings, bias_at, de, de_e, eps_text, ke
from ger_layout import dunkles_design, kopf_css
from ger_pdf_export import build_pdf
from ger_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from ger_scenario import check_szenario
from ger_visualization import build_cost, build_cost_norm, build_levels, build_run, build_shares, build_signal, build_single

st.set_page_config(page_title="Geschachtelte Erwartungen: was der Erwartungswert über einen Erwartungswert kostet – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Ebenen messen …")
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Schätzungen laufen lassen …")
def _run(settings):
    return ev.run_experiment(settings)


@st.cache_data(show_spinner="Signal und Schwelle messen …")
def _signal(settings):
    return ev.signal_experiment(settings)


@st.cache_data(show_spinner="Entzerrten Schätzer laufen lassen …")
def _single(settings):
    return ev.single_term_experiment(settings)


@st.cache_data(show_spinner=False)
def _custom(settings, rho, sigma, q, kmul):
    sc = check_szenario(rho, sigma, q, kmul)
    return sc, ev.analyse(settings, sc)


def _rho_label(r):
    return f"{r:g}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in ger_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🪆 Geschachtelte Erwartungen: was der Erwartungswert über einen Erwartungswert kostet")
st.markdown(
    """
Wer **E_X[ f( E_Y[ g(X, Y) | X ] ) ]** braucht (den Wert einer Information, ein Risikomaß, einen Sensitivitätsindex), kann die innere Erwartung nicht durch eine Ziehung ersetzen, sobald f nicht linear ist: das **geschachtelte Monte Carlo** zieht N äußere und je M innere Werte und wertet f am inneren Mittel aus.
Die Verzerrung fällt nur wie 1/M, also braucht man M ∝ ε⁻¹ und N ∝ ε⁻²: Kosten ε⁻³. **Multilevel** nimmt M_l = 2^l innere Ziehungen je Ebene und koppelt die Ebenen **antithetisch** (die grobe Stufe nutzt beide Hälften der inneren Ziehungen): die Differenzen streuen wenig, und beim Mittelwert sinken die Kosten auf etwa ε⁻², beim Knick auf wenig mehr.
Bei einem **Sprung** gelingt das nicht; bei lockerer Toleranz und schwachem Signal gewinnt das einfache Schachteln, und ist die innere Erwartung bekannt, ist beides Verschwendung.

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert das Schachteln und seine Multilevel-Version", expanded=True):
    st.markdown(
        """
Die Nachfrage D = exp(σ(ρX + √(1−ρ²)Y)) hängt von einem **Signal** X (außen) und einem **inneren Rauschen** Y ab; die innere Ausgabe ist der Fehlbestand g = (D − q)⁺. Die bedingte Erwartung μ(X) = E[g | X] ist hier geschlossen bekannt (so lässt sich alles gegen die Wahrheit prüfen); die Schätzer benutzen sie nicht. Gesucht ist E_X f(μ(X)) mit f = m² (Mittelwert), (m − k)⁺ (Knick) oder 1[m > k] (Sprung), Schwelle k beim Mittel E g.

1. **Geschachtelt:** N äußere Ziehungen X_i, je M innere Y_ij; Schätzer f(Mittel_j g(X_i, Y_ij)). Das Mittel streut um μ(X_i), und f ist nichtlinear: die Verzerrung ist ≈ c/M (beim Quadrat exakt E Var(g | X)/M). Für Fehler ε braucht man M ≈ ε⁻¹ und N ≈ ε⁻² äußere Ziehungen.
2. **Multilevel:** Ebene l hat M = 2^l innere Ziehungen. Y_l = f(Mittel aller M) − ½(f(Mittel der ersten Hälfte) + f(Mittel der zweiten Hälfte)) (**antithetisch**). Die Teleskopsumme E[P_L] = E[P_0] + Σ E[Y_l] wird mit eigenen äußeren Ziehungen je Ebene geschätzt; je kleiner V_l = Var Y_l, desto weniger Ziehungen auf den teuren Ebenen.
3. **Andere Kopplungen** zum Vergleich: **einseitig** (grob = f(Mittel der ersten Hälfte)) und **unabhängig** (grob = f(Mittel frischer innerer Ziehungen)). Nur die antithetische hebt die Rate.
4. **Bedingt:** wäre μ(X) bekannt, würde man f(μ(X)) direkt mitteln; im Kostenmodell (eine Auswertung je äußere Ziehung) ist das die Grenze nach unten für jedes Schachteln.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Äußere Funktion und Kopplung**")
    ausgabe = st.selectbox("Äußere Funktion f(m)", options("ausgabe_select"), format_func=C.AUSGABEN.get, key="ausgabe_select",
                           help="Mittelwert f(m) = m² (glatt), Knick f(m) = (m − k)⁺, Sprung f(m) = 1[m > k]; die Schwelle k liegt beim Mittel E g(D) = 0,1655.")
    kopplung = st.selectbox("Kopplung der Ebenen", options("kopplung_select"), format_func=C.KOPPLUNGEN.get, key="kopplung_select",
                            help="Bestimmt Kosten und Läufe (Experimente 2 bis 5). Experiment 1 zeigt alle drei nebeneinander.")
    rho = st.select_slider("Signalstärke ρ", options("rho_select"), key="rho_select", format_func=_rho_label, help="Wie stark das Signal X die Nachfrage bestimmt. Bei kleinem ρ ist das innere Rauschen groß.")
    st.markdown("**Toleranz und Ebenen**")
    eps = st.select_slider("Toleranz ε (Wurzel des mittleren quadratischen Fehlers)", options("eps_select"), key="eps_select", format_func=eps_text, help="Die gelaufenen Schätzungen (Experimente 3 und 5) zielen auf diese Toleranz; die Kostentabelle zeigt alle.")
    lmax = st.select_slider("Größte gemessene Ebene", options("lmax_select"), key="lmax_select", help="Ebene l hat 2^l innere Ziehungen. Mehr Ebenen kosten Rechenzeit, geben aber eine verlässlichere Rate.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"ausgabe_select": ausgabe, "kopplung_select": kopplung, "rho_select": float(rho), "eps_select": float(eps), "lmax_select": int(lmax), "seed_input": int(seed)})

settings = Settings(str(ausgabe), str(kopplung), float(rho), float(eps), int(lmax), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

row = ev.cost_row(a, settings.eps)
st.markdown("## 🎯 Ergebnis")
st.caption(f"{a.label}; Nachfrage mit σ = {de(a.sigma, 1)}, ρ = {_rho_label(a.rho)}, q = 1,3, k = {de(a.k, 4)}; Ebenen 0 bis {a.lmax}, {de(a.n, 0)} äußere Ziehungen je Ebene; Kosten in inneren Auswertungen, idealisiert (Ebenenvarianzen und Verzerrung bekannt).")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Exakter Wert", de(a.truth, 4), help="Quadratur der bedingten Erwartung (Gauß–Hermite, Gauß–Legendre, Φ), keine Simulation.")
m2.metric("Innere Ziehungen M", str(2 ** row["L"]), delta=f"Verzerrung ≈ {de(a.c, 4)}/M", delta_color="off", help="Kleinste Zweierpotenz mit Verzerrung ≤ ε/√2 beim geschachtelten Monte Carlo.")
m3.metric("Rate β der Ebenenvarianz", ev.dn(a.beta, 2), delta=ev.regime(a.beta).split(":")[0], delta_color="off", help="V_l ∝ 2^(−β l), Regression über die Ebenen ab 3. Kostenrate je Ebene γ = 1.")
m4.metric("Geschachtelt / Multilevel", f"{ev.dn(row['ratio'], 2)}×", delta=f"bei ε = {eps_text(settings.eps)}", delta_color="off", help="Verhältnis der idealisierten Kosten; unter 1 ist Multilevel teurer.")
st.plotly_chart(build_levels(a), width="stretch", key="main_chart")
st.caption("Varianz der Ebenendifferenz gegen die Ebene, logarithmisch, für die drei Kopplungen. Nur die antithetische fällt mit der besten Rate; einseitig und unabhängig fallen langsamer. Gepunktet: Steigungen −1 und −2.")

if st.button("📄 PDF der Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="geschachtelte-erwartungen.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Lohnt Multilevel hier, und wie viel spart es?")
kind, text = ev.verdict_main(a, settings)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Die Ebenen")
st.caption(f"Für jede Ebene {de(a.n, 0)} äußere Ziehungen mit M = 2^l inneren; gezeigt sind Mittel und Varianz der Differenz der gewählten Kopplung, die Varianz der beiden anderen Kopplungen und die Verzerrung der Ebene gegen den exakten Wert.")
st.dataframe(
    {"Ebene l": [r["l"] for r in a.levels], "M = 2^l": [2 ** r["l"] for r in a.levels], "E[Y_l]": [de_e(r["mean"]) if r["l"] > 0 else "–" for r in a.levels],
     f"V_l {C.KOPPLUNG_KURZ[a.variant]}": [de_e(r["var"]) for r in a.levels], "V_l / V_(l−1)": [de(r["var"] / a.levels[i - 1]["var"], 2) if i > 0 and a.levels[i - 1]["var"] > 0 else "–" for i, r in enumerate(a.levels)],
     **{f"V_l {C.KOPPLUNG_KURZ[v]}": [de_e(x) if i > 0 else "–" for i, x in enumerate(a.var_by_variant[v])] for v in C.KOPPLUNGEN if v != a.variant},
     "Kurtosis": [de(r["kurt"], 0) if r["l"] > 0 else "–" for r in a.levels], "Verzerrung E[P_l] − exakt": [de_e(bias_at(a, r["l"])) for r in a.levels]}, hide_index=True, width="stretch", key="level_table")
kind, text = ev.verdict_levels(a)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Kosten gegen Toleranz")
st.caption("Idealisierte Kosten für einen mittleren quadratischen Fehler ≤ ε² (Verzerrung² ≤ ε²/2, Varianz ≤ ε²/2), aus den gemessenen Ebenenvarianzen; jenseits der gemessenen Ebenen mit der Rate fortgesetzt. Geschachtelt: M = 2^L innere und N ≈ 2 Var(P_L)/ε² äußere Ziehungen. Multilevel: optimale Zuteilung. Bedingt: die innere Erwartung wäre bekannt.")
table = ev.cost_table(a)
c1, c2 = st.columns(2)
c1.plotly_chart(build_cost(table), width="stretch", key="cost_chart")
c2.plotly_chart(build_cost_norm(table), width="stretch", key="cost_norm_chart")
st.dataframe(
    {"Toleranz ε": [eps_text(r["eps"]) for r in table], "Ebene L": [r["L"] for r in table], "geschachtelt": [ke(r["nested"]) for r in table], "Multilevel": [ke(r["ml"]) for r in table], "bedingt": [ke(r["cond"]) for r in table],
     "geschachtelt / Multilevel": [ev.dn(r["ratio"], 2) for r in table], "ε² · geschachtelt": [de(r["eps2_nested"], 2) for r in table], "ε² · Multilevel": [de(r["eps2_ml"], 2) for r in table]},
    hide_index=True, width="stretch", key="cost_table")
st.plotly_chart(build_shares(row), width="stretch", key="shares_chart")
st.caption(f"Kostenanteil je Ebene des Multilevel-Schätzers bei ε = {eps_text(settings.eps)} (Ebene 0 bis {row['L']}): bei β > γ liegt er auf den groben Ebenen, bei β < γ auf den feinen.")
kind, text = ev.verdict_cost(a)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Gelaufene Schätzungen")
st.caption(f"Giles' adaptives Verfahren (Pilot {de(C.PILOT, 0)} äußere Ziehungen auf den Ebenen 0 bis 2, Zuteilung aus den geschätzten Varianzen, neue Ebene bei nicht bestandenem Abbruchtest) gegen geschachteltes Monte Carlo mit Orakelwissen über M und Varianz, {C.RUN_REPS} Wiederholungen bei ε = {eps_text(settings.eps)}.")
if st.button("▶ Messung starten", key="run_gelaufen"):
    st.session_state["run_done"] = True
if st.session_state.get("run_done"):
    ex3 = _run(settings)
    st.plotly_chart(build_run(ex3), width="stretch", key="run_chart")
    st.dataframe({"Verfahren": ["geschachtelt (Orakelwissen)", "Multilevel (Giles)"], "RMSE / ε": [de(ex3["nested"]["rmse"] / ex3["eps"], 2), de(ex3["ml"]["rmse"] / ex3["eps"], 2)],
                  "mittlere Kosten": [ke(ex3["nested"]["cost"]), ke(ex3["ml"]["cost"])], "Fehler über ε": [ev.prozent(ex3["nested"]["over"], 0), ev.prozent(ex3["ml"]["over"], 0)]}, hide_index=True, width="stretch", key="run_table")
    kind, text = ev.verdict_run(ex3)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Signal und Schwelle")
st.caption("Wie stark das Signal ρ und die Lage der Schwelle k den Gewinn bestimmen: idealisiertes Kostenverhältnis geschachtelt/Multilevel bei ε = 0,001 und 0,00025 für alle drei Signalstärken und für k = 0,5 / 1 / 2 mal Mittel (bei der gewählten Signalstärke). Gewählte Funktion und Kopplung gelten.")
if st.button("▶ Messung starten", key="run_signal"):
    st.session_state["signal_done"] = True
if st.session_state.get("signal_done"):
    ex4 = _signal(settings)
    st.plotly_chart(build_signal(ex4), width="stretch", key="signal_chart")
    labels = [f"ρ = {_rho_label(r['rho'])}" for r in ex4["rho_rows"]] + [f"k = {de(r['factor'], 1)}·Mittel" for r in ex4["k_rows"]]
    rows4 = ex4["rho_rows"] + ex4["k_rows"]
    st.dataframe({"Einstellung": labels, "Rate β": [ev.dn(r["beta"], 2) for r in rows4], "Verzerrungskonstante c": [de(r["c"], 4) for r in rows4],
                  **{f"geschachtelt / Multilevel bei ε = {eps_text(e)}": [ev.dn(r["ratios"][i], 2) for r in rows4] for i, e in enumerate(ex4["eps"])}}, hide_index=True, width="stretch", key="signal_table")
    kind, text = ev.verdict_signal(ex4)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Der entzerrte Schätzer")
st.caption("Ein-Term-Schätzer: eine zufällige Ebene L mit Wahrscheinlichkeit p_l ∝ 2^(−s·l) wird gezogen und Y_L / p_L gemittelt. Kein Abbruchfehler, aber die Varianz kann mit der Abschneideebene unbeschränkt wachsen. Gezeigt: Varianz gegen die Abschneideebene für drei s (aus den gemessenen zweiten Momenten) und ein gelaufener Lauf.")
if st.button("▶ Messung starten", key="run_single"):
    st.session_state["single_done"] = True
if st.session_state.get("single_done"):
    ex5 = _single(settings)
    st.plotly_chart(build_single(ex5), width="stretch", key="single_chart")
    lms = [r["lmax"] for r in next(iter(ex5["curves"].values()))]
    st.dataframe({"Abschneideebene": lms, **{f"Varianz, s = {de(s, 1)}": [de(r["var"], 2) for r in cv] for s, cv in ex5["curves"].items()}, **{f"Varianz·Kosten, s = {de(s, 1)}": [de(r["work"], 2) for r in cv] for s, cv in ex5["curves"].items()}},
                 hide_index=True, width="stretch", key="single_table")
    kind, text = ev.verdict_single(ex5)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigenes Modell"):
    st.caption("Eigene Signalstärke ρ, Streuung σ, Schwelle q der inneren Ausgabe und Schwellenfaktor k (Vielfaches des Mittels E g). Es gelten die gewählte Funktion, Kopplung, Toleranz und Ebenenzahl; die exakte Größe bleibt geschlossen bekannt.")
    s1, s2, s3, s4 = st.columns(4)
    rho_in = s1.number_input("Signalstärke ρ", C.CUSTOM_RHO_MIN, C.CUSTOM_RHO_MAX, 0.6, 0.05, key="eigen_rho", format="%.2f")
    sigma_in = s2.number_input("Streuung σ", C.CUSTOM_SIGMA_MIN, C.CUSTOM_SIGMA_MAX, 0.5, 0.05, key="eigen_sigma", format="%.2f")
    q_in = s3.number_input("Schwelle q", C.CUSTOM_Q_MIN, C.CUSTOM_Q_MAX, 1.3, 0.05, key="eigen_q", format="%.2f")
    k_in = s4.number_input("Schwellenfaktor k", C.CUSTOM_K_MIN, C.CUSTOM_K_MAX, 1.0, 0.25, key="eigen_k", format="%.2f")
    try:
        csc, ca = _custom(settings, float(rho_in), float(sigma_in), float(q_in), float(k_in))
    except ValueError as err:
        st.error(str(err))
    else:
        st.plotly_chart(build_levels(ca), width="stretch", key="custom_levels")
        st.plotly_chart(build_cost(ev.cost_table(ca)), width="stretch", key="custom_cost")
        kind, text = ev.verdict_main(ca, settings)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die innere Erwartung ist nur durch Simulation zu haben** | Ist sie geschlossen bekannt, genügt eine Auswertung je äußerer Ziehung: 14- bis 640-mal billiger als Multilevel (Experiment 2, Spalte »bedingt«). | Stück 6 |
| **Die Ebenen sind antithetisch gekoppelt** | Mit einseitiger oder unabhängiger Kopplung fällt die Ebenenvarianz nur mit etwa 2^(−l) (gemessen 1,0 bis 1,2) statt mit 2^(−2,06 l) (Mittelwert) bzw. 2^(−1,33 l) (Knick); der Gewinn schrumpft oder verschwindet (Experiment 1, Preset »Ohne Antithetik«). | Stück 2 |
| **Die äußere Funktion ist glatt oder geknickt** | Bei einem Sprung fällt die Ebenenvarianz nur mit etwa 2^(−l/2): Multilevel kostet bei den gezeigten Toleranzen und ρ = 0,6 im Idealfall mehr als einfaches Schachteln, bei sehr enger Toleranz oder starkem Signal nicht mehr (Experiment 4, Preset »Sprung«). | – |
| **Die Toleranz ist eng genug** | Bei lockerem ε kostet der Pilot mehr, als er spart (Experiment 3). | – |
| **Das Signal ist stark genug** | Bei kleiner Signalstärke dominiert die Ebene 0, und der Gewinn schrumpft (Experiment 4, Preset »Schwaches Signal«). | – |
| **Der Abbruchfehler stört** | Der entzerrte Schätzer verzichtet auf die Ebenenwahl, braucht dafür γ < s < β; beim Sprung wächst seine Varianz für jedes s (Experiment 5). | Stück 12 |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Das Modell (log-normale Nachfrage mit Signal und innerem Rauschen) ist **erfunden**, und seine bedingte Erwartung ist **geschlossen bekannt**; sie dient nur als Orakel für die Wahrheit, die Schätzer ziehen innere Werte. Die Zufallszahlen kommen aus festen Strömen (SplitMix64, Box–Muller), nie aus einem einzelnen Seed.
Kosten zählen innere Auswertungen, nicht Laufzeit; die äußere Ziehung zählt nicht mit.

**Grenzen.** Es gibt nur **ein Modell** und drei äußere Funktionen; **keinen Entscheidungsfall** (Minimum über mehrere Aktionen wie beim Wert der Information), keine mehrdimensionalen Eingaben, keine Tiefe über zwei Stufen, keine adaptiven inneren Stichproben und kein Quasi-Monte-Carlo.
Giles' Verfahren läuft mit fest angesetzter schwacher Ordnung 1; die Rate β wird für die Kostenrechnung aus den Ebenen ab 3 geschätzt. **Geschachteltes Monte Carlo ist mit Orakelwissen gerechnet** (es kennt M und die Varianz) und wird dadurch begünstigt.

**Bekannte Probleme und Unsicherheiten.** Die Ebenenvarianz beim Sprung beruht auf seltenen Ereignissen und ist mit 40.000 äußeren Ziehungen nur grob messbar (hohe Kurtosis); die Verzerrungskonstante rauscht, die gemessene Verzerrung beim Sprung liegt schon ab Ebene 3 im Rauschen (eine Kontrollrechnung mit gemeinsamen äußeren Ziehungen, nicht Teil der Demo, ergibt etwa 2^(−1,6) je Ebene).
Die Fortsetzung der Varianzen jenseits der gemessenen Ebenen nimmt die geschätzte Rate an. Die Ein-Term-Varianzen stammen aus den gemessenen zweiten Momenten; der Lauf in Experiment 5 prüft sie an einem Fall, und bei wachsender Varianz ist die Stichprobenvarianz selbst unzuverlässig.
Die Rate ε^(−2−1/α) und die Sätze zu Multilevel stehen im Mathematik-Expander aus dem Gedächtnis; die Quellen (Hironaka, Giles, Goda und Thom; Rainforth u. a.) sind nur im Abstract eingesehen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** $\theta=\mathbb E_X\bigl[f(\mu(X))\bigr]$ mit $\mu(X)=\mathbb E[g(X,Y)\mid X]$. Hier $D=\exp\bigl(\sigma(\rho X+\sqrt{1-\rho^2}\,Y)\bigr)$, $g=(D-q)^+$, $f(m)\in\{m^2,(m-k)^+,\mathbf 1[m>k]\}$, $k=\mathbb E g(D)$ (kmul = 1).

**Bedingte Erwartung.** Mit $m=\sigma\rho x$, $s=\sigma\sqrt{1-\rho^2}$, $d_2=(m-\ln q)/s$, $d_1=d_2+s$: $\mu(x)=e^{m+s^2/2}\Phi(d_1)-q\Phi(d_2)$. $\mathbb E g(D)=e^{\sigma^2/2}\Phi(\sigma-a)-q\Phi(-a)$ mit $a=\ln q/\sigma$.
Exakt: $\mathbb E\mu(X)^2$ per Gauß–Hermite; $\int_{x^*}^\infty(\mu-k)\varphi$ per Gauß–Legendre mit $\mu(x^*)=k$; $\mathbb P(\mu(X)>k)=1-\Phi(x^*)$.

**Geschachtelt.** $\hat\theta=\frac1N\sum_if\bigl(\bar g_i\bigr)$, $\bar g_i=\frac1M\sum_jg(X_i,Y_{ij})$. Verzerrung $\approx c\,M^{-\alpha}$ (hier $\alpha=1$); beim Quadrat exakt $\mathbb E\operatorname{Var}(g\mid X)/M$. Für MSE $\le\varepsilon^2$: $M\approx\varepsilon^{-1/\alpha}$, $N\approx\varepsilon^{-2}$, Kosten $\varepsilon^{-2-1/\alpha}$.

**Multilevel.** $M_l=2^l$, $\bar g^{(l)}$ das Mittel aller, $\bar g^{(l),a}$ und $\bar g^{(l),b}$ die Mittel der Hälften. $Y_l=f(\bar g^{(l)})-\tfrac12\bigl(f(\bar g^{(l),a})+f(\bar g^{(l),b})\bigr)$, $Y_0=f(g)$. $\mathbb E[P_L]=\mathbb E[Y_0]+\sum_{l\ge1}\mathbb E[Y_l]$.
Beim Quadrat: $Y_l=-(\bar g^a-\bar g^b)^2/4$, $\mathbb E Y_l=-\mathbb E\operatorname{Var}(g\mid X)/2^l$. Optimale Zuteilung wie in Stück 12 mit $C_l=2^l$: $N_l=2\varepsilon^{-2}\sqrt{V_l/C_l}\sum_k\sqrt{V_kC_k}$, Kosten $2\varepsilon^{-2}(\sum_l\sqrt{V_lC_l})^2$.
Dreifallunterscheidung (aus dem Gedächtnis, nicht aus dem Volltext; die Demo misst sie): $V_l\propto2^{-\beta l}$, $C_l\propto2^{\gamma l}$ mit $\gamma=1$: $\beta>\gamma$ gibt $\varepsilon^{-2}$, $\beta=\gamma$ gibt $\varepsilon^{-2}$ mal Logarithmen, $\beta<\gamma$ gibt $\varepsilon^{-2-(\gamma-\beta)/\alpha}$.

**Warum die Antithetik.** Der Fehler der inneren Mittelung hat in erster Näherung ein Vorzeichen, das sich zwischen den Hälften aufhebt; bei glattem $f$ bleibt nur ein Term zweiter Ordnung ($\beta=2$), bei Lipschitz-stetigem $f$ (Theorie $\beta=3/2$, gemessen $\beta\approx1{,}3$ bis $1{,}4$), beim Indikator nur $\beta\approx1/2$.

**Bedingt.** Wäre $\mu$ bekannt: $N=2\operatorname{Var}f(\mu(X))\,\varepsilon^{-2}$ Ziehungen zu je einer Auswertung.

**Ein-Term-Schätzer.** $L\sim p$, $Z=Y_L/p_L$: $\mathbb E Z=\sum_l\mathbb E Y_l$, $\operatorname{Var}Z=\sum_l\mathbb E[Y_l^2]/p_l-\bigl(\sum_l\mathbb EY_l\bigr)^2$, erwartete Kosten $\sum_lp_lC_l$; endliche Varianz und Kosten verlangen $\gamma<s<\beta$.

**Literatur (aus dem Gedächtnis, wenn nicht anders vermerkt).** Hironaka, Giles, Goda und Thom 2019, „Multilevel Monte Carlo estimation of the expected value of sample information“ (arXiv 1909.00549, nur Abstract eingesehen); Rainforth, Cornish, Yang, Warrington und Wood 2018, „On nesting Monte Carlo estimators“ (ICML; arXiv 1709.06181, nur Abstract eingesehen); Giles 2015, „Multilevel Monte Carlo methods“ (Acta Numerica); Rhee und Glynn 2015 (Ein-Term-Schätzer).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
