# Geschachtelte Erwartungen: was der Erwartungswert über einen Erwartungswert kostet – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-geschachtelte-erwartungen-demo.streamlit.app/)**

Dreizehntes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [multilevel-monte-carlo-demo](https://github.com/sebastian-hanisch/multilevel-monte-carlo-demo) (Stück 12, Multilevel über Zeitschritte): Hier geht es um **E_X[ f( E_Y[ g(X, Y) | X ] ) ]**, einen Erwartungswert über eine nichtlineare Funktion einer inneren bedingten Erwartung, und um die Frage, wann der **Multilevel-Schätzer** (Ebenen über die Zahl der inneren Ziehungen) das geschachtelte Monte Carlo rettet. Vehikel ist die log-normale Nachfrage der Stücke 11 und 12 als **zweistufiges Modell** mit Signal (außen) und innerem Rauschen; die bedingte Erwartung ist geschlossen bekannt und dient nur als Orakel.

## Warum dieses Problem

Wer den Wert einer Information, ein Risikomaß oder einen Sensitivitätsindex schätzt, braucht eine äußere Erwartung über eine nichtlineare Funktion f einer inneren bedingten Erwartung μ(X). Eine Ziehung je Stufe geht nicht (f ist nicht linear); das **geschachtelte Monte Carlo** zieht N äußere und je M innere Werte und wertet f am inneren Mittel aus. Die Verzerrung fällt nur wie 1/M, also braucht man M ∝ ε⁻¹ und N ∝ ε⁻² äußere Ziehungen: Kosten ε⁻³ (allgemein ε^(−2−1/α), α die Ordnung der Verzerrung; der Abstract von Hironaka, Giles, Goda und Thom, arXiv 1909.00549, nennt dies für den Erwartungswert der Stichprobeninformation und senkt die Kosten mit Multilevel auf O(ε⁻²); hier nur der Abstract gelesen).
Multilevel nimmt M_l = 2^l innere Ziehungen je Ebene und koppelt aufeinanderfolgende Ebenen; die Demo misst, **wann** das reicht und wann nicht.

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat oder beim Bau dazukam:

- **„Naiv ε^(−2−1/α), mit Multilevel deutlich weniger“ gilt nur bei Mittelwert und Knick.** Geschachtelt kostet ε⁻³ bei allen drei äußeren Funktionen. Multilevel kostet beim Mittelwert etwa ε⁻² (ε²·Kosten nahezu konstant, 5,6 bis 6,4), beim Knick wenig mehr (ε²·Kosten wächst langsam, 0,65 bis 1,21). Es ist aber erst ab einer Toleranz billiger (idealisiert: Mittelwert ab etwa ε = 0,0025, Knick schon bei 0,01; in der gelaufenen Schätzung mit Pilot, siehe unten, kippt der Knick erst zwischen 0,005 und 0,0025). Bei einem **Sprung** (Indikator als äußere Funktion) kostet es bei **allen Tabellentoleranzen** bis ε = 0,0003 mehr als geschachteltes Monte Carlo (Verhältnis 0,40 bis 0,82, steigend; bei ε = 0,00025 und ρ = 0,6 bereits 1,04).
- **Die Antithetik entscheidet.** Mit der antithetischen Kopplung (die grobe Stufe nutzt beide Hälften der inneren Ziehungen) fällt die Ebenenvarianz beim Mittelwert mit der Rate 2,06 (Verhältnis je Ebene 0,26), beim Knick mit 1,33 (Regression ab Ebene 3; die Verhältnisse fallen noch, die letzten Stufen liegen bei 0,37, etwa 2^(−1,4)); mit nur der ersten Hälfte oder mit unabhängigen Ziehungen fällt sie beim Mittelwert mit 1,18 bzw. 1,13, beim Knick mit 1,00 bzw. 0,99. Beim Sprung liegt die Rate bei allen Kopplungen um 0,4 (die Antithetik verkleinert nur die Konstante: auf Ebene 8 auf die Hälfte der einseitigen und etwa ein Drittel der unabhängigen Varianz). Auf Ebene 8 ist die Varianz beim Mittelwert antithetisch 1,8e-6 gegen 5,5e-4 einseitig und 1,6e-3 unabhängig.
- **Die Verzerrung beim Mittelwert ist exakt c/M** mit c = E Var(g | X) = E g² − E μ² = 0,12386; eine Beziehung, die ein Orakeltest ohne Simulation prüft. Beim Sprung liegt die gemessene Verzerrung schon ab Ebene 3 im Rauschen; eine Kontrollrechnung mit gemeinsamen äußeren Ziehungen (nicht Teil der Demo) ergibt etwa 2^(−1,6) je Ebene, nicht 2^(−l/2): die Rate ε^(−2−1/α) mit α = 1/2 zeigte sich hier nicht.
- **Wenn die innere Erwartung bekannt ist, ist Schachteln Verschwendung:** „bedingt“ (μ(X) direkt auswerten, eine Auswertung je äußerer Ziehung) ist 14- bis 25-mal billiger als Multilevel beim Knick, 85- bis 98-mal beim Mittelwert und 40- bis 620-mal beim Sprung (Stück 6).
- **Bei lockerer Toleranz gewinnt das einfache Schachteln** (der Pilot von Giles' Verfahren kostet zu viel): gelaufen, Kostenverhältnis geschachtelt/Multilevel bei ε = 0,01 / 0,005 / 0,0025: Mittelwert 0,22 / 0,52 / 1,08, Knick 0,27 / 0,81 / 1,85, Sprung 0,26 / 0,36 / 0,43.
- **Das Signal steuert den Gewinn:** Kostenverhältnis geschachtelt/Multilevel (idealisiert, ε = 0,001) bei Signalstärke ρ = 0,3 / 0,6 / 0,9: Mittelwert 0,13 / 2,6 / 31, Knick 1,25 / 6,2 / 12,0, Sprung 0,41 / 0,66 / 0,87. Die Schwelle k = 0,5 / 1 / 2 mal Mittel gibt beim Knick 11,0 / 6,2 / 2,8; der Mittelwert hängt nicht von k ab.
- **Der entzerrte Ein-Term-Schätzer** (Rhee und Glynn, wie in Stück 12) braucht γ < s < β (γ = 1 die Kostenrate je Ebene; bei s = γ = 1 wachsen die Kosten mit der Abschneideebene): Varianz bei Abschneideebene 3 → 8 beim Mittelwert 1,73 → 1,88 (s = 1), beim Knick 0,28 → 0,32 (s = 1) und 0,24 → 0,39 (s = 1,5), 0,25 → 1,41 (s = 2); beim Sprung wächst sie für jedes s (s = 1: 2,1 → 24,8).
- **Kein scipy zur Laufzeit:** Normalverteilungsfunktion über `math.erf`, Box–Muller; Gauß–Hermite- und Gauß–Legendre-Knoten für die exakte Größe kommen aus numpy.

## Modell

Nachfrage D = exp(σ (ρ X + √(1 − ρ²) Y)) mit unabhängigen Standardnormalen X (Signal, außen) und Y (inneres Rauschen), σ = 0,5, Signalstärke ρ = 0,6; marginal ist D log-normal mit μ = 0 wie in den Stücken 11 und 12. Innere Ausgabe g(D) = (D − q)⁺ mit q = 1,3 (Fehlbestand); die bedingte Erwartung μ(X) = E[g | X] ist geschlossen (Black-Scholes-Form). Äußere Funktion in drei Glattheitsstufen mit Schwelle k = E g(D) = 0,16554: **Mittelwert** f(m) = m² (exakt 0,064501), **Knick** f(m) = (m − k)⁺ (exakt 0,067545), **Sprung** f(m) = 1[m > k] (exakt 0,339560). Die exakte Größe kommt aus Gauß–Hermite (Mittelwert), Gauß–Legendre ab der Knickstelle x* = 0,4137 (Knick) und Φ (Sprung), gegen `scipy.integrate.quad` geprüft. Kosten zählen **innere Auswertungen**: 2^l je äußere Ziehung (die antithetische grobe Hälfte nutzt dieselben Werte); die äußere Ziehung zählt nicht mit.

## Methodik

Je Ebene 40 000 äußere Ziehungen mit M = 2^l inneren aus festen SplitMix64-Strömen, Box–Muller. Ebene l: Y_l = f(Mittel aller 2^l) − ½ (f(Mittel der ersten Hälfte) + f(Mittel der zweiten Hälfte)) (**antithetisch**); Vergleichsvarianten **einseitig** (grob = f(Mittel der ersten Hälfte)) und **unabhängig** (grob = f(Mittel frischer innerer Ziehungen zur selben äußeren)). Gemessen: Mittel, Varianz und Kurtosis der Differenz, die Verzerrung E[P_l] − exakt (aus dem geschlossenen E[P_0] plus den Ebenenmitteln). Rate β aus der Regression von log₂ V_l über die Ebenen ab 3, Verzerrungskonstante c aus dem Median von |E[Y_l]|·2^l über die Ebenen 2 bis 5.
**Idealisierte Kosten** für einen mittleren quadratischen Fehler ≤ ε² (Verzerrung² ≤ ε²/2, Varianz ≤ ε²/2): geschachtelt 2 Var(P_L) ε⁻² 2^L mit der kleinsten Ebene L, für die c·2^(−L) ≤ ε/√2 gilt; Multilevel 2 ε⁻² (Σ √(V_l C_l))² mit C_l = 2^l und der optimalen Zuteilung N_l ∝ √(V_l/C_l); bedingt 2 Var f(μ(X)) ε⁻². **Gelaufene Schätzungen:** Giles' adaptives Verfahren wie in Stück 12 (Pilot 1000 äußere Ziehungen auf den Ebenen 0 bis 2, schwache Ordnung fest 1) gegen geschachteltes Monte Carlo **mit Orakelwissen** über M und Varianz, 30 Wiederholungen. **Signal und Schwelle:** das idealisierte Kostenverhältnis für alle ρ und für k = 0,5 / 1 / 2 mal Mittel. **Ein-Term-Schätzer:** zufällige Ebene mit p_l ∝ 2^(−s l), Z = Y_L/p_L; Varianz aus den gemessenen zweiten Momenten und ein gelaufener Lauf mit 20 000 Proben.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Strom 37, σ = 0,5, ρ = 0,6 (wenn nicht anders genannt), Ebenen 0 bis 8, `tests/test_claims.py` rechnet sie nach (in teils weiten Bändern). Die Zeilen der Tabelle bei ε = 0,0003125 und die letzten Zeilen der Kostentabellen beruhen auf den Ebenen 9 und 10, die mit der geschätzten Rate fortgesetzt sind. Kosten in inneren Auswertungen, idealisiert.

| äußere Funktion | β antithetisch | β einseitig | β unabhängig | gesch./Multilevel bei ε = 0,01 | bei ε = 0,0003125 | Multilevel/bedingt |
|---|---|---|---|---|---|---|
| Mittelwert | 2,06 | 1,18 | 1,13 | 0,42 | 10,0 | 85 bis 98 |
| Knick | 1,33 | 1,00 | 0,99 | 1,45 | 20,6 | 13,5 bis 25 |
| Sprung | 0,41 | 0,41 | 0,43 | 0,40 | 0,82 | 40 bis 620 |

- **ε²·Kosten, Multilevel:** Mittelwert 5,6 → 6,4 (nahezu konstant), Knick 0,65 → 1,21, Sprung 17,7 → 278; geschachtelt wächst überall mit 1/ε (Mittelwert 2,4 → 64).
- **Kostenanteil je Ebene (ε = 0,000625):** Mittelwert und Knick tragen auf Ebene 0 jeweils 47 %; beim Knick liegen die übrigen Ebenen bei 5 bis 8 %, beim Mittelwert fallen sie von 19 % auf 1 %; beim Sprung wachsen sie zur feinen Ebene (4 % → 22 %).
- **Ohne Antithetik** (einseitige Kopplung, Mittelwert): Kostenverhältnis 0,21 bei ε = 0,0025 und 0,59 bei ε = 0,000625, also bei diesen beiden Toleranzen teurer als geschachtelt (das Verhältnis steigt mit sinkendem ε; bei 0,000625 ist die Ebene 9 fortgesetzt).
- **Schwaches Signal** (ρ = 0,3, Mittelwert, ε = 0,0025): Verzerrungskonstante c = 0,154 (statt 0,124) und Kostenverhältnis 0,08; starkes Signal (ρ = 0,9): 8,4.
- **Ein-Term-Lauf** (Knick, s = 1, Abschneideebene 8): Mittel 0,0655 gegen exakt 0,0675, Stichprobenvarianz 0,316 (erwartet 0,321), Kosten 4,62 innere Auswertungen je Probe; Varianz·Kosten/ε² = 1,45/ε².

## Ehrliche Grenzen

- **Nur ein Modell** (log-normale Nachfrage mit einem Fehlbestand als innerer Ausgabe, äußere Funktionen Quadrat, Knick, Sprung); **kein Entscheidungsfall** (Minimum über mehrere Aktionen wie beim Wert der Information in `newsvendor-evpi-demo`), obwohl die Quellen dort ansetzen; keine mehrdimensionalen Eingaben, keine Tiefe über zwei Stufen, keine adaptiven inneren Stichproben, kein Quasi-Monte-Carlo.
- **Geschachteltes Monte Carlo ist mit Orakelwissen gerechnet** (es kennt M und die Varianz) und wird dadurch begünstigt; Giles' Verfahren zahlt für den Pilot und nimmt die schwache Ordnung 1 an.
- **Kosten sind innere Auswertungen,** nicht Laufzeit; die äußere Ziehung zählt nicht mit.
- **Die Ebenenvarianz beim Sprung** beruht auf seltenen Ereignissen und ist mit 40 000 äußeren Ziehungen nur grob gemessen (hohe Kurtosis); die gemessene Verzerrung beim Sprung liegt schon ab Ebene 3 im Rauschen, und die daraus geschätzte Verzerrungskonstante (und damit die Ebene L des geschachtelten Verfahrens beim Sprung) ist unsicher (die Bänder der Tests sind dort weit).
- **Das Kostenverhältnis ändert sich nur mit der Ebenenzahl L** (es ist von ε unabhängig, solange L gleich bleibt); die Tabelle zeigt deshalb Stufen.
- **Ein-Term-Schätzer bei wachsender Varianz:** die Stichprobenvarianz im Lauf ist wegen schwerer Ränder unzuverlässig (beim Sprung 35,6 gegen erwartet 24,8); die README nennt nur den Knick-Lauf.
- **Der Kostensatz (Fälle β gegen γ) und Giles' Verfahren stehen aus dem Gedächtnis im Mathematik-Expander;** Hironaka u. a. und Rainforth u. a. sind nur im Abstract gelesen. Literaturangaben im Mathematik-Expander stammen aus dem Gedächtnis.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): Ebenenpaar mit zwei inneren Werten (0,3; 0,1) von Hand für alle drei Funktionen und die einseitige Kopplung (Mittelwert: fein 0,04, grob 0,05, Y = −(a − b)²/4 = −0,01; Knick: Y = −0,025; Sprung: Y = 0,5), Symmetrie der Hälften, Ebene 0, Kosten je Probe, optimale Zuteilung von Hand (V = (1, ½, ¼), C = (1, 2, 4): 6 / 3 / 1,5 und 18 ε⁻²), Schwelle, E[P_0] für Knick und Sprung von Hand, bedingte Erwartung bei x = 0 von Hand, Ein-Term-Wahrscheinlichkeiten, Zufallsstrom.
- **Orakel** (`tests/test_oracle_ger.py`): **bedingte Erwartung gegen Quadratur der inneren Verteilung**, **Schwelle gegen Quadratur**, **exakter Wert und Varianz der bedingten Funktion gegen `scipy.integrate.quad`** (sieben Parametersätze, drei Funktionen), **E[P_0] gegen Quadratur der Marginalverteilung**, die **Beziehung E g² = E μ² + E Var(g | X)** (die Verzerrungskonstante c = 0,12386 aus dem geschlossenen E[P_0] und der Wahrheit), **Ebenenmittel beim Mittelwert gleich Wahrheit + c/M**, **Y_l = −(Hälftendifferenz)²/4 je Probe**, Teleskopsumme, **Vertauschen von Ziehungen innerhalb und zwischen den Hälften ändert die antithetische grobe Stufe nicht**, Wahrheit gegen Monte Carlo mit exakter bedingter Erwartung, Normalzahlen gegen `scipy.stats` (Kolmogorov–Smirnov), jede zufällige Störung der optimalen Zuteilung ist nicht billiger.
- **Eigenschaften** (`tests/test_properties.py`): Raten je Funktion und Kopplung, antithetisch hat die kleinste Varianz auf den tiefen Ebenen, Verzerrung halbiert sich (beim Mittelwert gleich c/M), Ebene und Kosten wachsen mit enger werdendem ε, „bedingt“ ist am billigsten, Multilevel gewinnt beim Knick und nicht beim Sprung, Kostenanteile summieren sich zu 1, stärkeres Signal gibt mehr Gewinn, Reproduzierbarkeit, gelaufener Schätzer trifft die Toleranz, Ein-Term-Schätzer unverzerrt beim Knick und mit wachsender Varianz beim Sprung.
- **Randfälle** (`tests/test_edge_cases.py`): sehr lockere und sehr enge Toleranz, konstante Funktion (ehrliches Verdikt statt Division durch null), Null-Varianz in Giles' Verfahren, eine äußere Ziehung, Regression mit zu wenigen Punkten, unbekannte Namen, Eingabeprüfung der Spielwiese, Zahlenformate, äußerste Signalstärken, Extrapolation.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten, die drei Kostenfälle der Rate, Ebene und Kosten von Hand.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes mit dem erwarteten Verdikt; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jede Auswahl jedes Reglers, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims** (`tests/test_claims.py`).

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `ger_algorithm.py` | Kern: Strom, bedingte Erwartung und Wahrheit, Ebenenpaare mit drei Kopplungen, Ebenenstatistik, Giles' Verfahren, geschachteltes Monte Carlo, Ein-Term-Schätzer |
| `ger_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `ger_evaluation.py` | Kennzahlen, Kostenrechnung, Verdikt-Logik, Experimente über feste Ströme |
| `ger_layout.py` | Fester Kopf (Stylesheet) |
| `ger_pdf_export.py` | PDF der aktuellen Einstellung |
| `ger_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufallsstrom-Knopf |
| `ger_scenario.py` | Szenario der Spielwiese (ρ, σ, q, Schwellenfaktor) und seine Prüfung |
| `ger_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| Die innere Erwartung exakt ausrechnen | [bedingtes-monte-carlo-demo](https://github.com/sebastian-hanisch/bedingtes-monte-carlo-demo) (Stück 6) |
| Ebenen über Zeitschritte, Giles' Verfahren | [multilevel-monte-carlo-demo](https://github.com/sebastian-hanisch/multilevel-monte-carlo-demo) (Stück 12) |
| Sobol-Indizes als geschachtelte Erwartung (Pick-Freeze statt Schachteln) | [sobol-indizes-demo](https://github.com/sebastian-hanisch/sobol-indizes-demo) (Stück 10) |
| Wert der Information beim Bestand (Minimum über Aktionen als äußere Funktion) | [newsvendor-evpi-demo](https://github.com/sebastian-hanisch/newsvendor-evpi-demo) |
| Mehrdimensionale Eingaben, Tiefe über zwei Stufen, adaptive innere Stichproben, Quasi-Monte-Carlo | nicht gebaut |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Monte-Carlo-Schätzung: vom Schätzer zur Wahl](https://sebastianhanisch.net/konzepte-monte-carlo.html).
