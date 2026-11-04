"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ger_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_gelaufen", "Pilot 1.000 je Ebene"),
    ("run_signal", "Signalstärke ρ = 0,3"),
    ("run_single", "Varianz des Ein-Term-Schätzers bei wachsender Abschneideebene"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_kernfrage_ebenen_und_kostenverdikt_stehen_ohne_klick_da():
    t = _texte(_app())
    assert "geschachteltes Monte Carlo braucht M =" in t
    assert "Rate β der Ebenenvarianz (aus den Ebenen 3 bis 7)" in t
    assert "Geschachtelt/Multilevel bei ε = 0,01:" in t


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_gelaufen").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="eps_select").set_value(0.01).run(timeout=TIMEOUT)
    assert not at.exception
    assert "Toleranz ε = 0,01, 30 Wiederholungen" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    assert "teurer als einfaches Schachteln" not in _texte(at)
    at.sidebar.button(key="preset_Sprung: Multilevel verliert").click().run(timeout=TIMEOUT)
    assert "teurer als einfaches Schachteln" in _texte(at)
    at.sidebar.button(key="preset_Mittelwert: Multilevel rettet das Schachteln").click().run(timeout=TIMEOUT)
    assert "mindestens die Hälfte" in _texte(at)


def test_jede_ausgabe_und_jede_kopplung_laeuft_durch():
    at = _app()
    for ausgabe in C.AUSGABEN:
        for kopplung in C.KOPPLUNGEN:
            at.sidebar.selectbox(key="ausgabe_select").set_value(ausgabe).run(timeout=TIMEOUT)
            at.sidebar.selectbox(key="kopplung_select").set_value(kopplung).run(timeout=TIMEOUT)
            assert not at.exception, (ausgabe, kopplung, [str(e) for e in at.exception])


def test_spielwiese_rechnet_mit_eigenen_werten():
    at = _app()
    assert not at.error
    at.number_input(key="eigen_rho").set_value(0.9).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception] and not at.error
    at.number_input(key="eigen_q").set_value(2.5)
    at.number_input(key="eigen_sigma").set_value(0.1).run(timeout=TIMEOUT)        # innere Ausgabe nie positiv: keine Streuung
    assert not at.exception, [str(e) for e in at.exception]
    assert "nichts zu schätzen" in _texte(at)


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="lmax_select").set_value(6).run(timeout=TIMEOUT)         # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["r"] = "0.5"
    at.query_params["e"] = "0.003"
    at.query_params["l"] = "99"
    at.query_params["a"] = "unbekannt"        # unbekannte Ausgabe: verworfen
    at.query_params["k"] = "one"
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.select_slider(key="rho_select").value == 0.6                           # 0,5 liegt näher an 0,6 als an 0,3
    assert at.sidebar.select_slider(key="eps_select").value == 0.0025                        # 0,003 liegt näher an 0,0025 als an 0,005
    assert at.sidebar.select_slider(key="lmax_select").value == 8                            # oberhalb des Größten: das Größte
    assert at.sidebar.selectbox(key="ausgabe_select").value == C.DEFAULT_AUSGABE
    assert at.sidebar.selectbox(key="kopplung_select").value == "one"
    assert at.sidebar.number_input(key="seed_input").value == 0
