"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset rechnet und hat das erwartete Verdikt."""

import pytest

import ger_constants as C
import ger_evaluation as ev
import ger_presets as P

STANDARD = "Standardfall (Voreinstellung)"


def _settings(name):
    return ev.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert 0 <= preset["seed"] <= C.SEED_MAX, name
        assert preset["ausgabe"] in C.AUSGABEN and preset["kopplung"] in C.KOPPLUNGEN, name
        assert preset["rho"] in C.RHO_CHOICES and preset["eps"] in C.EPS_CHOICES and preset["lmax"] in C.LMAX_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[STANDARD]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard():
    for name, preset in C.PRESETS.items():
        if name != STANDARD:
            assert preset != C.PRESETS[STANDARD], name


def test_standardfall_zeigt_beim_knick_bei_lockerer_toleranz_noch_keinen_grossen_vorteil():
    s = _settings(STANDARD)
    a = ev.analyse(s)
    assert a.kind == "knick" and 1.2 < ev.cost_row(a, s.eps)["ratio"] < 3.5 and a.beta > 1.0


def test_preset_mittelwert_laesst_multilevel_deutlich_gewinnen():
    s = _settings("Mittelwert: Multilevel rettet das Schachteln")
    a = ev.analyse(s)
    kind, text = ev.verdict_main(a, s)
    assert kind == "success" and ev.cost_row(a, s.eps)["ratio"] > 5 and a.beta > 1.7


def test_preset_sprung_zeigt_multilevel_teurer():
    s = _settings("Sprung: Multilevel verliert")
    a = ev.analyse(s)
    kind, text = ev.verdict_main(a, s)
    assert kind == "warning" and ev.cost_row(a, s.eps)["ratio"] < 1.0 and "teurer als einfaches Schachteln" in text


def test_preset_ohne_antithetik_hat_die_rate_eins():
    s = _settings("Ohne Antithetik")
    a = ev.analyse(s)
    assert s.kopplung == "one" and 0.9 < a.beta < 1.5 and a.beta_by_variant["anti"] > a.beta + 0.3


def test_preset_schwaches_signal_laesst_das_schachteln_bei_dieser_toleranz_vorn():
    s = _settings("Schwaches Signal")
    a = ev.analyse(s)
    assert s.rho == 0.3 and ev.cost_row(a, s.eps)["ratio"] < 1.0


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_rechnet(name):
    a = ev.analyse(_settings(name))
    assert a.truth > 0 and len(a.levels) == C.PRESETS[name]["lmax"] + 1
