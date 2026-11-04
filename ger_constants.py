"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

AUSGABEN = {"glatt": "Mittelwert: f(m) = m²", "knick": "Knick: f(m) = (m − k)⁺", "sprung": "Sprung: f(m) = 1[m > k]"}
DEFAULT_AUSGABE = "knick"
KOPPLUNGEN = {"anti": "antithetisch (beide Hälften)", "one": "einseitig (nur die erste Hälfte)", "indep": "unabhängig (eigene Ziehungen)"}
KOPPLUNG_KURZ = {"anti": "antithetisch", "one": "einseitig", "indep": "unabhängig"}
DEFAULT_KOPPLUNG = "anti"
SIGMA = 0.5                   # Streuung der Nachfrage D = exp(σ (ρ X + √(1 − ρ²) Y))
Q = 1.3                       # Schwelle der inneren Ausgabe g(D) = (D − q)⁺
RHO_CHOICES = (0.3, 0.6, 0.9)
RHO_MIN, RHO_MAX, DEFAULT_RHO = RHO_CHOICES[0], RHO_CHOICES[-1], 0.6
EPS_CHOICES = (0.01, 0.005, 0.0025)                         # Toleranz der gelaufenen Schätzungen
EPS_MIN, EPS_MAX, DEFAULT_EPS = EPS_CHOICES[-1], EPS_CHOICES[0], 0.005
EPS_GRID = (0.01, 0.005, 0.0025, 0.00125, 0.000625, 0.0003125)    # Toleranzen der idealisierten Kostentabelle
EPS_SIGNAL = (0.001, 0.00025)                               # Toleranzen im Signal-Experiment
LMAX_CHOICES = (6, 7, 8)                                    # größte gemessene Ebene (2^l innere Ziehungen)
LMAX_MIN, LMAX_MAX, DEFAULT_LMAX = LMAX_CHOICES[0], LMAX_CHOICES[-1], 7
SEED_MAX, DEFAULT_SEED = 999999, 37

N_LEVEL = 40000               # äußere Ziehungen je Ebene (Experiment 1)
N_SINGLE = 20000              # Proben des entzerrten Schätzers im Lauf (Experiment 5)
RUN_REPS = 30                 # Wiederholungen der gelaufenen Schätzungen (Experiment 3)
PILOT = 1000                  # Pilotproben je Ebene in Giles' Verfahren
SINGLE_S = (1.0, 1.5, 2.0)    # Abfallexponenten p_l ∝ 2^(−s l)
FIT_FROM = 3                  # Ebenen ab hier gehen in die Regression der Rate ein
BIAS_FROM, BIAS_TO = 2, 5     # Ebenen für die Verzerrungskonstante
K_FACTORS = (0.5, 1.0, 2.0)   # Schwelle k als Vielfaches des Mittels E g(D) (Signal-Experiment)

CUSTOM_RHO_MIN, CUSTOM_RHO_MAX = 0.1, 0.95
CUSTOM_SIGMA_MIN, CUSTOM_SIGMA_MAX = 0.1, 1.0
CUSTOM_Q_MIN, CUSTOM_Q_MAX = 0.5, 2.5
CUSTOM_K_MIN, CUSTOM_K_MAX = 0.25, 4.0

COLORS = {"anti": "#2F6B65", "one": "#E08E0B", "indep": "#C0392B", "bias": "#7f8c3a", "nested": "#C0392B", "ml": "#2F6B65", "cond": "#4c78a8", "ref": "#222222"}
COLOR_BAD = "#C0392B"
COLOR_OK = "#2F6B65"

PRESETS = {
    "Standardfall (Voreinstellung)": {"ausgabe": DEFAULT_AUSGABE, "kopplung": DEFAULT_KOPPLUNG, "rho": DEFAULT_RHO, "eps": DEFAULT_EPS, "lmax": DEFAULT_LMAX, "seed": DEFAULT_SEED},
    "Mittelwert: Multilevel rettet das Schachteln": {"ausgabe": "glatt", "kopplung": DEFAULT_KOPPLUNG, "rho": 0.9, "eps": 0.0025, "lmax": 8, "seed": DEFAULT_SEED},
    "Sprung: Multilevel verliert": {"ausgabe": "sprung", "kopplung": DEFAULT_KOPPLUNG, "rho": DEFAULT_RHO, "eps": 0.0025, "lmax": 8, "seed": DEFAULT_SEED},
    "Ohne Antithetik": {"ausgabe": "glatt", "kopplung": "one", "rho": DEFAULT_RHO, "eps": 0.0025, "lmax": 8, "seed": DEFAULT_SEED},
    "Schwaches Signal": {"ausgabe": "glatt", "kopplung": DEFAULT_KOPPLUNG, "rho": 0.3, "eps": 0.0025, "lmax": 8, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Äußere Funktion (m − k)⁺ mit Schwelle k beim Mittel, Signalstärke 0,6, antithetische Kopplung, Toleranz 0,005: Multilevel ist bei dieser Toleranz noch nicht billiger als einfaches Schachteln, bei engerer deutlich.",
    "Mittelwert: Multilevel rettet das Schachteln": "Bei f(m) = m², starkem Signal und Toleranz 0,0025 fällt die Ebenenvarianz mit 2^(−2l): Multilevel kostet einen Bruchteil des geschachtelten Monte Carlo.",
    "Sprung: Multilevel verliert": "Bei einem Indikator als äußerer Funktion fällt die Ebenenvarianz nur mit 2^(−l/2): Multilevel ist im Idealfall teurer als einfaches Schachteln.",
    "Ohne Antithetik": "Benutzt die grobe Stufe nur die erste Hälfte der inneren Ziehungen, fällt die Ebenenvarianz nur mit 2^(−l) statt 2^(−2l): der Gewinn schrumpft oder verschwindet.",
    "Schwaches Signal": "Bei ρ = 0,3 ist das innere Rauschen groß: die Ebene 0 dominiert und Multilevel lohnt erst bei sehr enger Toleranz.",
}
