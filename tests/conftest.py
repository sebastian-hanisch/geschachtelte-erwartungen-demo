"""Gemeinsame Fixtures: die Mini-Instanz mit festen inneren Werten, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STD = dict(rho=0.6, sigma=0.5, q=1.3)          # Voreinstellung


@pytest.fixture(scope="session")
def std():
    return dict(STD)
