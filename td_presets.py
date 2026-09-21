"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio, siehe mc_presets.py in multicriteria-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import td_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _choice(options):
    def cast(value):
        value = str(value)
        if value not in [str(o) for o in options]:
            raise ValueError(value)
        return type(options[0])(value)
    return cast


SETTING_SPECS = {
    "net_select": SettingSpec("net", _choice(C.NETS), C.DEFAULT_NET),
    "strength_slider": SettingSpec("strength", float, C.DEFAULT_STRENGTH, C.STRENGTH_MIN, C.STRENGTH_MAX),
    "depart_slider": SettingSpec("depart", int, C.DEFAULT_DEPART, C.DEPART_MIN, C.DEPART_MAX - 15),
    "side_slider": SettingSpec("side", int, C.DEFAULT_SIDE, C.SIDE_MIN, C.SIDE_MAX),
    "sites_slider": SettingSpec("sites", int, C.DEFAULT_SITES, C.SITES_MIN, C.SITES_MAX),
    "nodes_slider": SettingSpec("nodes", int, C.DEFAULT_NODES, C.NODES_MIN, C.NODES_MAX),
    "degree_slider": SettingSpec("deg", float, C.DEFAULT_DEGREE, C.DEGREE_MIN, C.DEGREE_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
}
PRESET_KEYS = {"net": "net_select", "strength": "strength_slider", "depart": "depart_slider", "side": "side_slider", "sites": "sites_slider", "nodes": "nodes_slider", "degree": "degree_slider", "seed": "seed_input"}
# Regler, die je nach Netz ausgeblendet sind: Streamlit löscht ihren Zustand, sobald sie nicht gezeichnet werden - der zuletzt gewählte Wert bleibt hier erhalten
KEPT = {key: f"_kept_{key}" for key in ("side_slider", "sites_slider", "nodes_slider", "degree_slider", "seed_input")}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = st.session_state.get(KEPT[state_key], spec.default) if state_key in KEPT else spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    for key, step in (("strength_slider", 0.25), ("degree_slider", 0.5)):
        if key in st.session_state:
            st.session_state[key] = round(round(st.session_state[key] / step) * step, 2)
    if "depart_slider" in st.session_state:
        st.session_state["depart_slider"] = int(round(st.session_state["depart_slider"] / 15) * 15) % 1440
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][key]
        if state_key in KEPT:
            st.session_state[KEPT[state_key]] = C.PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
