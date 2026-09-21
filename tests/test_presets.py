"""Presets, Permalink-Angaben und Regler-Grenzen sind untereinander stimmig."""

import pytest

import td_constants as C
import td_presets as P
from td_scenario import make_network


def test_every_preset_sets_every_control_within_bounds_and_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 4
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and p["net"] in C.NETS
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
        assert p["depart"] in C.DEPART_OPTIONS
        make_network(p["net"], p["strength"], p["side"], p["sites"], p["nodes"], p["degree"], p["seed"])
        assert C.PRESET_HELP[name]
    assert [p["net"] for p in C.PRESETS.values()] == list(C.NETS)


def test_setting_specs_and_kept_keys_are_consistent():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) and set(P.KEPT) <= set(P.SETTING_SPECS)
    for spec in P.SETTING_SPECS.values():
        assert spec.lo is None or spec.lo < spec.hi                                 # kein Regler mit gleichen Grenzen (Streamlit bricht ab)
    assert len({s.url_param for s in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_defaults_lie_inside_the_bounds_and_the_largest_net_has_at_most_400_nodes():
    for lo, hi, d in ((C.SIDE_MIN, C.SIDE_MAX, C.DEFAULT_SIDE), (C.STRENGTH_MIN, C.STRENGTH_MAX, C.DEFAULT_STRENGTH), (C.NODES_MIN, C.NODES_MAX, C.DEFAULT_NODES),
                      (C.DEGREE_MIN, C.DEGREE_MAX, C.DEFAULT_DEGREE)):
        assert lo < d < hi
    assert C.SITES_MIN <= C.DEFAULT_SITES <= C.SITES_MAX and C.DEFAULT_DEPART in C.DEPART_OPTIONS and C.DEFAULT_NET == "small"
    assert C.SIDE_MAX ** 2 <= 400 and C.NODES_MAX <= 400 and C.DEPART_OPTIONS[0] == 0 and C.DEPART_OPTIONS[-1] == 1425 and C.DEPART_MAX - 15 == C.DEPART_OPTIONS[-1]


def test_choice_casters_reject_unknown_values():
    cast = P.SETTING_SPECS["net_select"].caster
    assert cast("site") == "site"
    with pytest.raises(ValueError):
        cast("ring")
