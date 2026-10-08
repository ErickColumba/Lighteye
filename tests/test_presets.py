from app.core.presets import (
    apply_look,
    delete_preset,
    list_presets,
    load_preset,
    look_values,
    save_preset,
)
from app.core.settings import Settings


def test_save_list_load_delete(tmp_path):
    s = Settings({"exposure": 0.5, "saturation": -20, "rotate": 1, "crop": [0, 0, 0.5, 0.5]})
    p = save_preset("Cálido / suave", s, tmp_path)
    assert p.path.parent == tmp_path and p.path.suffix == ".json"
    assert p.values == {"exposure": 0.5, "saturation": -20.0}  # sin geometría
    save_preset("Blanco y negro", Settings({"bw": 1}), tmp_path)
    names = [x.name for x in list_presets(tmp_path)]
    assert names == ["Blanco y negro", "Cálido / suave"]
    assert load_preset(p.path).values == p.values
    delete_preset(p)
    assert [x.name for x in list_presets(tmp_path)] == ["Blanco y negro"]


def test_apply_look_keeps_geometry_and_replaces_the_rest():
    photo = Settings({"contrast": 40, "angle": 3.5, "crop": [0.1, 0.1, 0.5, 0.5]})
    result = apply_look(photo, {"exposure": 1.0})
    assert result["exposure"] == 1.0
    assert result["contrast"] == 0  # no estaba en el preset
    assert result["angle"] == 3.5 and result["crop"] == (0.1, 0.1, 0.5, 0.5)


def test_invalid_files_are_ignored(tmp_path):
    (tmp_path / "roto.json").write_text("{no es json")
    (tmp_path / "raro.json").write_text('{"settings": [1, 2]}')
    (tmp_path / "otro.json").write_text('{"settings": {"exposure": 1, "inventado": 3}}')
    presets = list_presets(tmp_path)
    assert [p.name for p in presets] == ["otro"]
    assert presets[0].values == {"exposure": 1.0}


def test_look_values_excludes_geometry():
    assert look_values(Settings({"rotate": 2, "flip_h": 1, "grain": 10})) == {"grain": 10.0}


def test_builtin_presets(tmp_path, monkeypatch):
    import pytest

    from app.core.presets import BUILTIN_DIR

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))  # sin presets del usuario
    save_preset("Mío", Settings({"exposure": 1}))
    presets = list_presets()
    builtin = [p for p in presets if p.builtin]
    assert len(builtin) >= 10
    assert presets[-1].name == "Mío" and not presets[-1].builtin  # los incluidos van primero
    assert {p.path.parent for p in builtin} == {BUILTIN_DIR}
    for p in builtin:
        assert p.values, p.name  # cada preset cambia algo
        assert not {"rotate", "flip_h", "flip_v", "angle", "crop"} & set(p.values)
        assert Settings(p.values).non_default() == p.values  # todos los valores son válidos
    with pytest.raises(ValueError):
        delete_preset(builtin[0])
    assert builtin[0].path.exists()


def test_file_names_are_clean(tmp_path):
    p = save_preset("Retrato · Natural / v2", Settings({"exposure": 1}), tmp_path)
    assert p.path.name == "Retrato Natural v2.json"


def test_none_preset_clears_look_but_keeps_crop():
    from app.core.presets import NONE_PRESET

    photo = Settings({"exposure": 1, "lut_path": "lighteye:calido.cube", "angle": 2,
                      "crop": [0.1, 0.1, 0.5, 0.5]})
    result = apply_look(photo, NONE_PRESET.values)
    assert look_values(result) == {}
    assert result["angle"] == 2 and result["crop"] == (0.1, 0.1, 0.5, 0.5)


def test_categories():
    from app.core.presets import category_of

    assert category_of({"exposure": 1}) == "ajustes"
    assert category_of({"exposure": 1, "skin_smooth": 30}) == "ia"
    assert category_of({"bg_remove": 1.0}) == "ia"
    builtin = [p for p in list_presets() if p.builtin]
    assert sum(p.category == "ia" for p in builtin) >= 10
    assert all(p.name.startswith("IA · ") for p in builtin if p.category == "ia")


def test_saved_ai_preset_goes_to_ia_tab(tmp_path):
    p = save_preset("Mi retoque", Settings({"skin_smooth": 50, "exposure": 0.2}), tmp_path)
    assert p.category == "ia"
    assert list_presets(tmp_path)[0].category == "ia"


# --- Presets acumulados ------------------------------------------------------------

def test_presets_stack_and_can_be_removed_one_by_one():
    from app.core.presets import NONE_PRESET, Preset, preset_owners, remove_stacked_preset, stack_preset
    from pathlib import Path

    a = Preset("A", Path(), {"exposure": 0.5, "contrast": 20.0})
    b = Preset("B", Path(), {"contrast": 40.0, "saturation": -10.0})
    s = Settings({"shadows": 15})  # ajuste manual previo
    s = stack_preset(stack_preset(s, a), b)
    assert s["exposure"] == 0.5 and s["contrast"] == 40 and s["saturation"] == -10
    assert s["shadows"] == 15  # lo manual se conserva
    assert preset_owners(s) == {"exposure": "A", "contrast": "B", "saturation": "B", "shadows": None}

    without_b = remove_stacked_preset(s, "B")
    assert without_b["contrast"] == 20  # vuelve al valor de A
    assert without_b["saturation"] == 0 and without_b["exposure"] == 0.5
    assert [e["name"] for e in without_b["preset_stack"]] == ["A"]

    s["exposure"] = 1.0  # cambio a mano sobre un ajuste de A
    without_a = remove_stacked_preset(s, "A")
    assert without_a["exposure"] == 1.0  # se respeta
    assert preset_owners(without_a)["exposure"] is None

    cleared = stack_preset(s, NONE_PRESET)
    assert look_values(cleared) == {} and cleared["preset_stack"] == ()


def test_preset_stack_survives_json_and_is_not_saved_in_presets(tmp_path):
    import json

    from app.core.presets import Preset, preset_owners, stack_preset

    s = stack_preset(Settings(), Preset("Fondo", tmp_path, {"bg_remove": 1.0, "bg_color": (1.0, 1.0, 1.0)}))
    again = Settings(json.loads(json.dumps(s.non_default())))  # como el sidecar
    assert again == s
    assert preset_owners(again) == {"bg_remove": "Fondo", "bg_color": "Fondo"}
    assert "preset_stack" not in look_values(s)
    assert "preset_stack" not in save_preset("x", s, tmp_path).values


def test_removing_a_preset_restores_manual_values():
    from pathlib import Path

    from app.core.presets import Preset, remove_stacked_preset, stack_preset

    s = Settings({"shadows": 20})  # a mano
    s = stack_preset(s, Preset("Dorado", Path(), {"shadows": 15.0, "temperature": 25.0}))
    assert s["shadows"] == 15
    back = remove_stacked_preset(s, "Dorado")
    assert back["shadows"] == 20 and back["temperature"] == 0
    # A, luego B encima del mismo ajuste; quitar B vuelve al de A.
    s = stack_preset(stack_preset(Settings(), Preset("A", Path(), {"contrast": 10.0})),
                     Preset("B", Path(), {"contrast": 30.0}))
    assert remove_stacked_preset(s, "B")["contrast"] == 10
