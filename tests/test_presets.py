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
