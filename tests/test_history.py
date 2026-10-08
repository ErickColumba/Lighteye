from app.core.history import History, describe_changes
from app.core.settings import Settings


def s(**values):
    return Settings(values)


def test_undo_redo_cycle():
    h = History(s())
    assert not h.can_undo() and not h.can_redo()
    assert h.push(s(exposure=1))
    assert h.push(s(exposure=1, contrast=20))
    state, what = h.undo()
    assert state == s(exposure=1) and what == "Contraste"
    state, what = h.undo()
    assert state == s() and what == "Exposición"
    assert h.undo() is None
    state, what = h.redo()
    assert state == s(exposure=1) and what == "Exposición"


def test_push_same_state_is_ignored():
    h = History(s(exposure=1))
    assert not h.push(s(exposure=1))
    assert not h.can_undo()


def test_new_change_after_undo_discards_redo():
    h = History(s())
    h.push(s(exposure=1))
    h.push(s(exposure=2))
    h.undo()
    h.push(s(saturation=10))
    assert not h.can_redo()
    assert h.current == s(saturation=10)
    assert h.undo()[0] == s(exposure=1)


def test_history_returns_copies():
    h = History(s())
    h.push(s(exposure=1))
    state = h.current
    state["exposure"] = 3
    assert h.current["exposure"] == 1


def test_limit():
    h = History(s(), limit=3)
    for v in (1, 2, 3, 4):
        h.push(s(exposure=v))
    assert h.undo()[0] == s(exposure=3)
    assert h.undo()[0] == s(exposure=2)
    assert h.undo() is None


def test_describe_changes():
    assert describe_changes(s(), s(hsl_s_blue=10)) == "Azul (Saturación)"
    assert describe_changes(s(), s(curves={"rgb": [[0, 0], [0.5, 0.6], [1, 1]]})) == "Curvas"
    assert describe_changes(s(), s(exposure=1, contrast=1, shadows=1, blacks=1)) == "4 ajustes"
