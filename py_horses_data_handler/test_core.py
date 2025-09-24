import pytest

from py_horses_data_handler.core import _get_time, _get_iteration


def test_get_time_valid():
    out = "Some logs\nTime: 123.456\nMore logs"
    assert _get_time(out) == pytest.approx(123.456)


def test_get_time_missing():
    out = "No time here"
    with pytest.raises(ValueError):
        _get_time(out)


def test_get_time_none():
    with pytest.raises(ValueError):
        _get_time(None)


def test_get_iteration_valid():
    out = "Start\nIteration: 42\nEnd"
    assert _get_iteration(out) == 42


def test_get_iteration_missing():
    out = "No iter"
    with pytest.raises(ValueError):
        _get_iteration(out)


def test_get_iteration_none():
    with pytest.raises(ValueError):
        _get_iteration(None)
