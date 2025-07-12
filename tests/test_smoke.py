import pandas as pd

from battery_cycle_life.data import load_data
from battery_cycle_life.analysis import derivative_curves
from battery_cycle_life.models import extract_features, train_ensemble


def test_load_and_features():
    df = pd.read_csv("data/cycle_data_demo.csv")
    assert not df.empty
    feats = extract_features(df)
    assert "init_capacity" in feats


def test_derivative_curves():
    df = pd.read_csv("data/cycle_data_demo.csv")
    curves = derivative_curves(df, cycles=[1], kind="dQdV")
    assert 1 in curves and not curves[1].empty 