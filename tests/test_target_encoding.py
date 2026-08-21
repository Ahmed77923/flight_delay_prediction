import pandas as pd

from src.preprocessing.preprocess import build_preprocessor
from src.preprocessing.target_encoder import TargetEncoder


def test_build_preprocessor_includes_target_encoding() -> None:
    preprocessor = build_preprocessor()
    names = [name for name, _, _ in preprocessor.transformers]

    assert "target_encoding" in names


def test_target_encoder_oof_is_not_full_data_fit() -> None:
    X = pd.DataFrame({"cat": ["A", "A", "B", "B"]})
    y = pd.Series([1, 100, 2, 200])

    encoder = TargetEncoder(columns=["cat"], smoothing=0, n_splits=2, random_state=0)
    oof_encoded = encoder.fit_transform_oof(X, y)
    full_fit_encoded = encoder.fit(X, y).transform(X)

    assert not oof_encoded["cat"].equals(full_fit_encoded["cat"])
