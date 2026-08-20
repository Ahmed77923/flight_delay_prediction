import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Leakage-safe Target Encoder.

    Supports:
        - Smoothed target encoding
        - K-fold Out-of-Fold (OOF) encoding
        - Unknown category handling
        - Reproducible encoding
        - Pandas DataFrames

    Parameters
    ----------
    columns : list[str]
        Categorical columns to encode.

    smoothing : float
        Smoothing strength.

    n_splits : int
        Number of folds used for OOF encoding.

    random_state : int
        Random state for reproducibility.
    """

    def __init__(
        self,
        columns,
        smoothing=10.0,
        n_splits=5,
        random_state=42,
    ):
        self.columns = columns
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state

        self.global_mean_ = None
        self.mapping_ = {}

    def _validate_input(self, X):
        """Validate input DataFrame and required columns."""

        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        missing_columns = [
            column
            for column in self.columns
            if column not in X.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns: {missing_columns}"
            )

    def _calculate_mapping(self, X, y):
        """
        Calculate smoothed target statistics.

        Formula:

        encoded =
            (count * category_mean + smoothing * global_mean)
            / (count + smoothing)
        """

        data = X[self.columns].copy()

        # Convert target to Series with matching index
        y = pd.Series(
            y,
            index=X.index,
            name="target"
        )

        global_mean = y.mean()

        mapping = {}

        for column in self.columns:

            temp = pd.DataFrame({
                "category": data[column],
                "target": y,
            })

            # Handle missing categories explicitly
            temp["category"] = temp["category"].fillna(
                "__MISSING__"
            )

            stats = (
                temp
                .groupby("category")["target"]
                .agg(["mean", "count"])
            )

            stats["encoded"] = (
                (
                    stats["count"] * stats["mean"]
                    + self.smoothing * global_mean
                )
                /
                (
                    stats["count"] + self.smoothing
                )
            )

            mapping[column] = stats["encoded"]

        return global_mean, mapping

    def fit(self, X, y):
        """
        Fit encoder using all available training data.

        This should ONLY be called on training data.
        Never call this using the test target.
        """

        self._validate_input(X)

        y = pd.Series(
            y,
            index=X.index
        )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of rows."
            )

        (
            self.global_mean_,
            self.mapping_
        ) = self._calculate_mapping(
            X,
            y
        )

        return self

    def transform(self, X):
        """
        Transform categorical columns using fitted statistics.

        Unknown categories receive the global mean.
        """

        if self.global_mean_ is None:
            raise RuntimeError(
                "TargetEncoder must be fitted before transform()."
            )

        self._validate_input(X)

        encoded = pd.DataFrame(
            index=X.index
        )

        for column in self.columns:

            values = X[column].fillna(
                "__MISSING__"
            )

            encoded[column] = (
                values
                .map(self.mapping_[column])
                .fillna(self.global_mean_)
                .astype(np.float64)
            )

        return encoded

    def get_feature_names_out(self, input_features=None):
        """Return the encoded column names for sklearn transformers."""

        if input_features is None:
            input_features = self.columns

        return np.asarray(input_features, dtype=object)

    def fit_transform_oof(self, X, y):
        """
        Generate leakage-safe Out-of-Fold target encoding.

        For every fold:
            training folds -> calculate statistics
            validation fold -> receive encoding

        After OOF encoding is created:
            final statistics are calculated using ALL training data.

        These final statistics are then used by transform()
        for future/test data.
        """

        self._validate_input(X)

        y = pd.Series(
            y,
            index=X.index,
            name="target"
        )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of rows."
            )

        if self.n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2."
            )

        if len(X) < self.n_splits:
            raise ValueError(
                "n_splits cannot be greater than the number of rows."
            )

        # Preserve original index
        oof_encoded = pd.DataFrame(
            index=X.index,
            columns=self.columns,
            dtype=np.float64
        )

        kfold = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        # Work with positional indices because KFold
        # returns integer positions.
        X_reset = X.reset_index(drop=True)
        y_reset = y.reset_index(drop=True)

        for train_positions, validation_positions in kfold.split(
            X_reset
        ):

            X_fold_train = X_reset.iloc[
                train_positions
            ]

            y_fold_train = y_reset.iloc[
                train_positions
            ]

            X_fold_validation = X_reset.iloc[
                validation_positions
            ]

            # Calculate statistics ONLY from
            # the other four folds.
            fold_global_mean, fold_mapping = (
                self._calculate_mapping(
                    X_fold_train,
                    y_fold_train
                )
            )

            for column in self.columns:

                values = X_fold_validation[column].fillna(
                    "__MISSING__"
                )

                encoded_values = (
                    values
                    .map(fold_mapping[column])
                    .fillna(fold_global_mean)
                    .astype(np.float64)
                )

                validation_original_index = X.index[
                    validation_positions
                ]

                oof_encoded.loc[
                    validation_original_index,
                    column
                ] = encoded_values.to_numpy()

        # IMPORTANT:
        # After generating OOF values, fit the final
        # statistics using ALL training data.
        self.fit(X, y)

        return oof_encoded