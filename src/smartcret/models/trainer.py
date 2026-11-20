"""Model trainer, cross-validation, hyperparameter selection, and artifact versioning."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml

from smartcret.models.predictor import ConcretePredictor


@dataclass
class ModelMetrics:
    rmse: float
    mae: float
    r2: float
    mape: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse": round(self.rmse, 3),
            "mae": round(self.mae, 3),
            "r2": round(self.r2, 4),
            "mape": round(self.mape, 4),
        }


@dataclass
class TrainingResult:
    best_model_name: str
    metrics_resistencia: ModelMetrics
    metrics_slump: ModelMetrics
    comparison_table: pd.DataFrame
    model_version: str
    artifact_path: str


class ModelTrainer:
    """Trains, benchmarks, and versions concrete performance prediction models."""

    def __init__(self, config_path: Union[str, Path] = "configs/model.yaml"):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.num_features = self.config["features"]["numerical"]
        self.cat_features = self.config["features"].get("categorical", [])
        self.all_features = self.num_features + self.cat_features
        self.targets = self.config["targets"]

    def _build_preprocessor(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    self.cat_features,
                ),
            ],
            remainder="drop",
        )

    def _get_candidate_models(self) -> Dict[str, Any]:
        return {
            "Ridge (Línea base)": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=100, max_depth=12, random_state=42
            ),
            "Gradient Boosting": HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.08, max_leaf_nodes=31, random_state=42
            ),
        }

    def evaluate_predictions(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> ModelMetrics:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        # Handle zero divisions gracefully
        non_zero_mask = y_true > 1e-3
        if non_zero_mask.sum() > 0:
            mape = float(
                np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask]))
            )
        else:
            mape = 0.0
        return ModelMetrics(rmse=rmse, mae=mae, r2=r2, mape=mape)

    def cross_validate_models(
        self, train_df: pd.DataFrame
    ) -> Tuple[str, pd.DataFrame, Dict[str, Pipeline], Dict[str, Pipeline]]:
        """Compares models via 5-Fold Cross Validation and selects top performer."""
        X = train_df[self.all_features]
        y_res = train_df["resistencia_mpa"].values
        y_slump = train_df["slump_mm"].values

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        candidates = self._get_candidate_models()

        summary_rows = []
        fitted_res_pipelines = {}
        fitted_slump_pipelines = {}

        for name, reg in candidates.items():
            pipe_res = Pipeline([("prep", self._build_preprocessor()), ("reg", clone(reg))])
            pipe_slump = Pipeline([("prep", self._build_preprocessor()), ("reg", clone(reg))])

            res_cv_mape = []
            res_cv_r2 = []
            slump_cv_mape = []
            slump_cv_r2 = []

            for train_idx, val_idx in kf.split(X):
                X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
                y_r_tr, y_r_va = y_res[train_idx], y_res[val_idx]
                y_s_tr, y_s_va = y_slump[train_idx], y_slump[val_idx]

                pipe_res.fit(X_tr, y_r_tr)
                pipe_slump.fit(X_tr, y_s_tr)

                p_r = pipe_res.predict(X_va)
                p_s = pipe_slump.predict(X_va)

                res_cv_mape.append(np.mean(np.abs((y_r_va - p_r) / y_r_va)))
                res_cv_r2.append(r2_score(y_r_va, p_r))
                slump_cv_mape.append(
                    np.mean(np.abs((y_s_va[y_s_va > 1] - p_s[y_s_va > 1]) / y_s_va[y_s_va > 1]))
                )
                slump_cv_r2.append(r2_score(y_s_va, p_s))

            # Full fit on training set
            pipe_res.fit(X, y_res)
            pipe_slump.fit(X, y_slump)
            fitted_res_pipelines[name] = pipe_res
            fitted_slump_pipelines[name] = pipe_slump

            summary_rows.append(
                {
                    "Modelo": name,
                    "R² Resistencia (CV)": round(float(np.mean(res_cv_r2)), 3),
                    "MAPE Resistencia (CV)": f"{float(np.mean(res_cv_mape)) * 100:.2f}%",
                    "R² Slump (CV)": round(float(np.mean(slump_cv_r2)), 3),
                    "MAPE Slump (CV)": f"{float(np.mean(slump_cv_mape)) * 100:.2f}%",
                }
            )

        comp_df = pd.DataFrame(summary_rows)

        # Select best model based on mean R²
        best_name = max(
            summary_rows,
            key=lambda x: x["R² Resistencia (CV)"] + x["R² Slump (CV)"],
        )["Modelo"]

        return best_name, comp_df, fitted_res_pipelines, fitted_slump_pipelines

    def train_and_evaluate(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        version: str = "v1",
        output_dir: Union[str, Path] = "models",
    ) -> TrainingResult:
        """Complete workflow: benchmark, calibrate conformal intervals on validation,

        evaluate on reserved test set, and serialize versioned artifacts.
        """
        # 1. Benchmark & Select Best Model
        best_name, comp_df, res_pipes, slump_pipes = self.cross_validate_models(train_df)
        best_pipe_res = res_pipes[best_name]
        best_pipe_slump = slump_pipes[best_name]

        # 2. Conformal Interval Calibration on Validation Set
        X_val = val_df[self.all_features]
        val_res_pred = best_pipe_res.predict(X_val)
        val_slump_pred = best_pipe_slump.predict(X_val)

        res_residuals = np.abs(val_df["resistencia_mpa"].values - val_res_pred)
        slump_residuals = np.abs(val_df["slump_mm"].values - val_slump_pred)

        # 90% confidence level -> 90th percentile of absolute residuals
        q_level = float(self.config["uncertainty"].get("confidence_level", 0.90))
        q_res = (0.0, float(np.percentile(res_residuals, q_level * 100)))
        q_slump = (0.0, float(np.percentile(slump_residuals, q_level * 100)))

        # 3. Calculate Training Bounding Box (Extrapolation detector)
        training_bounds = {}
        for feat in self.num_features:
            training_bounds[feat] = (
                float(train_df[feat].min()),
                float(train_df[feat].max()),
            )

        # 4. Final Evaluation on Reserved Test Set
        X_test = test_df[self.all_features]
        test_res_pred = best_pipe_res.predict(X_test)
        test_slump_pred = best_pipe_slump.predict(X_test)

        metrics_res = self.evaluate_predictions(
            test_df["resistencia_mpa"].values, test_res_pred
        )
        metrics_slump = self.evaluate_predictions(
            test_df["slump_mm"].values, test_slump_pred
        )

        # 5. Build Predictor Object
        metadata = {
            "version": version,
            "trained_at": datetime.now().isoformat(),
            "best_model_name": best_name,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "metrics_test_resistencia": metrics_res.to_dict(),
            "metrics_test_slump": metrics_slump.to_dict(),
            "features_numerical": self.num_features,
            "features_categorical": self.cat_features,
            "uncertainty_q90_resistencia_mpa": round(q_res[1], 2),
            "uncertainty_q90_slump_mm": round(q_slump[1], 1),
        }

        predictor = ConcretePredictor(
            pipeline_resistencia=best_pipe_res,
            pipeline_slump=best_pipe_slump,
            quantiles_resistencia=q_res,
            quantiles_slump=q_slump,
            numerical_features=self.num_features,
            categorical_features=self.cat_features,
            training_bounds=training_bounds,
            metadata=metadata,
        )

        # 6. Save Versioned Artifacts
        version_dir = Path(output_dir) / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_file = version_dir / "predictor.pkl"
        predictor.save(model_file)

        meta_file = version_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        comp_file = version_dir / "model_comparison.csv"
        comp_df.to_csv(comp_file, index=False)

        return TrainingResult(
            best_model_name=best_name,
            metrics_resistencia=metrics_res,
            metrics_slump=metrics_slump,
            comparison_table=comp_df,
            model_version=version,
            artifact_path=str(model_file),
        )
