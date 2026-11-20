"""Inference model and uncertainty predictor for concrete properties."""

from dataclasses import dataclass, field
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


@dataclass
class PredictionOutput:
    resistencia_mpa: float
    resistencia_lower: float
    resistencia_upper: float
    resistencia_std: float
    slump_mm: float
    slump_lower: float
    slump_upper: float
    slump_std: float
    is_extrapolated: bool = False
    extrapolated_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resistencia_mpa": round(self.resistencia_mpa, 2),
            "resistencia_intervalo_90": [
                round(self.resistencia_lower, 2),
                round(self.resistencia_upper, 2),
            ],
            "resistencia_incertidumbre_mpa": round(self.resistencia_std, 2),
            "slump_mm": round(self.slump_mm, 1),
            "slump_intervalo_90": [
                round(self.slump_lower, 1),
                round(self.slump_upper, 1),
            ],
            "slump_incertidumbre_mm": round(self.slump_std, 1),
            "es_extrapolacion": self.is_extrapolated,
            "variables_fuera_de_rango": self.extrapolated_features,
        }


class ConcretePredictor:
    """Predicts compressive strength and slump with calibrated prediction intervals

    and extrapolation detection.
    """

    def __init__(
        self,
        pipeline_resistencia: Any,
        pipeline_slump: Any,
        quantiles_resistencia: Tuple[float, float],  # (lower_err, upper_err)
        quantiles_slump: Tuple[float, float],
        numerical_features: List[str],
        categorical_features: List[str],
        training_bounds: Dict[str, Tuple[float, float]],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.pipeline_resistencia = pipeline_resistencia
        self.pipeline_slump = pipeline_slump
        self.q_res = quantiles_resistencia
        self.q_slump = quantiles_slump
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.all_features = numerical_features + categorical_features
        self.training_bounds = training_bounds
        self.metadata = metadata or {}

    def check_extrapolation(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[List[str]]]:
        """Detects whether inputs fall outside the convex bounding box of training data."""
        n_samples = len(df)
        is_extrap = np.zeros(n_samples, dtype=bool)
        extrap_features: List[List[str]] = [[] for _ in range(n_samples)]

        for feat, (fmin, fmax) in self.training_bounds.items():
            if feat in df.columns:
                vals = df[feat].values
                margin = 0.02 * (fmax - fmin if fmax > fmin else 1.0)
                out_mask = (vals < (fmin - margin)) | (vals > (fmax + margin))
                for idx in np.where(out_mask)[0]:
                    is_extrap[idx] = True
                    extrap_features[idx].append(
                        f"{feat}={vals[idx]:.1f} fuera de [{fmin:.1f}, {fmax:.1f}]"
                    )

        return is_extrap, extrap_features

    def _prepare_df(self, X: Union[pd.DataFrame, Dict[str, Any]]) -> pd.DataFrame:
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        else:
            df = X.copy()

        # Compute relacion_ac if missing
        if "relacion_ac" not in df.columns or df["relacion_ac"].isna().any():
            cementitious = (
                df["cemento_kg"].fillna(0)
                + df.get("ceniza_volante_kg", 0)
                + df.get("escoria_kg", 0)
                + df.get("puzolana_kg", 0)
            )
            df["relacion_ac"] = (df["agua_kg"] / cementitious.replace(0, np.nan)).round(3)

        # Set default values for optional columns
        for col in self.numerical_features:
            if col not in df.columns:
                df[col] = 0.0

        for col in self.categorical_features:
            if col not in df.columns:
                df[col] = "Tipo I"

        return df

    def predict(
        self, X: Union[pd.DataFrame, Dict[str, Any]]
    ) -> Union[PredictionOutput, List[PredictionOutput]]:
        is_single = isinstance(X, dict) or (isinstance(X, pd.DataFrame) and len(X) == 1)
        df = self._prepare_df(X)

        # Extrapolation check
        is_extrap, extrap_feats = self.check_extrapolation(df)

        # Point predictions
        pred_res = self.pipeline_resistencia.predict(df)
        pred_slump = self.pipeline_slump.predict(df)

        # Interval bounds (conformal residuals)
        res_lower = np.maximum(pred_res - self.q_res[1], 5.0)
        res_upper = pred_res + self.q_res[1]
        res_std = (res_upper - res_lower) / (2.0 * 1.645)

        slump_lower = np.maximum(pred_slump - self.q_slump[1], 0.0)
        slump_upper = np.minimum(pred_slump + self.q_slump[1], 250.0)
        slump_std = (slump_upper - slump_lower) / (2.0 * 1.645)

        outputs = []
        for i in range(len(df)):
            out = PredictionOutput(
                resistencia_mpa=float(pred_res[i]),
                resistencia_lower=float(res_lower[i]),
                resistencia_upper=float(res_upper[i]),
                resistencia_std=float(res_std[i]),
                slump_mm=float(pred_slump[i]),
                slump_lower=float(slump_lower[i]),
                slump_upper=float(slump_upper[i]),
                slump_std=float(slump_std[i]),
                is_extrapolated=bool(is_extrap[i]),
                extrapolated_features=extrap_feats[i],
            )
            outputs.append(out)

        return outputs[0] if is_single and isinstance(X, dict) else outputs

    def save(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "ConcretePredictor":
        path = Path(filepath)
        with open(path, "rb") as f:
            return pickle.load(f)
