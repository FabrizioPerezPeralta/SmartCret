"""Data loader and validator conforming to configs/schema.yaml."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml


@dataclass
class ValidationReport:
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    duplicates_removed: int = 0
    missing_imputed: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.valid_rows > 0 and len(self.errors) == 0


class DataLoader:
    """Loads and validates concrete mix datasets according to schema.yaml."""

    def __init__(self, schema_path: Optional[Union[str, Path]] = None):
        if schema_path is None:
            schema_path = Path("configs/schema.yaml")
        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()
        self._build_alias_map()

    def _load_schema(self) -> Dict[str, Any]:
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema not found at {self.schema_path}")
        with open(self.schema_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_alias_map(self) -> None:
        self.alias_map: Dict[str, str] = {}
        all_defs = {}
        all_defs.update(self.schema.get("columns", {}))
        all_defs.update(self.schema.get("targets", {}))
        all_defs.update(self.schema.get("metadata", {}))

        for canonical_name, config in all_defs.items():
            self.alias_map[canonical_name.lower()] = canonical_name
            for alias in config.get("aliases", []):
                self.alias_map[alias.lower().strip()] = canonical_name

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using alias dictionary."""
        renamed = {}
        for col in df.columns:
            clean_col = str(col).strip().lower().replace(" ", "_")
            if clean_col in self.alias_map:
                renamed[col] = self.alias_map[clean_col]
            else:
                renamed[col] = clean_col
        return df.rename(columns=renamed)

    def validate_and_clean(
        self, df: pd.DataFrame, drop_invalid: bool = True
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Validates dataframe against schema and cleans/imputes where configured."""
        report = ValidationReport(total_rows=len(df))
        df = self.normalize_columns(df.copy())

        # Check duplicates
        init_len = len(df)
        df = df.drop_duplicates()
        report.duplicates_removed = init_len - len(df)

        col_defs = self.schema.get("columns", {})
        target_defs = self.schema.get("targets", {})
        meta_defs = self.schema.get("metadata", {})
        all_defs = {**col_defs, **target_defs, **meta_defs}

        # Ensure required columns exist
        for col_name, spec in all_defs.items():
            if spec.get("required", False) and col_name not in df.columns:
                report.errors.append(f"Columna requerida ausente: '{col_name}'")

        if report.errors:
            report.valid_rows = 0
            report.invalid_rows = len(df)
            return df, report

        # Apply defaults for optional columns
        for col_name, spec in col_defs.items():
            if col_name not in df.columns and "default" in spec:
                df[col_name] = spec["default"]
                report.warnings.append(
                    f"Columna '{col_name}' creada con valor por defecto: {spec['default']}"
                )

        # Calculate relacion_ac if missing or ensure consistency
        cementitious = (
            df["cemento_kg"].fillna(0)
            + df.get("ceniza_volante_kg", 0)
            + df.get("escoria_kg", 0)
            + df.get("puzolana_kg", 0)
        )
        calculated_ac = (df["agua_kg"] / cementitious.replace(0, np.nan)).round(3)
        if "relacion_ac" not in df.columns:
            df["relacion_ac"] = calculated_ac
        else:
            df["relacion_ac"] = df["relacion_ac"].fillna(calculated_ac)

        # Ensure metadata fuente exists
        if "fuente" not in df.columns:
            df["fuente"] = "sintetico"
            report.warnings.append("Columna 'fuente' no encontrada; asignada como 'sintetico'")

        # Range and type checks
        valid_mask = pd.Series(True, index=df.index)

        for col_name, spec in all_defs.items():
            if col_name not in df.columns:
                continue

            col = df[col_name]

            # Allowed values
            if "allowed_values" in spec:
                allowed = spec["allowed_values"]
                in_allowed = col.isin(allowed)
                if not in_allowed.all():
                    invalid_cnt = (~in_allowed).sum()
                    report.warnings.append(
                        f"Columna '{col_name}': {invalid_cnt} filas con categorías fuera de las permitidas."
                    )
                    valid_mask &= in_allowed

            # Numeric min/max bounds
            if "min" in spec:
                min_val = spec["min"]
                below_min = col < min_val
                if below_min.any():
                    cnt = below_min.sum()
                    report.warnings.append(
                        f"Columna '{col_name}': {cnt} filas con valor < {min_val}."
                    )
                    valid_mask &= ~below_min

            if "max" in spec:
                max_val = spec["max"]
                above_max = col > max_val
                if above_max.any():
                    cnt = above_max.sum()
                    report.warnings.append(
                        f"Columna '{col_name}': {cnt} filas con valor > {max_val}."
                    )
                    valid_mask &= ~above_max

        # Check NaN values in required features
        nan_mask = df[list(col_defs.keys()) + list(target_defs.keys())].isna().any(axis=1)
        if nan_mask.any():
            report.warnings.append(f"{nan_mask.sum()} filas tienen valores nulos (NaN).")
            valid_mask &= ~nan_mask

        if drop_invalid:
            clean_df = df[valid_mask].copy().reset_index(drop=True)
            report.invalid_rows = len(df) - len(clean_df)
            report.valid_rows = len(clean_df)
            return clean_df, report
        else:
            report.valid_rows = valid_mask.sum()
            report.invalid_rows = (~valid_mask).sum()
            return df, report

    def load_file(
        self, filepath: Union[str, Path], drop_invalid: bool = True
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Loads a CSV or Excel file and validates it against the schema."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {path}")

        if path.suffix.lower() in [".csv", ".txt"]:
            df = pd.read_csv(path)
        elif path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(path)
        else:
            raise ValueError(f"Formato no soportado: {path.suffix}")

        return self.validate_and_clean(df, drop_invalid=drop_invalid)

    def load_directory(
        self, dirpath: Union[str, Path], drop_invalid: bool = True
    ) -> Tuple[pd.DataFrame, ValidationReport]:
        """Loads all CSV/Excel files from a directory and concatenates them."""
        path = Path(dirpath)
        if not path.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {path}")

        files = list(path.glob("*.csv")) + list(path.glob("*.xlsx")) + list(path.glob("*.xls"))
        if not files:
            raise ValueError(f"No se encontraron archivos de datos en {path}")

        dfs = []
        combined_report = ValidationReport()
        for f in files:
            df, r = self.load_file(f, drop_invalid=drop_invalid)
            dfs.append(df)
            combined_report.total_rows += r.total_rows
            combined_report.duplicates_removed += r.duplicates_removed
            combined_report.errors.extend(r.errors)
            combined_report.warnings.extend(r.warnings)

        combined_df = pd.concat(dfs, ignore_index=True)
        if combined_df.duplicated().any():
            init_cnt = len(combined_df)
            combined_df = combined_df.drop_duplicates().reset_index(drop=True)
            combined_report.duplicates_removed += init_cnt - len(combined_df)

        combined_report.valid_rows = len(combined_df)
        combined_report.invalid_rows = combined_report.total_rows - combined_report.valid_rows
        return combined_df, combined_report
