import os
import sys
import json

import pandas as pd

from logger import logger
from exception import CustomException
from config import (
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    VALIDATION_REPORT_PATH,
    TARGET_COLUMN,
    NUMERICAL_FEATURES,
    DROP_COLUMNS,
)
from entity.config_entity import DataValidationConfig
from entity.artifact_entity import DataValidationArtifact


# All columns that MUST be present in the dataset (before engineering)
REQUIRED_COLUMNS = [
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF",
]


class DataValidation:

    def __init__(self):
        self.validation_config = DataValidationConfig(
            train_data_path=TRAIN_DATA_PATH,
            test_data_path=TEST_DATA_PATH,
            validation_report_path=VALIDATION_REPORT_PATH,
            required_columns=REQUIRED_COLUMNS,
            target_column=TARGET_COLUMN,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame, label: str) -> dict:
        missing = [c for c in self.validation_config.required_columns if c not in df.columns]
        extra   = [c for c in df.columns if c not in self.validation_config.required_columns]
        return {
            "dataset": label,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "missing_required_columns": missing,
            "extra_columns": extra,
            "columns_ok": len(missing) == 0,
        }

    def _validate_nulls(self, df: pd.DataFrame, label: str) -> dict:
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        return {
            "dataset": label,
            "columns_with_nulls": cols_with_nulls,
            "nulls_ok": len(cols_with_nulls) == 0,
        }

    def _validate_target(self, df: pd.DataFrame, label: str) -> dict:
        target = self.validation_config.target_column
        if target not in df.columns:
            return {"dataset": label, "target_ok": False, "reason": "target column missing"}
        dist = df[target].value_counts().to_dict()
        return {
            "dataset": label,
            "target_distribution": {str(k): int(v) for k, v in dist.items()},
            "target_ok": True,
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.info("=" * 60)
            logger.info("DATA VALIDATION STARTED")
            logger.info("=" * 60)

            train_df = pd.read_csv(self.validation_config.train_data_path)
            test_df  = pd.read_csv(self.validation_config.test_data_path)

            report = {
                "column_validation": [
                    self._validate_columns(train_df, "train"),
                    self._validate_columns(test_df,  "test"),
                ],
                "null_validation": [
                    self._validate_nulls(train_df, "train"),
                    self._validate_nulls(test_df,  "test"),
                ],
                "target_validation": [
                    self._validate_target(train_df, "train"),
                    self._validate_target(test_df,  "test"),
                ],
            }

            # Determine overall status
            col_ok    = all(r["columns_ok"] for r in report["column_validation"])
            null_ok   = all(r["nulls_ok"]   for r in report["null_validation"])
            target_ok = all(r["target_ok"]  for r in report["target_validation"])

            validation_status = col_ok and null_ok and target_ok

            report["overall_status"] = "PASSED" if validation_status else "FAILED"
            report["columns_ok"]     = col_ok
            report["nulls_ok"]       = null_ok
            report["target_ok"]      = target_ok

            # Save report
            os.makedirs(os.path.dirname(self.validation_config.validation_report_path), exist_ok=True)
            with open(self.validation_config.validation_report_path, "w") as f:
                json.dump(report, f, indent=4)

            status_str = "[PASSED]" if validation_status else "[FAILED]"
            logger.info(f"Data Validation Status : {status_str}")
            logger.info("DATA VALIDATION COMPLETED")

            print(f"\nData Validation : {status_str}")
            print(f"Report saved to : {self.validation_config.validation_report_path}")

            message = f"Validation {report['overall_status']} | columns_ok={col_ok} | nulls_ok={null_ok} | target_ok={target_ok}"

            return DataValidationArtifact(
                validation_status=validation_status,
                validation_report_path=self.validation_config.validation_report_path,
                message=message,
            )

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


# ===================================================
# Testing
# ===================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Data Validation")
    print("=" * 60)

    obj      = DataValidation()
    artifact = obj.initiate_data_validation()

    print("\nValidation Status :", artifact.validation_status)
    print("Message           :", artifact.message)
    print("=" * 60)
