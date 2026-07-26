import sys

from logger import logger
from exception import CustomException

from components.data_ingestion      import DataIngestion
from components.data_validation     import DataValidation
from components.feature_engineering import FeatureEngineering
from components.data_transformation import DataTransformation
from components.model_trainer       import ModelTrainer
from components.model_evaluation    import ModelEvaluation


class TrainingPipeline:
    """
    Orchestrates the full ML pipeline in sequence:

        DataIngestion
            → DataValidation
                → FeatureEngineering
                    → DataTransformation
                        → ModelTrainer
                            → ModelEvaluation
    """

    def run(self):
        try:
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            # ----------------------------------------------------------
            # Stage 1: Data Ingestion
            # ----------------------------------------------------------
            print("\n[1/6] Data Ingestion ...")
            ingestion  = DataIngestion()
            ing_artifact = ingestion.initiate_data_ingestion()
            logger.info(f"Ingestion Artifact : {ing_artifact}")

            # ----------------------------------------------------------
            # Stage 2: Data Validation
            # ----------------------------------------------------------
            print("\n[2/6] Data Validation ...")
            validation  = DataValidation()
            val_artifact = validation.initiate_data_validation()
            logger.info(f"Validation Artifact : {val_artifact}")

            if not val_artifact.validation_status:
                raise Exception(
                    f"Data validation FAILED. Check report at: "
                    f"{val_artifact.validation_report_path}"
                )

            # ----------------------------------------------------------
            # Stage 3: Feature Engineering
            # ----------------------------------------------------------
            print("\n[3/6] Feature Engineering ...")
            fe          = FeatureEngineering()
            fe_artifact = fe.initiate_feature_engineering()
            logger.info(f"Feature Engineering Artifact : {fe_artifact}")

            # ----------------------------------------------------------
            # Stage 4: Data Transformation
            # ----------------------------------------------------------
            print("\n[4/6] Data Transformation (Scaling + SMOTE) ...")
            transformation   = DataTransformation(fe_artifact)
            trans_artifact   = transformation.initiate_data_transformation()
            logger.info(f"Transformation Artifact : {trans_artifact}")

            # ----------------------------------------------------------
            # Stage 5: Model Training
            # ----------------------------------------------------------
            print("\n[5/6] Model Training ...")
            trainer           = ModelTrainer(trans_artifact)
            trainer_artifact  = trainer.initiate_model_training()
            logger.info(f"Trainer Artifact : {trainer_artifact}")

            # ----------------------------------------------------------
            # Stage 6: Model Evaluation
            # ----------------------------------------------------------
            print("\n[6/6] Model Evaluation ...")
            evaluator         = ModelEvaluation(trainer_artifact)
            eval_artifact     = evaluator.initiate_model_evaluation()
            logger.info(f"Evaluation Artifact : {eval_artifact}")

            # ----------------------------------------------------------
            # Summary
            # ----------------------------------------------------------
            logger.info("=" * 60)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            print("\n" + "=" * 60)
            print("  TRAINING PIPELINE — SUMMARY")
            print("=" * 60)
            print(f"  Best Model  : {trainer_artifact.best_model_name}")
            print(f"  Train F1    : {trainer_artifact.train_f1_score}")
            print(f"  Test  F1    : {trainer_artifact.test_f1_score}")
            print(f"  Accuracy    : {eval_artifact.accuracy}")
            print(f"  Precision   : {eval_artifact.precision}")
            print(f"  Recall      : {eval_artifact.recall}")
            print(f"  ROC AUC     : {eval_artifact.roc_auc}")
            print(f"  Accepted    : {'YES' if eval_artifact.model_accepted else 'NO'}")
            print("=" * 60)

            return eval_artifact

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)
