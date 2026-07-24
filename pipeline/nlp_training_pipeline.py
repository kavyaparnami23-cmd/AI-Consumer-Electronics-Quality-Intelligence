"""
pipeline/nlp_training_pipeline.py -- Orchestrates the full NLP training pipeline.
"""

import sys

from logger import logger
from exception import CustomException
from nlp.nlp_data_prep import NLPDataPrep
from nlp.nlp_trainer import NLPTrainer
from nlp.nlp_evaluator import NLPEvaluator


class NLPTrainingPipeline:
    """
    Orchestrates the complete NLP sentiment analysis pipeline:
        1. Data Preparation (Cleaning, Label Mapping, Train/Test Split)
        2. Model Training (TF-IDF Baseline & DistilBERT Fine-tuning)
        3. Model Evaluation & Artifact Generation
    """

    def run(self, run_bert: bool = True):
        try:
            logger.info("=" * 60)
            logger.info("NLP TRAINING PIPELINE STARTED")
            logger.info("=" * 60)

            # Stage 1: Data Preparation
            print("\n[1/3] NLP Data Preparation ...")
            prep = NLPDataPrep()
            prep_data = prep.prepare()
            train_df = prep_data["train_df"]
            test_df = prep_data["test_df"]

            # Stage 2: Model Training
            print("\n[2/3] NLP Model Training ...")
            trainer = NLPTrainer()
            tfidf_model = trainer.train_tfidf(train_df)

            if run_bert:
                bert_model = trainer.train_distilbert(train_df, val_df=test_df)

            # Stage 3: Model Evaluation
            print("\n[3/3] NLP Model Evaluation ...")
            evaluator = NLPEvaluator()

            if run_bert:
                report = evaluator.evaluate_all(test_df)
            else:
                report = evaluator.evaluate_tfidf(test_df)

            logger.info("NLP TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            print("\nNLP Training Pipeline Finished Successfully.")
            return report

        except Exception as e:
            logger.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = NLPTrainingPipeline()
    pipeline.run(run_bert=False)
