import sys
import os

from src.deep_learning.dl_config import CLEAN_AI4I_PATH, LSTM_MODEL_PATH, CNN_MODEL_PATH
from src.deep_learning.dl_data_prep import prepare_dl_data
from src.deep_learning.lstm_model import SensorLSTM
from src.deep_learning.cnn_model import Sensor1DCNN
from src.deep_learning.dl_trainer import DLTrainer
from src.deep_learning.dl_evaluator import DLEvaluator

class DLTrainingPipeline:
    def run(self):
        print("=" * 60)
        print("  DEEP LEARNING SENSOR FAILURE PIPELINE (LSTM & 1D-CNN)")
        print("=" * 60)

        # 1. Prepare Data
        print("\n[1/3] Preparing Sliding Window Sequences...")
        train_loader, val_loader, input_dim = prepare_dl_data(CLEAN_AI4I_PATH)
        print(f"  Input Features: {input_dim} | Sequences Prepared.")

        # 2. Train LSTM Model
        print("\n[2/3] Training Sensor LSTM Model...")
        lstm_model = SensorLSTM(input_dim=input_dim)
        lstm_trainer = DLTrainer(lstm_model, LSTM_MODEL_PATH)
        lstm_f1 = lstm_trainer.train(train_loader, val_loader)

        # 3. Train 1D-CNN Model
        print("\n[3/3] Training Sensor 1D-CNN Model...")
        cnn_model = Sensor1DCNN(input_dim=input_dim)
        cnn_trainer = DLTrainer(cnn_model, CNN_MODEL_PATH)
        cnn_f1 = cnn_trainer.train(train_loader, val_loader)

        # 4. Evaluation & Save Report
        evaluator = DLEvaluator()
        evaluator.evaluate_model(lstm_model, val_loader, "LSTM")
        evaluator.evaluate_model(cnn_model, val_loader, "1D-CNN")
        evaluator.save_report()

        print("\n" + "=" * 60)
        print(f"  DL Training Complete | LSTM F1: {lstm_f1:.4f} | 1D-CNN F1: {cnn_f1:.4f}")
        print("=" * 60)

if __name__ == "__main__":
    pipeline = DLTrainingPipeline()
    pipeline.run()
