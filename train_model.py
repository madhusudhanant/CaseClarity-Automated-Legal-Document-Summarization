"""
Train the T5-small model on legal document summarization dataset.
Run this script to train the model without interruptions.
"""

import os
os.environ['TRANSFORMERS_NO_TF'] = '1'  # Disable TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

from datascience.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from datascience.logging import logger

if __name__ == "__main__":
    STAGE_NAME = "Model Training stage"
    try:
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<<<")
        logger.info("This will take 1-2 hours depending on your hardware (GPU vs CPU)")
        logger.info("Training 6000 examples with T5-small model")
        
        pipeline = ModelTrainerTrainingPipeline()
        pipeline.main()
        
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<\n\nx==========x")
        logger.info("Model saved to: artifacts/model_trainer/t5-legal-model/")
        logger.info("Tokenizer saved to: artifacts/model_trainer/tokenizer/")
        
    except Exception as e:
        logger.exception(e)
        raise e
