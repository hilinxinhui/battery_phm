import logging
from logger import Logger
import numpy as np

def train(logger):
    # logger.info("Training started...")
    epochs = 100
    for epoch in range(epochs):
        # print(epoch)
        data = epoch ** 2
        if epoch % 10 == 0:
            logger.info(f"current epoch: {epoch}, current data: {data}.")
    # logger.info("Training Ended.")

if __name__ == "__main__":
    logger = Logger(
        log_root_path = "../logs/",
        log_level = logging.DEBUG,
        logger_name = "logger_test"
    ).get_logger()

    train(logger)