import logging
import os.path
import time


class Logger(object):
    def __init__(self, log_root_path, log_level, logger_name):
        """
        指定保存日志的文件路径，日志级别，以及调用文件
        将日志存入到指定的文件中
        """

        # 创建一个logger
        self.__logger = logging.getLogger(logger_name)
        self.__logger.setLevel(log_level)

        # 创建一个handler，用于写入日志文件
        local_time = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
        log_path = os.path.join(log_root_path, local_time + "_" + self.__logger.name + ".log")
        print(f"Save log to: {log_path}")
        
        # 日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.__logger.addHandler(file_handler)

        # 再创建一个handler，用于输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.__logger.addHandler(console_handler)

    def get_logger(self):
        return self.__logger
