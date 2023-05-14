import logging

# 创建logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件
log_path = "../logs/log.txt"
file_handler = logging.FileHandler(log_path, mode="a")
file_handler.setLevel(logging.DEBUG)

# 创建一个handler将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)

# 定义handler的输出格式
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将handler添加到logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 日志
logger.debug("这是 logger debug message")
logger.info("这是 logger info message")
logger.warning("这是 logger warning message")
logger.error("这是 logger error message")
logger.critical("这是 logger critical message")