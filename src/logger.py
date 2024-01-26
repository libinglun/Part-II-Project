import logging
import os
from datetime import datetime
from .utils.const import PROJ_PATH

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(PROJ_PATH, 'logs', f"{datetime.now().strftime('%m_%d_%Y')}")

# create a directory for each day
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE_NAME)

# debug(), info(), warning(), error(), critical() will automatically basicConfig
# Define the behaviour of the root logger
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO              # default logging level
)

mylogger = logging.getLogger("my_logger")

# Test my logger
if __name__=="__main__":
    logging.info("Logging has started")
