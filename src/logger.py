import os
import logging
from datetime import datetime

# ==========================
# Create Logs Directory
# ==========================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ==========================
# Log File Name
# ==========================
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# ==========================
# Logger Configuration
# ==========================
logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s : %(message)s"
)

logger = logging.getLogger("MLPipeline")


# ==========================
# Test Function
# ==========================
def test_logger():
    logger.info("Logger Test Started")
    logger.info("This is an INFO message")
    logger.warning("This is a WARNING message")
    logger.error("This is an ERROR message")

    print("✅ Logger is working successfully.")
    print(f"📁 Log file created at: {LOG_PATH}")


# ==========================
# Run Test
# ==========================
if __name__ == "__main__":
    test_logger()