import logging
import os

# Ensure the 'log' directory exists
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

# Define log file path
log_file = os.path.join(log_dir, "app.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a"  # Append mode
)

def log_message(message, level="info"):
    """
    Logs a message to the file.
    
    :param message: The message to log
    :param level: The logging level ('info', 'warning', 'error', 'debug', 'critical')
    """
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)
    elif level == "debug":
        logging.debug(message)
    elif level == "critical":
        logging.critical(message)
    else:
        logging.info(message)  # Default to info level
    
    print(f"✅ Logged: {message}")  # Optional: Print confirmation