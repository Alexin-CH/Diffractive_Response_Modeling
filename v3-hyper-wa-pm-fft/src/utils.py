import logging
import os
from datetime import datetime

class LoggerManager:
    def __init__(self, log_dir="logs", log_name="train_log"):
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{log_name}_{timestamp}.log")
        
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(logging.DEBUG)

        # File handler
        self.fh = logging.FileHandler(self.log_path)
        self.fh.setLevel(logging.DEBUG)

        # Console handler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)

        # Add handlers (prevent duplicate handlers)
        if not self.logger.handlers:
            self.logger.addHandler(self.fh)
            self.logger.addHandler(self.ch)

    def set_level(self, level):
        """Change the logging level dynamically."""
        self.logger.setLevel(level)
        self.fh.setLevel(level)
        self.ch.setLevel(level)

    def get_logger(self):
        """Return the logger instance."""
        return self.logger
