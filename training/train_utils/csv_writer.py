import logging
import csv
from pathlib import Path

class CsvLogger:
    def __init__(self, filepath):
        """
        Initializes a CSV logger.

        Args:
            filepath (str or Path): The path to the CSV file.
        """
        self.filepath = Path(filepath)
        
        # Create directory if it doesn't exist
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        new_filename = f"{self.filepath.stem}_v{self.filepath.suffix}"
        self.val_filepath = self.filepath.with_name(new_filename)
        
    def log(self, data_dict, val=False):
        """
        Logs a row of data to the CSV file and prints keys/values to the log.

        Args:
            data_dict (dict): A dictionary of data to log.
        """
        filepath = self.filepath
        if val:
            filepath = self.val_filepath
        
        row = []
        for k, v in data_dict.items():
            row.append(k)
            row.append(v)

        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
# # --- Example Usage ---
# # 1. Define header and initialize logger
# log_header = ['epoch', 'train_loss', 'val_acc']
# my_logger = CsvLogger('experiment_results.csv', log_header)

# # 2. Log data during your script's execution
# print("Logging epoch 1...")
# my_logger.log({'epoch': 1, 'train_loss': 0.54, 'val_acc': 0.88})

# print("Logging epoch 2...")
# my_logger.log({'epoch': 2, 'train_loss': 0.32, 'val_acc': 0.91})