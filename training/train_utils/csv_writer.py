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

    def _get_filepath(self, val=False):
        return self.val_filepath if val else self.filepath

    def _collect_fieldnames(self, rows):
        fieldnames = []
        seen = set()
        if any("epoch" in row for row in rows):
            fieldnames.append("epoch")
            seen.add("epoch")
        for row in rows:
            for key in row.keys():
                if key in seen:
                    continue
                fieldnames.append(key)
                seen.add(key)
        return fieldnames

    def write_history(self, rows, val=False):
        """
        Rewrites the per-phase CSV from full in-memory history.

        Args:
            rows (list[dict]): All rows accumulated for this phase.
        """
        filepath = self._get_filepath(val=val)
        if not rows:
            if filepath.exists():
                filepath.unlink()
            return

        fieldnames = self._collect_fieldnames(rows)
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

    def log(self, data_dict, val=False):
        """
        Compatibility wrapper for single-row writes.

        Args:
            data_dict (dict): A dictionary of data to log.
        """
        self.write_history([data_dict], val=val)
# # --- Example Usage ---
# # 1. Define header and initialize logger
# log_header = ['epoch', 'train_loss', 'val_acc']
# my_logger = CsvLogger('experiment_results.csv', log_header)

# # 2. Log data during your script's execution
# print("Logging epoch 1...")
# my_logger.log({'epoch': 1, 'train_loss': 0.54, 'val_acc': 0.88})

# print("Logging epoch 2...")
# my_logger.log({'epoch': 2, 'train_loss': 0.32, 'val_acc': 0.91})
