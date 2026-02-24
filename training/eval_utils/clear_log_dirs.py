# for subdirs under (iterative search) LOG_ROOT="/lc/code/3D/vggt-training/training/logs", if a subdir A has a subdir with name of csv, but this csv subdir is empty, then delete the subdir A

import os
import shutil
import argparse

def find_empty_csv_dirs(log_root: str):
    """
    Finds experiment directories that contain an empty 'csv' subdirectory.

    Args:
        log_root (str): The root directory to start the search from.

    Returns:
        list: A list of directories to be deleted.
    """
    dirs_to_delete = []
    # We walk from top down. This is important because we don't want to check
    # inside a directory that is already marked for deletion.
    for root, dirs, files in os.walk(log_root, topdown=True):
        # If the current directory is already a child of a directory marked for deletion, skip it.
        if any(root.startswith(d) and root != d for d in dirs_to_delete):
            continue

        if 'log.txt' not in files:
            continue

        if 'csv' in dirs:
            # The current `root` is a potential candidate for deletion (subdir A).
            csv_dir = os.path.join(root, 'csv')

            # Check if the 'csv' subdirectory does NOT contain "epoch_metrics.csv".
            if "epoch_metrics.csv" in os.listdir(csv_dir):
                continue
        dirs_to_delete.append(root)
        # We don't need to look further down this path
        dirs[:] = [] 
    return dirs_to_delete

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean up incomplete log directories.")
    parser.add_argument(
        '--log_root', 
        type=str, 
        default="/lc/code/3D/vggt-training/training/logs",
        help="The root directory of the logs to check."
    )
    args = parser.parse_args()

    print(f"Searching for log directories without 'epoch_metrics.csv' in: {args.log_root}\n")
    dirs_to_delete = find_empty_csv_dirs(args.log_root)

    if not dirs_to_delete:
        print("No incomplete log directories found.")
    else:
        print("The following directories will be deleted:")
        for d in dirs_to_delete:
            print(f"- {d}")
        
        if input("\nProceed with deletion? (y/N): ").lower() == 'y':
            for d in dirs_to_delete:
                try:
                    shutil.rmtree(os.path.dirname(d))
                    # shutil.rmtree(d)
                    print(f"Deleted: {d}")
                except OSError as e:
                    print(f"Error deleting {d}: {e}")
            print("\nCleanup complete.")
        else:
            print("\nDeletion cancelled.")