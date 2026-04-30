import os
from pathlib import Path

def get_size(start_path):
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(start_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except Exception:
        pass
    return total_size

base = r"e:\ML project\smart-omnisentinel-final\smart-omnisentinel"
for item in os.listdir(base):
    path = os.path.join(base, item)
    if os.path.isdir(path):
        size = get_size(path) / (1024**3)
        print(f"{item}: {size:.2f} GB")
