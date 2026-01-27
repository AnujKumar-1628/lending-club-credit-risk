from pathlib import Path


def get_project_root() -> Path:
    """
    Returns project root directory.

    """
    return Path(__file__).resolve().parents[3]


# Project root directory
project_root = get_project_root()

data_dir = project_root / "data"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
samples_dir = data_dir / "samples"


def create_dirs():
    for path in [data_dir, raw_dir, processed_dir, samples_dir]:
        path.mkdir(parents=True, exist_ok=True)
