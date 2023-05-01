from importlib import resources

VALID_CHROMS = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]

DATA_DIR = resources.files("proxbias").joinpath("data")  # type:ignore[attr-defined]
