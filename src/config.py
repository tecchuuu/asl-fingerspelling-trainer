from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_TRAIN_DIR = DATA_DIR / "raw" / "asl_alphabet_train"
PROCESSED_DIR = DATA_DIR / "processed"
WEBCAM_TEST_DIR = DATA_DIR / "webcam_test"

HAND_MODEL_PATH = ROOT / "models" / "hand_landmarker.task"

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
RUNS_CSV = RESULTS_DIR / "runs.csv"

# J and Z need motion. "nothing" has no hand, so MediaPipe finding no hand plays that role.
LETTERS = list("ABCDEFGHIKLMNOPQRSTUVWXY")
NUM_CLASSES = len(LETTERS)

NUM_LANDMARKS = 21
CROP_SIZE = 64
CROP_MARGIN = 0.25

SEED = 42
