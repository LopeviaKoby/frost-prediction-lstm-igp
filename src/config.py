from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_LSTM_DIR = MODELS_DIR / "lstm"
MODELS_BASELINES_DIR = MODELS_DIR / "baselines"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
OUTPUTS_METRICS_DIR = OUTPUTS_DIR / "metrics"
OUTPUTS_LOGS_DIR = OUTPUTS_DIR / "logs"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_FIGURES_DIR = REPORTS_DIR / "figures"
REPORTS_TABLES_DIR = REPORTS_DIR / "tables"
REPORTS_PAPER_NOTES_DIR = REPORTS_DIR / "paper_notes"

BASELINE_VERSION = "v01"
RANDOM_STATE = 42
FROST_THRESHOLD_C = 0.0
FORECAST_HORIZON_HOURS = 12
SEQUENCE_LENGTH_HOURS = 24

TRAIN_TARGET_END = "2023-12-31 23:00:00"
VALIDATION_TARGET_END = "2024-12-31 23:00:00"

RAW_DATASETS = {
    "tempsup": "tempsup_hourly_2018_2025.csv",
    "HR": "HR_hourly_2018_2025.csv",
    "radinf": "radinf_hourly_2018_2025.csv",
    "dir": "dir_hourly_2018_2025.csv",
    "vel": "vel_hourly_2018_2025.csv",
    "pp": "pp_hourly_2018_2025.csv",
    "press": "press_hourly_2018_2025.csv",
}

GROUP_LIMITS = {
    "tempsup": (-30.0, 40.0),
    "HR": (0.0, 100.0),
    "radinf": (150.0, 500.0),
    "dir": (0.0, 360.0),
    "vel": (0.0, 40.0),
    "pp": (0.0, 100.0),
    "press": (500.0, 750.0),
}

EXCLUDED_FEATURE_COLUMNS = {
    "frost_event_current",
    f"frost_event_t_plus_{FORECAST_HORIZON_HOURS}h",
    "target_timestamp",
}
