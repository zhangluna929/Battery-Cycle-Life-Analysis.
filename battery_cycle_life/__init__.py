from importlib import metadata

try:
    __version__ = metadata.version(__package__ or "battery_cycle_life")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

from .data import DataPipeline, load_data, quick_clean  # noqa: F401
from .analysis import derivative_curves, fit_eis_nyquist  # noqa: F401
from .models import extract_features, train_ensemble  # noqa: F401 