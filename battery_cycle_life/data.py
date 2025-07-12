from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from data_pipeline import DataPipeline as _DataPipeline

__all__ = ["DataPipeline", "load_data", "quick_clean"]

DataPipeline = _DataPipeline  # re-export


def load_data(source: Union[str, Path, Dict[str, Any]], **kwargs) -> pd.DataFrame:  # noqa: D401
    """Load battery cycling data using the shared DataPipeline."""
    return _DataPipeline().load(source, **kwargs)


def quick_clean(
    source: Union[str, Path, Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Shortcut: load → optional metadata bind → outlier removal."""
    pipe = _DataPipeline()
    df = pipe.load(source)
    if metadata:
        df = pipe.bind_metadata(df, metadata)
    return pipe.detect_outliers(df, **kwargs) 