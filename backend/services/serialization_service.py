"""
Aprixia Serialization Service: Safe JSON encoding for ML/NumPy arrays
"""
import json
import math
from typing import Any
import numpy as np
import pandas as pd


class SafeJSONEncoder(json.JSONEncoder):
    """Encodes NumPy arrays, scalars, and sanitizes NaN / Infinity to None."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        if isinstance(obj, (pd.Timestamp, pd.Period)):
            return str(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)


def sanitize_floats(obj: Any) -> Any:
    """Recursively converts NaN and Inf into None for strict JSON RFC 8259 compliance."""
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_floats(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    return obj


def dumps_safe_json(data: Any) -> str:
    """Serializes data to JSON string ensuring all numerical types are clean."""
    sanitized = sanitize_floats(data)
    return json.dumps(sanitized, cls=SafeJSONEncoder)


def loads_safe_json(json_str: str) -> Any:
    """Loads JSON string back to Python dict."""
    if not json_str:
        return None
    return json.loads(json_str)
