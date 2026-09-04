"""sentinel - streaming anomaly detection for financial time series."""

from .engine import AlertPolicy, Engine
from .evaluate import evaluate, format_report
from .synth import generate
from .types import Alert, Severity, Snapshot, Tick

__version__ = "0.1.0"
__all__ = [
    "AlertPolicy", "Engine", "Alert", "Severity", "Snapshot", "Tick",
    "generate", "evaluate", "format_report",
]
