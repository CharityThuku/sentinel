from .base import Detector
from .correlation import CorrelationBreakdown
from .cross_sectional import CrossSectionalResidual
from .digits import DigitDistribution
from .drift import DriftDetector
from .integrity import FeedIntegrity
from .robust_zscore import RobustZScore

__all__ = [
    "Detector",
    "DigitDistribution",
    "CorrelationBreakdown",
    "CrossSectionalResidual",
    "DriftDetector",
    "FeedIntegrity",
    "RobustZScore",
]


def default_stack() -> list[Detector]:
    """The six-layer stack, cheapest and most certain first."""
    return [
        FeedIntegrity(),
        RobustZScore(),
        CrossSectionalResidual(),
        DriftDetector(),
        CorrelationBreakdown(),
        DigitDistribution(),
    ]
