"""
Value Function Library for MAVT
================================
Transforms raw criterion values into normalized value scores [0,1].

Supported types:
- linear: Proportional transformation
- concave: Diminishing returns (risk-averse)
- convex: Accelerating returns (risk-seeking)
- piecewise: Two-segment linear (threshold-based)

All functions map [min_val, max_val] → [0, 1]
"""

import numpy as np
from typing import Dict, Any, Literal

ValueFunctionType = Literal["linear", "concave", "convex", "piecewise"]


class ValueFunction:
    """Base class for all value functions."""

    def __init__(self, vmin: float, vmax: float):
        """
        Args:
            vmin: Minimum possible value for this criterion
            vmax: Maximum possible value for this criterion
        """
        if vmin >= vmax:
            raise ValueError(f"vmin ({vmin}) must be < vmax ({vmax})")
        self.vmin = vmin
        self.vmax = vmax

    def transform(self, x: float) -> float:
        """
        Transform raw value x to normalized value score [0,1].

        Args:
            x: Raw criterion value

        Returns:
            Normalized value score (0 = worst, 1 = best)
        """
        # Clamp to valid range
        x = np.clip(x, self.vmin, self.vmax)

        # Normalize to [0,1]
        x_norm = (x - self.vmin) / (self.vmax - self.vmin)

        # Apply specific transformation
        return self._apply_shape(x_norm)

    def _apply_shape(self, x_norm: float) -> float:
        """Override in subclasses to define shape."""
        raise NotImplementedError


class LinearValueFunction(ValueFunction):
    """Linear transformation: v(x) = x"""

    def _apply_shape(self, x_norm: float) -> float:
        return x_norm


class ConcaveValueFunction(ValueFunction):
    """
    Concave (diminishing returns): v(x) = x^alpha, where 0 < alpha < 1

    Used when early gains matter most (e.g. reducing energy cost from $10→$5
    is more valuable than $2→$1).
    """

    def __init__(self, vmin: float, vmax: float, alpha: float = 0.5):
        """
        Args:
            alpha: Concavity parameter (0 < alpha < 1)
                   Lower = more concave (stronger diminishing returns)
        """
        super().__init__(vmin, vmax)
        if not (0 < alpha < 1):
            raise ValueError(f"Concave alpha must be in (0,1), got {alpha}")
        self.alpha = alpha

    def _apply_shape(self, x_norm: float) -> float:
        return x_norm ** self.alpha


class ConvexValueFunction(ValueFunction):
    """
    Convex (accelerating returns): v(x) = x^beta, where beta > 1

    Used when only high performance matters (e.g. comfort is only "good"
    if it's near-optimal).
    """

    def __init__(self, vmin: float, vmax: float, beta: float = 2.0):
        """
        Args:
            beta: Convexity parameter (beta > 1)
                  Higher = more convex (stronger accelerating returns)
        """
        super().__init__(vmin, vmax)
        if beta <= 1:
            raise ValueError(f"Convex beta must be > 1, got {beta}")
        self.beta = beta

    def _apply_shape(self, x_norm: float) -> float:
        return x_norm ** self.beta


class PiecewiseValueFunction(ValueFunction):
    """
    Piecewise linear with threshold: different slopes before/after threshold.

    Used when there's a qualitative change (e.g. shower <7min is acceptable,
    >7min starts feeling rushed).
    """

    def __init__(self, vmin: float, vmax: float, threshold: float = 0.5,
                 slope_low: float = 0.3, slope_high: float = 0.7):
        """
        Args:
            threshold: Normalized breakpoint in [0,1]
            slope_low: Slope for x < threshold (as fraction of total rise)
            slope_high: Slope for x >= threshold (as fraction of total rise)
        """
        super().__init__(vmin, vmax)
        if not (0 < threshold < 1):
            raise ValueError(f"Threshold must be in (0,1), got {threshold}")
        if slope_low + slope_high <= 0:
            raise ValueError("Slopes must be positive")

        self.threshold = threshold
        self.slope_low = slope_low
        self.slope_high = slope_high

        # Calculate intercept to ensure continuity
        self.intercept = slope_low * threshold

    def _apply_shape(self, x_norm: float) -> float:
        if x_norm < self.threshold:
            return self.slope_low * x_norm
        else:
            return self.intercept + self.slope_high * (x_norm - self.threshold)


def create_value_function(
        func_type: ValueFunctionType,
        vmin: float,
        vmax: float,
        params: Dict[str, Any] = None
) -> ValueFunction:
    """
    Factory function to create value functions from config.

    Args:
        func_type: Type of value function
        vmin: Minimum criterion value
        vmax: Maximum criterion value
        params: Type-specific parameters (optional)

    Returns:
        Configured ValueFunction instance

    Example:
        >>> vf = create_value_function("concave", 0, 100, {"alpha": 0.6})
        >>> vf.transform(50)  # Transform raw value 50
        0.707...
    """
    params = params or {}

    if func_type == "linear":
        return LinearValueFunction(vmin, vmax)
    elif func_type == "concave":
        return ConcaveValueFunction(vmin, vmax, **params)
    elif func_type == "convex":
        return ConvexValueFunction(vmin, vmax, **params)
    elif func_type == "piecewise":
        return PiecewiseValueFunction(vmin, vmax, **params)
    else:
        raise ValueError(f"Unknown value function type: {func_type}")


# Define allowed parameter ranges for validation
PARAM_CONSTRAINTS = {
    "linear": {},  # No parameters
    "concave": {
        "alpha": (0.1, 0.9)  # Must be in (0,1), restrict to reasonable range
    },
    "convex": {
        "beta": (1.1, 3.0)  # Must be >1, cap at 3.0 to avoid extreme curves
    },
    "piecewise": {
        "threshold": (0.2, 0.8),  # Avoid extreme breakpoints
        "slope_low": (0.1, 0.9),
        "slope_high": (0.1, 0.9)
    }
}