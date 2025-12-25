"""
Pure Prompting Architecture - MAVT-based Decision System
=========================================================
AI generates MAVT configuration (weights + value functions),
then we apply it to rank alternatives.

Architecture: AI parameterizes → MAVT calculates → Compare to ground truth
"""

import os
import json
import numpy as np
import requests
from dotenv import load_dotenv
from value_functions import create_value_function, PARAM_CONSTRAINTS

load_dotenv()
CRITERION_RANGES = {
    "energy_cost": (0.0, 10.0),  # $0 to $10/day (lower is better)
    "environmental": (0.0, 30.0),  # 0 to 30 lbs CO2/day (lower is better)
    "comfort": (1.0, 10.0),  # 1-10 subjective scale (higher is better)
    "practicality": (1.0, 10.0)  # 1-10 subjective scale (higher is better)
}

# Criteria where lower raw value = higher desirability (need inversion)
INVERT_CRITERIA = {"energy_cost", "environmental"}

# Default weights (AI will override these)
DEFAULT_WEIGHTS = {
    "energy_cost": 0.30,
    "environmental": 0.35,
    "comfort": 0.20,
    "practicality": 0.15
}

def transform_criterion(raw_value: float, criterion_name: str, value_function) -> float:
    """ Transform raw criterion value to normalized MAVT value [0,1].
    functions+variables:
        raw_value: Raw score in criterion's native units
        criterion_name: Name of criterion (for inversion check)
        value_function: ValueFunction object from value_functions.py
    Returns:
        Normalized value score where 1.0 = best, 0.0 = worst
    Example:
        vf = create_value_function("linear", 0, 10, {})
        transform_criterion(2.5, "energy_cost", vf)
        0.75  # $2.5 cost becomes 0.25 normalized, inverted to 0.75
    """
    # Apply value function (maps raw → [0,1])
    normalized = value_function.transform(raw_value)

    # For cost/emissions, invert so lower raw = higher value
    if criterion_name in INVERT_CRITERIA:
        normalized = 1.0 - normalized
    return normalized