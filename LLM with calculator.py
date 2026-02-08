import os
import json
import csv
import requests
import time
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Import ground truth calculators
# NOTE: Make sure these are in the same directory or Python path
try:
    from ground_truth_calculators import (
        HVACGroundTruthCalculator,
        ApplianceGroundTruthCalculator,
        ShowerGroundTruthCalculator
    )

except ImportError:
    print("Ground truth calculators not found!")
    load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables!")

MODEL_ID = "mistralai/mistral-small-3.2"
TEMPERATURE = 0.3
CRITERION_WEIGHTS = {
    'energy_cost': 0.35,
    'environmental': 0.30,
    'comfort': 0.20,
    'practicality': 0.15
}

MAX_RETRIES = 3
RETRY_DELAY = 2
EXTRACTION_MAX_RETRIES = 1

OUTPUT_CSV = '/mnt/user-data/outputs/hybrid_results.csv'
OUTPUT_DIAGNOSTICS = '/mnt/user-data/outputs/hybrid_diagnostics.json'

#May remove this; does this even line up with our proposed Hybrid Structure?
# may be giving it too much of an edge
FALLBACK_DEFAULTS = {
    'HVAC': {
        'r_value': {'Poor': 11, 'Medium': 13, 'Good': 19},
        'hvac_age': 15,  # Default to mid-range
        'seer': {'Poor': 11, 'Medium': 13, 'Good': 15},
    },
    'Appliance': {
        'kwh_per_cycle': {'Dishwasher': 1.2, 'Washer': 0.35, 'Dryer': 3.0},
        'appliance_age': 10,
    },
    'Shower': {
        'gpm': 2.5,  # Standard pre-WaterSense
        'water_heater_type': 'Electric',
        'tank_size': 40,
        'water_heater_temp': 120,
    }
}

EXTRACTION_PROMPT_HVAC = """[PLACEHOLDER PROMPT]

You are a technical expert in residential HVAC systems. Extract specific numeric 
parameters from this scenario description.

Extract these parameters as JSON:
1. r_value: Insulation R-value (Poor=11, Medium=13, Good=19)
2. hvac_age: Estimated HVAC system age in years
3. seer: Estimated SEER rating (old: 10-12, medium: 13-14, new: 15-18+)
4. alternatives: Array of 3 temperature setpoints (format: "72F", "76F", "80F")

Return ONLY valid JSON:
{{"r_value": X, "hvac_age": X, "seer": X, "alternatives": ["XF", "XF", "XF"]}}

SCENARIO:
{scenario_text}
"""

EXTRACTION_PROMPT_APPLIANCE = """[PLACEHOLDER PROMPT]

You are an expert in residential appliances and energy consumption. Extract 
specific parameters from this scenario.

Extract these parameters as JSON:
1. appliance_type: "Dishwasher", "Washer", or "Dryer"
2. kwh_per_cycle: Estimated energy consumption per cycle
3. appliance_age: Estimated appliance age in years
4. alternatives: Array of 3 time options (format: "Run at 7pm", "Run at 11pm", etc.)

Return ONLY valid JSON:
{{"appliance_type": "...", "kwh_per_cycle": X, "appliance_age": X, "alternatives": [...]}}

SCENARIO:
{scenario_text}
"""

EXTRACTION_PROMPT_SHOWER = """[PLACEHOLDER PROMPT]

You are an expert in residential water heating and plumbing. Extract specific 
parameters from this scenario.

Extract these parameters as JSON:
1. gpm: Flow rate in gallons per minute
2. water_heater_type: "Electric" or "Gas"
3. tank_size: Water heater tank size in gallons
4. water_heater_temp: Setpoint temperature in °F
5. alternatives: Array of 3 shower durations in minutes (format: 5, 8, 12)

Return ONLY valid JSON:
{{"gpm": X, "water_heater_type": "...", "tank_size": X, 
 "water_heater_temp": X, "alternatives": [X, X, X]}}

SCENARIO:
{scenario_text}
"""
def query_openrouter(messages: List[Dict], model: str = MODEL_ID,
                     temperature: float = TEMPERATURE) -> Tuple[Dict, Dict]:
    """
    Query OpenRouter API with retry logic.
    EXACT COPY from pure_prompting.py and rag_enhanced.py
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }

    for attempt in range(MAX_RETRIES):
        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            latency = time.time() - start_time

            if response.status_code == 200:
                data = response.json()

                usage = data.get('usage', {})
                diagnostics = {
                    'prompt_tokens': usage.get('prompt_tokens', 0),
                    'completion_tokens': usage.get('completion_tokens', 0),
                    'total_tokens': usage.get('total_tokens', 0),
                    'latency_seconds': latency,
                    'model': model
                }

                return data, diagnostics
            else:
                print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {response.status_code}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        except Exception as e:
            print(f"  Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    raise Exception(f"Failed to get response after {MAX_RETRIES} attempts")


def format_scenario_for_extraction(scenario: Dict) -> str:
    """
    Convert scenario dict to natural language text for extraction prompt.
    """
    lines = []
    for key, value in scenario.items():
        if key not in ['Question']:
            lines.append(f"- {key}: {value}")
    return '\n'.join(lines)

def extract_hvac_parameters(scenario: Dict) -> Tuple[Optional[Dict], Dict]:
    """
    Extract HVAC-specific parameters using AI.

    Returns:
        (extracted_params_dict, extraction_diagnostics)
    """
    scenario_text = format_scenario_for_extraction(scenario)
    prompt = EXTRACTION_PROMPT_HVAC.format(scenario_text=scenario_text)

    messages = [
        {"role": "user", "content": prompt}
    ]

    extraction_diagnostics = {
        'attempts': 0,
        'success': False,
        'used_fallback': False,
        'extraction_error': None
    }

    for attempt in range(EXTRACTION_MAX_RETRIES + 1):
        extraction_diagnostics['attempts'] += 1

        try:
            response, api_diagnostics = query_openrouter(messages)
            response_text = response['choices'][0]['message']['content']
            import re
            json_match = re.search(r'\{[^}]+\}', response_text)

            if json_match:
                extracted = json.loads(json_match.group())

                # Validate required keys
                required_keys = ['r_value', 'hvac_age', 'seer', 'alternatives']
                if all(k in extracted for k in required_keys):
                    extraction_diagnostics['success'] = True
                    extraction_diagnostics.update(api_diagnostics)
                    return extracted, extraction_diagnostics

            print(f"Extraction attempt {attempt + 1} failed to parse JSON")
            extraction_diagnostics['extraction_error'] = "Invalid JSON format"

        except Exception as e:
            print(f"extraction attempt{attempt + 1} error: {e}")
            extraction_diagnostics['extraction_error'] = str(e)
    print(" Using fallback defaults for HVAC parameters")
    extraction_diagnostics['used_fallback'] = True

    insulation = scenario.get('Insulation', 'Medium')
    fallback = {
        'r_value': FALLBACK_DEFAULTS['HVAC']['r_value'].get(insulation, 13),
        'hvac_age': min(15, FALLBACK_DEFAULTS['HVAC']['hvac_age']),
        'seer': FALLBACK_DEFAULTS['HVAC']['seer'].get(insulation, 13),
        'alternatives': ['72F', '76F', '80F']
    }

    return fallback, extraction_diagnostics


def extract_appliance_parameters(scenario: Dict) -> Tuple[Optional[Dict], Dict]:
    """
    Extract Appliance-specific parameters using AI.
    Returns:
        (extracted_params_dict, extraction_diagnostics)
    """
    scenario_text = format_scenario_for_extraction(scenario)
    prompt = EXTRACTION_PROMPT_APPLIANCE.format(scenario_text=scenario_text)

    messages = [
        {"role": "user", "content": prompt}
    ]

    extraction_diagnostics = {
        'attempts': 0,
        'success': False,
        'used_fallback': False,
        'extraction_error': None
    }

    for attempt in range(EXTRACTION_MAX_RETRIES + 1):
        extraction_diagnostics['attempts'] += 1

        try:
            response, api_diagnostics = query_openrouter(messages)
            response_text = response['choices'][0]['message']['content']

            import re
            json_match = re.search(r'\{[^}]+\}', response_text)

            if json_match:
                extracted = json.loads(json_match.group())

                required_keys = ['appliance_type', 'kwh_per_cycle', 'appliance_age', 'alternatives']
                if all(k in extracted for k in required_keys):
                    extraction_diagnostics['success'] = True
                    extraction_diagnostics.update(api_diagnostics)
                    return extracted, extraction_diagnostics

            print(f"Extraction attempt {attempt + 1} failed to parse JSON")
            extraction_diagnostics['extraction_error'] = "Invalid JSON format"

        except Exception as e:
            print(f"extraction attempt {attempt + 1} error: {e}")
            extraction_diagnostics['extraction_error'] = str(e)
    print("Using fallback defaults for Appliance parameters")
    extraction_diagnostics['used_fallback'] = True
    fallback = {
        'appliance_type': "dishwasher",
        'kwh_per_cycle': FALLBACK_DEFAULTS['Appliance']['kwh_per_cycle'].get(dishwasher, 1.2),
        'appliance_age': FALLBACK_DEFAULTS['Appliance']['appliance_age'],
        'alternatives': ['Run at 7pm', 'Run at 11pm', 'Run at 6am']
    }

    return fallback, extraction_diagnostics


def extract_shower_parameters(scenario: Dict) -> Tuple[Optional[Dict], Dict]:
    """
    Extract Shower-specific parameters using AI.

    Returns:
        (extracted_params_dict, extraction_diagnostics)
    """
    scenario_text = format_scenario_for_extraction(scenario)
    prompt = EXTRACTION_PROMPT_SHOWER.format(scenario_text=scenario_text)

    messages = [
        {"role": "user", "content": prompt}
    ]

    extraction_diagnostics = {
        'attempts': 0,
        'success': False,
        'used_fallback': False,
        'extraction_error': None
    }

    for attempt in range(EXTRACTION_MAX_RETRIES + 1):
        extraction_diagnostics['attempts'] += 1

        try:
            response, api_diagnostics = query_openrouter(messages)
            response_text = response['choices'][0]['message']['content']

            import re
            json_match = re.search(r'\{[^}]+\}', response_text)

            if json_match:
                extracted = json.loads(json_match.group())

                required_keys = ['gpm', 'water_heater_type', 'tank_size', 'water_heater_temp', 'alternatives']
                if all(k in extracted for k in required_keys):
                    extraction_diagnostics['success'] = True
                    extraction_diagnostics.update(api_diagnostics)
                    return extracted, extraction_diagnostics

            print(f"  ⚠ Extraction attempt {attempt + 1} failed to parse JSON")
            extraction_diagnostics['extraction_error'] = "Invalid JSON format"

        except Exception as e:
            print(f"  ⚠ Extraction attempt {attempt + 1} error: {e}")
            extraction_diagnostics['extraction_error'] = str(e)

    # Fallback
    print("  ⚠ Using fallback defaults for Shower parameters")
    extraction_diagnostics['used_fallback'] = True

    fallback = {
        'gpm': FALLBACK_DEFAULTS['Shower']['gpm'],
        'water_heater_type': FALLBACK_DEFAULTS['Shower']['water_heater_type'],
        'tank_size': FALLBACK_DEFAULTS['Shower']['tank_size'],
        'water_heater_temp': FALLBACK_DEFAULTS['Shower']['water_heater_temp'],
        'alternatives': [5, 8, 12]  # Duration in minutes
    }

    return fallback, extraction_diagnostics


def score_with_ground_truth(scenario: Dict, extracted_params: Dict,
                            decision_type: str) -> Dict:
    """
    Feed extracted parameters to appropriate ground truth calculator.

    Returns:
        Dict with scores for all alternatives
    """
    gt_scenario = {**scenario, **extracted_params}
    alternatives = extracted_params.get('alternatives', [])
    for i, alt in enumerate(alternatives[:3], 1):
        gt_scenario[f'Alternative {i}'] = alt
    if decision_type == 'HVAC':
        result = HVACGroundTruthCalculator.calculate_scenario_scores(gt_scenario)
    elif decision_type == 'Appliance':
        result = ApplianceGroundTruthCalculator.calculate_scenario_scores(gt_scenario)
    elif decision_type == 'Shower':
        result = ShowerGroundTruthCalculator.calculate_scenario_scores(gt_scenario)
    else:
        raise ValueError(f"Unknown decision type: {decision_type}")
    alternatives_scores = []
    for alt_data in result['alternatives']:
        alternatives_scores.append({
            'alternative': alt_data['alternative'],
            'scores': alt_data['transformed_values']
        })

    return alternatives_scores


def apply_mavt_ranking(alternatives_scores: List[Dict]) -> Dict:
    """
    Apply MAVT weighted sum to rank alternatives.
    """
    weighted_scores = []

    for alt_data in alternatives_scores:
        scores = alt_data['scores']
        weighted_sum = (
                CRITERION_WEIGHTS['energy_cost'] * scores['energy_cost'] +
                CRITERION_WEIGHTS['environmental'] * scores['environmental'] +
                CRITERION_WEIGHTS['comfort'] * scores['comfort'] +
                CRITERION_WEIGHTS['practicality'] * scores['practicality']
        )
        weighted_scores.append({
            'alternative': alt_data['alternative'],
            'weighted_score': weighted_sum,
            'raw_scores': scores
        })

    # Sort by weighted score (descending)
    ranked = sorted(weighted_scores, key=lambda x: x['weighted_score'], reverse=True)

    return {
        'ranked_alternatives': [r['alternative'] for r in ranked],
        'weighted_scores': [r['weighted_score'] for r in ranked],
        'details': ranked
    }


# ========== ORCHESTRATION ==========

def run_scenario(scenario: Dict) -> Dict:
    """
    Run Hybrid approach on a single scenario.

    Process:
    1. Detect decision type
    2. Extract parameters with AI
    3. Feed to ground truth calculator
    4. Apply MAVT ranking

    Returns:
        Dict with results and diagnostics
    """
    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {scenario.get('Question', 'N/A')}")
    print(f"{'=' * 70}")

    # Step 1: Detect decision type
    decision_type = detect_decision_type(scenario)
    print(f"Decision type: {decision_type}")

    # Step 2: Extract parameters
    print(f"Extracting {decision_type} parameters...")

    if decision_type == 'HVAC':
        extracted_params, extraction_diag = extract_hvac_parameters(scenario)
    elif decision_type == 'Appliance':
        extracted_params, extraction_diag = extract_appliance_parameters(scenario)
    elif decision_type == 'Shower':
        extracted_params, extraction_diag = extract_shower_parameters(scenario)
    else:
        raise ValueError(f"Unknown decision type: {decision_type}")

    print(f"  Extraction {'succeeded' if extraction_diag['success'] else 'used fallback'}")
    print(f"  Extracted: {extracted_params}")

    # Step 3: Score with ground truth
    print(f"Calculating ground truth scores...")

    try:
        alternatives_scores = score_with_ground_truth(scenario, extracted_params, decision_type)

        for alt_data in alternatives_scores:
            scores = alt_data['scores']
            print(f"  {alt_data['alternative']}: "
                  f"Energy={scores['energy_cost']:.1f}, "
                  f"Env={scores['environmental']:.1f}, "
                  f"Comfort={scores['comfort']:.1f}, "
                  f"Pract={scores['practicality']:.1f}")

    except Exception as e:
        print(f"  ⚠ Ground truth calculation failed: {e}")
        # Return empty result
        return {
            'scenario': scenario.get('Question', 'N/A'),
            'decision_type': decision_type,
            'error': str(e),
            'extraction_diagnostics': extraction_diag
        }
    ranking_result = apply_mavt_ranking(alternatives_scores)

    print(f"\nRANKING:")
    for i, (alt, score) in enumerate(zip(
            ranking_result['ranked_alternatives'],
            ranking_result['weighted_scores']
    ), 1):
        print(f"  {i}. {alt} (weighted score: {score:.2f})")

    return {
        'scenario': scenario.get('Question', 'N/A'),
        'decision_type': decision_type,
        'extracted_parameters': extracted_params,
        'alternatives_scores': alternatives_scores,
        'ranking_result': ranking_result,
        'extraction_diagnostics': extraction_diag
    }

