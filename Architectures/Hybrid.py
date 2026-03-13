import os
import json
import requests
import time
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

try:
    from HVACGroundTruthCalculator import (
        HVACGroundTruthCalculator
    )
except ModuleNotFoundError:
    print("couldnt get HVAC ground truth calculator")
try:
    from ApplianceGroundTruthCalculator import (
        ApplianceGroundTruthCalculator
    )
except ModuleNotFoundError:
    print("couldnt get appliance  ground truth calculator")
try:
    from ShowerGroundTruthCalculator import (
        ShowerGroundTruthCalculator
    )
except ModuleNotFoundError:
    print("couldnt get shower ground truth calculator")


load_dotenv()
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in environment variables!")

MODEL_ID = "mistralai/mistral-small-3.2-24b-instruct"
TEMPERATURE = 0.3
CRITERION_WEIGHTS = {
    'energy_cost': 0.30,
    'environmental': 0.35,
    'comfort': 0.20,
    'practicality': 0.15
}

MAX_RETRIES = 3
RETRY_DELAY = 2
EXTRACTION_MAX_RETRIES = 1
OUTPUT_CSV = 'hybrid_results.csv'
OUTPUT_DIAGNOSTICS = 'hybrid_diagnostics.json'
UNIFIED_EXTRACTION_PROMPT = """You are a household decision expert. Analyze this scenario and extract ALL required information in a single response.

SCENARIO:
{scenario_text}

QUESTION: {question}

YOUR TASK:
1. Read the Decision Type from the scenario (HVAC, Appliance, or Shower)
2. Extract the specific parameters needed for that decision type
3. No field should be left blank; if a value is not apparent, it is mandatory to reasonably estimate it based off of available information
3. Select the appropriate ground truth calculator
4. Format alternatives exactly as shown below

Return ONLY valid JSON with this structure:

For HVAC decisions:
{{
  "decision_type": "HVAC",
  "calculator": "HVACGroundTruthCalculator",
  "parameters": {{
    "Location": "<city, state>",
    "square_footage": <number>,
    "Insulation": "<Poor/Medium/Good>",
    "r_value": <number>,
    "household_size": <number>,
    "outdoor_temp": <number>,
    "seer": <number>,
    "hvac_age": <number>,
    "Household Type": "<Apartment/Single-family/Townhouse>",
    "utility_budget": <number>
    "Occupancy Context": "occupied_all_day|unoccupied_<hours>|occupied_sleep",
    "alternatives": ["<temp>", "<temp>", "<temp>"]
  }}
}}

For Appliance decisions:
{{
  "decision_type": "Appliance",
  "calculator": "ApplianceGroundTruthCalculator",
  "parameters": {{
    "Location": "<city, state>",
    "Appliance": "Dishwasher|Washer|Dryer",
    "kwh/cycle": <number>,
    "Appliance Age/Type": "<age> OR <type>",
    "Baseline Time": "<time like 7pm, 8am, 9am>",
    "Peak Rate": <number>,
    "Off-Peak Rate": <number>,
    "Occupants": <number>,
    "Housing Type": "<Apartment/Single-family/Townhouse>",
    "utility_budget": <number>,
    "alternatives": ["<time>", "<time>", "<time>"]
  }}
}}

For Shower decisions:
{{
  "decision_type": "Shower",
  "calculator": "ShowerGroundTruthCalculator",
  "parameters": {{
    "Location": "<city, state>",
    "GPM": <number>,
    "Tank Size": <number>,
    "Water Heater Temp": <number>,
    "outdoor_temp": <number>,
    "Occupants": <number>,
    "Housing Type": "<Apartment/Single-family/Townhouse>",
    "utility_budget": <number>,
    "alternatives": ["<minutes>", "<minutes>", "<minutes>"]
  }}
}}

CRITICAL: Alternative formats must match exactly:
- HVAC: "72", "76", "80" (No suffix)
- Appliance: "7pm", "10pm", "2am" 
- Shower: "5", "10", "15" (No suffix)

Return ONLY the JSON, no explanation.
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
        if key not in ['Question']:  # Don't repeat question in details
            lines.append(f"- {key}: {value}")
    return '\n'.join(lines)


def extract_all_with_ai(scenario: Dict) -> Tuple[Optional[Dict], Dict]:
    
    scenario_text = format_scenario_for_extraction(scenario)
    question = scenario.get('Question', '')

    prompt = UNIFIED_EXTRACTION_PROMPT.format(
        scenario_text=scenario_text,
        question=question
    )

    messages = [{"role": "user", "content": prompt}]

    extraction_diagnostics = {
        'attempts': 0,
        'success': False,
        'extraction_error': None
    }

    for attempt in range(EXTRACTION_MAX_RETRIES + 1):
        extraction_diagnostics['attempts'] += 1

        try:
            response, api_diagnostics = query_openrouter(messages)
            response_text = response['choices'][0]['message']['content']
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)

            if json_match:
                extracted = json.loads(json_match.group())

                required_top_level = ['decision_type', 'calculator', 'parameters']
                if all(k in extracted for k in required_top_level):

                    if extracted['decision_type'] not in ['HVAC', 'Appliance', 'Shower']:
                        print(f" iinvalid decision_type: {extracted['decision_type']}")
                        extraction_diagnostics['extraction_error'] = "Invalid decision_type"
                        continue

                    valid_calculators = ['HVACGroundTruthCalculator', 'ApplianceGroundTruthCalculator',
                                         'ShowerGroundTruthCalculator']
                    if extracted['calculator'] not in valid_calculators:
                        print(f" invalid calculator: {extracted['calculator']}")
                        extraction_diagnostics['extraction_error'] = "Invalid calculator"
                        continue

                    params = extracted['parameters']
                    decision_type = extracted['decision_type']

                    params = extracted['parameters']
                    decision_type = extracted['decision_type']

                    if decision_type == 'HVAC':
                        required_params = ['Location', 'square_footage', 'Insulation', 'r_value',
                                           'seer', 'hvac_age', 'outdoor_temp', 'alternatives']
                    elif decision_type == 'Appliance':
                        required_params = ['Location', 'Appliance', 'kwh/cycle', 'Appliance Age/Type',
                                           'Baseline Time', 'Peak Rate', 'Off-Peak Rate', 'alternatives']
                    elif decision_type == 'Shower':
                        required_params = ['Location', 'GPM', 'Tank Size',
                                           'Water Heater Temp', 'outdoor_temp', 'alternatives']
                    if all(k in params for k in required_params):
                        extraction_diagnostics['success'] = True
                        extraction_diagnostics.update({
                            'prompt_tokens': api_diagnostics.get('prompt_tokens', 0),
                            'completion_tokens': api_diagnostics.get('completion_tokens', 0),
                            'latency_ms': api_diagnostics.get('latency_seconds', 0) * 1000
                        })
                        return extracted, extraction_diagnostics
                    else:
                        print(f"Missing required parameters for {decision_type}")
                        extraction_diagnostics['extraction_error'] = f"Missing parameters: {required_params}"
                        continue

            print(f"Extraction attempt {attempt + 1} failed to parse JSON")
            extraction_diagnostics['extraction_error'] = "Invalid JSON format"

        except Exception as e:
            print(f"Extraction attempt {attempt + 1} error: {e}")
            extraction_diagnostics['extraction_error'] = str(e)

    print("  extraction fail")
    return None, extraction_diagnostics

def score_with_ground_truth(extracted_result: Dict, scenario: Dict) -> List[Dict]:
    gt_scenario = {**scenario, **extracted_result['parameters']}

    if 'utility_budget' in gt_scenario:
        gt_scenario['Utility Budget'] = gt_scenario['utility_budget']
    alternatives = extracted_result['parameters'].get('alternatives', [])
    for i, alt in enumerate(alternatives[:3], 1):
        gt_scenario[f'Alternative {i}'] = alt
    for key in ['Utility Budget', 'Occupants', 'Peak Rate', 'Off-Peak Rate', 'kwh/cycle']:
        if key in gt_scenario and isinstance(gt_scenario[key], str):
            try:
                gt_scenario[key] = float(gt_scenario[key])
            except (ValueError, TypeError):
                gt_scenario[key] = 0.0

    calculator_name = extracted_result['calculator']
    print(f"  Using AI-selected calculator: {calculator_name}")
    
    if calculator_name == 'HVACGroundTruthCalculator':
        calc = HVACGroundTruthCalculator()
        result = calc.calculate_scenario_scores(gt_scenario)
    elif calculator_name == 'ApplianceGroundTruthCalculator':
        calc = ApplianceGroundTruthCalculator()
        result = calc.calculate_scenario_scores(gt_scenario)
    elif calculator_name == 'ShowerGroundTruthCalculator':
        calc = ShowerGroundTruthCalculator()
        result = calc.calculate_scenario_scores(gt_scenario)
    else:
        raise ValueError(f"Unknown calculator: {calculator_name}")
    alternatives_scores = []
    if calculator_name == 'ShowerGroundTruthCalculator':
        for alt_data in result['alternatives']:
            alternatives_scores.append({
                'alternative': str(alt_data['alternative']),
                'scores': {
                    'energy_cost': alt_data['transformed_values']['energy_cost'],
                    'environmental': alt_data['transformed_values']['environmental'],
                    'comfort': alt_data['transformed_values']['comfort'],
                    'practicality': alt_data['transformed_values']['practicality']
                }
            })
    else:  # HVAC and Appliance — identical return structure
        for alt_key, alt_data in result.items():
            alternatives_scores.append({
                'alternative': str(alt_key),
                'scores': {
                    'energy_cost': alt_data['energy_cost_score'],
                    'environmental': alt_data['environmental_score'],
                    'comfort': alt_data['comfort_score'],
                    'practicality': alt_data['practicality_score']
                }
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

def run_scenario(scenario: Dict) -> Dict:
    """
    Run Hybrid approach on a single scenario.

    Process:
    1. SINGLE AI CALL extracts: decision type + parameters + calculator selection
    2. If extraction fails → output zeros and mark as failed
    3. Feed to ground truth calculator (AI-selected)
    4. Apply MAVT ranking

    Returns:
        Dict with results and diagnostics
    """
  
    print(f"SCENARIO: {scenario.get('Question', 'N/A')}")
   
    print(f"AI extracting all information (decision type + parameters + calculator)...")

    extraction_result, extraction_diag = extract_all_with_ai(scenario)

    # Step 2: Check if extraction failed
    if extraction_result is None:
        print(f" EXTRACTION FAILEd. Outputting zero scores")

        # Create zero-score alternatives
        zero_alternatives = []
        for i in range(1, 4):
            zero_alternatives.append({
                'alternative': f'Alternative {i} (extraction failed)',
                'scores': {
                    'energy_cost': 0.0,
                    'environmental': 0.0,
                    'comfort': 0.0,
                    'practicality': 0.0
                }
            })

        ranking_result = apply_mavt_ranking(zero_alternatives)

        return {
            'scenario': scenario.get('Question', 'N/A'),
            'decision_type': 'UNKNOWN',
            'calculator': 'NONE',
            'extraction_failed': True,
            'extracted_result': None,
            'alternatives_scores': zero_alternatives,
            'ranking_result': ranking_result,
            'extraction_diagnostics': extraction_diag
        }

    decision_type = extraction_result['decision_type']
    calculator = extraction_result['calculator']
    parameters = extraction_result['parameters']

    print(f"   Extraction succeeded")
    print(f"  Decision type: {decision_type}")
    print(f"  Calculator: {calculator}")
    print(f"  Parameters: {parameters}")
    print(f"Calculating ground truth scores")

    try:
        alternatives_scores = score_with_ground_truth(extraction_result, scenario)

        for alt_data in alternatives_scores:
            scores = alt_data['scores']
            print(f"  {alt_data['alternative']}: "
                  f"Energy={scores['energy_cost']:.1f}, "
                  f"Env={scores['environmental']:.1f}, "
                  f"Comfort={scores['comfort']:.1f}, "
                  f"Pract={scores['practicality']:.1f}")

    except Exception as e:
        print(f" hround truth calculation failed: {e}")

        # Output zeros on GT calculation failure
        zero_alternatives = []
        for i, alt in enumerate(parameters.get('alternatives', ['Alt1', 'Alt2', 'Alt3'])[:3], 1):
            zero_alternatives.append({
                'alternative': str(alt),
                'scores': {
                    'energy_cost': 0.0,
                    'environmental': 0.0,
                    'comfort': 0.0,
                    'practicality': 0.0
                }
            })

        ranking_result = apply_mavt_ranking(zero_alternatives)

        return {
            'scenario': scenario.get('Question', 'N/A'),
            'decision_type': decision_type,
            'calculator': calculator,
            'extraction_failed': False,
            'gt_calculation_failed': True,
            'extracted_result': extraction_result,
            'alternatives_scores': zero_alternatives,
            'ranking_result': ranking_result,
            'error': str(e),
            'extraction_diagnostics': extraction_diag
        }

    # Step 4: Apply MAVT ranking
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
        'calculator': calculator,
        'extraction_failed': False,
        'gt_calculation_failed': False,
        'extracted_result': extraction_result,
        'alternatives_scores': alternatives_scores,
        'ranking_result': ranking_result,
        'extraction_diagnostics': extraction_diag
    }


def run_test_set(test_csv_path: str, output_csv_path: str,
                 output_diagnostics_path: str) -> Dict:
    """
    Run Hybrid approach on full test set.

    Args:
        test_csv_path: Path to test scenarios CSV
        output_csv_path: Path to save results CSV
        output_diagnostics_path: Path to save diagnostics JSON

    Returns:
        Summary statistics dict
    """
    import csv as csv_module


    print(f"Loading test scenarios from: {test_csv_path}")

    scenarios = []
    with open(test_csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv_module.DictReader(f)
        first_row = next(reader)

        # Validate required columns
        required_cols = ['Question', 'Decision Type']
        missing_cols = [col for col in required_cols if col not in first_row]

        if missing_cols:
            raise ValueError(f" Missing required columns: {missing_cols}")

        scenarios.append(first_row)
        scenarios.extend(list(reader))

    print(f" Loaded {len(scenarios)} test scenarios")
    print(f"  Decision types: {set([s.get('Decision Type', 'UNKNOWN') for s in scenarios])}\n")

    # Process all scenarios
    all_results = []
    cumulative_diagnostics = {
        'total_scenarios': len(scenarios),
        'total_api_calls': 0,
        'total_latency_ms': 0.0,
        'total_tokens_input': 0,
        'total_tokens_output': 0,
        'successful_calls': 0,
        'failed_calls': 0
    }
    for i, scenario in enumerate(scenarios):
        print(f"\n[{i + 1}/{len(scenarios)}] Processing: {scenario.get('Question', 'N/A')[:60]}...")

        result = run_scenario(scenario)
        all_results.append(result)

        cumulative_diagnostics['total_api_calls'] += 1

        ext_diag = result.get('extraction_diagnostics', {})
        cumulative_diagnostics['total_tokens_input'] += ext_diag.get('prompt_tokens', 0)
        cumulative_diagnostics['total_tokens_output'] += ext_diag.get('completion_tokens', 0)
        cumulative_diagnostics['total_latency_ms'] += ext_diag.get('latency_ms', 0.0)

        if result.get('extraction_failed', False) or result.get('gt_calculation_failed', False):
            cumulative_diagnostics['failed_calls'] += 1
        else:
            cumulative_diagnostics['successful_calls'] += 1
    cumulative_diagnostics['avg_latency_ms'] = (
            cumulative_diagnostics['total_latency_ms'] /
            max(cumulative_diagnostics['total_api_calls'], 1)
    )
    cumulative_diagnostics['success_rate'] = (
            cumulative_diagnostics['successful_calls'] /
            max(cumulative_diagnostics['total_api_calls'], 1)
    )
    print(f"\nSaving results to: {output_csv_path}")

    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = [
            'scenario_id', 'question', 'location', 'decision_type', 'calculator',
            'extraction_failed', 'gt_calculation_failed',
            'alternative', 'energy_cost', 'environmental', 'comfort', 'practicality',
            'rank', 'weighted_score'
        ]
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scenario_id, result in enumerate(all_results, 1):
            question = result['scenario']
            decision_type = result['decision_type']
            calculator = result['calculator']
            extraction_failed = result.get('extraction_failed', False)
            gt_calc_failed = result.get('gt_calculation_failed', False)

            location = scenarios[scenario_id - 1].get('Location', 'N/A')

            # Get ranking details
            ranked_alts = result['ranking_result']['ranked_alternatives']
            weighted_scores = result['ranking_result']['weighted_scores']

            # Write each alternative
            for alt_idx, alt_data in enumerate(result['alternatives_scores']):
                alternative = alt_data['alternative']
                scores = alt_data['scores']

                # Find rank (1-based)
                rank = ranked_alts.index(alternative) + 1
                weighted_score = weighted_scores[ranked_alts.index(alternative)]

                writer.writerow({
                    'scenario_id': scenario_id,
                    'question': question,
                    'location': location,
                    'decision_type': decision_type,
                    'calculator': calculator,
                    'extraction_failed': extraction_failed,
                    'gt_calculation_failed': gt_calc_failed,
                    'alternative': alternative,
                    'energy_cost': scores['energy_cost'],
                    'environmental': scores['environmental'],
                    'comfort': scores['comfort'],
                    'practicality': scores['practicality'],
                    'rank': rank,
                    'weighted_score': weighted_score
                })

    print(f" Results saved to: {output_csv_path}")

    # Save diagnostics
    print(f"Saving diagnostics to: {output_diagnostics_path}")

    with open(output_diagnostics_path, 'w', encoding='utf-8-sig') as f:
        json.dump(cumulative_diagnostics, f, indent=2)

    print(f" Diagnostics saved to: {output_diagnostics_path}")


    print(f"HYBRID TEST COMPLETE")
    print(f"Total scenarios: {cumulative_diagnostics['total_scenarios']}")
    print(f"Total API calls: {cumulative_diagnostics['total_api_calls']}")
    print(f"Successful calls: {cumulative_diagnostics['successful_calls']}")
    print(f"Failed calls: {cumulative_diagnostics['failed_calls']}")
    print(f"Total tokens (input): {cumulative_diagnostics['total_tokens_input']}")
    print(f"Total tokens (output): {cumulative_diagnostics['total_tokens_output']}")
    print(f"Average latency: {cumulative_diagnostics['avg_latency_ms']:.0f} ms")
    print(f"Success rate: {cumulative_diagnostics['success_rate']:.1%}")

    return cumulative_diagnostics


if __name__ == "__main__":
    import sys
    test_csv = 'TestScenarios.csv'

    if not os.path.exists(test_csv):
        print(f" ERROR: Test scenarios file not found: {test_csv}")
        print("Please upload your test scenarios CSV first.")
        sys.exit(1)

    try:
        from HVACGroundTruthCalculator import HVACGroundTruthCalculator
        from ApplianceGroundTruthCalculator import ApplianceGroundTruthCalculator
        from ShowerGroundTruthCalculator import ShowerGroundTruthCalculator

        print(" Ground truth calculators loaded")
    except ImportError as e:
        print(f" ERROR: Could not load ground truth calculators: {e}")
        print("Please ensure HVACGroundTruthCalculator.py, ApplianceGroundTruthCalculator.py, and ShowerGroundTruthCalculator.py are in the same directory.")
        sys.exit(1)

    run_test_set(
        test_csv_path=test_csv,
        output_csv_path=OUTPUT_CSV,
        output_diagnostics_path=OUTPUT_DIAGNOSTICS
    )