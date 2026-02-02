import os
import json
import csv
import time
import logging
from typing import Dict, List, Tuple
import numpy as np
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

API_CONFIG = {
    "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    "model": "google/gemini-flash-1.5-8b",
    "max_tokens": 500,
    "temperature": 0.3  # Need determinism for reliability
}
CRITERION_WEIGHTS = {
    "energy_cost": 0.35,
    "environmental": 0.30,
    "comfort": 0.20,
    "practicality": 0.15
}

# Criterion polarities for TOPSIS
CRITERION_POLARITIES = {
    "energy_cost": "maximize",
    "environmental": "maximize",
    "comfort": "maximize",
    "practicality": "maximize"
}
def query_openrouter(messages: List[Dict], max_retries: int = 3) -> Tuple[str, Dict]:
    """
    Query OpenRouter API with retry logic

    Args:
        messages: List of message dicts with role and content
        max_retries: Number of retry attempts

    Returns:
        Tuple of (response_text, diagnostics_dict)
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": API_CONFIG["model"],
        "messages": messages,
        "max_tokens": API_CONFIG["max_tokens"],
        "temperature": API_CONFIG["temperature"],
        "response_format": {"type": "json_object"}
    }

    diagnostics = {
        "tokens_input": 0,
        "tokens_output": 0,
        "latency_ms": 0,
        "retries": 0,
        "success": False
    }

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(
                API_CONFIG["endpoint"],
                headers=headers,
                json=payload,
                timeout=30
            )
            latency = (time.time() - start_time) * 1000

            if response.status_code == 200:
                data = response.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                # Extract token usage if available
                usage = data.get("usage", {})
                diagnostics["tokens_input"] = usage.get("prompt_tokens", 0)
                diagnostics["tokens_output"] = usage.get("completion_tokens", 0)
                diagnostics["latency_ms"] = latency
                diagnostics["success"] = True
                diagnostics["retries"] = attempt

                return content, diagnostics
            else:
                logging.warning(f"API error {response.status_code}: {response.text}")
                diagnostics["retries"] = attempt + 1

        except Exception as e:
            logging.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
            diagnostics["retries"] = attempt + 1

        # Exponential backoff
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    # All retries failed
    return None, diagnostics



def build_user_prompt(scenario: Dict, alternative: str) -> str:
    return f"""Score this alternative: "{alternative}"

For the decision: "{scenario.get('Question', 'N/A')}"

SCENARIO CONTEXT:
- Location: {scenario.get('Location', 'N/A')}
- Outdoor Temperature: {scenario.get('Outdoor Temp', 'N/A')}°F
- Home Size: {scenario.get('Square Footage', 'N/A')} sq ft
- Insulation Quality: {scenario.get('Insulation', 'N/A')} (R-value: {scenario.get('R-Value', 'N/A')})
- Household Size: {scenario.get('Household Size', 'N/A')} people
- Utility Budget: ${scenario.get('Utility Budget', 'N/A')}/month
- Housing Type: {scenario.get('Housing Type', 'N/A')}
- House Age: {scenario.get('House Age', 'N/A')}
- HVAC Age: {scenario.get('HVAC Age', 'N/A')} years
- HVAC SEER Rating: {scenario.get('SEER', 'N/A')}

Provide scores (0-10) for all 4 criteria using the calibrations in the system prompt.
Consider how this specific alternative performs given the scenario context.
"""


def score_alternative(scenario: Dict, alternative: str) -> Tuple[Dict, Dict]:
    """
    Score a single alternative using LLM

    Args:
        scenario: Full scenario context
        alternative: Alternative to score

    Returns:
        Tuple of (scores_dict, diagnostics_dict)
    """
    system_prompt = f"""You are an expert HVAC energy analyst trained on peer-reviewed research. 
Your task is to score household energy decisions on 4 criteria using research-backed calibrations.

CRITICAL INSTRUCTIONS:
1. Score each criterion on a 0-10 scale
2. Use the research references to anchor your scores
3. Consider the specific scenario context (outdoor temperature, insulation, HVAC specs)
4. Return ONLY valid JSON with this EXACT format:

{{
    "energy_cost": X.X,
    "environmental": X.X,
    "comfort": X.X,
    "practicality": X.X,
    "reasoning": "Brief 1-2 sentence explanation of scores"
}}

Each score MUST be a number between 0.0 and 10.0.
"""
    user_prompt = build_user_prompt(scenario, alternative)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response, diagnostics = query_openrouter(messages)

    # Default scores if API fails
    default_scores = {
        "energy_cost": 5.0,
        "environmental": 5.0,
        "comfort": 5.0,
        "practicality": 5.0,
        "reasoning": "API failure - using default scores"
    }

    if not response:
        logging.error(f"LLM scoring failed for alternative: {alternative}")
        return default_scores, diagnostics


    try:
        scores = json.loads(response)

        validated_scores = {}
        for criterion in ["energy_cost", "environmental", "comfort", "practicality"]:
            raw_score = scores.get(criterion, 5.0)

            if isinstance(raw_score, (int, float)):
                validated_scores[criterion] = max(0.0, min(10.0, float(raw_score)))
            else:
                logging.warning(f"Invalid score type for {criterion}: {raw_score}")
                validated_scores[criterion] = 5.0

        validated_scores["reasoning"] = scores.get("reasoning", "No reasoning provided")

        return validated_scores, diagnostics

    except json.JSONDecodeError as e:
        logging.error(f"JSON parse failed: {e}. Raw response: {response[:200]}")
        return default_scores, diagnostics


def apply_mavt_ranking(alternatives_scores: List[Dict]) -> Dict:
    """
    Apply MAVT weighted sum to rank alternatives

    Args:
        alternatives_scores: List of dicts with keys: alternative, energy_cost, environmental, comfort, practicality

    Returns:
        Dict with ranked_alternatives, ranks, weighted_scores
    """
    try:
        alternatives = [alt["alternative"] for alt in alternatives_scores]

        # Calculate weighted sum for each alternative
        weighted_scores = []
        for alt_scores in alternatives_scores:
            weighted_sum = (
                    CRITERION_WEIGHTS["energy_cost"] * alt_scores["energy_cost"] +
                    CRITERION_WEIGHTS["environmental"] * alt_scores["environmental"] +
                    CRITERION_WEIGHTS["comfort"] * alt_scores["comfort"] +
                    CRITERION_WEIGHTS["practicality"] * alt_scores["practicality"]
            )
            weighted_scores.append(weighted_sum)

        # Rank alternatives (higher weighted sum = better = lower rank number)
        ranked_indices = np.argsort(weighted_scores)[::-1]  # Descending order
        ranked_alternatives = [alternatives[i] for i in ranked_indices]

        # Create rank numbers (1 = best, 2 = second, 3 = third)
        ranks = [0] * len(alternatives)
        for rank_position, alt_index in enumerate(ranked_indices):
            ranks[alt_index] = rank_position + 1

        return {
            "ranked_alternatives": ranked_alternatives,
            "ranks": ranks,
            "weighted_scores": weighted_scores
        }

    except Exception as e:
        logging.error(f"MAVT ranking failed: {e}")

        # Fallback: rank by average score
        avg_scores = []
        for alt_scores in alternatives_scores:
            avg = np.mean([
                alt_scores["energy_cost"],
                alt_scores["environmental"],
                alt_scores["comfort"],
                alt_scores["practicality"]
            ])
            avg_scores.append(avg)

        ranked_indices = np.argsort(avg_scores)[::-1]
        ranked_alternatives = [alternatives[i] for i in ranked_indices]

        ranks = [0] * len(alternatives)
        for rank_position, alt_index in enumerate(ranked_indices):
            ranks[alt_index] = rank_position + 1

        return {
            "ranked_alternatives": ranked_alternatives,
            "ranks": ranks,
            "weighted_scores": avg_scores,
            "error": str(e)
        }

def run_scenario(scenario: Dict) -> Dict:
    """
    Process one scenario: score all alternatives and rank them

    Args:
        scenario: Full scenario dict

    Returns:
        Results dict with rankings, scores, and diagnostics
    """
    # Extract alternatives
    alternatives = [
        scenario.get("Alternative 1", ""),
        scenario.get("Alternative 2", ""),
        scenario.get("Alternative 3", "")
    ]

    # Score each alternative
    alternatives_scores = []
    total_diagnostics = {
        "api_calls": 0,
        "total_latency_ms": 0,
        "total_tokens_input": 0,
        "total_tokens_output": 0,
        "successful_calls": 0,
        "failed_calls": 0
    }

    for alt in alternatives:
        scores, diagnostics = score_alternative(scenario, alt)

        alternatives_scores.append({
            "alternative": alt,
            **scores
        })

        # Aggregate diagnostics
        total_diagnostics["api_calls"] += 1
        total_diagnostics["total_latency_ms"] += diagnostics["latency_ms"]
        total_diagnostics["total_tokens_input"] += diagnostics["tokens_input"]
        total_diagnostics["total_tokens_output"] += diagnostics["tokens_output"]

        if diagnostics["success"]:
            total_diagnostics["successful_calls"] += 1
        else:
            total_diagnostics["failed_calls"] += 1

    # Rank alternatives using TOPSIS
    ranking_results = apply_mavt_ranking(alternatives_scores)

    return {
        "scenario_id": scenario.get("scenario_id", "N/A"),
        "question": scenario.get("Question", "N/A"),
        "location": scenario.get("Location", "N/A"),
        "outdoor_temp": scenario.get("Outdoor Temp", "N/A"),
        "alternatives_scores": alternatives_scores,
        "ranking_results": ranking_results,
        "diagnostics": total_diagnostics
    }


def run_test_set(test_csv_path: str, output_csv_path: str) -> Dict:
    """
    Run Pure Prompting on test set

    Args:
        test_csv_path: Path to test scenarios CSV
        output_csv_path: Path to save results CSV

    Returns:
        Summary statistics dict
    """
    # Load test scenarios
    scenarios = []
    with open(test_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            row['scenario_id'] = i + 1
            scenarios.append(row)

    logging.info(f"Loaded {len(scenarios)} test scenarios from {test_csv_path}")

    # Process each scenario
    all_results = []
    cumulative_diagnostics = {
        "total_scenarios": len(scenarios),
        "total_api_calls": 0,
        "total_latency_ms": 0,
        "total_tokens_input": 0,
        "total_tokens_output": 0,
        "successful_calls": 0,
        "failed_calls": 0
    }

    for i, scenario in enumerate(scenarios):
        logging.info(f"Processing scenario {i + 1}/{len(scenarios)}: {scenario.get('Question', 'N/A')[:50]}...")

        result = run_scenario(scenario)
        all_results.append(result)

        # Aggregate diagnostics
        diag = result["diagnostics"]
        cumulative_diagnostics["total_api_calls"] += diag["api_calls"]
        cumulative_diagnostics["total_latency_ms"] += diag["total_latency_ms"]
        cumulative_diagnostics["total_tokens_input"] += diag["total_tokens_input"]
        cumulative_diagnostics["total_tokens_output"] += diag["total_tokens_output"]
        cumulative_diagnostics["successful_calls"] += diag["successful_calls"]
        cumulative_diagnostics["failed_calls"] += diag["failed_calls"]

    # Calculate summary statistics
    avg_latency = cumulative_diagnostics["total_latency_ms"] / max(cumulative_diagnostics["total_api_calls"], 1)
    success_rate = cumulative_diagnostics["successful_calls"] / max(cumulative_diagnostics["total_api_calls"], 1)

    # Estimate cost (Gemini Flash is typically free or very cheap, but we'll track tokens)
    cumulative_diagnostics["avg_latency_ms"] = avg_latency
    cumulative_diagnostics["success_rate"] = success_rate

    # Save results to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "scenario_id", "question", "location", "outdoor_temp",
            "alternative", "energy_cost", "environmental", "comfort", "practicality",
            "rank", "topsis_score"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            scenario_id = result["scenario_id"]
            question = result["question"]
            location = result["location"]
            outdoor_temp = result["outdoor_temp"]

            # Get ranking
            ranked_alts = result["ranking_results"]["ranked_alternatives"]
            ranks = result["ranking_results"]["ranks"]
            topsis_scores = result["ranking_results"]["topsis_scores"]

            # Write each alternative as a row
            for alt_scores in result["alternatives_scores"]:
                alt = alt_scores["alternative"]

                # Find rank for this alternative
                try:
                    alt_index = result["ranking_results"]["ranked_alternatives"].index(alt)
                    rank = alt_index + 1
                    topsis_score = topsis_scores[alt_index]
                except ValueError:
                    rank = "N/A"
                    topsis_score = "N/A"

                writer.writerow({
                    "scenario_id": scenario_id,
                    "question": question,
                    "location": location,
                    "outdoor_temp": outdoor_temp,
                    "alternative": alt,
                    "energy_cost": alt_scores["energy_cost"],
                    "environmental": alt_scores["environmental"],
                    "comfort": alt_scores["comfort"],
                    "practicality": alt_scores["practicality"],
                    "rank": rank,
                    "topsis_score": topsis_score
                })

    logging.info(f"Results saved to {output_csv_path}")

    # Save diagnostics
    diagnostics_path = output_csv_path.replace('.csv', '_diagnostics.json')
    with open(diagnostics_path, 'w') as f:
        json.dump(cumulative_diagnostics, f, indent=2)

    logging.info(f"Diagnostics saved to {diagnostics_path}")

    return cumulative_diagnostics


def main():
    """Main execution function"""

    # Check for API key
    if not os.getenv("OPENROUTER_API_KEY"):
        logging.error("OPENROUTER_API_KEY not found in .env file")
        return

    # Example: Run on test set
    test_csv = "/mnt/user-data/uploads/test_scenarios.csv"  # User will provide this
    output_csv = "/mnt/user-data/outputs/pure_prompting_results.csv"

    logging.info("Starting Pure Prompting Architecture Test...")
    logging.info(f"Model: {API_CONFIG['model']}")
    logging.info(f"Temperature: {API_CONFIG['temperature']}")

    # Run test set
    diagnostics = run_test_set(test_csv, output_csv)

    # Print summary
    logging.info("\n" + "="*60)
    logging.info("PURE PROMPTING TEST COMPLETE")
    logging.info("="*60)
    logging.info(f"Total scenarios: {diagnostics['total_scenarios']}")
    logging.info(f"Total API calls: {diagnostics['total_api_calls']}")
    logging.info(f"Success rate: {diagnostics['success_rate']:.1%}")
    logging.info(f"Average latency: {diagnostics['avg_latency_ms']:.0f} ms")
    logging.info(f"Total tokens (input): {diagnostics['total_tokens_input']}")
    logging.info(f"Total tokens (output): {diagnostics['total_tokens_output']}")
    logging.info("="*60)


main()