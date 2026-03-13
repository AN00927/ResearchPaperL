import os
import json
import csv
import time
import logging
import numpy as np
from typing import Dict, List, Tuple
import numpy as np
import requests
from dotenv import load_dotenv
import pandas as pd

"""ACROSS ALL THREE ARCHITECTURES
These sources were used to help build prompts:
 Prompt engineering components and their citations:

 1. Role prompting ("You are an expert in [domain]...")
 Shanahan, M., McDonell, K., & Reynolds, L. (2023).
 Role play with large language models. Nature, 623(7987), 493–498.
 https://doi.org/10.1038/s41586-023-06647-8

 2. Structured output constraint ("return ONLY JSON")
 Wei, J., et al. (2022). Chain-of-thought prompting elicits reasoning in large language models.
 NeurIPS 2022. https://proceedings.neurips.cc/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf

 3. RAG context injection ("use retrieved examples as reference but score independently")
 Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks.
 NeurIPS 2020. https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf

 4. Hybrid parameter extraction ("extract parameters then compute")
 Khot, T., et al. (2023). Decomposed prompting: A modular approach for solving complex tasks.
 ICLR 2023. https://arxiv.org/abs/2210.02406

 5. "Reasonably estimate if not apparent" (bias mitigation)
 Galaz, V., et al. (2021). Artificial intelligence, systemic risks, and sustainability.
 Technology in Society, 67, 101741. https://doi.org/10.1016/j.techsoc.2021.101741

"""


df = pd.read_csv('../Scenario Files + Ground Truth/TestScenarios.csv', encoding='utf-8-sig')
print("Columns found:", df.columns.tolist())
print("First column repr():", repr(df.columns[0]))

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

API_CONFIG = {
    "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    "model": "mistralai/mistral-small-3.2-24b-instruct",
    "max_tokens": 500,
    "temperature": 0.3  # Need determinism for reliability
}
CRITERION_WEIGHTS = {
    "energy_cost": 0.30,
    "environmental": 0.35,
    "comfort": 0.20,
    "practicality": 0.15
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
    decision_type = scenario.get('Decision Type', 'HVAC')
    prompt = f'Score this alternative: "{alternative}"\n\n'
    prompt += f'For the decision: "{scenario.get("Question", "N/A")}"\n\n'
    prompt += "SCENARIO CONTEXT:\n"
    prompt += f"- Location: {scenario.get('Location', 'N/A')}\n"

    if decision_type == 'HVAC':
        # HVAC-specific fields
        prompt += f"- Outdoor Temperature: {scenario.get('Outdoor Temp', 'N/A')}°F\n"
        prompt += f"- Home Size: {scenario.get('Square Footage', 'N/A')} sq ft\n"
        prompt += f"- Insulation: {scenario.get('Insulation', 'N/A')} (R-value: {scenario.get('R-Value', 'N/A')})\n"
        prompt += f"- Household Size: {scenario.get('Household Size', 'N/A')} people\n"
        prompt += f"- Housing Type: {scenario.get('Housing Type', 'N/A')}\n"
        prompt += f"- HVAC Age: {scenario.get('HVAC Age', 'N/A')} years\n"
        prompt += f"- Occupancy Pattern: {scenario.get('Occupancy Context', 'N/A')}\n"
        prompt += f"- HVAC SEER Rating: {scenario.get('SEER', 'N/A')}\n"
        prompt += f"- Utility Budget: ${scenario.get('Utility Budget', 'N/A')}/month\n"

    elif decision_type == 'Appliance':
        # Appliance-specific fields
        prompt += f"- Appliance Type: {scenario.get('Appliance', 'N/A')}\n"
        prompt += f"- Energy per Cycle: {scenario.get('kwh/cycle', 'N/A')} kWh\n"
        prompt += f"- Appliance Age: {scenario.get('Appliance Age/Type', 'N/A')}\n"
        prompt += f"- Baseline Time: {scenario.get('Baseline Time', '7pm')}\n"
        prompt += f"- Peak Rate: ${scenario.get('Peak Rate', 'N/A')}/kWh\n"
        prompt += f"- Off-Peak Rate: ${scenario.get('Off-Peak Rate', 'N/A')}/kWh\n"
        prompt += f"- Household Size: {scenario.get('Occupants', 'N/A')} people\n"
        prompt += f"- Housing Type: {scenario.get('Housing Type', 'N/A')}\n"
        prompt += f"- Utility Budget: ${scenario.get('Utility Budget', 'N/A')}/month\n"

    elif decision_type == 'Shower':
        # Shower-specific fields
        prompt += f"- Flow Rate: {scenario.get('GPM', 'N/A')} GPM\n"
        prompt += f"- Tank Size: {scenario.get('Tank Size', 'N/A')} gallons\n"
        prompt += f"- Water Heater Temperature: {scenario.get('Water Heater Temp', 'N/A')}°F\n"
        prompt += f"- Outdoor Temperature: {scenario.get('Outdoor Temp', 'N/A')}°F\n"
        prompt += f"- Household Size: {scenario.get('Occupants', 'N/A')} people\n"
        prompt += f"- Housing Type: {scenario.get('Housing Type', 'N/A')}\n"
        prompt += f"- Utility Budget: ${scenario.get('Utility Budget', 'N/A')}/month\n"

    prompt += "\nProvide scores (0-10) for all 4 criteria using the calibrations in the system prompt.\n"
    prompt += "Consider how this specific alternative performs given the scenario context.\n"

    return prompt


def score_alternative(scenario: Dict, alternative: str) -> Tuple[Dict, Dict]:
    """
    Score a single alternative using LLM

    Args:
        scenario: Full scenario context
        alternative: Alternative to score

    Returns:
        Tuple of (scores_dict, diagnostics_dict)
    """
    system_prompt = f"""You are an expert household decision analyst specializing in Multi-Criteria Decision Analysis (MCDA). 
    You consistently utilize all information given in the scenario context. You must take into account all factors and how they may affect all 4 criteria.

Your task is to score alternatives on four criteria:
1. Energy Cost (0-10): Lower energy costs = higher score
2. Environmental Impact (0-10): Lower emissions = higher score
3. Comfort (0-10): Higher user comfort = higher score
4. Practicality (0-10): Easier to implement/maintain = higher score

Scoring guidelines:
- Use the full 0-10 scale
- Consider tradeoffs between criteria
- Base scores on engineering principles, behavioral research, and practical constraints
- Be consistent across similar scenarios

Return ONLY a JSON object with four numeric scores (0-10):
{{"energy_cost": X, "environmental": X, "comfort": X, "practicality": X}}
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
        "decision_type": scenario.get("Decision Type", "N/A"),
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
    with open(test_csv_path, 'r', encoding='utf-8-sig') as f:
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

    cumulative_diagnostics["avg_latency_ms"] = avg_latency
    cumulative_diagnostics["success_rate"] = success_rate

    # Save results to CSV
    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = [
            "scenario_id", "decision_type", "question", "location", "outdoor_temp",
            "alternative", "energy_cost", "environmental", "comfort", "practicality",
            "rank", "weighted_score"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            scenario_id = result["scenario_id"]
            question = result["question"]
            location = result["location"]
            outdoor_temp = result["outdoor_temp"]
            decision_type = result["decision_type"]

            ranks = result["ranking_results"]["ranks"]
            weighted_scores = result["ranking_results"]["weighted_scores"]

            # Get list of alternatives in original order
            alternatives = [alt["alternative"] for alt in result["alternatives_scores"]]

            # Write each alternative as a row
            for alt_idx, alt_scores in enumerate(result["alternatives_scores"]):
                alt = alt_scores["alternative"]

                rank = ranks[alt_idx]
                weighted_score = weighted_scores[alt_idx]

                writer.writerow({
                    "scenario_id": scenario_id,
                    "decision_type": decision_type,
                    "question": question,
                    "location": location,
                    "outdoor_temp": outdoor_temp,
                    "alternative": alt,
                    "energy_cost": alt_scores["energy_cost"],
                    "environmental": alt_scores["environmental"],
                    "comfort": alt_scores["comfort"],
                    "practicality": alt_scores["practicality"],
                    "rank": rank,
                    "weighted_score": weighted_score
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

    test_csv = "TestScenarios.csv"
    output_csv = "pure_prompting_results.csv"

    logging.info("Starting Pure Prompting Architecture Test...")
    logging.info(f"Model: {API_CONFIG['model']}")
    logging.info(f"Temperature: {API_CONFIG['temperature']}")
    # Validate CSV has required columns
    import csv as csv_module
    try:
        with open(test_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv_module.DictReader(f)
            first_row = next(reader)

            required_cols = ['Question', 'Decision Type', 'Alternative 1', 'Alternative 2', 'Alternative 3']
            missing_cols = [col for col in required_cols if col not in first_row]

            if missing_cols:
                logging.error(f"Missing required columns: {missing_cols}")
                logging.error("CSV must have: Question, Decision Type, Alternative 1, Alternative 2, Alternative 3")
                logging.error("Plus decision-type-specific columns")
                return

            # Check decision types
            f.seek(0)
            next(reader)  # Skip header
            decision_types = set([row.get('Decision Type', 'UNKNOWN') for row in reader])

            logging.info(f"CSV validation passed")
            logging.info(f"  Decision types found: {decision_types}")

    except FileNotFoundError:
        logging.error(f"Test file not found: {test_csv}")
        return
    except Exception as e:
        logging.error(f" CSV validation error: {e}")
        return

    diagnostics = run_test_set(test_csv, output_csv)

    logging.info("PURE PROMPTING TEST COMPLETE")
    logging.info(f"Total scenarios: {diagnostics['total_scenarios']}")
    logging.info(f"Total API calls: {diagnostics['total_api_calls']}")
    logging.info(f"Success rate: {diagnostics['success_rate']:.1%}")
    logging.info(f"Average latency: {diagnostics['avg_latency_ms']:.0f} ms")
    logging.info(f"Total tokens (input): {diagnostics['total_tokens_input']}")
    logging.info(f"Total tokens (output): {diagnostics['total_tokens_output']}")



main()