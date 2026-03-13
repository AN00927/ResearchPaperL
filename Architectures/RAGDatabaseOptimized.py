import os
import json
import csv
import requests
import time
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer


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

CHROMA_DB_PATH = '../chroma_rag_db'
COLLECTION_NAME = 'mcda_scenarios'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
RETRIEVE_K = 3  # Number of similar scenarios to retrieve

MAX_RETRIES = 3
RETRY_DELAY = 2

OUTPUT_CSV = 'RAGResults.csv'
OUTPUT_DIAGNOSTICS = 'RAGDiagnostics.json'

print("Loading ChromaDB and embedding model")
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Loaded RAG database: {chroma_collection.count()} scenarios available")
except Exception as e:
    print(f" WARNING: Could not load RAG database: {e}")
    print("  Make sure to run build_rag_database.py first!")
    chroma_collection = None
    embedding_model = None


def query_openrouter(messages: List[Dict], model: str = MODEL_ID,
                     temperature: float = TEMPERATURE) -> Tuple[Dict, Dict]:
    """
    Returns:
        (response_dict, diagnostics_dict)
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

    last_error = None
    response = None

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
                    'latency_ms': latency *1000,
                    'model': model
                }

                return data, diagnostics
            else:
                last_error = f"Status {response.status_code}: {response.text}"
                print(f"  API error (attempt {attempt + 1}/{MAX_RETRIES}): {response.status_code}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)

        except Exception as e:
            last_error = str(e)
            print(f"  Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    raise Exception(f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}")

def build_system_prompt() -> str:
    """
    Build system prompt for MCDA scoring.
    """
    return """You are an expert household decision analyst specializing in Multi-Criteria Decision Analysis (MCDA).
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
{"energy_cost": X, "environmental": X, "comfort": X, "practicality": X}"""


def format_scenario_text_for_retrieval(scenario: Dict) -> Tuple[str, str]:
    """
    Convert scenario to text for RAG retrieval.
    Returns:
        (scenario_text, decision_type)
    """
    # Read decision type from CSV (not keyword detection)
    decision_type = scenario.get('Decision Type', 'HVAC')

    if decision_type == 'HVAC':
        scenario_text = (
            f"{scenario.get('Outdoor Temp', 'N/A')}°F outdoor, "
            f"{scenario.get('Insulation', 'N/A')} insulation R-{scenario.get('R-Value', 'N/A')}, "
            f"SEER {scenario.get('SEER', 'N/A')}, "
            f"- Occupancy Pattern: {scenario.get('Occupancy Context', 'N/A')}\n"
            f"{scenario.get('Square Footage', 'N/A')} sqft, "
            f"{scenario.get('Household Size', 'N/A')} occupants, "
            f"{scenario.get('Housing Type', 'N/A')}"
        )
    elif decision_type == 'Appliance':
        scenario_text = (
            f"{scenario.get('Question', 'N/A')}, "
            f"{scenario.get('Household Size', 'N/A')} occupants, "
            f"{scenario.get('Housing Type', 'N/A')}, "
            f"appliance age range: {scenario.get('Appliance Age', 'N/A')}, "
            f"budget ${scenario.get('Utility Budget', 'N/A')}/month"
        )
    elif decision_type == 'Shower':
        scenario_text = (
            f"{scenario.get('Flow rate', 'N/A')} showerhead, "
            f"{scenario.get('Outdoor Temp', 'N/A')}°F outdoor, "
            f"{scenario.get('Household Size', 'N/A')} occupants, "
            f"{scenario.get('Housing Type', 'N/A')}, "
            f"budget ${scenario.get('Utility Budget', 'N/A')}/month"
        )
    else:
        scenario_text = scenario.get('Question', f'Unknown decision type: {decision_type}')
        print(f"   Warning: Unknown decision type '{decision_type}'")

    return scenario_text, decision_type

def retrieve_similar_scenarios(scenario: Dict, k: int = RETRIEVE_K) -> List[Dict]:
    """
    Retrieve k most similar scenarios from RAG database.

    Args:
        scenario: Current test scenario dict
        k: Number of similar scenarios to retrieve

    Returns:
        List of dicts with retrieved scenario info and scores
    """
    if chroma_collection is None or embedding_model is None:
        print("   RAG database not available, skipping retrieval")
        return []

    # Convert scenario to text
    scenario_text, decision_type = format_scenario_text_for_retrieval(scenario)

    # Generate embedding
    query_embedding = embedding_model.encode(scenario_text).tolist()

    # Retrieve from database (filtered by decision type)
    try:
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where={"decision_type": decision_type}
        )
    except Exception as e:
        print(f"   Retrieval error: {e}")
        return []

    retrieved = []
    if results['ids'] and len(results['ids'][0]) > 0:
        for doc_id, doc_text, metadata in zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0]
        ):
            retrieved.append({
                'id': doc_id,
                'text': doc_text,
                'decision_type': metadata.get('decision_type', 'Unknown'),
                'question': metadata.get('question', 'N/A'),
                'alternatives': [
                    {
                        'name': metadata.get('alt1', 'N/A'),
                        'scores': {
                            'energy_cost': metadata.get('alt1_energy_cost', 0.0),
                            'environmental': metadata.get('alt1_environmental', 0.0),
                            'comfort': metadata.get('alt1_comfort', 0.0),
                            'practicality': metadata.get('alt1_practicality', 0.0)
                        }
                    },
                    {
                        'name': metadata.get('alt2', 'N/A'),
                        'scores': {
                            'energy_cost': metadata.get('alt2_energy_cost', 0.0),
                            'environmental': metadata.get('alt2_environmental', 0.0),
                            'comfort': metadata.get('alt2_comfort', 0.0),
                            'practicality': metadata.get('alt2_practicality', 0.0)
                        }
                    },
                    {
                        'name': metadata.get('alt3', 'N/A'),
                        'scores': {
                            'energy_cost': metadata.get('alt3_energy_cost', 0.0),
                            'environmental': metadata.get('alt3_environmental', 0.0),
                            'comfort': metadata.get('alt3_comfort', 0.0),
                            'practicality': metadata.get('alt3_practicality', 0.0)
                        }
                    }
                ]
            })

    return retrieved


def format_rag_context(retrieved_scenarios: List[Dict]) -> str:
    """
    Format retrieved scenarios as context for LLM prompt.

    Args:
        retrieved_scenarios: List of retrieved scenario dicts

    Returns:
        Formatted context string
    """
    if not retrieved_scenarios:
        return ""

    context = "RELEVANT SIMILAR SCENARIOS WITH EXPERT SCORES:\n\n"

    for i, scenario in enumerate(retrieved_scenarios, 1):
        context += f"Example {i}: {scenario['text']}\n"
        context += f"  Question: {scenario['question']}\n"

        for alt in scenario['alternatives']:
            scores = alt['scores']
            context += (
                f"  • {alt['name']}: "
                f"Energy Cost: {scores['energy_cost']:.1f}/10, "
                f"Environmental: {scores['environmental']:.1f}/10, "
                f"Comfort: {scores['comfort']:.1f}/10, "
                f"Practicality: {scores['practicality']:.1f}/10\n"
            )
        context += "\n"

    context += "Use these examples as reference, but score based on the specific scenario below.\n"
    context += "Just because a reference scenario has an extreme value does not mean that the scenario you are analyzing has the same characteristics.\n"

    return context

def build_user_prompt_with_rag(scenario: Dict, alternative: str, rag_context: str) -> str:
    prompt = rag_context
    prompt += f'Score this alternative: "{alternative}"\n\n'
    prompt += f'For the decision: "{scenario.get("Question", "N/A")}"\n\n'
    prompt += "SCENARIO CONTEXT:\n"

    decision_type = scenario.get('Decision Type', 'HVAC')

    if decision_type == 'HVAC':
        prompt += (
            f"- Location: {scenario.get('Location', 'N/A')}\n"
            f"- Outdoor Temp: {scenario.get('Outdoor Temp', 'N/A')}°F\n"
            f"- Square Footage: {scenario.get('Square Footage', 'N/A')} sqft\n"
            f"- Insulation: {scenario.get('Insulation', 'N/A')}\n"
            f"- Household Size: {scenario.get('Household Size', 'N/A')} occupants\n"
            f"- Housing Type: {scenario.get('Housing Type', 'N/A')}\n"
            f"- House Age: {scenario.get('House Age', 'N/A')}\n"
            f"- Utility Budget: ${scenario.get('Utility Budget', 'N/A')}/month\n"
        )

    elif decision_type == 'Appliance':
        prompt += (
            f"- Location: {scenario.get('Location', 'N/A')}\n"
            f"- Household Size: {scenario.get('Household Size', 'N/A')} occupants\n"
            f"- Housing Type: {scenario.get('Housing Type', 'N/A')}\n"
            f"- Utility Budget: ${scenario.get('Utility Budget', 'N/A')}/month\n"
            f"- Appliance Age Range: {scenario.get('Appliance Age', 'N/A')} years\n"
        )

    elif decision_type == 'Shower':
        prompt += (
            f"- Location: {scenario.get('Location', 'N/A')}\n"
            f"- Outdoor Temp: {scenario.get('Outdoor Temp', 'N/A')}°F\n"
            f"- Household Size: {scenario.get('Household Size', 'N/A')} occupants\n"
            f"- Housing Type: {scenario.get('Housing Type', 'N/A')}\n"
            f"- Flow Rate: {scenario.get('Flow rate', 'N/A')}\n"
            f"- Utility Budget: ${scenario.get('Utility Budget', 'N/A')}/month\n"
        )

    return prompt

def parse_llm_scores(response_text: str) -> Dict[str, float]:
    """
    Parse JSON scores from LLM response.
    EXACT COPY from pure_prompting.py
    """
    import re
    json_match = re.search(r'\{[^}]+\}', response_text)

    if json_match:
        try:
            scores = json.loads(json_match.group())

            # Validate keys
            required_keys = ['energy_cost', 'environmental', 'comfort', 'practicality']
            if all(k in scores for k in required_keys):
                # Clamp to [0, 10]
                return {k: max(0.0, min(10.0, float(scores[k]))) for k in required_keys}
        except:
            pass

    print("   Could not parse scores; failed")
    return {'energy_cost': None, 'environmental': None, 'comfort': None, 'practicality': None, '_failed': True}


def score_alternative_with_rag(scenario: Dict, alternative: str) -> Tuple[Dict, Dict]:
    """
    Score an alternative using RAG-Enhanced approach.

    Process:
    1. Retrieve k similar scenarios from database
    2. Format as context
    3. Build prompt with context + scenario
    4. Query LLM (single call)
    5. Parse scores

    Returns:
        (scores_dict, diagnostics_dict)
    """
    # Step 1: Retrieve similar scenarios
    retrieved = retrieve_similar_scenarios(scenario, k=RETRIEVE_K)

    # Step 2: Format RAG context
    rag_context = format_rag_context(retrieved)

    # Step 3: Build prompt
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt_with_rag(scenario, alternative, rag_context)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Step 4: Query LLM
    response, diagnostics = query_openrouter(messages)

    # Step 5: Parse scores

    response_text = response['choices'][0]['message']['content']
    scores = parse_llm_scores(response_text)
    diagnostics['success'] = not scores.get('_failed', False)
    # Add RAG metadata to diagnostics
    diagnostics['rag_retrieved_count'] = len(retrieved)
    diagnostics['rag_context_length'] = len(rag_context)

    return scores, diagnostics


def apply_mavt_ranking(alternatives_scores: List[Dict]) -> Dict:
    """
    Apply MAVT weighted sum to rank alternatives.
    EXACT COPY from pure_prompting.py

    Args:
        alternatives_scores: List of dicts with 'alternative' and 'scores'

    Returns:
        Dict with ranked alternatives and weighted scores
    """
    weighted_scores = []

    for alt_data in alternatives_scores:
        if alt_data.get('failed'):
            continue
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
    ranked = sorted(weighted_scores, key=lambda x: x['weighted_score'], reverse=True)

    return {
        'ranked_alternatives': [r['alternative'] for r in ranked],
        'weighted_scores': [r['weighted_score'] for r in ranked],
        'details': ranked
    }


def run_scenario(scenario: Dict) -> Dict:
    """
    Run RAG-Enhanced scoring on all alternatives in a scenario.

    Returns:
        Dict with scores and ranking results
    """
    print(f"SCENARIO: {scenario.get('Question', 'N/A')}")

    alternatives_scores = []
    total_diagnostics = {
        'api_calls': 0,
        'total_tokens_input': 0,
        'total_tokens_output': 0,
        'total_latency_ms': 0.0,
        'successful_calls': 0,
        'failed_calls': 0
    }
    for i in range(1, 4):
        alt_key = f'Alternative {i}'
        if alt_key not in scenario:
            continue

        alternative = scenario[alt_key]
        print(f"\nScoring: {alternative}")

        scores, diagnostics = score_alternative_with_rag(scenario, alternative)
        if scores.get('_failed'):
            print(f" FAILED — skipping alternative")
            total_diagnostics['failed_calls'] += 1
            alternatives_scores.append({
                'alternative': alternative,
                'scores': {'energy_cost': None, 'environmental': None, 'comfort': None, 'practicality': None},
                'failed': True
            })
            continue
        print(f"  Scores: Energy={scores['energy_cost']:.1f}, "
              f"Env={scores['environmental']:.1f}, "
              f"Comfort={scores['comfort']:.1f}, "
              f"Pract={scores['practicality']:.1f}")
        print(f"  Retrieved {diagnostics.get('rag_retrieved_count', 0)} similar scenarios")

        alternatives_scores.append({
            'alternative': alternative,
            'scores': scores
        })
        total_diagnostics['api_calls'] += 1
        total_diagnostics['total_tokens_input'] += diagnostics.get('prompt_tokens', 0)
        total_diagnostics['total_tokens_output'] += diagnostics.get('completion_tokens', 0)
        total_diagnostics['total_latency_ms'] += diagnostics.get('latency_ms', 0.0)

        if diagnostics.get('success', True):
            total_diagnostics['successful_calls'] += 1
        else:
            total_diagnostics['failed_calls'] += 1
    ranking_result = apply_mavt_ranking(alternatives_scores)

    print(f"\nRANKING:")
    for i, (alt, score) in enumerate(zip(
            ranking_result['ranked_alternatives'],
            ranking_result['weighted_scores']
    ), 1):
        print(f"  {i}. {alt} (weighted score: {score:.2f})")

    return {
        'scenario': scenario.get('Question', 'N/A'),
        'alternatives_scores': alternatives_scores,
        'ranking_result': ranking_result,
        'diagnostics': total_diagnostics
    }


def run_test_set(test_csv_path: str, output_csv_path: str,
                 output_diagnostics_path: str) -> Dict:
    """
    Run RAG-Enhanced on full test set.

    Args:
        test_csv_path: Path to test scenarios CSV
        output_csv_path: Path to save results CSV
        output_diagnostics_path: Path to save diagnostics JSON

    Returns:
        Summary statistics dict
    """
    import csv as csv_module

    print(f"RAG-ENHANCED MCDA ARCHITECTURE - TEST SET")

    print(f"Loading test scenarios from: {test_csv_path}")

    scenarios = []
    with open(test_csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv_module.DictReader(f)
        first_row = next(reader)

        # Validate required columns
        required_cols = ['Question', 'Decision Type', 'Alternative 1', 'Alternative 2', 'Alternative 3']
        missing_cols = [col for col in required_cols if col not in first_row]

        if missing_cols:
            raise ValueError(f" Missing required columns: {missing_cols}")

        scenarios.append(first_row)
        scenarios.extend(list(reader))

    print(f"✓ Loaded {len(scenarios)} test scenarios")
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

        # Aggregate diagnostics
        diag = result['diagnostics']
        cumulative_diagnostics['total_api_calls'] += diag['api_calls']
        cumulative_diagnostics['total_tokens_input'] += diag['total_tokens_input']
        cumulative_diagnostics['total_tokens_output'] += diag['total_tokens_output']
        cumulative_diagnostics['total_latency_ms'] += diag['total_latency_ms']
        cumulative_diagnostics['successful_calls'] += diag['successful_calls']
        cumulative_diagnostics['failed_calls'] += diag['failed_calls']
        # Track by decision type

        cumulative_diagnostics['avg_latency_ms'] = (
            cumulative_diagnostics['total_latency_ms'] /
            max(cumulative_diagnostics['total_api_calls'], 1)
    )
    cumulative_diagnostics['success_rate'] = (
            cumulative_diagnostics['successful_calls'] /
            max(cumulative_diagnostics['total_api_calls'], 1)
    )
    # Save results to CSV
    print(f"\nSaving results to: {output_csv_path}")

    with open(output_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = [
            'scenario_id', 'question', 'location', 'decision_type',
            'alternative', 'energy_cost', 'environmental', 'comfort', 'practicality',
            'rank', 'weighted_score'
        ]
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for scenario_id, result in enumerate(all_results, 1):
            question = result['scenario']
            decision_type = scenarios[scenario_id - 1].get('Decision Type', 'UNKNOWN')
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
                    'alternative': alternative,
                    'energy_cost': scores['energy_cost'],
                    'environmental': scores['environmental'],
                    'comfort': scores['comfort'],
                    'practicality': scores['practicality'],
                    'rank': rank,
                    'weighted_score': weighted_score
                })

    print(f"✓ Results saved to: {output_csv_path}")

    # Save diagnostics
    print(f"Saving diagnostics to: {output_diagnostics_path}")

    with open(output_diagnostics_path, 'w', encoding='utf-8-sig') as f:
        json.dump(cumulative_diagnostics, f, indent=2)

    print(f"✓ Diagnostics saved to: {output_diagnostics_path}")

    print(f"RAG-ENHANCED TEST COMPLETE")
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
        print(f"Test scenarios file not found: {test_csv}")
        print("Please upload your test scenarios CSV first.")
        sys.exit(1)

    if chroma_collection is None:
        print(f"RAG database not available.")
        print("Please run build_rag_database.py first to create the RAG database.")
        sys.exit(1)

    # Run test set
    run_test_set(
        test_csv_path=test_csv,
        output_csv_path=OUTPUT_CSV,
        output_diagnostics_path=OUTPUT_DIAGNOSTICS
    )