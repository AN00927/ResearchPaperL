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

MODEL_ID = "mistralai/mistral-small-3.2"
TEMPERATURE = 0.3
CRITERION_WEIGHTS = {
    'energy_cost': 0.35,
    'environmental': 0.30,
    'comfort': 0.20,
    'practicality': 0.15
}

CHROMA_DB_PATH = './chroma_rag_db'
COLLECTION_NAME = 'mcda_scenarios'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
RETRIEVE_K = 3  # Number of similar scenarios to retrieve

MAX_RETRIES = 3
RETRY_DELAY = 2

OUTPUT_CSV = '/mnt/user-data/outputs/rag_enhanced_results.csv'
OUTPUT_DIAGNOSTICS = '/mnt/user-data/outputs/rag_enhanced_diagnostics.json'

print("Loading ChromaDB and embedding model")
try:
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = chroma_client.get_collection(COLLECTION_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"✓ Loaded RAG database: {chroma_collection.count()} scenarios available")
except Exception as e:
    print(f"⚠ WARNING: Could not load RAG database: {e}")
    print("  Make sure to run build_rag_database.py first!")
    chroma_collection = None
    embedding_model = None


def query_openrouter(messages: List[Dict], model: str = MODEL_ID,
                     temperature: float = TEMPERATURE) -> Tuple[Dict, Dict]:
    """
    Query OpenRouter API with retry logic.
    EXACT COPY from pure_prompting.py

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


def build_system_prompt() -> str:
    """
    Build system prompt for MCDA scoring.
    EXACT COPY from pure_prompting.py - NO numerical anchors
    """
    return """You are an expert decision analyst specializing in Multi-Criteria Decision Analysis (MCDA).

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
    question = scenario.get('Description', '').lower()
# AHAAN THIS IS INCORRECT; THE DECISION TYPE WILL BE INCLUDED AS ANOTHER PARAMETER, IT DOESNT HAVE TO DO THIS!
# Update input variables later
    if any(keyword in question for keyword in ['temperature', 'thermostat', 'ac', 'heat', 'hvac']):
        decision_type = 'HVAC'
        scenario_text = (
            f"{scenario.get('Outdoor Temp', 'N/A')}°F outdoor, "
            f"{scenario.get('Insulation', 'N/A')} insulation R-{scenario.get('R-Value', 'N/A')}, "
            f"SEER {scenario.get('SEER', 'N/A')}, "
            f"{scenario.get('Square Footage', 'N/A')} sqft, "
            f"{scenario.get('Household Size', 'N/A')} occupants, "
            f"{scenario.get('Housing Type', 'N/A')}"
        )

    elif any(keyword in question for keyword in ['dishwasher', 'washer', 'dryer', 'appliance', 'run at', 'wash at']):
        decision_type = 'Appliance'
        scenario_text = (
            f"{scenario.get('Appliance', 'N/A')}, "
            f"{scenario.get('kwh/cycle', 'N/A')} kWh/cycle, "
            f"{scenario.get('Appliance Age/Type', 'N/A')} years old, "
            f"{scenario.get('Occupants', 'N/A')} occupants, "
            f"{scenario.get('Housing Type', 'N/A')}, "
            f"peak ${scenario.get('Peak Rate', 'N/A')}/kWh"
        )

    else:
        decision_type = 'Shower'
        scenario_text = (
            f"{scenario.get('GPM', 'N/A')} GPM, "
            f"{scenario.get('Water Heater', 'N/A')} water heater, "
            f"{scenario.get('Tank Size', 'N/A')} gal tank, "
            f"{scenario.get('Outdoor Temp', 'N/A')}°F outdoor, "
            f"{scenario.get('Occupants', 'N/A')} occupants, "
            f"{scenario.get('Housing Type', 'N/A')}"
        )

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
        print("  ⚠ RAG database not available, skipping retrieval")
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
        print(f"  ⚠ Retrieval error: {e}")
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
    context += "=" * 70 + "\n\n"

    return context


def build_user_prompt_with_rag(scenario: Dict, alternative: str,
                               rag_context: str) -> str:
    """
    Build user prompt with RAG context + scenario details.

    Args:
        scenario: Scenario dict
        alternative: Alternative to score
        rag_context: Formatted RAG context from retrieved scenarios

    Returns:
        Complete user prompt string
    """
    # RAG context comes first as context
    prompt = rag_context

    # Then the scoring task (same structure as Pure Prompting)
    prompt += f'Score this alternative: "{alternative}"\n\n'
    prompt += f'For the decision: "{scenario.get("Description", "N/A")}"\n\n'
    prompt += "SCENARIO CONTEXT:\n"

    for key, value in scenario.items():
        if key not in ['Description', 'Alternative 1', 'Alternative 2', 'Alternative 3']:
            prompt += f"- {key}: {value}\n"

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

    print("culd not parse scores, using defaults")
    return {'energy_cost': 5.0, 'environmental': 5.0, 'comfort': 5.0, 'practicality': 5.0}


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
    print(f"\n{'=' * 70}")
    print(f"SCENARIO: {scenario.get('Description', 'N/A')}")
    print(f"{'=' * 70}")

    alternatives_scores = []
    total_diagnostics = {
        'api_calls': 0,
        'total_tokens': 0,
        'total_latency': 0.0,
        'rag_retrieved_total': 0
    }
    for i in range(1, 4):
        alt_key = f'Alternative {i}'
        if alt_key not in scenario:
            continue

        alternative = scenario[alt_key]
        print(f"\nScoring: {alternative}")

        scores, diagnostics = score_alternative_with_rag(scenario, alternative)

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
        total_diagnostics['total_tokens'] += diagnostics.get('total_tokens', 0)
        total_diagnostics['total_latency'] += diagnostics.get('latency_seconds', 0.0)
        total_diagnostics['rag_retrieved_total'] += diagnostics.get('rag_retrieved_count', 0)
    ranking_result = apply_mavt_ranking(alternatives_scores)

    print(f"\nRANKING:")
    for i, (alt, score) in enumerate(zip(
            ranking_result['ranked_alternatives'],
            ranking_result['weighted_scores']
    ), 1):
        print(f"  {i}. {alt} (weighted score: {score:.2f})")

    return {
        'scenario': scenario.get('Description', 'N/A'),
        'alternatives_scores': alternatives_scores,
        'ranking_result': ranking_result,
        'diagnostics': total_diagnostics
    }

#add a run Test set based on the scenarios once they are finished
#do this for other two architectures asw