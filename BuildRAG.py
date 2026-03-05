import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

RAG_FILES = {
    'HVAC': {

        'ground_truth': 'HVACRagScenarios.csv',
        'merge_required': False
    },
    'Appliance': {
        'ground_truth': 'ApplianceRAGScenariosGT.csv',
        'merge_required': False
    },
    'Shower': {
        'scenarios': 'ShowerRAGScenarios.csv',
        'merge_required': False
    }
}

CHROMA_DB_PATH = './chroma_rag_db'  # Will be created in current directory
COLLECTION_NAME = 'mcda_scenarios'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'


def load_hvac_data(csv_dir: str) -> pd.DataFrame:
    """Load HVAC data (single file contains everything)."""
    gt_path = os.path.join(csv_dir, RAG_FILES['HVAC']['ground_truth'])
    df = pd.read_csv(gt_path)

    # Rename columns to match   format_scenario_text expectations
    df['Question'] = df['question']
    df['Location'] = df['location']
    df['Square Footage'] = df['square_footage']
    df['Insulation'] = df['insulation']
    df['Household Size'] = df['household_size']
    df['Housing Type'] = df['housing_type']
    df['Outdoor Temp'] = df['outdoor_temp']
    df['alternative_num'] = df.groupby('scenario_id').cumcount() + 1

    return df


def load_appliance_data(csv_dir: str) -> pd.DataFrame:
    """Load Appliance data (GT file contains everything)."""
    gt_path = os.path.join(csv_dir, RAG_FILES['Appliance']['ground_truth'])
    df = pd.read_csv(gt_path)

    #rename columns to match expected format
    df['Question'] = df['description']
    df['Location'] = df['location']
    df['Household Size'] = df['occupants']
    df['Housing Type'] = df['housing_type']

    return df


def load_shower_data(csv_dir: str) -> pd.DataFrame:
    """Load Shower data (scenario file contains everything)."""
    scenarios_path = os.path.join(csv_dir, RAG_FILES['Shower']['scenarios'])
    df = pd.read_csv(scenarios_path)

    #rename columns to match expected format
    df['Question'] = df['description']
    df['Location'] = df['location']
    df['Household Size'] = df['occupants']

    return df


def format_scenario_text(row: pd.Series, decision_type: str) -> str:
    """
    Convert scenario to natural language text for embedding.
    Decision-type-specific formatting.
    Args:
        row: Pandas Series with scenario fields
        decision_type: 'HVAC', 'Appliance', or 'Shower'\
    Returns:
        Formatted text string
    """
    if decision_type == 'HVAC':
        return (
            f"{row.get('Outdoor Temp', 'N/A')}°F outdoor, "
            f"{row.get('Insulation', 'N/A')} insulation, "
            f"{row.get('Square Footage', 'N/A')} sqft, "
            f"{row.get('Household Size', 'N/A')} occupants, "
            f"{row.get('Housing Type', 'N/A')}"
        )

    elif decision_type == 'Appliance':
        return (
            f"{row.get('appliance', 'N/A')}, "
            f"{row.get('kwh_per_cycle', 'N/A')} kWh/cycle, "
            f"{row.get('Household Size', 'N/A')} occupants, "
            f"{row.get('Housing Type', 'N/A')}, "
            f"peak ${row.get('peak_rate', 'N/A')}/kWh, "
            f"off-peak ${row.get('offpeak_rate', 'N/A')}/kWh"
        )

    elif decision_type == 'Shower':
        return (
            f"{row.get('gpm', 'N/A')} GPM, "
            f"water heater {row.get('water_heater_temp', 'N/A')}°F, "
            f"{row.get('tank_size', 'N/A')} gal tank, "
            f"{row.get('outdoor_temp', 'N/A')}°F outdoor, "
            f"{row.get('Household Size', 'N/A')} occupants"
        )

    else:
        raise ValueError(f"Unknown decision type: {decision_type}")


def build_rag_database(csv_dir=r'C:\Users\ishaa\PycharmProjects\LLM-MCDA'):
    """
    Build ChromaDB vector database from RAG scenario CSV files.
    Process:
    1. Load all scenario files (with appropriate merging)
    2. Convert each scenario to text using decision-type-specific formatting
    3. Generate embeddings using sentence-transformers
    4. Store in ChromaDB with metadata (decision type, ground truth scores)
    Args:
        csv_dir: Directory containing the CSV files
    """
    print("BUILDING RAG DATABASE")

    # Initialize embedding model
    print(f"\nLoading embedding mo del: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Model loaded (embedding dim: {embedding_model.get_sentence_embedding_dimension()})")

    # Initialize ChromaDB
    print(f"\nInitializing ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    # Delete existing collection if it exists (fresh build)
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection: {COLLECTION_NAME}")
    except:
        pass

    # Create new collection
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "MCDA scenarios with ground truth scores"}
    )
    print(f"Created collection: {COLLECTION_NAME}")

    # Process each decision type
    total_scenarios = 0

    # HVAC
    
    print(f"Processing HVAC scenarios")
    
    try:
        hvac_df = load_hvac_data(csv_dir)
        print(f"Loaded {len(hvac_df)} HVAC scenario-alternative combinations")

        # Group by scenario_id to get unique scenarios
        for scenario_id, group in hvac_df.groupby('scenario_id'):
            # Get scenario info from first row
            first_row = group.iloc[0]
            scenario_text = format_scenario_text(first_row, 'HVAC')

            # Get all three alternatives
            alts_data = {}
            for _, row in group.iterrows():
                alt_num = int(row['alternative_num'])
                alts_data[f'alt{alt_num}'] = row['alternative']
                alts_data[f'alt{alt_num}_energy_cost'] = float(row['energy_cost_score'])
                alts_data[f'alt{alt_num}_environmental'] = float(row['environmental_score'])
                alts_data[f'alt{alt_num}_comfort'] = float(row['comfort_score'])
                alts_data[f'alt{alt_num}_practicality'] = float(row['practicality_score'])

            # Generate embedding
            embedding = embedding_model.encode(scenario_text).tolist()

            #prepare data
            metadata = {
                'decision_type': 'HVAC',
                'scenario_id': f'hvac_{scenario_id}',
                'question': first_row['Question'],
                'location': first_row['Location'],
                **alts_data
            }

            # Add to ChromaDB
            collection.add(
                ids=[f'hvac_{scenario_id}'],
                embeddings=[embedding],
                documents=[scenario_text],
                metadatas=[metadata]
            )

        unique_scenarios = len(hvac_df['scenario_id'].unique())
        print(f"Added {unique_scenarios} HVAC scenarios to database")
        total_scenarios += unique_scenarios

    except Exception as e:
        print(f" Error processing HVAC scenarios: {e}")

    # Appliance
    
    print(f"Processing Appliance scenarios")

    try:
        appliance_df = load_appliance_data(csv_dir)
        print(f"Loaded {len(appliance_df)} Appliance scenario-alternative combinations")

        # Group by scenario_id
        for scenario_id, group in appliance_df.groupby('scenario_id'):
            first_row = group.iloc[0]
            scenario_text = format_scenario_text(first_row, 'Appliance')

            # Get all alternatives
            alts_data = {}
            for idx, (_, row) in enumerate(group.iterrows(), 1):
                alts_data[f'alt{idx}'] = row['alternative']
                alts_data[f'alt{idx}_energy_cost'] = float(row['energy_cost_score'])
                alts_data[f'alt{idx}_environmental'] = float(row['environmental_score'])
                alts_data[f'alt{idx}_comfort'] = float(row['comfort_score'])
                alts_data[f'alt{idx}_practicality'] = float(row['practicality_score'])

            embedding = embedding_model.encode(scenario_text).tolist()

            metadata = {
                'decision_type': 'Appliance',
                'scenario_id': f'appliance_{scenario_id}',
                'question': first_row['Question'],
                'location': first_row['Location'],
                **alts_data
            }

            collection.add(
                ids=[f'appliance_{scenario_id}'],
                embeddings=[embedding],
                documents=[scenario_text],
                metadatas=[metadata]
            )

        unique_scenarios = len(appliance_df['scenario_id'].unique())
        print(f"Added {unique_scenarios} Appliance scenarios to database")
        total_scenarios += unique_scenarios

    except Exception as e:
        print(f" Error processing Appliance scenarios: {e}")

    # Shower
    
    print(f"Processing Shower scenarios")
    
    try:
        shower_df = load_shower_data(csv_dir)
        print(f"Loaded {len(shower_df)} Shower scenario-alternative combinations")
        for scenario_id, group in shower_df.groupby('scenario_id'):
            first_row = group.iloc[0]
            scenario_text = format_scenario_text(first_row, 'Shower')

            # Get all alternatives
            alts_data = {}
            for idx, (_, row) in enumerate(group.iterrows(), 1):
                alts_data[f'alt{idx}'] = f"{row['duration_min']} min"
                alts_data[f'alt{idx}_energy_cost'] = float(row['energy_cost_score'])
                alts_data[f'alt{idx}_environmental'] = float(row['environmental_score'])
                alts_data[f'alt{idx}_comfort'] = float(row['comfort_score'])
                alts_data[f'alt{idx}_practicality'] = float(row['practicality_score'])

            embedding = embedding_model.encode(scenario_text).tolist()

            metadata = {
                'decision_type': 'Shower',
                'scenario_id': f'shower_{scenario_id}',
                'question': first_row['Question'],
                'location': first_row['Location'],
                **alts_data
            }

            collection.add(
                ids=[f'shower_{scenario_id}'],
                embeddings=[embedding],
                documents=[scenario_text],
                metadatas=[metadata]
            )

        unique_scenarios = len(shower_df['scenario_id'].unique())
        print(f"added {unique_scenarios} Shower scenarios to database")
        total_scenarios += unique_scenarios

    except Exception as e:
        print(f" Error processing Shower scenarios: {e}")

    # Summary
    
    print(f"DATABASE BUILD COMPLETE")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Database location: {CHROMA_DB_PATH}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"\nTo use in rag_enhanced.py:")
    print(f"  client = chromadb.PersistentClient(path='{CHROMA_DB_PATH}')")
    print(f"  collection = client.get_collection('{COLLECTION_NAME}')")


def test_retrieval(test_scenario_text: str, decision_type: str, k: int = 3):
    """
    Test retrieval from built database.

    Args:
        test_scenario_text: Text description of test scenario
        decision_type: 'HVAC', 'Appliance', or 'Shower'
        k: Number of similar scenarios to retrieve
    """
    
    print(f"TESTING RETRIEVAL")
    print(f"Query: {test_scenario_text}")
    print(f"Decision type filter: {decision_type}")
    print(f"Retrieving top-{k} similar scenarios...\n")

    # Load database
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # Load embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    # Generate query embedding
    query_embedding = embedding_model.encode(test_scenario_text).tolist()

    # Retrieve similar scenarios (filtered by decision type)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where={"decision_type": decision_type}
    )

    # Display results
    if results['ids'] and len(results['ids'][0]) > 0:
        for i, (doc_id, doc_text, metadata) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0]
        )):
            print(f"Result {i + 1}: {doc_id}")
            print(f"  Text: {doc_text}")
            print(f"  Question: {metadata.get('question', 'N/A')}")
            print(f"  Alternative 1 ({metadata.get('alt1', 'N/A')}):")
            print(f"    Energy: {metadata.get('alt1_energy_cost', 0):.2f}, "
                  f"Env: {metadata.get('alt1_environmental', 0):.2f}, "
                  f"Comfort: {metadata.get('alt1_comfort', 0):.2f}, "
                  f"Pract: {metadata.get('alt1_practicality', 0):.2f}")
            print()
    else:
        print("No results foun")


if __name__ == "__main__":
    # Build database
    build_rag_database()

    # Test retrieval with middle 3 appliance scenarios
    test_cases = [
        (
            "dishwasher, 1.4 kWh/cycle, 4 occupants, Townhouse, peak $0.18/kWh, off-peak $0.08/kWh",
            "Appliance",
            "Scenario 9: I want to run my dishwasher this afternoon (around 2 PM). When should I start it?"
        ),
        (
            "dishwasher, 1.4 kWh/cycle, 3 occupants, Townhouse, peak $0.18/kWh, off-peak $0.09/kWh",
            "Appliance",
            "Scenario 10: Planning to clean up from lunch around 2 PM. When's the best time for the dishwasher?"
        ),
        (
            "dishwasher, 0.98 kWh/cycle, 4 occupants, Single-family, peak $0.17/kWh, off-peak $0.09/kWh",
            "Appliance",
            "Scenario 11: When should I do dishes today?"
        ),
    ]

    for scenario_text, decision_type, label in test_cases:
        print(f"\n{label}")
        test_retrieval(scenario_text, decision_type, k=3)