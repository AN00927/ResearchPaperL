# LLM-MCDA
Creating a LLM-assisted MCDA tool for Household Emission Optimization
# CliMate: AI-MCDA for Household Emission Optimization

## Project Overview

This project investigates the reliability of AI-generated Multi-Criteria Decision Analysis (MCDA) weights for household energy decisions by comparing three LLM integration architectures against physics-based ground truth calculations.

**Research Question:** Which AI integration approach most accurately replicates research-backed environmental impact calculations for household decisions?

**Author:** Ahaan Nigam  
**Institution:** Downingtown East High School  
**Advisor:** Dr. Melissa Bilec  
**External Consultation:** Dr. He Huang (Paul Scherrer Institut)

---

## Three Architectures Being Compared on 4 criterion and overall ranking of alternatives (3 per scenario)

### 1. Pure Prompting
- **Approach:** LLM scores all four criteria (Energy Cost, Environmental Impact, Comfort, Practicality) directly using calibrated system prompts
- **Input:** Natural language user question + household context
- **Output:** Four 0-10 scores per alternative

### 2. RAG-Enhanced
- **Approach:** LLM retrieves relevant research chunks from vector database before scoring
- **Input:** User question → retrieves from 50+ ground truth scenarios → LLM scores with context
- 

### 3. Hybrid (AI Extraction + Calculator)
- **Approach:** LLM extracts structured parameters (SEER tier, appliance age, etc.) → deterministic calculator computes scores using physics formulas
- **Input:** User description → AI maps to parameters → calculator runs
- **Output:** Four 1-10 scores from research-backed formulas

---

## Ground Truth Methodology

Ground truth scores are calculated using:

- **HVAC Energy:** ASHRAE load calculations, SEER/HSPF efficiency standards (Kim et al. 2024, Huyen & Cetin 2019)
- **Appliance Energy:** DOE consumption benchmarks, Energy Star data (Porras et al. 2020, Chen-Yu & Emmel 2018)
- **Water Heating:** Field-tested consumption models (Booysen et al. 2019, Yildiz et al. 2021)
- **Shower Energy:** Flow rate measurements (Shahmohammadi et al. 2019)
- **Emissions:** EPA eGRID PJM region factor (0.6574 lbs CO₂/kWh)
- **Comfort/Practicality:** ASHRAE 55 thermal comfort standards, behavioral adoption research

All calculations documented with peer-reviewed citations in `CITATIONS.md`.

---

## Validation Metrics

Each architecture will be evaluated on:

**Accuracy:**
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Spearman's ρ (rank correlation)
- Kendall's τ (rank correlation)
- Top-1 and Top-2 accuracy

**Consistency:**
- Standard deviation across 3 runs per scenario (temperature=0.3)

**Practical Performance:**
- API cost per decision (tokens × rate)
- Mean latency (query → ranked output)

---

## Project Structure

/src
  /architectures
    - pure_prompting.py         # Architecture 1 implementation
    - rag_enhanced.py            # Architecture 2 implementation
    - hybrid.py                  # Architecture 3 implementation
  /ground_truth
    - calculator.py              # Physics-based score calculations
    - scenarios_generator.py     # (Future: auto-generate 100 scenarios)
  /evaluation
    - metrics.py                 # RMSE, MAE, Kendall's Tau, etc.
    - validation.py              # Run all architectures on test set
  /data
    - scenarios_train.json       # 50 training scenarios (for tuning)
    - scenarios_test.json        # 50 test scenarios (locked until final eval)
    - ground_truth.json          # Calculated ground truth for all 100

/docs
  - ITERATION_LOG.md             # Development log for each architecture
  - CITATIONS.md                 # All research sources with full references

---

## Key Dependencies

- Python 3.12
- OpenRouter API (LLM access)

See `requirements.txt` for full list.

---

## Citation

If you use this work, please cite:

```
Nigam, A. (2026). Comparing AI Integration Architectures for Multi-Criteria 
Decision Analysis in Household Energy Optimization. Delaware Valley Science Fair.
```

---

## License

This project is currently private. Contact author for access or collaboration inquiries.

**Contact:** Ishaannigam27@gmail.com
