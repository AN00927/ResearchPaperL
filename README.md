# LLM-MCDA: AI-Assisted Multi-Criteria Decision Analysis for Household Energy Optimization

**Author:** Ahaan Nigam  
**Institution:** Downingtown East High School  
**Collaborator:** Dr. River Huang, Paul Scherrer Institut (PSI), Switzerland  

---

## Research Question

Which AI-MCDA architecture most accurately replicates physics-based ground truth for household energy decisions, while still maintaining reasonable failure rate and API costs?

---

## Project Overview

[Abstract](Abstract.pdf)

This project compares three LLM-MCDA architectures for household energy decision-making against a physics-based Multi-Attribute Value Theory (MAVT) ground truth calculator across 100 scenarios (60 HVAC, 25 Appliance, 15 Shower). Each scenario presents three alternatives; each architecture ranks them on four criteria.

**MAVT Criterion Weights:**
| Criterion | Weight |
|---|---|
| Environmental Impact | 35% |
| Energy Cost | 30% |
| Comfort | 20% |
| Practicality | 15% |

**Underlying LLM:** Mistral Small 3.2 24B via OpenRouter  
**RAG Database:** ChromaDB  
**Score Scale:** 0–10

---

## Repository Structure

```
LLM-MCDA/
├── Architectures/
│   ├── Hybrid.py
│   ├── PurePrompting.py
│   └── RAGDatabaseOptimized.py
├── Ground Truth/
│   ├── ground_truth_appliance.csv
│   ├── ground_truth_hvac.csv
│   └── ground_truth_shower.csv
├── Ground Truth Calculators/
│   ├── ApplianceGroundTruthCalculator.py
│   ├── HVACGroundTruthCalculator.py
│   └── ShowerGroundTruthCalculator.py
├── Miscellaneous Files/
│   └── CalculateMetrics.py
├── Output Files/
│   ├── hybrid_diagnostics.json
│   ├── hybrid_results.csv
│   ├── metrics_summary.csv
│   ├── pure_prompting_results.csv
│   ├── pure_prompting_results_diagnostics.json
│   ├── RAGDiagnostics.json
│   └── RAGResults.csv
├── Scenario Files/
│   ├── ApplianceRAGScenariosGT.csv
│   ├── ApplianceScenarios.csv
│   ├── HVACRagScenarios.csv
│   ├── HVACScenarios.csv
│   ├── ShowerRAGScenarios.csv
│   ├── ShowerScenarios.csv
│   └── TestScenarios.csv
├── BuildRAG.py
├── MCDA Files Consolidated.xlsx
├── README.md
└── requirements.txt
```

---

## Three Architectures

### 1. Pure Prompting
- **Approach:** LLM scores all four criteria directly via calibrated system prompts
- **Input:** Natural language scenario description
- **Output:** Four 0–10 scores per alternative
- **API calls per scenario:** 3 (one per alternative)

### 2. RAG-Enhanced
- **Approach:** LLM retrieves relevant ground truth scenario chunks from ChromaDB vector database before scoring
- **Input:** User description → semantic retrieval → LLM scores with retrieved context
- **Output:** Four 0–10 scores per alternative
- **API calls per scenario:** 3 (one per alternative)

### 3. Hybrid (AI Extraction + Deterministic Calculator)
- **Approach:** LLM extracts structured parameters (SEER tier, appliance age, flow rate, etc.) → deterministic MAVT calculator computes scores using physics formulas
- **Input:** User description → AI maps to parameters → calculator runs
- **Output:** Four 0–10 scores from physics-backed formulas
- **API calls per scenario:** 1 (single call processes all three alternatives)

---

## Ground Truth Methodology

Ground truth scores are calculated using deterministic MAVT value functions with empirically derived reference ranges (5th–95th percentile from actual scenario data).

**Value function structure (identical across all three calculators):**
- Energy Cost & Environmental Impact: Linear value function
- Comfort: Logarithmic (a = 1.5)
- Practicality: Logarithmic (a = 1.2)

**Budget penalty tiers** (Thaler 1999; Heath & Soll 1996; Prelec & Loewenstein 1998; Gathergood 2012):
- Less than 80% of budget: no penalty
- 80–100%: linear penalty
- 100–150%: exponential penalty
- Greater than 150%: eliminated

**Domain-specific methods:**
| Domain | Method |
|---|---|
| HVAC Energy | ASHRAE cooling/heating load calculations, SEER degradation (Domanski 2014) |
| Appliance Energy | DOE consumption benchmarks, Energy Star data |
| Shower Energy | Flow rate × temperature × duration |
| Emissions | EPA eGRID PJM factor: **0.6458 lbs CO₂/kWh** |
| Comfort | ASHRAE 55 thermal comfort standards |
| Practicality | Behavioral adoption research; floor = 1.5 |

**Reference ranges:**
| Domain | Energy Cost | Environmental |
|---|---|---|
| HVAC | $0.47–$3.31 | 1.60–11.25 lbs CO₂ |
| Appliance | $0.02–$0.90 | 0.09–3.83 lbs CO₂ |
| Shower | $0.20–$1.40 | 1.10–5.90 lbs CO₂ |

> Pre-transformation clamping is NOT applied — values extrapolate beyond reference bounds freely; final clamping occurs only after value function transformation to preserve MAVT independence.

---

## Results

### Overall Performance (100 scenarios)

| Metric | Pure Prompting | RAG-Enhanced | Hybrid |
|---|---|---|---|
| Kendall's τ | -0.06 | 0.43 | **0.80** |
| Spearman's ρ | -0.055 | 0.49 | **0.82** |
| Top-1 Accuracy | 30% | 58% | **83%** |
| Top-2 Accuracy | 62% | 89% | **97%** |
| Overall MAE | 2.52 | 1.61 | **0.69** |
| Overall RMSE | 3.07 | 2.40 | **1.52** |

> Pure Prompting's τ ≈ −0.06 is near-random — end-to-end LLM scoring cannot reliably replicate physics-based MAVT rankings.

---

### Criterion-Level MAE (Overall)

| Criterion | Pure | RAG | Hybrid |
|---|---|---|---|
| Energy Cost | 2.60 | 1.60 | 0.96 |
| Environmental | 2.80 | 1.59 | 1.02 |
| Comfort | 2.51 | 1.97 | 0.32 |
| Practicality | 2.16 | 1.29 | 0.43 |

---

### Performance by Decision Type

#### HVAC (60 scenarios)
| Metric | Pure | RAG | Hybrid |
|---|---|---|---|
| Kendall's τ | -0.111 | 0.522 | **0.90** |
| Top-1 Accuracy | 25% | 63% | **92%** |
| Top-2 Accuracy | 57% | 93% | **100%** |
| Overall MAE | 2.11 | 1.46 | **0.62** |

#### Appliance (25 scenarios)
| Metric | Pure | RAG | Hybrid |
|---|---|---|---|
| Kendall's τ | 0.093 | 0.413 | **0.68** |
| Top-1 Accuracy | 40% | 56% | **72%** |
| Top-2 Accuracy | 72% | 88% | **92%** |
| Overall MAE | 3.31 | 2.15 | **0.97** |

#### Shower (15 scenarios)
| Metric | Pure | RAG | Hybrid |
|---|---|---|---|
| Kendall's τ | -0.111 | 0.111 | **0.60** |
| Top-1 Accuracy | 33% | 40% | **67%** |
| Top-2 Accuracy | 67% | 73% | **93%** |
| Overall MAE | 2.85 | 1.34 | **0.46** |

---

### Efficiency (100 scenarios)

| Metric | Pure Prompting | RAG-Enhanced | Hybrid |
|---|---|---|---|
| Total API Calls | 300 | 300 | **100** |
| Input Tokens | 114,739 | 295,174 | **81,741** |
| Output Tokens | 9,970 | 13,463 | 16,275 |
| **Total Tokens** | **124,709** | **308,637** | **98,016** |
| Total Latency | 437.1s | 480.8s | **275.6s** |
| Avg Latency/Call | 1,457ms | 1,603ms | **2,756ms** |
| Success Rate | 100% | 100% | 100% |

---

## Key Findings

1. **Pure Prompting performs near-randomly** (τ = −0.06). LLMs cannot reliably score household energy tradeoffs without physics grounding.
2. **RAG improves ranking** (τ = 0.43) but still struggles — especially on Appliance Comfort (MAE = 3.88), where retrieval context doesn't translate to accurate scoring.
3. **Hybrid dominates** on every metric: τ = 0.80, Top-1 = 83%, Top-2 = 97%, MAE = 0.69. Separating parameter extraction (LLM strength) from calculation (deterministic) produces the best results.
4. **Hybrid is also the most efficient**: fewest total tokens (98,016), fewest API calls (100), and lowest total latency (275.6s) — despite higher per-call complexity.

---

## Validation Metrics

| Category | Metrics |
|---|---|
| Accuracy | MAE, RMSE, Kendall's τ, Spearman's ρ, Top-1 Accuracy, Top-2 Accuracy |
| Architecture Diagnostics | API calls, input/output tokens, total latency, success rate, failure rate |

---

[Notebook](Notebook.pdf) | [Evaluation Metrics](Evaluation_Metrics_Derivations.pdf) | [How Budget Penalties Were applied](Budget_Penalties.pdf) | [Reference Ranges for Value Functions](Reference_Ranges_for_Value_Functions) | [Worked Calculator Examples](Calculator_Examples.pdf)

---

## Citation / Collaborator

This project is being developed into a journal paper with **Dr. River Huang (Paul Scherrer Institut, Switzerland)**.
