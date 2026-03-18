# Multi-Agent Guided Measure for PHQ-8 Assessment (MAGMA)
MAGMA is a multi-agent framework for automated PHQ-8 administration that replaces linear LLM assessment with structured, evidence-verified clinical reasoning. Through a verify-then-score mechanism and active consistency checks, the system prevents diagnostic drift and logic hallucinations. The result is a scalable screening tool grounded in explicit, non-contradictory clinical evidence.

## 📖 Overview
This repository contains the implementation of a multi-agent system called MAGMA (Multi-Agent Guided Measure for PHQ-8 Assessment) designed to automate the administration of the Patient Health Questionnaire-8 (PHQ-8). (LLMs) that treat assessment as a linear slot-filling task—often leading to "diagnostic drift" and hallucinations—this framework operates as a coordinated feedback loop.

Designed to mimic the cognitive process of a human clinician, the system employs a "Verify-then-Score" mechanism. It decomposes the clinical interview into specialized roles (Navigation, Questioning, Clarification, Alignment, and Scoring) to ensure that every diagnostic decision is grounded in explicit, non-contradictory evidence.

## ✨ Key Unlike standard single-agent Large Language Models 
* **5-Agent Specialized Architecture**: distinct agents for Navigation, Questioning, Clarification, Alignment, and Scoring.
* **Active Detection Modules**: Real-time validation of patient responses for Timeframe, Vagueness, and Relevance.
* **Logical Consistency Checks**: An Alignment Agent that cross-references new responses against previous history (e.g., ensuring "Insomnia" aligns with "Fatigue") to prevent logic hallucinations.
* **Simulated Client Pipeline**: A robust testing framework using DAIC-WOZ transcripts to generate high-fidelity patient personas with variable response styles.

## 🏗️ System Architecture
The workflow is divided into four chronological stages: Interaction, Analysis, Navigation, and Final Assessment.

1. The Agents
* **Question Agent** - Facilitates the dialogue, asking PHQ-8 items and follow-up using 2 probing techniques (Funneling and Clarification) based on Network Approach to Psychopathology and Shea's Clinical Interviewing (Investigtion Loops).
* **Clarification Agent**(Detection) - Monitors for "Missing Key Domains" (Timeframe, Severity, Relevance). Uses a hybrid NLI + GPT detection architecturee to trigger specific follow-ups.
* **Alignment Agent**(Detection) - Monitors for logical consistency. Uses an Item-Dependency Map to verify that symptoms match (e.g., Item 1 links to Items 2, 4, 7, 8).
    * **Four Adaptive Question Selection Strategies**: Baseline (fixed order), Information Gain (PMI × MIRT gain), Threshold (MIRT evidence ceiling), and Hybrid OR (either condition fires) — all sharing identical agent architecture with only the skip condition differing.
    * **PMI-Weighted Symptom Co-occurrence Graph**: Population-level co-occurrence between PHQ-8 symptoms quantified via Pointwise Mutual Information (PMI), enabling evidence propagation from confirmed symptoms to correlated domains and principled adaptive skipping.
* **Navigation Agent**(Control) - The logic gatekeeper. Evaluates signals from detection agents to decide whether to loop back for clarification or proceed to the next item.
* **Scoring Agent**(Assessment) - Aggregates verified evidence to produce the final PHQ-8 severity score.

2. Operational Workflow
The system actively halts linear progression if validity criteria are not met.
**Input**: User responds to a PHQ-8 item.
**Analysis**: Clarification and Alignment agents scan the response.
**Decision**:
* If Ambiguous/Contradictory: Navigation Agent triggers a Clarification Loop.
* If Valid: Navigation Agent authorizes the Scoring Agent to log the evidence and proceed.

## 🧪 Simulation & Testing Protocol
To rigorously evaluate the system without risking live patients, we implement a *Simulated Client* pipeline based on the *DAIC-WOZ dataset*.

**Patient Profile Generation**
Raw transcripts are processed into structured Participant Profile Cards covering 9 clinical domains (Symptoms, Affective Tone, Emotion, Behavioral, Cognitive Pattern, Core belief, Intermediate Belief, Relational Context and Conversation Style).

**Response Style Hierarchies**
To test the system's robustness against real-world linguistic variability, the simulated clients operate on three difficulty levels based on Rappport:**Response Style Hierarchies**Simulated clients operate on three rapport-driven difficulty levels (scale 1--5):* **Level 1 (Direct)** — Rapport ≥ 4: Clear, explicit answers with full symptom disclosure. All psychological domains unmasked.* **Level 2 (Targeted Ambiguity)** — Rapport = 3: Introduces one specific clinical ambiguity (Vagueness, Timeframe, or Relevance). External domains accessible, Internal domains masked.* **Level 3 (Complex Ambiguity)** — Rapport ≤ 2: Introduces two simultaneous flaws. All domains masked, representing active resistance.

## 🔬 How to run the model
To run the system, use the following command:

# Baseline (fixed order, no skipping)
python3 MAGMA_Backup_NLI-GPT_.py --pid <participant_id>

# Information Gain (PMI × MIRT skip)
python3 MAGMA_InformationGain.py --pid <participant_id>

# Threshold (MIRT evidence ceiling skip)
python3 MAGMA_Threshold.py --pid <participant_id>

# Hybrid OR (either condition fires)
python3 MAGMA-Hybrid.py --pid <participant_id>

Replace <participant_id> with the ID of the participant you want to run.

## 📊 Evaluation Metrics & Performance Analysis
This system is evaluated across three dimensions critical for clinical AI: **Robustness**, **Scoring Accuracy**, **Clinical Coherence**.
The system is evaluated across the following metrics:

| Metric | Description |
|---|---|
| Macro-F1 | Balanced accuracy across severity classes |
| MAE | Mean deviation from ground truth PHQ-8 score |
| Exact Match | Perfect score agreement rate |
| False Positive Rate | Over-diagnosis prevention |
| Avg. Questions Asked | Interview efficiency per strategy |
| Avg. Skipped Domains | Adaptive selection effectiveness |
| Missing Domain Rate | Clarification agent recall |
| Alignment Score | Logical coherence across symptom network |

## ⚙️ Configuration
Key parameters shared across all four files:

| Parameter | Value | Description |
|---|---|---|
| GAIN_THRESHOLD | 1.5 | PMI × MIRT skip threshold |
| CONFIRM_THRESHOLD | 1.5 | MIRT evidence ceiling |
| CORR_THRESHOLD | 0.5 | Spearman correlation threshold |
| CONTRADICT_THRESHOLD | 0.7 | NLI contradiction threshold |
| NEUTRAL_THRESHOLD | 0.6 | NLI vagueness threshold |

## 📁 Dataset
- **DAIC-WOZ**: Place ground truth labels at 
`Dataset/PHQ8 Mapping/GrouthTruth_PHQ8_Labels.csv`
- **Participant profiles**: Place at 
`Clean_Dataset/profiles/<pid>_client_profile.json`
