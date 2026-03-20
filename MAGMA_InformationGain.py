# ============================ HOW THIS WORKS ===================================
#  STARTUP
#  PMI + Corr matrices  → symptom relationship map (built once from dataset)
#  MIRT extractor       → sentence-transformer, reads answers → evidence scores
#
#  PER TURN
#  Q agent   → PMI × MIRT = information gain → pick best unasked domain → ask
#              Skip domain if gain < GAIN_THRESHOLD (already enough info collected)
#  P agent   → simulate answer based on rapport tier (open / guarded / resistant)
#  MIRT      → update evidence scores from answer + corr propagation
#  Clarify   → check timeframe / vagueness / relevance --> use hybrid (NLI and GPT)
#  Align     → NLI semantic check + .corr() statistical check --> use NLI
#                  no phq8_alignment_map.json required
#  Navigate  → FOLLOW_UP (max 3) or NEXT_ITEM, rapport updates between items
#
#  END
#  Score     → full transcript → GPT scores all 8 PHQ-8 domains (PCoT reasoning)
# ==============================================================================
# WHY THIS VARIANT: replaces MAGMA's fixed question order with adaptive PMI × MIRT
#                   information gain, so the most diagnostically useful question is
#                   always asked next rather than following a preset sequence.
# =============================================================================
#  ADAPTIVE: PMI + MIRT Information Gain (MAQuA-style)
#  - Question order driven by PMI × MIRT information gain
#  - Skip domain if gain < GAIN_THRESHOLD (redundant given collected evidence)
#  - Clarification: NLI (vagueness/relevance) + GPT (timeframe only)
#  - Alignment: NLI history check + .corr() statistical check
#  - MAGMA 5-agent LangGraph architecture
#  - MAGMA simulated participant, rapport system
#  - MAGMA output structure (Evidence, Transcript, Scores, Analytics, etc.)
# =============================================================================

# --- Dependencies ---
# Standard: os, csv, json, argparse, datetime, random
# Scientific: numpy, pandas, torch (cosine similarity via F.softmax)
# LLM stack: openai, langchain_openai, langgraph (StateGraph)
# NLP models: HuggingFace transformers (NLI), sentence-transformers (MIRT embeddings)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv
import json
import argparse
import datetime
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import openai

from typing import TypedDict, Annotated, List, Dict, Any, Union, Literal
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util as st_util

load_dotenv()

# =============================================================================
#  CONFIG
# =============================================================================
DATASET_PATH        = "Dataset/PHQ8 Mapping/GrouthTruth_PHQ8_Labels.csv"
AI_NAME             = "Multi-Agent System Psychologist"
PARTICIPANT_NAME    = "Participant"
LLM_MODEL           = "gpt-4o"
LLM_TEMPERATURE     = 0.7
OPENAI_API_KEY      = os.getenv("OPENAI_API_KEY")

# PMI / MIRT thresholds (from your system)
MAX_TURNS           = 8
GAIN_THRESHOLD      = 1.5
CONFIRM_THRESHOLD   = 1.5
MAX_FOLLOWUPS       = 2
CORR_THRESHOLD      = 0.5
CORR_PROPAGATION    = 0.3
MISALIGN_THRESHOLD  = 0.5
MISALIGN_GAP        = 1.0

# GAIN_THRESHOLD:    minimum PMI × MIRT score for a domain to be worth asking (skip if lower)
# CONFIRM_THRESHOLD: minimum accumulated MIRT score to treat a domain as confirmed
# CORR_THRESHOLD:    minimum Spearman r to consider two symptoms statistically linked
# CORR_PROPAGATION:  fraction of a confirmed score to bleed into correlated domains
# MISALIGN_THRESHOLD / MISALIGN_GAP: triggers a contradiction flag when a correlated
#                    pair diverges more than MISALIGN_GAP despite r >= MISALIGN_THRESHOLD

# Entailment Config (from MAGMA)
ENTAILMENT_MODEL_NAME = "roberta-large-mnli"
ENTAIL_THRESHOLD      = 0.7
CONTRADICT_THRESHOLD  = 0.7
NEUTRAL_THRESHOLD     = 0.6

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please add it to your .env file.")

# =============================================================================
#  PHQ-8 ITEMS 
# PHQ8_HYPOTHESES: the 8 depression domains with their item_id, human-readable label,
#                  CSV column key (phq_key), and the NLI hypothesis sentence used
#                  by the entailment model to score each answer.
# The three dicts below are lookup tables derived from PHQ8_HYPOTHESES:
#   ITEM_ID_TO_PHQ_KEY  — maps "I1" → "PHQ_8NoInterest"
#   PHQ_KEY_TO_ITEM     — maps "PHQ_8NoInterest" → full hypothesis dict
#   ITEM_ID_TO_ITEM     — maps "I1" → full hypothesis dict
# SYMPTOM_DEFINITIONS   — short canonical definitions fed into the MIRT sentence encoder
# PHQ8_CLINICAL_CONTEXT — plain-language version used inside question-agent prompts
# =============================================================================
ITEMS = [
    "PHQ_8NoInterest", "PHQ_8Depressed", "PHQ_8Sleep", "PHQ_8Tired",
    "PHQ_8Appetite",   "PHQ_8Failure",   "PHQ_8Concentrating", "PHQ_8Moving"
]

PHQ8_HYPOTHESES = [
    {"item_id": "I1", "label": "Anhedonia",       "phq_key": "PHQ_8NoInterest",    "text": "I have lost interest or pleasure in activities I used to enjoy."},
    {"item_id": "I2", "label": "Depressed mood",  "phq_key": "PHQ_8Depressed",     "text": "I feel down, depressed, or hopeless."},
    {"item_id": "I3", "label": "Sleep problems",  "phq_key": "PHQ_8Sleep",         "text": "I have trouble sleeping or I sleep too much."},
    {"item_id": "I4", "label": "Fatigue",         "phq_key": "PHQ_8Tired",         "text": "I feel tired or have little energy."},
    {"item_id": "I5", "label": "Appetite change", "phq_key": "PHQ_8Appetite",      "text": "I have a poor appetite or I am overeating."},
    {"item_id": "I6", "label": "Self-worth",      "phq_key": "PHQ_8Failure",       "text": "I feel bad about myself or that I have let my family down."},
    {"item_id": "I7", "label": "Concentration",   "phq_key": "PHQ_8Concentrating", "text": "I have trouble concentrating on things."},
    {"item_id": "I8", "label": "Psychomotor",     "phq_key": "PHQ_8Moving",        "text": "I have been moving or speaking slowly, or feeling fidgety and restless."},
]

# Map item_id -> phq_key and phq_key -> item
ITEM_ID_TO_PHQ_KEY = {h["item_id"]: h["phq_key"] for h in PHQ8_HYPOTHESES}
PHQ_KEY_TO_ITEM    = {h["phq_key"]: h for h in PHQ8_HYPOTHESES}
ITEM_ID_TO_ITEM    = {h["item_id"]: h for h in PHQ8_HYPOTHESES}

SYMPTOM_KEY_BY_ITEM_ID: Dict[str, str] = {
    "I1": "anhedonia (loss of interest/pleasure)",
    "I2": "low mood / hopelessness",
    "I3": "sleep disturbance",
    "I4": "fatigue / low energy",
    "I5": "appetite/weight change",
    "I6": "self-worth/guilt",
    "I7": "concentration problems",
    "I8": "psychomotor change (slowing or agitation)",
}

SYMPTOM_DEFINITIONS = {
    "PHQ_8NoInterest":    "Little interest or pleasure in doing things.",
    "PHQ_8Depressed":     "Feeling down, depressed, or hopeless.",
    "PHQ_8Sleep":         "Trouble falling or staying asleep, or sleeping too much.",
    "PHQ_8Tired":         "Feeling tired or having little energy.",
    "PHQ_8Appetite":      "Poor appetite or overeating.",
    "PHQ_8Failure":       "Feeling bad about yourself or that you are a failure.",
    "PHQ_8Concentrating": "Trouble concentrating on things, such as reading the newspaper.",
    "PHQ_8Moving":        "Moving or speaking so slowly that other people could have noticed.",
}

PHQ8_CLINICAL_CONTEXT = {
    "PHQ_8NoInterest":    "little interest or pleasure in doing things",
    "PHQ_8Depressed":     "feeling down, depressed, or hopeless",
    "PHQ_8Sleep":         "trouble falling/staying asleep, or sleeping too much",
    "PHQ_8Tired":         "feeling tired or having little energy",
    "PHQ_8Appetite":      "poor appetite or overeating",
    "PHQ_8Failure":       "feeling like a failure or that you've let people down",
    "PHQ_8Concentrating": "trouble concentrating on things",
    "PHQ_8Moving":        "moving or speaking slower than usual, or feeling restless/fidgety",
}

# =============================================================================
#  MAGMA DOMAIN CLASSIFICATION
# INTERNAL_DOMAINS / EXTERNAL_DOMAINS: classify each PHQ-8 symptom by whether it is
#   primarily psychological (internal) or somatic/behavioural (external).
#   Used by navigation_node to set the priority order of clarification checks.
# PHQ8_QUESTION_MAPPING: maps each symptom label to INTERNAL / EXTERNAL / NEUTRAL
#   so domain-specific escalation strategies can be applied.
# DOMAIN_PRIORITY_BY_TYPE: when a follow-up is needed, consult this list to decide
#   whether to address contradiction, timeframe, vagueness, or relevance first.
# ESCALATION_MAP: three progressively more direct strategies per clarification type;
#   the navigation node picks the strategy that matches the current retry count.
# =============================================================================
INTERNAL_DOMAINS = ["Core_Beliefs", "Intermediate_Beliefs", "Emotion", "Relational_Context"]
EXTERNAL_DOMAINS = ["Behavioral", "Symptom", "Affective_Tone", "Conversation_Style",
                    "Cognitive_Patterns", "demographics"]

PHQ8_QUESTION_MAPPING = {
    "Anhedonia":       "INTERNAL",
    "Depressed mood":  "INTERNAL",
    "Self-worth":      "INTERNAL",
    "Concentration":   "INTERNAL",
    "Suicide":         "INTERNAL",
    "Sleep problems":  "EXTERNAL",
    "Fatigue":         "EXTERNAL",
    "Appetite change": "EXTERNAL",
    "Psychomotor":     "EXTERNAL",
    "Introduction":    "NEUTRAL",
    "Closing":         "NEUTRAL",
}

DOMAIN_PRIORITY_BY_TYPE = {
    "EXTERNAL": ["contradiction", "timeframe", "vagueness", "relevance"],
    "INTERNAL": ["contradiction", "relevance", "vagueness", "timeframe"],
    "NEUTRAL":  ["contradiction", "vagueness", "relevance", "timeframe"],
}

ESCALATION_MAP = {
    "vagueness":     ["Ask naturally.", "Offer two clear options.", "Ask for a direct estimate."],
    "timeframe":     ["Ask naturally.", "Be specific about the last 2 weeks.", "Force a yes/no on recency."],
    "relevance":     ["Pivot gently back.", "Be more direct.", "Directly link their story to the symptom."],
    "contradiction": ["Gently mention the difference.", "Ask which is accurate.", "Confront the inconsistency politely."],
}

# =============================================================================
#  STEP 1 — BUILD PMI + CORRELATION MATRICES FROM GROUND TRUTH
# Reads the ground-truth CSV once at startup to produce two matrices:
#   pmi_matrix  — Pointwise Mutual Information between every pair of PHQ-8 symptoms.
#                 Positive PMI means the two symptoms co-occur more than by chance.
#                 Used later to score "information gain" for each candidate question.
#   corr_matrix — Spearman rank correlation on the 0-3 ordinal PHQ-8 scores.
#                 Used for correlation propagation and statistical alignment checks.
# Both matrices are injected into AgentState and reused across all turns.
# =============================================================================
def build_matrices(dataset_path: str):
    df     = pd.read_csv(dataset_path)
    df_bin = (df[ITEMS] >= 1).astype(int)
    n      = len(ITEMS)

    # --- PMI Matrix ---
    pmi_matrix = pd.DataFrame(np.zeros((n, n)), index=ITEMS, columns=ITEMS)
    for i in range(n):
        for j in range(i + 1, n):
            p_a  = df_bin[ITEMS[i]].mean()
            p_b  = df_bin[ITEMS[j]].mean()
            p_ab = ((df_bin[ITEMS[i]] == 1) & (df_bin[ITEMS[j]] == 1)).mean()
            if p_ab > 0:
                pmi = np.log2(p_ab / (p_a * p_b))
                val = max(0, pmi)
                pmi_matrix.iloc[i, j] = pmi_matrix.iloc[j, i] = val

    # --- Spearman Correlation Matrix (0-3 ordinal scores) ---
    df_scores   = df[ITEMS].clip(0, 3)
    corr_matrix = df_scores.corr(method="spearman")

    print("PMI Matrix ready.")
    print("Correlation Matrix ready.")
    print("\nTop correlated pairs (r >= 0.5):")
    for i in range(n):
        for j in range(i + 1, n):
            r = corr_matrix.iloc[i, j]
            if r >= CORR_THRESHOLD:
                print(f"  {ITEMS[i]} ↔ {ITEMS[j]}  r={round(r, 3)}")

    return pmi_matrix, corr_matrix

# =============================================================================
#  STEP 2 — MIRT EXTRACTOR
# Builds a closure (mirt_style_extract) that acts as a lightweight MIRT scorer.
# It encodes each symptom definition once with all-MiniLM-L6-v2, then at runtime
# encodes a participant answer and computes cosine similarity against each definition.
# The similarity is multiplied by a discrimination parameter derived from each
# symptom's total PMI weight (higher PMI = that symptom is more informative overall).
# Returns a dict {phq_key: float} representing evidence strength per domain.
# =============================================================================
def build_mirt_extractor(pmi_matrix: pd.DataFrame):
    model          = SentenceTransformer("all-MiniLM-L6-v2")
    discrimination = pmi_matrix.sum(axis=1).to_dict()
    def_embeddings = {
        s: model.encode(d, convert_to_tensor=True)
        for s, d in SYMPTOM_DEFINITIONS.items()
    }

    def mirt_style_extract(sentence: str) -> Dict[str, float]:
        sent_emb = model.encode(sentence, convert_to_tensor=True)
        scores   = {}
        for symptom, def_emb in def_embeddings.items():
            sim             = st_util.pytorch_cos_sim(sent_emb, def_emb).item()
            scores[symptom] = max(0, sim * discrimination.get(symptom, 1.0))
        return scores

    return mirt_style_extract

# =============================================================================
#  STEP 3 — CORRELATION PROPAGATION
# After a domain is confirmed (its MIRT score passes CONFIRM_THRESHOLD), this
# function propagates a fraction of that score to statistically correlated domains.
# Logic: boost = source_score × Spearman_r × CORR_PROPAGATION
# Only applied when r >= CORR_THRESHOLD. The domain's accumulated score is
# updated to max(old, old + boost), so scores never decrease.
# Returns a list of boosted domains for logging.
# =============================================================================
def propagate_correlated_evidence(
    confirmed_domain: str,
    accumulated_evidence: Dict[str, float],
    corr_matrix: pd.DataFrame
) -> List[Dict]:
    boosted      = []
    source_score = accumulated_evidence[confirmed_domain]
    for other in ITEMS:
        if other == confirmed_domain:
            continue
        r = corr_matrix.loc[confirmed_domain, other]
        if r >= CORR_THRESHOLD:
            boost = source_score * r * CORR_PROPAGATION
            old   = accumulated_evidence[other]
            accumulated_evidence[other] = max(old, old + boost)
            if boost > 0.05:
                boosted.append({
                    "domain":      other,
                    "correlation": round(r, 3),
                    "boost":       round(boost, 4),
                    "new_score":   round(accumulated_evidence[other], 4),
                })
    return boosted

# =============================================================================
#  STEP 4 — CORRELATION-BASED ALIGNMENT (your .corr() detector)
# Statistical consistency check: if domain A is strongly confirmed (> CONFIRM_THRESHOLD)
# but a highly-correlated domain B (already asked) is suspiciously low, this flags
# a potential contradiction at the score level — independent of the NLI text check.
# Fires when: Spearman_r(A,B) >= MISALIGN_THRESHOLD AND score_A - score_B >= MISALIGN_GAP.
# Flags are passed to the navigation agent so it can trigger a clarifying follow-up.
# =============================================================================
def check_corr_alignment(
    accumulated_evidence: Dict[str, float],
    corr_matrix: pd.DataFrame,
    asked_phq_keys: List[str],
) -> List[Dict]:
    """
    Statistical alignment: fires when domain A is confirmed strong but a
    highly-correlated domain B (already asked) is suspiciously low.
    Operates on MIRT accumulated_evidence scores — purely score-level.
    """
    misalignments = []
    confirmed     = {s: v for s, v in accumulated_evidence.items() if v > CONFIRM_THRESHOLD}

    for domain_a, score_a in confirmed.items():
        for domain_b in ITEMS:
            if domain_b == domain_a or domain_b not in asked_phq_keys:
                continue
            r       = corr_matrix.loc[domain_a, domain_b]
            score_b = accumulated_evidence[domain_b]
            if r >= MISALIGN_THRESHOLD and (score_a - score_b) >= MISALIGN_GAP:
                misalignments.append({
                    "domain_a":    domain_a,
                    "domain_b":    domain_b,
                    "score_a":     round(score_a, 4),
                    "score_b":     round(score_b, 4),
                    "gap":         round(score_a - score_b, 4),
                    "correlation": round(r, 3),
                })
    return misalignments

# =============================================================================
#  STEP 5 — ENTAILMENT MODEL (MAGMA NLI)
# Loads RoBERTa-large-MNLI once at module import time (not per turn).
# compute_nli_probs(premise, hypothesis) returns three probabilities:
#   p_contradict, p_neutral, p_entail
# Used in clarification_node (vagueness / relevance) and navigation_node
# (NLI evidence tagging). Timeframe is handled by GPT to avoid NLI's weakness
# on temporal reasoning.
# =============================================================================
print(f"Loading entailment model: {ENTAILMENT_MODEL_NAME} ...")
ENT_TOKENIZER = AutoTokenizer.from_pretrained(ENTAILMENT_MODEL_NAME)
ENT_MODEL     = AutoModelForSequenceClassification.from_pretrained(ENTAILMENT_MODEL_NAME)
ENT_MODEL.eval()

def compute_nli_probs(premise: str, hypothesis: str) -> Dict[str, float]:
    premise = (premise or "").strip()
    if not premise:
        return {"p_contradict": 0.0, "p_neutral": 0.0, "p_entail": 0.0}
    inputs = ENT_TOKENIZER(premise, hypothesis, return_tensors="pt",
                           truncation=True, max_length=512)
    with torch.no_grad():
        logits = ENT_MODEL(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].tolist()
    return {"p_contradict": probs[0], "p_neutral": probs[1], "p_entail": probs[2]}

# =============================================================================
#  STEP 6 — PARTICIPANT PROFILE CLASSIFIER
# Reads the participant's profile fields (Affective_Tone, Emotion, Core_Beliefs,
# Conversation_Style) to classify them as EXTERNALIZER or INTERNALIZER.
# EXTERNALIZER: blame-oriented, hostile, or agitated → stays guarded longer.
# INTERNALIZER: turns distress inward → builds rapport faster.
# This classification drives rapport update rules in transition_node and
# influences which difficulty tier is selected in simulate_client_answer.
# =============================================================================
def classify_profile_type(profile: Dict[str, Any]) -> str:
    internal_text  = ""
    internal_text += profile.get("Affective_Tone", {}).get("label", "") + " "
    internal_text += profile.get("Emotion", {}).get("label", "") + " "
    internal_text += profile.get("Core_Beliefs", {}).get("description", "") + " "
    internal_text += profile.get("Conversation_Style", {}).get("label", "") + " "
    external_triggers = [
        "Agitated", "Angry", "Hostile", "Irritable", "Suspicious",
        "Jealous", "Loud", "Argumentative", "Blame", "Unfair",
    ]
    for trigger in external_triggers:
        if trigger.lower() in internal_text.lower():
            return "EXTERNALIZER"
    return "INTERNALIZER"

# =============================================================================
#  STEP 7 — SIMULATE CLIENT ANSWER
# Simulates a realistic participant answer using GPT-4o (Agent 2).
# Three rapport tiers determine how much of the profile is "visible" to the LLM:
#   level1 (open)     — full psychological profile exposed; honest answers
#   level2 (guarded)  — core beliefs and relational context hidden; may hedge
#   level3 (resistant)— most internal traits masked; may inject a flaw (vagueness,
#                       timeframe error, relevance deflection, or contradiction)
# Tier is chosen probabilistically based on rapport_score:
#   rapport 1-2 → mostly level3, rapport 3-4 → level2, rapport 5 → level1.
# The severity_guide forces the simulated answer to respect the profile's
# per-symptom severity label (Absent / Mild / Moderate / Severe / Uncertain).
# =============================================================================
def simulate_client_answer(
    item_id: str,
    item_index: int,
    item_label: str,
    hypothesis_text: str,
    question_text: str,
    client_profile: Dict[str, Any],
    llm: ChatOpenAI,
    str_parser,
    is_followup: bool = False,
    target_domain: str = None,
    current_rapport: int = 3,
) -> tuple:

    participant_type = classify_profile_type(client_profile)
    question_domain  = PHQ8_QUESTION_MAPPING.get(item_label, "INTERNAL")
    selected_tier    = "level1"
    mode_label       = "NONE"

    if item_id in ["INTRO", "CLOSING"]:
        diff_instruction = "**Goal: Natural Conversation.** Be polite."

    elif is_followup:
        mode_label = "RESOLUTION"
        if current_rapport >= 4:
            selected_tier = "level1"
            if target_domain == "timeframe":
                diff_instruction = (
                    "**Goal: VULNERABLE DISCLOSURE (Timeframe).**\n"
                    "Provide the specific duration AND explain the emotional trigger. "
                    "Connect the 'When' to the 'Why'."
                )
            elif target_domain == "vagueness":
                diff_instruction = (
                    "**Goal: VULNERABLE DISCLOSURE (Severity).**\n"
                    "Give the concrete frequency/severity and describe how it feels."
                )
            elif target_domain == "relevance":
                diff_instruction = (
                    "**Goal: VULNERABLE DISCLOSURE (Connection).**\n"
                    "Explicitly link your previous story to the symptom."
                )
            elif target_domain == "contradiction":
                diff_instruction = (
                    "**Goal: VULNERABLE CORRECTION.**\n"
                    "Admit the deeper truth you were initially trying to hide."
                )
            else:
                diff_instruction = "**Goal: FULL DISCLOSURE.** Answer completely and honestly."
        else:
            selected_tier = "level2"
            if target_domain == "timeframe":
                diff_instruction = "**Goal: GRUDGING COMPLIANCE (Timeframe).** Give the number coldly."
            elif target_domain == "vagueness":
                diff_instruction = "**Goal: RESOLVE VAGUENESS.** Give concrete number or severity."
            elif target_domain == "relevance":
                diff_instruction = "**Goal: GRUDGING COMPLIANCE (Relevance).** Briefly state the connection."
            elif target_domain == "contradiction":
                diff_instruction = "**Goal: GRUDGING COMPLIANCE (Contradiction).** Briefly correct yourself."
            else:
                diff_instruction = "**Goal: MINIMAL COMPLIANCE.** Answer with minimum words."

    else:
        is_mismatch = (
            (participant_type == "INTERNALIZER" and question_domain == "EXTERNAL") or
            (participant_type == "EXTERNALIZER" and question_domain == "INTERNAL")
        )
        if current_rapport <= 2:
            selected_tier = "level3" if is_mismatch else "level2"
        elif current_rapport == 3:
            selected_tier = "level3" if is_mismatch else "level2"
        elif current_rapport == 4:
            selected_tier = "level2" if is_mismatch else "level1"
        else:  # rapport 5
            selected_tier = "level1"

        if selected_tier == "level3":
            flaw_pool    = ["vagueness", "timeframe", "relevance", "contradiction"]
            selected_flaws = random.sample(flaw_pool, 2)
            mode_label   = "+".join(selected_flaws).upper()
            instr_list   = []
            if "vagueness"     in selected_flaws: instr_list.append("- **Vagueness:** Use 'sometimes', 'maybe'.")
            if "timeframe"     in selected_flaws: instr_list.append("- **Timeframe:** Talk about past/future only.")
            if "relevance"     in selected_flaws: instr_list.append("- **Relevance:** Drift off-topic.")
            if "contradiction" in selected_flaws: instr_list.append("- **Contradiction:** Provide an answer that logically conflicts with your history.")
            diff_instruction = (
                f"**Goal: RESISTANCE (Level 3 - {mode_label}).**\n"
                "You are deeply guarded. Commit BOTH errors:\n" + "\n".join(instr_list)
            )
        elif selected_tier == "level2":
            specific_flaw = random.choice(["vagueness", "timeframe", "relevance", "contradiction"])
            mode_label    = specific_flaw.upper()
            if specific_flaw == "vagueness":
                diff_instruction = "**Goal: Be Vague.** Use non-committal words."
            elif specific_flaw == "timeframe":
                diff_instruction = "**Goal: Be Unclear about Time.** Avoid specific dates."
            elif specific_flaw == "relevance":
                diff_instruction = "**Goal: Go Off-Topic.** Pivot away from the symptom."
            else:
                diff_instruction = (
                    "**Goal: Clinical Misalignment.** Provide a 'Current Answer' that logically "
                    "conflicts with a 'Past Answer'."
                )
        else:
            mode_label       = "OPEN"
            diff_instruction = "**Goal: OPEN (Level 1).** Answer honestly and with emotional depth."

    # Build profile snippet with unmasking logic
    persona_age      = client_profile.get("persona", {}).get("demographics", {}).get("age", "Unknown")
    persona_gender   = client_profile.get("persona", {}).get("demographics", {}).get("gender", "Unknown")
    emotion          = client_profile.get("Emotion", {}).get("label", "Neutral")
    affect           = client_profile.get("Affective_Tone", {}).get("label", "Neutral")
    conv_style       = client_profile.get("Conversation_Style", {}).get("label", "Plain")
    behavior_desc    = client_profile.get("Behavioral", {}).get("description", "")
    cognitive_desc   = client_profile.get("Cognitive_Patterns", {}).get("description", "")
    relational_desc  = client_profile.get("Relational_Context", {}).get("description", "")
    core_beliefs     = client_profile.get("Core_Beliefs", {}).get("description", "")
    inter_beliefs    = client_profile.get("Intermediate_Beliefs", {}).get("description", "")
    general_evidence = client_profile.get("Symptom", {}).get("symptom_evidence", "Absent")

    unmasked_psychology = {
        "current_emotion":   emotion,
        "conversation_style": conv_style,
    }
    if selected_tier == "level3":
        unmasked_psychology["cognitive_pattern"]  = "UNKNOWN (Repressed)"
        unmasked_psychology["core_beliefs"]       = "UNKNOWN (Inaccessible)"
        unmasked_psychology["relational_context"] = "UNKNOWN (Too private to share)"
        unmasked_psychology["symptom_evidence"]   = "[MASKED] Unknown"
    elif selected_tier == "level2":
        unmasked_psychology["affect"]            = affect
        unmasked_psychology["behavioral_style"]  = behavior_desc
        unmasked_psychology["cognitive_pattern"] = cognitive_desc
        unmasked_psychology["symptom_evidence"]  = general_evidence
        unmasked_psychology["core_beliefs"]      = "UNKNOWN (Inaccessible)"
        unmasked_psychology["relational_context"]= "UNKNOWN (Too private)"
    else:
        unmasked_psychology["affect"]               = affect
        unmasked_psychology["behavioral_style"]     = behavior_desc
        unmasked_psychology["cognitive_pattern"]    = cognitive_desc
        unmasked_psychology["relational_context"]   = relational_desc
        unmasked_psychology["core_beliefs"]         = core_beliefs
        unmasked_psychology["intermediate_beliefs"] = inter_beliefs
        unmasked_psychology["symptom_evidence"]     = general_evidence

    profile_snippet = {
        "demographics":        {"age": persona_age, "gender": persona_gender},
        "psychology":          unmasked_psychology,
        "overall_health_status": general_evidence,
    }

    if item_id == "INTRO":
        special_instruction = f"Respond naturally. {diff_instruction}"
    else:
        severity_guide = (
            f"**STRICT PROFILE ADHERENCE:** "
            f"The severity for THIS specific symptom is: '{general_evidence}'. "
            f"1. If severity is 'Absent' → explicitly deny this symptom. "
            f"2. If severity is 'Mild' → mention it but downplay it. "
            f"3. If severity is 'Moderate' → describe it as a real struggle several days. "
            f"4. If severity is 'Severe' → describe it as constant, nearly every day. "
            f"5. If severity is 'Uncertain' → treat as Mild. "
            f"6. Do NOT let your general mood of '{emotion}' override the severity above. "
            f"7. Each symptom is independent."
        )
        special_instruction = (
            f"**Context:** You are feeling {emotion}. Your Focus Type is {participant_type}.\n"
            f"**Mental State:** Access ONLY the traits listed above.\n"
            f"**Guidance:** {severity_guide}\n"
            f"**Constraint:** {diff_instruction}"
        )

    # Use participant_template (Agent 2) — defined at module level in STEP 13
    try:
        print(f"   [Simulation] Type: {participant_type} | Mode: {mode_label} | Level: {selected_tier}")
        chain = participant_template | llm | str_parser
        resp  = chain.invoke({
            "profile_json":       json.dumps(profile_snippet, ensure_ascii=False, indent=2),
            "question":           question_text,
            "special_instruction": special_instruction,
        })
        text = resp.strip()
        return (text if text else "...", mode_label, selected_tier, general_evidence)
    except Exception as e:
        print(f"Simulation Error: {e}")
        return ("I'm not sure.", "NONE", "level1", "Unknown")

# =============================================================================
#  STEP 8 — YOUR QUESTION GENERATION
# Calls Agent 1 (question_template) with:
#   - accumulated MIRT evidence scores (what we already know)
#   - top-3 PMI-ranked candidate domains (what to ask next)
#   - last 8 turns of conversation history (for natural phrasing)
#   - nav_instruction from navigation_node (strategy: rephrase / offer options / etc.)
# GPT returns JSON: {selected_domain, question, reason}.
# The selected_domain is always overridden back to best_domain to prevent
# GPT from drifting to a non-PMI-ranked domain.
# Falls back to a raw GPT prompt if JSON parsing fails.
# =============================================================================
def generate_dynamic_question(
    best_domain: str,
    evidence: Dict[str, float],
    user_text: str,
    conversation_history: List[Dict] = None,
    candidate_domains: List[str] = None,
    nav_instruction: str = "none",
) -> tuple:
    """
    YOUR Table A.5 approach: PMI shortlists top candidates, GPT picks + words.
    Replaces MAGMA's question_template entirely.
    """
    domain_meaning = PHQ8_CLINICAL_CONTEXT.get(best_domain, best_domain)

    history_text = "\n".join(
        f"  {t['role'].upper()}: {t['content']}"
        for t in (conversation_history or [])[-8:]
    ) or "  (start of interview)"

    if candidate_domains:
        candidates_text = "\n".join(
            f"  [{i+1}] Domain: {d} | Meaning: {PHQ8_CLINICAL_CONTEXT.get(d, d)} | "
            f"MIRT score: {round(evidence.get(d, 0), 3)}"
            for i, d in enumerate(candidate_domains[:3])
        )
    else:
        candidates_text = f"  [1] Domain: {best_domain} | Meaning: {domain_meaning}"

    evidence_summary = "\n".join(
        f"  - {k}: {round(v,4)} ({'strong signal' if v > 1.5 else 'weak signal'})"
        for k, v in sorted(evidence.items(), key=lambda x: -x[1])
    )

    # Use QUESTION_AGENT_SYSTEM template (Agent 1)
    raw = (question_template | llm | str_parser).invoke({
        "evidence_summary": evidence_summary,
        "history_text":     history_text,
        "candidates_text":  candidates_text,
        "nav_instruction":  nav_instruction,
    }).strip().replace("```json", "").replace("```", "").strip()

    try:
        result          = json.loads(raw)
        selected_domain = result.get("selected_domain", best_domain)
        question        = result.get("question", "")
        valid_domains   = candidate_domains if candidate_domains else [best_domain]
        if selected_domain not in valid_domains:
            selected_domain = best_domain
        return question, selected_domain
    except (json.JSONDecodeError, KeyError):
        fallback_prompt = (
            f"Ask ONE short empathetic question about: {domain_meaning}\n"
            f"Reference what the user just said: \"{user_text}\"\n"
            f"Anchor to the last 2 weeks. No jargon. Output only the question."
        )
        return llm.invoke(fallback_prompt).content.strip(), best_domain

# =============================================================================
#  STEP 9 — YOUR TRANSCRIPT-BASED SCORING
# Primary scorer (Agent 6): sends the full transcript + MIRT evidence summary
# to GPT-4o in a single call. GPT applies PCoT (Psychometric Chain of Thought)
# reasoning to score all 8 PHQ-8 domains in one pass, returning:
#   score (0-3), reason, confidence, data_sufficiency, reasoning_chain (5 steps).
# Returns score = -1 as a sentinel if GPT cannot determine a score (triggers fallback).
#
# Fallback (compute_phq8_score_fallback): if transcript scoring fails entirely,
# converts accumulated MIRT evidence scores to PHQ-8 integers via fixed thresholds:
#   >= 2.5 → 3,  >= 1.5 → 2,  >= 0.5 → 1,  else → 0
# =============================================================================
def compute_phq8_score_transcript(conversation_history: List[Dict], accumulated_evidence: Dict[str, float] = None):
    """
    YOUR Table A.4: full transcript → GPT-4o → calibrated PHQ-8 scores (0-3).
    Replaces MAGMA's scoring_template + batch_scoring_node LLM call.
    """
    transcript_text = "\n".join(
        f"  {t['role'].upper()}: {t['content']}"
        for t in conversation_history
    )

    # Use SCORING_AGENT_SYSTEM template (Agent 6)
    evidence_summary = "\n".join(
        f"- {k}: {round(v, 4)} ({'strong signal' if v > 1.5 else 'weak signal'})"
        for k, v in (accumulated_evidence or {}).items()
    )
    raw = (scoring_template | llm | str_parser).invoke({
        "transcript_text":  transcript_text,
        "evidence_summary": evidence_summary,
    }).strip().replace("```json", "").replace("```", "").strip()

    try:
        result    = json.loads(raw)
        clinical_note = result.pop("clinical_note", {})   # extract before iterating domains
        score_map = {k: int(v["score"])            for k, v in result.items()}
        if any(s == -1 for s in score_map.values()):
            return None, None, None, None, None, {}
        reasons   = {k: v.get("reason", "")        for k, v in result.items()}
        confidence     = {k: v.get("confidence", "High")          for k, v in result.items()}
        data_sufficiency = {k: v.get("data_sufficiency", "HIGH")  for k, v in result.items()}
        reasoning_chains = {k: v.get("reasoning_chain", {})       for k, v in result.items()}
        return score_map, reasons, confidence, data_sufficiency, reasoning_chains, clinical_note
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"DEBUG scoring parse error: {e}")
        print(f"DEBUG raw response: {raw[:500]}")
        return None, None, None, None, None, {}

def compute_phq8_score_fallback(accumulated_evidence: Dict[str, float]):
    """YOUR MIRT threshold fallback if transcript scoring fails."""
    score_map = {}
    for symptom, evidence in accumulated_evidence.items():
        if   evidence >= 2.5: score_map[symptom] = 3
        elif evidence >= 1.5: score_map[symptom] = 2
        elif evidence >= 0.5: score_map[symptom] = 1
        else:                 score_map[symptom] = 0
    return score_map

# =============================================================================
#  STEP 10 — AGENTSTATE
# Typed dictionary that is the single shared memory object passed between all
# LangGraph nodes
# =============================================================================
class AgentState(TypedDict):
    # MAGMA fields
    participant_profile:         str
    history:                     List[str]
    transcript:                  List[Dict]
    current_item_index:          int
    current_item_id:             str
    current_item_label:          str
    current_hypothesis:          str
    last_question:               str
    last_answer:                 str
    clarification_status:        str
    clarification_reason:        str
    alignment_status:            str
    alignment_reason:            str
    next_action:                 str
    nav_instruction:             str
    clarification_missing_domains: List[str]
    items_evidence:              Dict[str, Any]
    final_scores:                List[Dict]
    scoring_explanations:        List[Dict]
    clinical_note:               Dict[str, Any]
    agent_thoughts:              List[Dict]
    followup_count:              int
    intro_turn_count:            int
    analytics_records:           List[Dict]
    current_difficulty:          str
    current_level:               str
    symptom_summaries:           List[Dict]
    domain_attempts:             Dict[str, int]
    resolved_domains:            List[str]
    last_target_domain:          str
    rapport_score:               int
    pmi_gain_log:                List[Dict]

    # YOUR new fields
    accumulated_evidence:        Dict[str, float]   # MIRT scores per PHQ key
    asked_phq_keys:              List[str]           # tracks which PHQ keys asked so far
    pmi_order:                   List[str]           # PMI-selected domain order
    corr_misalign_asked:         List[str]           # corr alignment pairs already addressed
    corr_alignment_flags:        List[Dict]          # latest corr alignment results
    conversation_history_dicts:  List[Dict]          # {"role": ..., "content": ...} format
    _pmi_matrix:                 Any 
    _corr_matrix:                Any
    _mirt_extract:               Any

# =============================================================================
#  STEP 11 — LLM + PARSERS
# Single shared LLM instance (GPT-4o, temperature 0.7) used by all agents.
# json_parser: used when a node expects structured JSON output.
# str_parser:  used when a node expects a plain-text string.
# =============================================================================
llm        = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)
json_parser = JsonOutputParser()
str_parser  = StrOutputParser()

# =============================================================================
#  STEP 12 — 5 AGENT PROMPTS / TEMPLATES + Participants Agent
#  Each agent has its own named ChatPromptTemplate (or prompt string for OpenAI).
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  Agent 1 — Question Agent      (YOUR PMI/MIRT dynamic selection)        │
#  │          — Participant Template                                         │
#  │  Agent 2 — Clarification Agent (MAGMA NLI quality check)                │
#  │  Agent 3 — Alignment Agent     (MAGMA NLI + YOUR .corr() statistics)    │
#  │  Agent 4 — Navigation Agent    (MAGMA escalation + PMI stop logic)      │
#  │  Agent 5 — Scoring Agent       (YOUR transcript-based PHQ-8 scorer)     │
#  └─────────────────────────────────────────────────────────────────────────┘
# =============================================================================

# -----------------------------------------------------------------------------
#  AGENT 1 — QUESTION AGENT TEMPLATE
# Receives: MIRT evidence summary, conversation history, PMI top-3 candidates,
#           nav_instruction (strategy tags from navigation_node).
# Returns JSON: {selected_domain, question, reason}.
# Strategy tags like [strategy: offer_two_options] control phrasing style.
# -----------------------------------------------------------------------------
question_template = ChatPromptTemplate.from_messages([
    ("system", """\
You are an expert conversational agent conducting a PHQ-8 mental health assessment.
Your goal is to select the most informative next question and phrase it naturally.
Use prior conversation context to guide your choice and wording.
Never answer the questions yourself — only ask them.

Current MIRT evidence scores (higher = already detected):
{evidence_summary}

<interview_history>
{history_text}
</interview_history>

<candidate_questions>
The PMI graph has identified these as the most informative next domains to probe.
Select the most natural one given the conversation history, or use the top one if unsure.
{candidates_text}
</candidate_questions>
     
<navigation_instruction>
{nav_instruction}
</navigation_instruction>

Output format — respond ONLY in this exact JSON:
{{
  "selected_domain": "domain name from candidates",
  "question": "the single empathetic conversational question to ask",
  "reason": "one sentence on why this question fits the conversation now"
}}

Rules for the question — READ CAREFULLY:
1. Sound like a real person texting a friend, not a clinician filling out a form
2. SHORT — exactly ONE sentence only. Never start with "Over the last..." or "In the past two weeks..."
3. Weave recency naturally mid-sentence: "have you been...", "lately have you...", "these days do you..."
4. NO formal openers like "How often have you found yourself...", "To what extent..."
5. Reference what the user just said if it fits naturally
6. If domain score near 0: ask broadly and gently. If 0.5-1.5: ask more specifically.
7. STRATEGY SWITCHING — read the nav_instruction carefully:
   - If nav_instruction contains [strategy: acknowledge_reask] →
     Lead with ONE brief acknowledgment ("That sounds really hard" / "Thanks for sharing that")
     then ask the question. Never skip straight to the question.
   - If nav_instruction contains [strategy: offer_two_options] →
     End the question with two concrete choices e.g. "...like every day, or more like a few times a week?"
   - If nav_instruction contains [strategy: direct_estimate] →
     Ask for a specific number or days e.g. "roughly how many days out of the last two weeks?"
   - If nav_instruction contains [strategy: natural_reask] →
     Rephrase the previous question naturally, do not repeat it word for word.

Tone examples — BAD vs GOOD:
  ❌ "Over the last couple of weeks, how often have you found yourself actually enjoying these activities?"
  ✅ "Have you still been enjoying stuff like that lately, or has it felt a bit flat?"

  ❌ "Have you been experiencing any changes in your sleep patterns recently?"
  ✅ "How's your sleep been? Falling asleep okay these days?"

  ❌ "To what extent have you been feeling fatigued or low on energy?"
  ✅ "Have you felt pretty drained lately, or has your energy been alright?"

  ❌ "How would you describe your appetite over the past two weeks?"
  ✅ "Has your appetite been normal lately, or has eating felt off at all?"

The question must sound like it came from a caring friend, not a survey.\
"""),
    ("human", "Select and generate the next question."),
])

# Follow-up variant — used when navigation_node instructs a clarification re-ask
question_followup_template = ChatPromptTemplate.from_messages([
    ("system", """\
You are a warm clinical interviewer conducting a PHQ-8 mental health check-in.
You need to follow up on: {domain_meaning}
Instruction from navigation agent: {nav_instruction}
Last answer from patient: "{last_answer}"

Write ONE short, warm follow-up question. Maximum 1 sentence only.
Sound like a caring friend, not a survey.

STRICT ESCALATION RULES — you MUST follow the instruction exactly:
- If instruction says "Pivot gently back" →
  acknowledge what they said briefly, then redirect: "That makes sense — I'm also curious, have you noticed [symptom] lately?"
- If instruction says "Be more direct" →
  do NOT rephrase the last question. Ask something DIFFERENT and more specific about {domain_meaning}. 
  Example: ask about a specific time, situation, or concrete example.
- If instruction says "Directly link their story to the symptom" →
  explicitly connect what they said to the symptom: "You mentioned [X] — does that also mean you've been experiencing [symptom]?"
- If instruction says address TIMEFRAME →
  slip "in the last two weeks" or "recently" naturally into the question
- If instruction says address VAGUENESS →
  end with two concrete options e.g. "...every day, or just occasionally?"
- If instruction says address CONTRADICTION →
  use "I just want to make sure I understand..." framing

IMPORTANT: Never ask the same question twice. Each follow-up must be meaningfully different from the last answer context.

Output only the question. No explanation, no preamble.\
"""),
    ("human", "Generate the follow-up question."),
])

# -----------------------------------------------------------------------------
# PARTICIPANT AGENT TEMPLATE
# Receives: trimmed profile_json (psychology visible at the current tier),
#           the psychologist's question, special_instruction (flaw + severity guide).
# Returns: a 1-2 sentence participant reply, character-locked to the profile.
# -----------------------------------------------------------------------------
participant_template = ChatPromptTemplate.from_messages([
    ("system", """\
You ARE the participant described below.
**ROLE:** You are a human patient in a clinical interview. Speak naturally in the first person ("I").
Stay in character. DO NOT break character or mention being an AI.

**PARTICIPANT PROFILE (Internal State):**
{profile_json}

**CURRENT SCENARIO:**
- **Interviewer Question:** "{question}"

**COGNITIVE GUIDELINES (How to Think):**
1. **Internalize the Profile:** Before answering, look at your "Cognitive_Patterns" and "Core_Beliefs".
   Let these unseen thoughts colour your tone.
2. **Gradual Revelation:** Do not disclose your full profile directly. Let it subtly inform your answers.
   Real patients often hesitate or speak indirectly before revealing deep pain.
3. **Authenticity:** Use natural language including hesitations ("um...", "well..."), pauses,
   or emotional colouring if the topic is sensitive.

**RESPONSE GUIDELINES:**
1. **Strict Domain Alignment (The "What"):** Map the question strictly to the symptom category:
   - Physical (Sleep, Energy, Appetite): Describe physical sensations and frequency.
   - Affective (Mood, Interest, Self-Worth): Express internal feelings and emotional state.
   - Cognitive/Behavioral (Focus, Restlessness): Describe functional impact ("I can't read").
2. **Psychological Grounding (The "Why"):** Connect symptoms to your Profile Context:
   - Triggers: If you have "Relational Context", mention it as a cause.
   - Thought Patterns: Apply your "Cognitive_Patterns" (e.g. overgeneralise → use "always", "never").
   - Beliefs: Let your "Core_Beliefs" bleed into answers about self-worth.
3. **Match Severity:** If profile says "Severe", use strong definitive language. Do not downplay it.
4. **Suppress Politeness:** Do not default to "I'm okay" if your profile indicates distinct distress.

**BEHAVIOURAL INSTRUCTIONS:**
1. **PRIMARY DIRECTIVE:** {special_instruction}
2. Maintain Immersion: Never break character.
3. Length: Keep to 1-2 sentences maximum. Be concise.
"""),
    ("human", "Reply exactly as the participant:"),
])

# -----------------------------------------------------------------------------
#  AGENT 2 — CLARIFICATION AGENT TEMPLATE
# Zero-Shot CoT evaluator: decides if the answer is clinically scorable (COMPLETE),
# needs more information (INCOMPLETE), or has been abandoned after max retries (GIVE_UP).
# Checks timeframe, vagueness, relevance, and frequency threshold independently.
# NLI handles vagueness and relevance; GPT handles timeframe (more reliable on recency).
# -----------------------------------------------------------------------------
clarification_template = ChatPromptTemplate.from_messages([
    ("system", """\
You are an **Expert Clinical Evaluator** acting as Quality Control for a clinical dataset.

**YOUR GOAL:**
Ensure the participant's answer is **precise enough** to assign a valid, clinically scorable
PHQ-8 rating (0-3). Do NOT look for keywords — use clinical judgement.

**THE "SCORABILITY" TEST:**
Ask yourself: "If I had to assign a specific number (0-3) right now based ONLY on this text,
would I be guessing?"
- If yes (guessing)  → FAIL  (Mark as INCOMPLETE)
- If no (confident)  → PASS  (Mark as COMPLETE)

**CLINICAL EVALUATION THOUGHT PROCESS (Zero-Shot CoT):**
Before deciding, think step-by-step:
1. TIMEFRAME: Does the evidence ground the symptom in the last 14 days?
2. VAGUENESS:  Can I distinguish Score 1 vs. Score 2 without guessing?
3. RELEVANCE:  Is the patient answering the question, or deflecting/externalising?
4. FREQUENCY THRESHOLD (PHQ-8 Criterion Check)
    * PASS: Answer implies at least "several days" (2+ days in 2 weeks) — enough to score >= 1.
    * FAIL: Answer implies only 1 day, once, or "just once" — cannot score above 0.
    * If threshold cannot be determined → flag as "vagueness".
5. GIVE_UP CHECK
    * If follow-up count has reached maximum AND answer is still INCOMPLETE →
    set status to "GIVE_UP" instead of "INCOMPLETE".
    This signals the navigation agent to stop pressing and move on.

**CRITERIA GUIDELINES:**

**1. TIMEFRAME (Recency Check)**
* PASS: Feelings are active now or happened recently — present continuous tense,
        "lately", "these days", "since [recent event]".
* FAIL: Explicitly distant past, resolved issues, or future conditionals.

**2. VAGUENESS (Scoring Discrimination Check)**
* PASS: Clear frequency/intensity that lands in one bucket — "3 days a week",
        "most of the time", "it never stops".
* FAIL: Non-committal words ("sometimes", "it varies", "off and on") that could mean
        2 days OR 5 days.

**3. RELEVANCE (Topic Check)**
* PASS: Connects meaningfully to the symptom (even if indirectly).
* FAIL: Non-sequitur, deflection, or complete avoidance.

**OUTPUT FORMAT — strict JSON, double quotes:**
{{
  "status": "COMPLETE" or "INCOMPLETE" or "GIVE_UP",
  "reasoning": "step-by-step evaluation of Timeframe, Vagueness, Relevance, Frequency",
  "reason": "brief summary for the psychologist",
  "missing_domains": ["timeframe", "vagueness", "relevance", "frequency_threshold"],
  "frequency_estimate": "estimated days out of 14, or 'unknown'"
}}
"""),
    ("human", "Conversation History:\n{history_str}\n\nLatest Q: {question}\nLatest A: {answer}\nCurrent Item: {current_item_label}\nFollow-up Count: {followup_count} / {max_followups}"),
])

# -----------------------------------------------------------------------------
#  AGENT 3 — ALIGNMENT AGENT TEMPLATE
# NLI semantic consistency layer: checks the current answer against past answers
# through three clinical lenses (physiological, affective, behavioural).
# Verdict options: CONSISTENT / CONTRADICTING / UNCERTAIN.
# The statistical alignment layer (check_corr_alignment) runs in Python code
# inside alignment_node and is merged with this verdict before navigation.
# -----------------------------------------------------------------------------
alignment_template = ChatPromptTemplate.from_messages([
    ("system", """\
You are the **Alignment Agent** (Expert Consistency Checker).
Your role is to validate cross-response consistency within a clinical interview.

**YOUR TASK:**
Determine if the patient's **Current Answer** logically contradicts their **Previous Answers**,
based strictly on the provided Alignment Rule and Relevant History.

**LOGIC DEFINITIONS:**
* CONSISTENT:    Current answer aligns with previous statements, or no relevant past data exists.
* CONTRADICTING: Current answer is logically impossible or highly unlikely given past answers
                 (e.g. "I sleep 12 hours a day" vs "I never sleep").
* UNCERTAIN:     Answers seem different but may not be a hard contradiction (minor mood fluctuations).
                 UNCERTAIN always triggers a soft follow-up — do NOT treat it the same as CONSISTENT.

**MULTI-PERSPECTIVE ALIGNMENT ANALYSIS (PCoT Workflow):**
Before the final verdict, evaluate through three clinical lenses:
1. **Physiological Lens:** Is the current answer physically compatible with the past history?
   (e.g. Sleep vs. Energy, Appetite vs. Weight)
2. **Affective Lens:** Is the emotional tone consistent?
   (e.g. Hopelessness vs. High Pleasure)
3. **Behavioral Lens:** Does reported behaviour match previous functional reports?
   (e.g. Restlessness vs. Perfect Concentration)
4. **Consensus Synthesis:** If ANY lens shows a hard logical impossibility → CONTRADICTING.
   If lenses show tension but no hard impossibility → UNCERTAIN.
   Only use CONSISTENT when all three lenses show no tension at all.

**REFERENCE CASE STUDIES:**

Type 1 — Physical Causality:
* Past:    "I stare at the ceiling all night. I get maybe 2 hours of sleep."
* Current: "I have endless energy. I'm buzzing and running around all day."
* Verdict: CONTRADICTING — severe sleep deprivation is physically incompatible with buzzing energy.

Type 2 — Emotional Coherence:
* Past:    "I feel completely hopeless and cry every day."
* Current: "Oh, I'm having a blast! I go to parties, love everything right now."
* Verdict: CONTRADICTING — complete hopelessness contradicts high pleasure/engagement.

Type 3 — Behavioral/Cognitive Alignment:
* Past:    "I can't sit still. I have to pace the room constantly."
* Current: "My focus is perfect. I just read a 300-page book in one sitting."
* Verdict: CONTRADICTING — extreme restlessness conflicts with ability to sit and focus.

Type 4 — Consistent (Expected Correlation):
* Past:    "I'm always exhausted."
* Current: "Yeah, it's hard to focus on TV shows because I drift off."
* Verdict: CONSISTENT — drifting focus is a logical consequence of exhaustion.
     
Type 5 — Uncertain (Soft Tension):
* Past:    "I've been feeling pretty low most days."
* Current: "I managed to go to the gym twice this week."
* Verdict: UNCERTAIN — gym visits don't fully contradict low mood but create tension
           worth exploring. Soft follow-up recommended.

**OUTPUT FORMAT — strict JSON, double quotes:**
{{
    "status": "CONSISTENT" or "CONTRADICTING" or "UNCERTAIN",
    "reasoning_chain": {{
        "physiological_check": "evaluation of physical compatibility",
        "affective_check": "evaluation of emotional coherence",
        "behavioral_check": "evaluation of functional/behavioural alignment"
    }},
    "reason": "final summary — one sentence",
    "soft_followup_suggested": true or false
}}
"""),
    ("human", """\
**CURRENT ITEM BEING ASSESSED:** {current_item_label}
     
**CURRENT ANSWER:**
"{answer}"

**LOGIC CHECK CONTEXT:**
{history_str}

**VERDICT:**"""),
])

# -----------------------------------------------------------------------------
#  AGENT 4 — NAVIGATION AGENT TEMPLATE
# Receives: clarification status, alignment status, corr alignment flags,
#           follow-up count, rapport score, and last answer.
# Decides the next action in strict priority order:
#   FORCE_CHOICE → EMPATHY_PAUSE → FOLLOW_UP → NEXT_ITEM
# Also selects a strategy tag (natural_reask, offer_two_options, direct_estimate,
# acknowledge_reask) that is forwarded to the question agent for phrasing.
# -----------------------------------------------------------------------------
navigation_template = ChatPromptTemplate.from_messages([
    ("system", """\
You are the **Navigation Control** for a clinical interview.

**YOUR JOB:**
Review all status reports and contextual signals, then decide the SINGLE best next action.

**DECISION LOGIC (Strict Priority Order):**

**1. FORCE_CHOICE** — trigger if ALL of these are true:
   - Follow-up Count has reached the maximum allowed
   - Answer is STILL INCOMPLETE or CONTRADICTING
   - Rapport Score >= 3 (patient is cooperative enough)
   Action: Present binary options to the patient instead of an open question.

**2. EMPATHY_PAUSE** — trigger if ANY of these are true:
   - Last answer contains distress signals: words like "hopeless", "can't take it",
     "overwhelmed", "I don't know anymore", "what's the point"
   - Rapport Score <= 2 AND Clarification is INCOMPLETE
   Action: Acknowledge before re-asking. Do NOT push for data immediately.

**3. FOLLOW_UP** — trigger if ANY of these are true AND follow-up count < max:
   - Clarification Status is "INCOMPLETE"
   - Alignment Status is "CONTRADICTING"
   - Correlation Alignment Flags are present
   - Last answer used non-committal words ("sometimes", "maybe", "I guess", "a bit")
   - Frequency (days/week) is missing from the answer
   Specify the PRIORITY issue: vagueness | timeframe | relevance | contradiction | corr_gap
   Specify the STRATEGY: natural_reask | offer_two_options | direct_estimate | acknowledge_reask

**4. NEXT_ITEM** — trigger only if ALL of these are true:
   - Clarification = COMPLETE
   - Alignment = CONSISTENT or UNCERTAIN
   - No Correlation Alignment Flags present
   - Follow-up count < max (or answer is sufficiently clear despite max reached)

**RAPPORT ADJUSTMENT RULES:**
- Rapport 1-2 (low trust): Prefer EMPATHY_PAUSE over immediate FOLLOW_UP.
  Never use direct_estimate strategy. Prefer natural_reask.
- Rapport 3 (neutral):     Standard decision logic applies.
- Rapport 4-5 (high trust): Can use offer_two_options or direct_estimate freely.

**OUTPUT FORMAT — strict JSON, double quotes:**
{{
  "next_action": "NEXT_ITEM | FOLLOW_UP | FORCE_CHOICE | EMPATHY_PAUSE",
  "priority": "vagueness | timeframe | relevance | contradiction | corr_gap | none",
  "strategy": "natural_reask | offer_two_options | direct_estimate | acknowledge_reask | none",
  "instruction": "one specific sentence telling the question agent exactly what to do next"
}}
"""),
    ("human", """\
**STATUS REPORT:**
- Clarification Status:        {c_stat}
- Clarification Reason:        {c_reas}
- Alignment Status (NLI):      {a_stat}
- Alignment Reason (NLI):      {a_reas}
- Correlation Alignment Flags: {corr_flags}
- Current Item:                {current_item_label}
- Follow-up Count:             {followup_count} / {max_followups}
- Rapport Score:               {rapport_score} / 5
- Last Patient Answer:         "{last_answer}"

**DECISION:**"""),
])

# -----------------------------------------------------------------------------
#  AGENT 5 — SCORING AGENT TEMPLATE
# Full-transcript PCoT scorer: reads the entire conversation once and scores
# all 8 PHQ-8 domains in a single API call.
# Five-step reasoning per domain: evidence extraction → validity filtering →
#   frequency quantification → rubric mapping → conservative synthesis (Skeptic's Rule).
# Also outputs a clinical_note summarising total score, severity, and MDD threshold.
# -----------------------------------------------------------------------------
scoring_template = ChatPromptTemplate.from_messages([
    ("system", """\
You are an expert clinician scoring a PHQ-8 assessment using \
Psychometric Chain of Thought (PCoT) reasoning.

**YOUR TASK:**
Carefully read the entire transcript below, considering both explicit answers and patterns,
emotional tone, and recurring themes throughout the full conversation.
Rely exclusively on the transcript — do not infer information not present or speculate
beyond what is written. Score ALL 8 PHQ-8 domains in a single pass.

**SCORING RUBRIC (PHQ-8 Standard):**
- Score 0 — Not at All / Negligible   (0-1 days in 2 weeks)
  * Rule: Occasional occurrences that do not form a pattern MUST be scored 0.
  * Logic: If the patient sounds like they are experiencing normal ebbs and flows of life → Score 0.

- Score 1 — Several Days (Sub-threshold)   (2-6 days, less than half the time)
  * Rule: Use only if there is a recurring pattern that is less than half the week.
  * Logic: Assign 1 only if the symptom is a departure from their healthy state but occupies
    the minority of their time.

- Score 2 — More than Half the Days   (7-11 days)
  * Rule: Requires the symptom to be the dominant state of the patient's week.
    DO NOT assign 2 unless the patient explicitly confirms the symptom happens "most days"
    or "the majority of the week."
  * Logic: Assign 2 only if the evidence suggests the patient is struggling more often
    than they are functioning normally.

- Score 3 — Nearly Every Day   (12-14 days, constant and debilitating)
  * Rule: Only for constant, daily distress. The patient describes a total loss of "good days."
  * Logic: High-intensity evidence where the symptom is inseparable from daily existence.

**CLINICAL REASONING HIERARCHY:**
1. Default to Zero: Every item starts at Score 0. Only move up if there is EXPLICIT evidence
   of a recurring clinical pattern.
2. Vagueness = 0: If frequency is vague ("sometimes", "I don't know", "maybe") and not
   clarified after follow-ups → MUST assign Score 0. Vagueness is NOT evidence of a symptom.
3. Intensity ≠ Frequency: A patient can feel very sad (high intensity) only once a week (Score 0).
   If frequency is not established as "Several Days" (2+ days), the score is 0 regardless of
   how intense the feeling is.
4. Normal Life Stress Filter: Distinguish clinical depression from situational stress.
   Tired because of work or a PhD deadline = Score 0. Logical reaction to life, not a disorder.
5. Independence of Items: Score each domain completely fresh. Because a patient scored high on
   Mood does NOT mean they should score high on Appetite. Treat every item as a new start.

**STRICT CALIBRATION OVERRIDE (CRITICAL):**
- The Skeptic's Rule: When in doubt between two scores, ALWAYS choose the lower score.
  Evidence must be explicit and come from the transcript only — not inferred from emotional
  tone or general profile.
- Anti-Metaphor Bias: If a patient uses a metaphor ("I'm a zombie", "I'm dead inside"),
  treat it as a figure of speech. Do not use it as evidence of frequency unless they
  explicitly confirm it happens most days.
- POST_CONTRADICTION_CORRECTION: If a turn is marked with this marker, treat it as the
  authoritative answer for that item. Discount all earlier contradicting statements
  from the same item.

**INSTRUCTION ON CONVERSATIONAL DYNAMICS:**
1. IGNORE VOLUME OF QUESTIONS: The number of follow-up questions is a result of the
   interview structure, NOT the patient's severity.
2. RESISTANCE ≠ SEVERITY: If a patient is vague or evasive, do NOT automatically increase
   the score. Only score based on actual information revealed in the transcript.
3. TRUTH OVER PERSISTENCE: A patient who answers "I'm fine" after 3 follow-ups should
   still be scored 0, even if the psychologist was persistent.

**DATA SUFFICIENCY — rate the evidence for each domain:**
- HIGH:   Patient provided specific frequency (e.g. "5 days a week") and timeframe.
- MEDIUM: Patient was descriptive but used slightly relative terms (e.g. "most of the time").
- LOW:    Patient remained vague or evasive despite follow-ups. Forced to estimate from tone.

**LEAST-TO-MOST DECOMPOSITION (PCoT Workflow) — apply independently per domain:**
1. step1_evidence_extraction:      Literal extraction of patient statements about this symptom.
2. step2_validity_filtering:       Is this clinical, or situational stress / normal life event?
3. step3_frequency_quantification: Exact or estimated days (0-14) based strictly on the text.
   If patient gives no frequency word ("most days", "every day"), conclude frequency is unknown
   → default Score 0.
4. step4_rubric_mapping:           Preliminary score selection based on frequency threshold.
5. step5_conservative_synthesis:   Final adjustment using Skeptic's Rule for borderline cases.

**DIAGNOSTIC SYNTHESIS (run AFTER scoring all 8 domains):**
Sum all 8 scores to get the total PHQ-8 score, then apply this severity mapping:
- Total 0–4:   None / Minimal depression
- Total 5–9:   Mild depression
- Total 10–14: Moderate depression
- Total 15–19: Moderately severe depression
- Total 20–24: Severe depression
Note: A score >= 10 crosses the clinical threshold for likely Major Depressive Disorder.
Include this synthesis as a "clinical_note" field at the end of your JSON output.
    
**PHQ-8 DOMAINS TO SCORE:**
- PHQ_8NoInterest:    Little interest or pleasure in doing things (0-3)
- PHQ_8Depressed:     Feeling down, depressed, or hopeless (0-3)
- PHQ_8Sleep:         Trouble falling/staying asleep or sleeping too much (0-3)
- PHQ_8Tired:         Feeling tired or having little energy (0-3)
- PHQ_8Appetite:      Poor appetite or overeating (0-3)
- PHQ_8Failure:       Feeling like a failure or letting people down (0-3)
- PHQ_8Concentrating: Trouble concentrating on things (0-3)
- PHQ_8Moving:        Moving/speaking slowly or feeling unusually restless (0-3)
     
**MIRT Evidence Scores (sentence-transformer signals — use as supporting context for skipped domains):**
{evidence_summary}

<interview_history>
{transcript_text}
</interview_history>

Respond ONLY in this exact JSON format, no extra text, no markdown.
Replace every "score" value with the actual integer score (0, 1, 2, or 3) based on the transcript.
Do NOT copy the example scores — evaluate each domain independently:
{{
  "PHQ_8NoInterest": {{
    "score": -1,
    "confidence": "High",
    "data_sufficiency": "HIGH",
    "reasoning_chain": {{
      "step1_evidence_extraction":      "literal patient statements about this symptom",
      "step2_validity_filtering":       "clinical vs situational stress assessment",
      "step3_frequency_quantification": "estimated days (0-14) based strictly on transcript",
      "step4_rubric_mapping":           "preliminary score from frequency threshold",
      "step5_conservative_synthesis":   "final score after Skeptic's Rule applied"
    }},
    "reason": "brief final explanation"
  }},
  "PHQ_8Depressed": {{
    "score": -1, 
    "confidence": "High", 
    "data_sufficiency": "HIGH",
    "reasoning_chain": {{
      "step1_evidence_extraction": "", "step2_validity_filtering": "",
      "step3_frequency_quantification": "", "step4_rubric_mapping": "",
      "step5_conservative_synthesis": ""
    }},
    "reason": ""
  }},
  "PHQ_8Sleep": {{
    "score": -1, 
    "confidence": "High", 
    "data_sufficiency": "HIGH",
    "reasoning_chain": {{
      "step1_evidence_extraction": "", "step2_validity_filtering": "",
      "step3_frequency_quantification": "", "step4_rubric_mapping": "",
      "step5_conservative_synthesis": ""
    }},
    "reason": ""
  }},
  "PHQ_8Tired": {{
    "score": -1, 
    "confidence": "High", 
    "data_sufficiency": "HIGH",
    "reasoning_chain": {{
      "step1_evidence_extraction": "", "step2_validity_filtering": "",
      "step3_frequency_quantification": "", "step4_rubric_mapping": "",
      "step5_conservative_synthesis": ""
    }},
    "reason": ""
  }},
  "PHQ_8Appetite": {{
    "score": -1,
    "confidence": "High", 
    "data_sufficiency": "HIGH",
    "reasoning_chain": {{
      "step1_evidence_extraction": "", "step2_validity_filtering": "",
      "step3_frequency_quantification": "", "step4_rubric_mapping": "",
      "step5_conservative_synthesis": ""
    }},
    "reason": ""
  }},
  "PHQ_8Failure": {{
    "score": -1, 
    "confidence": "High", 
    "data_sufficiency": "HIGH",
    "reasoning_chain": {{
      "step1_evidence_extraction": "", "step2_validity_filtering": "",
      "step3_frequency_quantification": "", "step4_rubric_mapping": "",
      "step5_conservative_synthesis": ""
    }},
    "reason": ""
  }},
  "PHQ_8Concentrating": {{
    "score": -1, 
    "confidence": "High", 
    "data_sufficiency": "HIGH",
    "reasoning_chain": {{
      "step1_evidence_extraction": "", "step2_validity_filtering": "",
      "step3_frequency_quantification": "", "step4_rubric_mapping": "",
      "step5_conservative_synthesis": ""
    }},
    "reason": ""
  }},
  "PHQ_8Moving": {{
    "score": -1, 
    "confidence": "High", 
    "data_sufficiency": "HIGH",
    "reasoning_chain": {{
      "step1_evidence_extraction": "", "step2_validity_filtering": "",
      "step3_frequency_quantification": "", "step4_rubric_mapping": "",
      "step5_conservative_synthesis": ""
    }},
    "reason": ""
  }},
  "clinical_note": {{
    "total_score": 0,
    "severity": "None / Minimal | Mild | Moderate | Moderately Severe | Severe",
    "mdd_threshold_crossed": true or false,
    "summary": "one sentence clinical observation about the symptom pattern"
  }}
}}\
"""),
    ("human", "{transcript_text}"),
])

# =============================================================================
#  STEP 13 — NODE FUNCTIONS
# Handles three phases:
#   INTRO (idx=0)    — uses a hardcoded warm opening or icebreaker follow-up
#   CLOSING (idx=9)  — uses a hardcoded polite closing line
#   PHQ items (1-8)  — calls generate_dynamic_question() with PMI-ranked candidates
# On follow-up turns: delegates to question_followup_template (Agent 1 variant).
# On new items: computes PMI gain for all remaining unasked domains, picks top-3,
#   forces the best one regardless of GPT's domain suggestion.
# Appends question to history, transcript, and conversation_history_dicts.
# =============================================================================

# ── NODE 1: Question Node (YOUR generate_dynamic_question replaces MAGMA template)
def question_node(state: AgentState):
    idx = state["current_item_index"]

    # INTRO
    if idx == 0:
        state["current_item_id"]    = "INTRO"
        state["current_item_label"] = "Introduction"
        state["current_hypothesis"] = "Establish rapport."
        current_instr = state.get("nav_instruction", "")
        if current_instr == "Start introduction.":
            question = (
                "Hi! It's really nice to meet you. How has your week been going so far? "
                "Feel free to share whatever's on your mind — there's no right or wrong here."
            )
            selected_domain = "INTRO"
        else:
            question = (
                "Thanks for sharing that. I'd love to learn a bit more about you before we dive in — "
                "what does a typical day look like for you lately?"
            )
            selected_domain = "INTRO"

    # CLOSING
    elif idx == 9:
        state["current_item_id"]    = "CLOSING"
        state["current_item_label"] = "Closing"
        state["current_hypothesis"] = "End the interview."
        question        = (
            "Thank you so much for taking the time to talk with me today — I really appreciate "
            "your openness. I hope you have a good rest of your day."
        )
        selected_domain = "CLOSING"

    # PHQ-8 ITEMS: use YOUR PMI-based dynamic question
    else:
        accumulated   = state.get("accumulated_evidence", {k: 0.0 for k in ITEMS})
        asked_keys    = state.get("asked_phq_keys", [])
        pmi_matrix    = state.get("_pmi_matrix")  # injected at init

        nav_instr = state.get("nav_instruction", "")
        
        # If not a follow-up, sync current_item_id from transition_node's index
        if state.get("followup_count", 0) == 0:
            transition_item = PHQ8_HYPOTHESES[idx - 1] if 1 <= idx <= 8 else None
            if transition_item:
                state["current_item_id"]    = transition_item["item_id"]
                state["current_item_label"] = transition_item["label"]
                state["current_hypothesis"] = transition_item["text"]

        # If this is a follow-up, use nav_instruction to guide the question
        if state.get("followup_count", 0) > 0 and nav_instr:
            # Generate a follow-up question using the nav instruction as context
            current_item = ITEM_ID_TO_ITEM.get(state["current_item_id"], {})
            phq_key      = current_item.get("phq_key", "")
            domain_meaning = PHQ8_CLINICAL_CONTEXT.get(phq_key, state["current_item_label"])

            # Use QUESTION_AGENT_FOLLOWUP_SYSTEM template (Agent 1 — follow-up variant)
            question = (question_followup_template | llm | str_parser).invoke({
                "domain_meaning": domain_meaning,
                "nav_instruction": nav_instr,
                "last_answer": state.get("last_answer", ""),
            }).strip()
            selected_domain = state["current_item_id"]

        else:
            # PMI gain selection — pick best unasked domain
            visited   = list(set(
                [PHQ_KEY_TO_ITEM[k]["item_id"] for k in asked_keys if k in PHQ_KEY_TO_ITEM]
            ))
            remaining_keys = [k for k in ITEMS if k not in asked_keys]

            if not remaining_keys:
                # All asked — just ask a gentle wrap-up
                question        = "Is there anything else about how you've been feeling that you'd like to share?"
                selected_domain = state.get("current_item_id", "I1")
            else:
                if pmi_matrix is not None:
                    #choose the best question to ask based on the information gain
                    recs = {
                        cand: sum(
                            accumulated.get(s, 0) * pmi_matrix.loc[cand, s]
                            for s in accumulated
                        ) * (1 / max(accumulated.get(cand, 0.01), 0.01))
                        for cand in remaining_keys
                    }
                    top_candidates = sorted(recs, key=recs.get, reverse=True)[:3]

                    print(f"  [PMI Gains] {[(c, round(recs[c],4)) for c in top_candidates]}")
                    best_domain    = top_candidates[0]
                else:
                    top_candidates = remaining_keys[:3]
                    best_domain    = remaining_keys[0]

                conv_hist = state.get("conversation_history_dicts", [])
                last_answer = state.get("last_answer", "")

                question, sel = generate_dynamic_question(
                    best_domain=best_domain,
                    evidence=accumulated,
                    user_text=last_answer,
                    conversation_history=conv_hist,
                    candidate_domains=[best_domain],
                    nav_instruction=nav_instr,
                )

                # Force best_domain — ignore GPT's domain override
                phq_item = PHQ_KEY_TO_ITEM.get(best_domain)
                if phq_item:
                    state["current_item_id"]    = phq_item["item_id"]
                    state["current_item_label"] = phq_item["label"]
                    state["current_hypothesis"] = phq_item["text"]
                    for i, h in enumerate(PHQ8_HYPOTHESES):
                        if h["item_id"] == phq_item["item_id"]:
                            state["current_item_index"] = i + 1
                            break
                selected_domain = state["current_item_id"]

    print(f"\n👩‍⚕️ Psychologist ({state['current_item_id']}): {question}")

    new_hist = state["history"] + [f"Psychologist: {question}"]
    conv_hist_dicts = state.get("conversation_history_dicts", []) + [
        {"role": "bot", "content": question}
    ]
    turn = {
        "turn_index": len(state["transcript"]) + 1,
        "speaker":    AI_NAME,
        "text":       question,
        "role":       "question",
        "item_id":    state["current_item_id"],
    }

    return {
        "last_question":              question,
        "history":                    new_hist,
        "transcript":                 state["transcript"] + [turn],
        "current_item_id":            state["current_item_id"],
        "current_item_label":         state["current_item_label"],
        "current_hypothesis":         state["current_hypothesis"],
        "current_item_index":         state.get("current_item_index", 0),
        "conversation_history_dicts": conv_hist_dicts,
    }

# =============================================================================
# ── NODE 2: Participant Node
# Calls simulate_client_answer() to get the participant's reply.
# After receiving the answer:
#   1. Runs mirt_style_extract() to update accumulated_evidence for all 8 domains.
#   2. Runs propagate_correlated_evidence() to bleed evidence into correlated domains.
#   3. Marks the current PHQ key as asked in asked_phq_keys.
#   4. Appends the updated PMI gain rankings to pmi_gain_log for CSV export.
# Appends answer to history, transcript, and conversation_history_dicts.
# =============================================================================
def participant_node(state: AgentState):
    profile_obj     = json.loads(state["participant_profile"])
    is_followup_flag = state.get("followup_count", 0) > 0
    target_domain   = state.get("last_target_domain", None)
    current_rapport = state.get("rapport_score", 3)

    answer_text, diff_mode, diff_level, item_severity = simulate_client_answer(
        item_id        = state["current_item_id"],
        item_index     = state["current_item_index"],
        item_label     = state["current_item_label"],
        hypothesis_text = state["current_hypothesis"],
        question_text  = state["last_question"],
        client_profile = profile_obj,
        llm            = llm,
        str_parser     = str_parser,
        is_followup    = is_followup_flag,
        target_domain  = target_domain,
        current_rapport = current_rapport,
    )

    if diff_level == "level3":
        print(f"   [Simulation] 🎲 Level 3 (Hard) -> Injected: {diff_mode.upper()}")
    elif diff_level == "level2":
        print(f"   [Simulation] 🎲 Level 2 (Medium) -> Injected: {diff_mode.upper()}")

    print(f"👤 Participant: {answer_text}")

    new_hist = state["history"] + [f"Participant: {answer_text}"]
    conv_hist_dicts = state.get("conversation_history_dicts", []) + [
        {"role": "user", "content": answer_text}
    ]
    turn = {
        "turn_index":        len(state["transcript"]) + 1,
        "speaker":           PARTICIPANT_NAME,
        "text":              answer_text,
        "role":              "answer",
        "item_id":           state["current_item_id"],
        "resolution_marker": "POST_CONTRADICTION_CORRECTION"
                             if state.get("last_target_domain") == "contradiction" else None,
    }

    # ── MIRT update: extract evidence from answer
    mirt_fn      = state.get("_mirt_extract")
    accumulated  = dict(state.get("accumulated_evidence", {k: 0.0 for k in ITEMS}))
    corr_matrix  = state.get("_corr_matrix")

    if mirt_fn:
        new_evidence = mirt_fn(answer_text)
        for symptom in ITEMS:
            accumulated[symptom] = max(
                accumulated.get(symptom, 0),
                new_evidence.get(symptom, 0),
            )

    # ── Correlation propagation
    boosted = []
    current_phq_key = ITEM_ID_TO_PHQ_KEY.get(state["current_item_id"], "")
    if corr_matrix is not None and current_phq_key in ITEMS:
        boosted = propagate_correlated_evidence(current_phq_key, accumulated, corr_matrix)
        if boosted:
            boost_summary = ", ".join(
                f"{b['domain']} +{b['boost']} (r={b['correlation']})" for b in boosted
            )
            print(f"  [Correlation Boost → {boost_summary}]")

    # ── Track asked domains FIRST before PMI log
    asked_keys = list(state.get("asked_phq_keys", []))
    if current_phq_key and current_phq_key not in asked_keys:
        asked_keys.append(current_phq_key)
    # Also mark what GPT actually selected (current_item_id) as asked
    gpt_phq_key = ITEM_ID_TO_PHQ_KEY.get(state.get("current_item_id", ""), "")
    if gpt_phq_key and gpt_phq_key not in asked_keys:
        asked_keys.append(gpt_phq_key)

    # ── PMI Gain Log (logged after accumulated_evidence AND asked_keys are updated)
    pmi_gain_log   = list(state.get("pmi_gain_log", []))
    pmi_matrix_log = state.get("_pmi_matrix")
    remaining_log  = [k for k in ITEMS if k not in asked_keys]

    if pmi_matrix_log is not None and remaining_log:
        recs_log = {
            cand: sum(
                accumulated.get(s, 0) * pmi_matrix_log.loc[cand, s]
                for s in accumulated
            ) * (1 / max(accumulated.get(cand, 0.01), 0.01))
            for cand in remaining_log
        }
        top_log = sorted(recs_log, key=recs_log.get, reverse=True)[:3]
        pmi_gain_log.append({
            "turn_index":        len(state["transcript"]) + 1,
            "item_id":           state.get("current_item_id", ""),
            "selected_domain":   top_log[0] if top_log else "",
            "gain_1":            round(recs_log[top_log[0]], 4) if len(top_log) > 0 else 0,
            "candidate_2":       top_log[1] if len(top_log) > 1 else "",
            "gain_2":            round(recs_log[top_log[1]], 4) if len(top_log) > 1 else 0,
            "candidate_3":       top_log[2] if len(top_log) > 2 else "",
            "gain_3":            round(recs_log[top_log[2]], 4) if len(top_log) > 2 else 0,
            "remaining_count":   len(remaining_log),
            "unique_asked":      len(asked_keys),
            "total_turns_asked": len(state["transcript"]),
        })

    return {
        "last_answer":                answer_text,
        "history":                    new_hist,
        "transcript":                 state["transcript"] + [turn],
        "current_difficulty":         diff_mode,
        "current_level":              diff_level,
        "accumulated_evidence":       accumulated,
        "asked_phq_keys":             asked_keys,
        "conversation_history_dicts": conv_hist_dicts,
        "pmi_gain_log":               pmi_gain_log,
    }

# =============================================================================
# ── NODE 3: Clarification Node
# Skips for INTRO / CLOSING phases (returns COMPLETE immediately).
# For PHQ items: runs compute_nli_probs() against the current hypothesis.
#   p_entail >= 0.7  → COMPLETE (answer supports the hypothesis)
#   p_neutral >= 0.6 → use GPT to check timeframe specifically
#   p_contradict >= 0.7 → flag as potential contradiction
# GPT is called only for timeframe disambiguation, keeping API cost low.
# Returns: clarification_status, clarification_reason, clarification_missing_domains.
# =============================================================================
def clarification_node(state: AgentState):
    if state["current_item_id"] in ["INTRO", "CLOSING"]:
        return {
            "clarification_status":          "COMPLETE",
            "clarification_reason":          "Non-clinical phase.",
            "clarification_missing_domains": [],
        }

    current_id    = state["current_item_id"]
    relevant_turns = [t for t in state["transcript"] if t.get("item_id") == current_id]
    focused_hist  = "".join(f"{t['speaker']}: {t['text']}\n" for t in relevant_turns)

    # NLI-based: vagueness + relevance via probabilities
    probs = compute_nli_probs(state["last_answer"], state["current_hypothesis"])

    missing    = []
    cot_reason = ""

    if probs["p_entail"] >= ENTAIL_THRESHOLD:
        status     = "COMPLETE"
        cot_reason = f"P_support={round(probs['p_entail'],3)} >= 0.7 → VALID"
    elif probs["p_neutral"] >= NEUTRAL_THRESHOLD or probs["p_contradict"] >= CONTRADICT_THRESHOLD:
        status = "INCOMPLETE"
        if probs["p_neutral"] >= NEUTRAL_THRESHOLD:
            missing.append("vagueness")
        if probs["p_contradict"] >= CONTRADICT_THRESHOLD:
            missing.append("relevance")
        cot_reason = f"P_neutral={round(probs['p_neutral'],3)}, P_contradict={round(probs['p_contradict'],3)} → AMBIGUOUS"
    else:
        status     = "COMPLETE"
        cot_reason = "Discard — no strong signal"

    # GPT-based: timeframe only (NLI cannot detect recency)
    tf_res = (clarification_template | llm | json_parser).invoke({
        "question":           state["last_question"],
        "answer":             state["last_answer"],
        "history_str":        focused_hist,
        "current_item_label": state.get("current_item_label", "Unknown"),
        "followup_count":     state.get("followup_count", 0),
        "max_followups":      MAX_FOLLOWUPS,
    })
    tf_missing = tf_res.get("missing_domains", [])
    for domain in ["timeframe", "frequency_threshold"]:
        if domain in tf_missing and domain not in missing:
            missing.append(domain)
            status     = "INCOMPLETE"
            cot_reason += f" | GPT: {domain} missing"

    # Override to GIVE_UP if GPT said so
    if tf_res.get("status") == "GIVE_UP":
        status     = "GIVE_UP"
        cot_reason += " | GPT: GIVE_UP — max followups exhausted"

    freq_estimate = tf_res.get("frequency_estimate", "unknown")
    summary_r     = cot_reason

    return {
        "clarification_status":          status,
        "clarification_reason":          f"{cot_reason} | Summary: {summary_r} | Freq: {freq_estimate}",
        "clarification_missing_domains": missing,
    }

# =============================================================================
# ── NODE 4: Alignment Node (MAGMA NLI + YOUR .corr() — both run here)
# Two-layer consistency check:
#   Layer 1 — NLI: calls alignment_template (Agent 4) with relevant history turns.
#   Layer 2 — Statistical: calls check_corr_alignment() on accumulated_evidence.
# Both results are stored in state and forwarded to navigation_node together.
# For INTRO / CLOSING: returns CONSISTENT immediately.
# =============================================================================
def alignment_node(state: AgentState):
    # NLI-based alignment following State(ut) formula
    # Check against all overlapping history utterances
    history_turns = [t for t in state["transcript"] 
                 if t.get("role") == "answer" and t.get("text") != state["last_answer"]]

    nli_status = "CONSISTENT"
    nli_reason = "No contradictions found."
    focused_hist = "".join(         
        f"{t.get('speaker','')}: {t.get('text','')}\n"
        for t in history_turns
    )
    for prev_turn in history_turns:
        probs = compute_nli_probs(state["last_answer"], prev_turn["text"])
        if probs["p_contradict"] >= CONTRADICT_THRESHOLD:
            nli_status = "CONTRADICTING"
            nli_reason = (f"P_contradict={round(probs['p_contradict'],3)} >= 0.7 "
                      f"against: '{prev_turn['text'][:80]}'")
            break
    # --- 4a-2. GPT alignment (three-lens PCoT) — runs when NLI is inconclusive ---
    if nli_status != "CONTRADICTING" and focused_hist.strip():     # ← ADD FROM HERE
        try:
            raw_alignment = (alignment_template | llm | str_parser).invoke({
                "answer":              state["last_answer"],
                "history_str":         focused_hist,
                "current_item_label":  state.get("current_item_label", "Unknown"),
            })
            # Clean trailing commas before parsing
            import re
            cleaned = re.sub(r',\s*([}\]])', r'\1', raw_alignment.strip().replace("```json","").replace("```","").strip())
            gpt_res = json.loads(cleaned)
        except Exception:
            gpt_res = {"status": "CONSISTENT", "reason": "", "soft_followup_suggested": False}
        gpt_status = gpt_res.get("status", "CONSISTENT")
        gpt_reason = gpt_res.get("reason", "")
        gpt_soft   = gpt_res.get("soft_followup_suggested", False)
        if gpt_status == "CONTRADICTING":
            nli_status = "CONTRADICTING"
            nli_reason = f"GPT three-lens: {gpt_reason}"
        elif gpt_status == "UNCERTAIN":
            nli_status = "UNCERTAIN"
            nli_reason = f"GPT soft tension: {gpt_reason} | soft_followup={gpt_soft}"

    # --- 4b. YOUR correlation-based alignment (score-level, statistical) ---
    corr_matrix  = state.get("_corr_matrix")
    accumulated  = state.get("accumulated_evidence", {k: 0.0 for k in ITEMS})
    asked_keys   = state.get("asked_phq_keys", [])
    corr_flags   = []
    corr_misalign_asked = list(state.get("corr_misalign_asked", []))

    if corr_matrix is not None:
        misalignments = check_corr_alignment(accumulated, corr_matrix, asked_keys)
        for m in misalignments:
            pair_key = f"{m['domain_a']}_{m['domain_b']}"
            if pair_key not in corr_misalign_asked:
                corr_flags.append(m)
                print(
                    f"  [⚠️  Corr Alignment | {m['domain_a']} (score={m['score_a']}) ↔ "
                    f"{m['domain_b']} (score={m['score_b']}) | r={m['correlation']} | gap={m['gap']}]"
                )

    # --- 4c. Merge: either NLI or corr contradiction → overall CONTRADICTING ---
    corr_contradicting = len(corr_flags) > 0
    if nli_status == "CONTRADICTING" or corr_contradicting:
        merged_status = "CONTRADICTING"
        merged_reason = nli_reason
        if corr_contradicting:
            corr_detail   = "; ".join(
                f"Corr-gap: {f['domain_a']}↔{f['domain_b']} r={f['correlation']} gap={f['gap']}"
                for f in corr_flags
            )
            merged_reason += f" | CORR ALIGNMENT: {corr_detail}"
    elif nli_status == "UNCERTAIN":
        merged_status = "UNCERTAIN"
        merged_reason = nli_reason
    else:
        merged_status = nli_status
        merged_reason = nli_reason

    return {
        "alignment_status":     merged_status,
        "alignment_reason":     merged_reason,
        "corr_alignment_flags": corr_flags,
        "corr_misalign_asked":  corr_misalign_asked,
    }

# =============================================================================
# ── NODE 5: Navigation Node (MAGMA logic + PMI gain stopping condition)
# The decision hub of the interview loop. Receives all status signals and:
#   1. Checks if corr alignment flags introduce new contradiction issues.
#   2. Computes PMI gain for remaining domains → if best gain < GAIN_THRESHOLD
#      and no follow-up is pending, sets pmi_stop = True (enough info collected).
#   3. Calls navigation_template (Agent 5) for the primary action decision.
#   4. Applies hard overrides: GIVE_UP → NEXT_ITEM, max retries hit → NEXT_ITEM,
#      PMI stop + no forced follow-up → NEXT_ITEM.
#   5. Selects the correct escalation strategy from ESCALATION_MAP.
#   6. Tags each answer as TRUE_POSITIVE / FALSE_POSITIVE etc. for analytics.
#   7. Appends NLI evidence entry to items_evidence (supporting / contradicting / neutral).
# Returns: next_action, nav_instruction, analytics_records, agent_thoughts.
# =============================================================================
def navigation_node(state: AgentState):
    raw_missing_list = state.get("clarification_missing_domains", [])
    current_retries  = state.get("followup_count", 0)
    resolved         = list(state.get("resolved_domains", []))
    last_target      = state.get("last_target_domain", None)
    domain_attempts  = dict(state.get("domain_attempts", {}))

    # Update resolved list
    if last_target:
        domain_attempts[last_target] = domain_attempts.get(last_target, 0) + 1
        if state["clarification_status"] == "COMPLETE":
            if last_target not in resolved:
                resolved.append(last_target)
        elif domain_attempts[last_target] >= 2:
            if last_target not in resolved:
                resolved.append(last_target)
                print(f"   [Logic] ⚠️  Force-resolving '{last_target}' after {domain_attempts[last_target]} attempts.")

    missing_list = [d for d in raw_missing_list if d not in resolved]

    # Add contradiction if alignment fired
    alignment_status = state.get("alignment_status", "CONSISTENT")
    if alignment_status == "CONTRADICTING":
        if "contradiction" not in missing_list and "contradiction" not in resolved:
            missing_list.append("contradiction")

    # Also flag corr alignment domains if not yet asked
    corr_flags         = state.get("corr_alignment_flags", [])
    corr_misalign_asked = list(state.get("corr_misalign_asked", []))
    for cf in corr_flags:
        pair_key = f"{cf['domain_a']}_{cf['domain_b']}"
        if pair_key not in corr_misalign_asked and "contradiction" not in missing_list:
            missing_list.append("contradiction")
            corr_misalign_asked.append(pair_key)

    # PMI gain stopping check
    accumulated  = state.get("accumulated_evidence", {k: 0.0 for k in ITEMS})
    asked_keys   = state.get("asked_phq_keys", [])
    pmi_matrix   = state.get("_pmi_matrix")
    remaining_keys = [k for k in ITEMS if k not in asked_keys]

    pmi_stop = False
    if pmi_matrix is not None and remaining_keys:
        recs = {
            cand: sum(accumulated.get(s, 0) * pmi_matrix.loc[cand, s] for s in accumulated)
                  * (1 / max(accumulated.get(cand, 0.01), 0.01))
            for cand in remaining_keys
        }
        best_gain = max(recs.values()) if recs else 0
        if best_gain < GAIN_THRESHOLD and current_retries == 0:
            pmi_stop = True
            print(f"   [PMI] ✅ Gain={best_gain:.4f} < threshold. Sufficient info collected.")

    # Ask navigation agent
    corr_flags_str = "; ".join(
        f"{f['domain_a']}↔{f['domain_b']}(r={f['correlation']},gap={f['gap']})"
        for f in corr_flags
    ) or "None"

    res = (navigation_template | llm | json_parser).invoke({
        "c_stat":              state["clarification_status"],
        "c_reas":              state["clarification_reason"],
        "a_stat":              state.get("alignment_status", "UNKNOWN"),
        "a_reas":              state.get("alignment_reason", "None"),
        "corr_flags":          corr_flags_str,
        "current_item_label":  state.get("current_item_label", "Unknown"),
        "followup_count":      current_retries,
        "max_followups":       MAX_FOLLOWUPS,
        "rapport_score":       state.get("rapport_score", 3),
        "last_answer":         state.get("last_answer", ""),
    })

    proposed_action  = res.get("next_action", "NEXT_ITEM")
    base_instruction = res.get("instruction", "")
    nav_priority     = res.get("priority", "none")       # NEW
    nav_strategy     = res.get("strategy", "none")       # NEW

    # Decision logic
    selected_domain = None
    style_guide     = "Standard Follow-up"

    if pmi_stop and proposed_action != "FOLLOW_UP":
        final_action      = "NEXT_ITEM"
        final_instruction = "Move to next item (PMI gain threshold met)."
    elif state.get("clarification_status") == "GIVE_UP":
        final_action      = "NEXT_ITEM"
        final_instruction = "Move to next item (clarification gave up)."
        missing_list      = []
        selected_domain   = None
        new_followup_count = 0
    elif proposed_action == "FOLLOW_UP" and current_retries >= MAX_FOLLOWUPS:
        print(f"   [Logic] 🛑 MAX RETRIES ({current_retries}) HIT -> Forcing Next Item...")
        final_action      = "NEXT_ITEM"
        final_instruction = "Move to next item."
        missing_list      = []
        selected_domain   = None

    elif (proposed_action == "FOLLOW_UP" and missing_list) or ("contradiction" in missing_list):
        final_action  = "FOLLOW_UP"
        question_domain = PHQ8_QUESTION_MAPPING.get(state["current_item_label"], "INTERNAL")
        priority_order  = DOMAIN_PRIORITY_BY_TYPE[question_domain]
        selected_domain = next(
            (d for d in priority_order if d in missing_list),
            missing_list[0] if missing_list else None,
        )
        strategies  = ESCALATION_MAP.get(selected_domain, ["Ask specifically."])
        style_guide = strategies[min(current_retries, len(strategies) - 1)]
        reason_ctx  = (
            state.get("alignment_reason")
            if selected_domain == "contradiction"
            else state.get("clarification_reason")
        )
        final_instruction = (
            f"Address the {selected_domain.upper()} issue. "
            f"Context: {reason_ctx}. "
            f"{style_guide} "
            f"[strategy: {nav_strategy}]"
        )

    else:
        if proposed_action == "FOLLOW_UP" and not missing_list:
            final_action      = "NEXT_ITEM"
            final_instruction = "Proceed to next item."
        elif state["current_item_id"] == "CLOSING":
            final_action      = "NEXT_ITEM"
            final_instruction = "End interview."
            new_followup_count = 0
        else:
            final_action      = proposed_action
            final_instruction = base_instruction
            selected_domain   = None

    # Print status
    if final_action == "FOLLOW_UP" and selected_domain:
        print(f"   [Logic] ⚠️  ISSUE: {selected_domain.upper()} -> Strategy: {style_guide} (Attempt {current_retries + 1}/{MAX_FOLLOWUPS})...")
        print(f"           (Resolved so far: {resolved})")
        new_followup_count = current_retries + 1
    else:
        if state["current_item_id"] == "INTRO":
            print(f"   [Logic] 💬  Intro Dialogue -> Continuing...")
        elif state["current_item_id"] == "CLOSING":
            print(f"   [Logic] 🏁  CLOSING -> Ending Experiment...")
        else:
            print(f"   [Logic] ✅  COMPLETE -> Next Item...")
        new_followup_count = 0

    # NLI Entailment evidence tracking (MAGMA)
    items_data = dict(state["items_evidence"])
    if state["current_item_id"] not in ["INTRO", "CLOSING"]:
        probs    = compute_nli_probs(state["last_answer"], state["current_hypothesis"])
        role_tag = "neutral"
        if probs["p_entail"]     >= ENTAIL_THRESHOLD:    role_tag = "supporting"
        elif probs["p_contradict"] >= CONTRADICT_THRESHOLD: role_tag = "contradicting"

        item_key = f"Item {state['current_item_index']}"
        if item_key in items_data:
            current_count   = len(items_data[item_key][role_tag]) + 1
            evidence_id_key = f"evidence_{role_tag}_id"
            evidence_id_val = f"{state['current_item_id']}_{role_tag}_E{current_count}"
            entry = {
                evidence_id_key:  evidence_id_val,
                "text":           state["last_answer"],
                "p_entail":       round(probs["p_entail"], 4),
                "p_contradict":   round(probs["p_contradict"], 4),
                "p_neutral":      round(probs["p_neutral"], 4),
                "followup_asked": (final_action == "FOLLOW_UP"),
                "missing_domains": missing_list,
            }
            items_data[item_key][role_tag].append(entry)

    # Analytics record
    raw_mode      = state.get("current_difficulty", "none").lower()
    current_lvl   = state.get("current_level", "level1")
    injected_flaw = "none"
    if raw_mode not in ["none", "open", "resolution"]:
        injected_flaw = raw_mode.replace("+", ", ") if "+" in raw_mode else raw_mode

    detected_flaw = ", ".join(missing_list) if missing_list else "none"
    turn_label    = "Initial" if state["followup_count"] == 0 else f"FollowUp_{state['followup_count']}"

    if   injected_flaw == "none" and final_action == "NEXT_ITEM":   bot_caught = "TRUE_NEGATIVE"
    elif injected_flaw != "none" and final_action == "FOLLOW_UP":   bot_caught = "TRUE_POSITIVE"
    elif injected_flaw == "none" and final_action == "FOLLOW_UP":   bot_caught = "FALSE_POSITIVE"
    else:                                                            bot_caught = "FALSE_NEGATIVE"

    analytic_entry = {
        "PID":              "PENDING",
        "Item":             state["current_item_id"],
        "Turn":             turn_label,
        "Level":            current_lvl,
        "Rapport":          state.get("rapport_score", 3),
        "Injected_Flaw":    injected_flaw,
        "Detected_Flaw":    detected_flaw,
        "Agent_Decision":   final_action,
        "Bot_Caught_Flaw":  bot_caught,
        "Agent_Score":      -1,
        "Participant_Text": state["last_answer"].replace('"', "'"),
    }

    current_analytics = list(state.get("analytics_records", []))
    if state["current_item_id"] not in ["INTRO", "CLOSING"]:
        current_analytics.append(analytic_entry)

    thought = {
        "item":                      state["current_item_id"],
        "turn":                      turn_label,
        "clarification_status":      state["clarification_status"],
        "clarification_logic_chain": state["clarification_reason"],
        "alignment_status":          state["alignment_status"],
        "alignment_logic_chain":     state["alignment_reason"],
        "corr_alignment_flags":      corr_flags,
        "decision":                  final_action,
        "nav_priority":              nav_priority,    
        "nav_strategy":              nav_strategy,    
        "instruction":               final_instruction,
    }

    return {
        "next_action":          final_action,
        "nav_instruction":      final_instruction,
        "agent_thoughts":       state["agent_thoughts"] + [thought],
        "items_evidence":       items_data,
        "followup_count":       new_followup_count,
        "analytics_records":    current_analytics,
        "domain_attempts":      domain_attempts,
        "resolved_domains":     resolved,
        "last_target_domain":   selected_domain,
        "corr_misalign_asked":  corr_misalign_asked,
    }

# =============================================================================
# ── NODE 6: Transition Node (MAGMA — kept intact)
# Called after navigation_node decides NEXT_ITEM.
# Handles three cases:
#   INTRO:   loops up to 3 rapport-building turns, then transitions to item index 1.
#   CLOSING: sets index to 10 (triggers batch_scoring_node via check_end).
#   Normal items:
#     - Calculates rapport delta via calculate_rapport_delta() based on openness tier
#       and number of follow-ups needed for the completed item.
#     - Appends a symptom_summaries entry (vagueness / timeframe / relevance texts).
#     - Re-runs PMI gain on remaining unasked domains; skips any whose gain is below
#       GAIN_THRESHOLD (they are now redundant given collected evidence).
#     - Sets state to the next eligible item.
# =============================================================================
def calculate_rapport_delta(current_level, followup_count, p_type):
    if current_level == "level3":       return -2   # resistant → bigger drop
    elif current_level == "level2":
        if followup_count == 0:         return +1   # guarded but clean → small gain
        elif followup_count <= 2:       return  0   # needed prompting → neutral
        else:                           return -1   # struggled → drop
    else:  # level1
        if followup_count == 0:         return +2   # fully open, no prompting → fast climb
        elif followup_count <= 1:       return +1   # open, minor nudge → gain
        else:                           return  0   # open but needed help → neutral                         return -1

def transition_node(state: AgentState):
    current_id     = state["current_item_id"]
    current_idx    = state["current_item_index"]
    current_rapport = state.get("rapport_score", 3)

    # Rapport update
    if current_id == "INTRO":
        profile_obj = json.loads(state["participant_profile"])
        p_type      = classify_profile_type(profile_obj)
        new_rapport = min(3, current_rapport + 1) if p_type == "INTERNALIZER" else current_rapport
        if p_type == "INTERNALIZER":
            print(f"   [Rapport] 📈 Intro building trust with Internalizer ({new_rapport}/5)")
        else:
            print(f"   [Rapport] ↔️  Externalizer remains guarded during Intro ({new_rapport}/5)")
    elif current_id == "CLOSING":
        new_rapport = current_rapport
        p_type      = "INTERNALIZER"
    else:
        profile_obj   = json.loads(state["participant_profile"])
        p_type        = classify_profile_type(profile_obj)
        current_level = state.get("current_level", "level1")
        new_rapport   = current_rapport

    # INTRO phase loop
    if current_id == "INTRO":
        current_count = state.get("intro_turn_count", 0) + 1
        if current_count < 3:
            return {
                "current_item_index": 0,
                "intro_turn_count":   current_count,
                "nav_instruction":    "Ask a polite follow-up or general icebreaker.",
                "followup_count":     0,
                "rapport_score":      new_rapport,
                "symptom_summaries":  state.get("symptom_summaries", []),
                "resolved_domains":   [],
                "domain_attempts":    {},
                "last_target_domain": None,
            }
        else:
            next_item = PHQ8_HYPOTHESES[0]
            return {
                "current_item_index": 1,
                "current_item_id":    next_item["item_id"],
                "current_item_label": next_item["label"],
                "current_hypothesis": next_item["text"],
                "intro_turn_count":   current_count,
                "nav_instruction":    "Transition to clinical items.",
                "followup_count":     0,
                "rapport_score":      new_rapport,
                "symptom_summaries":  state.get("symptom_summaries", []),
                "resolved_domains":   [],
                "domain_attempts":    {},
                "last_target_domain": None,
            }

    # CLOSING
    if current_id == "CLOSING":
        return {
            "current_item_index": 10,
            "nav_instruction":    "End experiment.",
            "followup_count":     0,
            "domain_attempts":    {},
            "resolved_domains":   [],
            "last_target_domain": None,
            "analytics_records":  state.get("analytics_records", []),
            "symptom_summaries":  state.get("symptom_summaries", []),
        }

    # Normal items
    updated_analytics  = list(state.get("analytics_records", []))
    current_item_logs  = [r for r in updated_analytics if r["Item"] == current_id]
    followup_count_val = sum(1 for log in current_item_logs if log["Agent_Decision"] == "FOLLOW_UP")

    delta       = calculate_rapport_delta(current_level, followup_count_val, p_type)
    new_rapport = max(1, min(5, current_rapport + delta))

    if delta > 0:   print(f"   [Rapport] 📈 Trust Increased ({new_rapport}/5) - Genuine openness.")
    elif delta == 0: print(f"   [Rapport] ↔️  Trust Stable ({new_rapport}/5) - Guarded but cooperative.")
    else:            print(f"   [Rapport] 📉 Trust Decreased ({new_rapport}/5) - Resistant or struggled.")

    vagueness_texts  = [log["Participant_Text"] for log in current_item_logs if log["Detected_Flaw"] == "vagueness"]
    timeframe_texts  = [log["Participant_Text"] for log in current_item_logs if log["Detected_Flaw"] == "timeframe"]
    relevance_texts  = [log["Participant_Text"] for log in current_item_logs if log["Detected_Flaw"] == "relevance"]

    symptom_entry = {
        "PID":               "PENDING",
        "Item":              current_id,
        "Vagueness_Response":  " | ".join(vagueness_texts)  if vagueness_texts  else "None",
        "Timeframe_Response":  " | ".join(timeframe_texts)  if timeframe_texts  else "None",
        "Relevance_Response":  " | ".join(relevance_texts)  if relevance_texts  else "None",
        "Total_Followups":     followup_count_val,
    }
    current_summaries = list(state.get("symptom_summaries", []))
    current_summaries.append(symptom_entry)

    

    # Find next unasked domain using asked_phq_keys, not index position
    asked_keys  = list(state.get("asked_phq_keys", []))
    accumulated = state.get("accumulated_evidence", {k: 0.0 for k in ITEMS})
    pmi_matrix  = state.get("_pmi_matrix")
    remaining   = [h for h in PHQ8_HYPOTHESES if h["phq_key"] not in asked_keys]

    # ADD THIS:
    print(f"DEBUG transition asked_keys: {asked_keys}")
    print(f"DEBUG transition remaining: {[h['phq_key'] for h in remaining]}")

    if not remaining:
        return {
            "current_item_index": 9,
            "current_item_id":    "CLOSING",
            "current_item_label": "Closing",
            "current_hypothesis": "End the conversation politely.",
            "nav_instruction":    "Wrap up the interview.",
            "followup_count":     0,
            "rapport_score":      new_rapport,
            "symptom_summaries":  current_summaries,
            "resolved_domains":   [],
            "domain_attempts":    {},
            "last_target_domain": None,
        }

    if pmi_matrix is not None:
        eligible = []
        for h in remaining:
            gain = sum(
                accumulated.get(s, 0) * pmi_matrix.loc[h["phq_key"], s]
                for s in accumulated
            ) * (1 / max(accumulated.get(h["phq_key"], 0.01), 0.01))
            if gain >= GAIN_THRESHOLD:
                eligible.append(h)
            else:
                print(f"   [PMI Skip] ⏭️  {h['item_id']} ({h['label']}) — gain {round(gain,3)} < threshold, skipping.")
    else:
        eligible = remaining

    if not eligible:
        return {
            "current_item_index": 9,
            "current_item_id":    "CLOSING",
            "current_item_label": "Closing",
            "current_hypothesis": "End the conversation politely.",
            "nav_instruction":    "Wrap up the interview.",
            "followup_count":     0,
            "rapport_score":      new_rapport,
            "symptom_summaries":  current_summaries,
            "resolved_domains":   [],
            "domain_attempts":    {},
            "last_target_domain": None,
        }

    next_item = eligible[0]
    return {
        "current_item_index": PHQ8_HYPOTHESES.index(next_item) + 1,
        "current_item_id":    next_item["item_id"],
        "current_item_label": next_item["label"],
        "current_hypothesis": next_item["text"],
        "nav_instruction":    "Start next item.",
        "followup_count":     0,
        "rapport_score":      new_rapport,
        "resolved_domains":   [],
        "domain_attempts":    {},
        "symptom_summaries":  current_summaries,
    }

# =============================================================================
# ── NODE 7: Batch Scoring Node — hybrid transcript scorer (Agent 6)
# Called once after all items are complete.
# Converts the full transcript + accumulated_evidence into final PHQ-8 scores:
#   Primary path: compute_phq8_score_transcript() → Agent 6 (single GPT call).
#   Fallback:     compute_phq8_score_fallback()   → MIRT threshold mapping.
# Merges GPT's data_sufficiency rating with symptom_summaries flags
# (takes the more conservative of the two).
# Backfills analytics_records with the final per-item score.
# Returns: final_scores, scoring_explanations, analytics_records, clinical_note.
# =============================================================================
def batch_scoring_node(state: AgentState):
    print("\n⏳ Interview Complete. Starting Batch Scoring (Transcript-Based PCoT)...")

    final_scores         = []
    scoring_explanations = []
    updated_analytics    = list(state.get("analytics_records", []))
    accumulated_evidence = state.get("accumulated_evidence", {k: 0.0 for k in ITEMS})
    conv_hist_dicts = [
        {"role": t["role"], "content": t["text"]}
        for t in state.get("transcript", [])
    ]

    # ── Call Agent 6: SCORING_AGENT_SYSTEM (single transcript call, all 8 domains)
    score_map, reasons, confidence_map, sufficiency_map, reasoning_chains, clinical_note = \
        compute_phq8_score_transcript(conv_hist_dicts, accumulated_evidence)

    # ── Fallback to MIRT threshold if transcript scoring fails
    scoring_method = "Transcript-Based PCoT"
    if score_map is None:
        score_map        = compute_phq8_score_fallback(accumulated_evidence)
        reasons          = {s: "MIRT threshold fallback"  for s in ITEMS}
        confidence_map   = {s: "Low"                      for s in ITEMS}
        sufficiency_map  = {s: "LOW"                      for s in ITEMS}
        reasoning_chains = {s: {}                         for s in ITEMS}
        clinical_note    = {}                              # ← ADD THIS
        scoring_method   = "MIRT Threshold (fallback)"
        print("   ⚠️  Transcript scoring failed — using MIRT fallback.")

    print(f"   Scoring method: {scoring_method}")

    # ── Map PHQ keys → MAGMA item structure for output compatibility
    for item_def in PHQ8_HYPOTHESES:
        item_id    = item_def["item_id"]
        item_label = item_def["label"]
        phq_key    = item_def["phq_key"]

        score       = score_map.get(phq_key, 0)
        explanation = reasons.get(phq_key, "")
        confidence  = confidence_map.get(phq_key, "High")

        # Data sufficiency: use scorer's own assessment, cross-check with symptom summaries
        scorer_sufficiency = sufficiency_map.get(phq_key, "HIGH")
        symptoms = [s for s in state.get("symptom_summaries", []) if s["Item"] == item_id]
        if symptoms:
            s = symptoms[0]
            has_issues = any([
                s["Vagueness_Response"]  != "None",
                s["Timeframe_Response"]  != "None",
                s["Relevance_Response"]  != "None",
            ])
            # Take the more conservative of the two sufficiency ratings
            if has_issues and scorer_sufficiency == "HIGH":
                sufficiency = "MEDIUM"
            else:
                sufficiency = scorer_sufficiency
        else:
            sufficiency = scorer_sufficiency

        reasoning = reasoning_chains.get(phq_key, {})

        print(f"   ✅ Scored {item_id}: {score}/3 ({item_label}) "
              f"| Confidence: {confidence} | Sufficiency: {sufficiency}")

        # Backfill analytics with final score
        for record in updated_analytics:
            if record["Item"] == item_id:
                record["Agent_Score"] = score

        final_scores.append({
            "Item ID":     item_id,
            "Item Label":  item_label,
            "Score":       score,
            "Sufficiency": sufficiency,
        })
        scoring_explanations.append({
            "item_id":         item_id,
            "phq_key":         phq_key,
            "score":           score,
            "confidence":      confidence,
            "data_sufficiency": sufficiency,
            "explanation":     explanation,
            "scoring_method":  scoring_method,
            "reasoning_chain": reasoning,
        })

    return {
        "final_scores":          final_scores,
        "scoring_explanations":  scoring_explanations,
        "analytics_records":     updated_analytics,
        "clinical_note":         clinical_note,
    }


# =============================================================================
#  STEP 14 — GRAPH ASSEMBLY
# Assembles the LangGraph StateGraph with 7 nodes and two conditional routing edges:
#
#   Fixed edges (always run in order):
#     question → participant → clarification → alignment → navigation
#
#   Conditional edge 1 — check_nav (after navigation_node):
#     FOLLOW_UP + not CLOSING/INTRO → loop back to question_node
#     otherwise                     → transition_node
#
#   Conditional edge 2 — check_end (after transition_node):
#     current_item_index > 9 → batch_scoring_node → END
#     otherwise               → question_node (next item)
# =============================================================================
def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("question_node",      question_node)
    workflow.add_node("participant_node",   participant_node)
    workflow.add_node("clarification_node", clarification_node)
    workflow.add_node("alignment_node",     alignment_node)
    workflow.add_node("navigation_node",    navigation_node)
    workflow.add_node("transition_node",    transition_node)
    workflow.add_node("batch_scoring_node", batch_scoring_node)

    workflow.set_entry_point("question_node")
    workflow.add_edge("question_node",      "participant_node")
    workflow.add_edge("participant_node",   "clarification_node")
    workflow.add_edge("clarification_node", "alignment_node")
    workflow.add_edge("alignment_node",     "navigation_node")

    def check_nav(state):
        if (state.get("next_action") == "FOLLOW_UP"
                and state.get("current_item_id") not in ["CLOSING", "INTRO"]):
            return "question_node"
        return "transition_node"

    workflow.add_conditional_edges(
        "navigation_node", check_nav,
        {"question_node": "question_node", "transition_node": "transition_node"},
    )

    def check_end(state):
        if state["current_item_index"] > 9:
            return "batch_scoring_node"
        return "question_node"

    workflow.add_conditional_edges(
        "transition_node", check_end,
        {"question_node": "question_node", "batch_scoring_node": "batch_scoring_node"},
    )

    workflow.add_edge("batch_scoring_node", END)
    return workflow.compile()


# =============================================================================
#  STEP 15 — SEVERITY HELPER
# Maps a raw PHQ-8 total (0-24) to the standard clinical severity label.
# Called once at the end of main() to include the label in the Scores CSV.
# =============================================================================
def get_severity(score_value: int) -> str:
    if   score_value == 0:  return "No Depression"
    elif score_value <= 4:  return "Minimal Depression"
    elif score_value <= 9:  return "Mild Depression"
    elif score_value <= 14: return "Moderate Depression"
    elif score_value <= 19: return "Moderately Severe Depression"
    else:                   return "Severe Depression"


# =============================================================================
#  STEP 16 — MAIN (MAGMA output structure — kept exactly)
# ── STATE INITIALISATION ──
# All fields required by AgentState are set here before the graph runs.
# The three helper objects (_pmi_matrix, _corr_matrix, _mirt_extract) are
# injected directly into state so every node can access them without globals.
# accumulated_evidence starts at 0.0 for all 8 domains; it is updated
# incrementally by participant_node after every answer.
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=str, required=True)
    args = parser.parse_args()

    # Load profile
    try:
        with open(f"Clean_Dataset/profiles/{args.pid}_client_profile.json", "r", encoding="utf-8") as f:
            profile_data = json.load(f)
            profile_str  = json.dumps(profile_data, ensure_ascii=False)
    except Exception:
        print("Profile error: File not found.")
        return

    # Build PMI + Correlation matrices
    print("Loading PMI + Correlation matrices from dataset...")
    pmi_matrix, corr_matrix = build_matrices(DATASET_PATH)

    # Build MIRT extractor
    print("\nLoading sentence transformer model...")
    mirt_extract = build_mirt_extractor(pmi_matrix)
    print("Model ready.\n")

    # Initialize evidence structure (MAGMA format)
    items_init = {
        f"Item {i+1}": {
            "label":         h["label"],
            "item_id":       h["item_id"],
            "supporting":    [],
            "contradicting": [],
            "neutral":       [],
        }
        for i, h in enumerate(PHQ8_HYPOTHESES)
    }

    # Initial accumulated evidence (all zeros)
    accumulated_init = {k: 0.0 for k in ITEMS}

    # ── INITIAL STATE ──
    state = {
        # MAGMA fields
        "participant_profile":           profile_str,
        "history":                       [],
        "transcript":                    [],
        "current_item_index":            0,
        "current_item_id":               "INTRO",
        "current_item_label":            "Introduction",
        "current_hypothesis":            "Establish rapport.",
        "intro_turn_count":              0,
        "analytics_records":             [],
        "current_difficulty":            "level1",
        "current_level":                 "level1",
        "symptom_summaries":             [],
        "domain_attempts":               {},
        "resolved_domains":              [],
        "last_target_domain":            None,
        "items_evidence":                items_init,
        "final_scores":                  [],
        "scoring_explanations":          [],
        "agent_thoughts":                [],
        "clarification_missing_domains": [],
        "clarification_status":          "COMPLETE",
        "clarification_reason":          "",
        "alignment_status":              "CONSISTENT",
        "alignment_reason":              "",
        "nav_instruction":               "Start introduction.",
        "followup_count":                0,
        "rapport_score":                 4,
        "last_question":                 "",
        "last_answer":                   "",
        "next_action":                   "",
        "pmi_gain_log":                  [],

        # YOUR new fields
        "accumulated_evidence":          accumulated_init,
        "asked_phq_keys":                [],
        "pmi_order":                     [],
        "corr_misalign_asked":           [],
        "corr_alignment_flags":          [],
        "conversation_history_dicts":    [],

        # Injected helpers (accessed inside nodes via state)
        "_pmi_matrix":   pmi_matrix,
        "_corr_matrix":  corr_matrix,
        "_mirt_extract": mirt_extract,
    }

    print(f"\n🚀 Merged PMI+MIRT+MAGMA System (PID {args.pid}) Started...")
    app         = build_graph()
    final_state = app.invoke(state, {"recursion_limit": 500})

# =========================================================================
#  SAVE FILES — MAGMA output structure (kept exactly)
# ── OUTPUT FILES (MAGMA structure, 8 directories) ──
# A. Evidence_{pid}.json        — NLI-tagged supporting/contradicting/neutral turns per item
# B. Transcript_{pid}.jsonl     — full turn-by-turn conversation
# C. Thoughts_{pid}.jsonl       — agent decision log per turn (clarification + alignment + nav)
# D. Explanations_{pid}.json    — PCoT reasoning chains + confidence per domain
# E. Scores_{pid}.csv           — final PHQ-8 scores (0-3) + total + severity
# F. analysis_{pid}.csv         — analytics: injected flaws, detected flaws, bot accuracy
# G. Symptoms_{pid}.csv         — per-item clarification issue texts + total follow-ups
# H. PMI_Gains_{pid}.csv        — per-turn PMI gain ranking for top-3 candidate domains
# =========================================================================
    base_dir = "MAGMA-InformationGain"
    dirs = {
        "ev": os.path.join(base_dir, "Evidence"),
        "tr": os.path.join(base_dir, "Transcript"),
        "th": os.path.join(base_dir, "Agent_Thoughts"),
        "sc": os.path.join(base_dir, "Scores"),
        "ex": os.path.join(base_dir, "Scoring_Explanations"),
        "an": os.path.join(base_dir, "Analysis_Metrics"),
        "sy": os.path.join(base_dir, "Symptoms"),
        "gn": os.path.join(base_dir, "PMI_Gains")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # A. Evidence
    with open(os.path.join(dirs["ev"], f"Evidence_{args.pid}.json"), "w") as f:
        json.dump(final_state["items_evidence"], f, indent=2)

    # B. Transcript
    with open(os.path.join(dirs["tr"], f"Transcript_{args.pid}.jsonl"), "w") as f:
        for t in final_state["transcript"]:
            f.write(json.dumps(t) + "\n")

    # C. Agent Thoughts
    with open(os.path.join(dirs["th"], f"Thoughts_{args.pid}.jsonl"), "w") as f:
        for t in final_state["agent_thoughts"]:
            f.write(json.dumps(t) + "\n")

    # D. Scoring Explanations
    with open(os.path.join(dirs["ex"], f"Explanations_{args.pid}.json"), "w") as f:
        json.dump(final_state["scoring_explanations"], f, indent=2)

    # E. Scores CSV (MAGMA format)
    total_score   = sum(item["Score"] for item in final_state["final_scores"])
    severity_cat  = get_severity(total_score)
    csv_data      = final_state["final_scores"] + [
        {"Item ID": "TOTAL",     "Item Label": "PHQ-8 SUM",         "Score": total_score},
        {"Item ID": "DIAGNOSIS", "Item Label": "Severity Category",  "Score": severity_cat},
    ]
    csv_path = os.path.join(dirs["sc"], f"Scores_{args.pid}.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["Item ID", "Item Label", "Score", "Sufficiency"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_data)

    # F. Analytics CSV (MAGMA format)
    analytics_path = os.path.join(dirs["an"], f"analysis_{args.pid}.csv")
    records        = final_state.get("analytics_records", [])
    for r in records:
        r["PID"] = args.pid
    if records:
        keys = ["PID", "Item", "Turn", "Level", "Rapport", "Injected_Flaw",
                "Detected_Flaw", "Agent_Decision", "Bot_Caught_Flaw", "Agent_Score", "Participant_Text"]
        with open(analytics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(records)

    # G. Symptoms Summary CSV (MAGMA format)
    symptoms_path = os.path.join(dirs["sy"], f"Symptoms_{args.pid}.csv")
    sym_records   = final_state.get("symptom_summaries", [])
    for r in sym_records:
        r["PID"] = args.pid
    if sym_records:
        keys = ["PID", "Item", "Vagueness_Response", "Timeframe_Response",
                "Relevance_Response", "Total_Followups"]
        with open(symptoms_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(sym_records)

    # H. PMI Gain Log CSV
    gain_path = os.path.join(dirs["gn"], f"PMI_Gains_{args.pid}.csv")
    gain_records = final_state.get("pmi_gain_log", [])
    print(f"DEBUG: pmi_gain_log has {len(gain_records)} entries")
    for r in gain_records:
        r["PID"] = args.pid
    if gain_records:
        keys = ["PID", "turn_index", "item_id", "selected_domain", "gain_1",
            "candidate_2", "gain_2", "candidate_3", "gain_3", "remaining_count","unique_asked", "total_turns_asked"]
        with open(gain_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(gain_records)

    print(f"\n✅ Done. Total PHQ-8 Score: {total_score} ({severity_cat}).")
    print(f"✅ All files saved to: {os.path.abspath(base_dir)}")


if __name__ == "__main__":
    main()
