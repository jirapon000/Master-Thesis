# python3 "Multiple-Agent(MAGMA).py" --pid (participant_id)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import datetime
import random
import csv
import torch
import torch.nn.functional as F
from typing import TypedDict, Annotated, List, Dict, Any, Union, Literal
from dotenv import load_dotenv

# --- LANGCHAIN & LANGGRAPH IMPORTS ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

# ================= CONFIG =================
AI_NAME = "Multi-Agent System (5-Agent Architecture)"
PARTICIPANT_NAME = "Participant"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.7
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Entailment Config (MATCHING BASELINE)
ENTAILMENT_MODEL_NAME = "roberta-large-mnli"
ENTAIL_THRESHOLD = 0.7
CONTRADICT_THRESHOLD = 0.7
NEUTRAL_THRESHOLD = 0.6

# ================= GLOBAL DATA =================
PHQ8_HYPOTHESES = [
    {"item_id": "I1", "label": "Anhedonia",       "text": "Little interest or pleasure in doing things."},
    {"item_id": "I2", "label": "Depressed mood",  "text": "Feeling depressed, down, or hopeless."},
    {"item_id": "I3", "label": "Sleep problems",  "text": "Trouble falling or staying asleep, or sleeping too much."},
    {"item_id": "I4", "label": "Fatigue",         "text": "Feeling tired or having little energy."},
    {"item_id": "I5", "label": "Appetite change", "text": "Poor appetite or overeating."},
    {"item_id": "I6", "label": "Self-worth",      "text": "Feeling bad about yourself or that you are a failure."},
    {"item_id": "I7", "label": "Concentration",   "text": "Trouble concentrating on things."},
    {"item_id": "I8", "label": "Psychomotor",     "text": "Moving or speaking so slowly (or being fidgety/restless)."}
]

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

# ================= ENTAILMENT MODEL SETUP =================
print(f"Loading entailment model: {ENTAILMENT_MODEL_NAME} ...")
ENT_TOKENIZER = AutoTokenizer.from_pretrained(ENTAILMENT_MODEL_NAME)
ENT_MODEL = AutoModelForSequenceClassification.from_pretrained(ENTAILMENT_MODEL_NAME)
ENT_MODEL.eval()

def compute_nli_probs(premise: str, hypothesis: str) -> Dict[str, float]:
    premise = (premise or "").strip()
    if not premise: return {"p_contradict": 0.0, "p_neutral": 0.0, "p_entail": 0.0}
    inputs = ENT_TOKENIZER(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        logits = ENT_MODEL(**inputs).logits
    probs = F.softmax(logits, dim=-1)[0].tolist()
    return {"p_contradict": probs[0], "p_neutral": probs[1], "p_entail": probs[2]}

# ================= HELPERS =================
class AlignmentHelper:
    def __init__(self, json_path):
        self.map = {}
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            self.map = {item['item_id']: item for item in data}
        else:
            print(f"Warning: {json_path} not found. Alignment will be generic.")

    def get_focused_history(self, current_item_id_str, evidence_data):
        try:
            curr_id = int(current_item_id_str.replace("I", ""))
        except:
            return "No alignment context available."

        if curr_id not in self.map:
            return "No specific alignment rules for this item."

        specs = self.map[curr_id]
        rule = specs['rule']
        related_ids = specs['check_against']

        relevant_answers = []
        for rid in related_ids:
            key = f"Item {rid}"
            if key in evidence_data:
                found_text = None
                for cat in ["supporting", "contradicting", "neutral"]:
                    if evidence_data[key][cat]:
                        found_text = evidence_data[key][cat][-1]["text"]
                        break
                if found_text:
                    relevant_answers.append(f"Item {rid}: {found_text}")

        history_text = "\n".join(relevant_answers) if relevant_answers else "No relevant previous answers yet."
        return f"--- ALIGNMENT RULE ---\n{rule}\n\n--- RELEVANT HISTORY ---\n{history_text}"

alignment_helper = AlignmentHelper("phq8_alignment_map.json")

# 2. Participant Simulator Response
def simulate_client_answer(
    item_id: str,
    item_index: int,
    item_label: str,
    hypothesis_text: str,
    question_text: str,
    client_profile: Dict[str, Any],
    llm: ChatOpenAI,
    is_followup: bool = False,
) -> tuple[str, str]:
    """
    Generate a simulated client answer with DYNAMIC difficulty.
    It randomly selects a difficulty level based on defined weights.
    """
    # --- 1. DYNAMIC DIFFICULTY SELECTION ---
    DIFFICULTY_MODES = ["level1", "level2", "level3"]
    # Weights: 20% Direct, 50% Paraphrase, 30% Ambiguous
    WEIGHTS = [0.2, 0.5, 0.3] 
    
    selected_tier = random.choices(DIFFICULTY_MODES, weights=WEIGHTS, k=1)[0]

    # --- 2. DEFINE INSTRUCTIONS & MODE LABEL ---
    if selected_tier == "level3":
        # LEVEL 3: FLAWS (Randomly pick WHICH flaw to inject)
        flaw_types = ["vagueness", "timeframe", "relevance"]
        specific_flaw = random.choice(flaw_types)
        
        mode_label = specific_flaw # This becomes the 'Injected_Flaw'

        if item_id in ["INTRO", "CLOSING"]:
            selected_tier = "level1"
        
        if specific_flaw == "vagueness":
            diff_instruction = (
                "**Goal: Be Vague.**\n"
                "- Use non-committal words like 'sometimes', 'maybe', 'sort of'.\n"
                "- Do NOT specify if it happens 1 day or 7 days a week.\n"
                "- Make it impossible to decide between Score 1 and Score 2."
            )
        elif specific_flaw == "timeframe":
            diff_instruction = (
                "**Goal: Be Unclear about Time.**\n"
                "- Talk about how you felt 'years ago' or 'in the past'.\n"
                "- Do NOT confirm if this is happening *currently* (in the last 2 weeks).\n"
                "- Use phrases like 'I used to feel...' or 'Back when I was working...'"
            )
        elif specific_flaw == "relevance":
            diff_instruction = (
                "**Goal: Go Off-Topic (Irrelevant).**\n"
                "- Ignore the specific symptom asked.\n"
                "- Talk about a tangential topic (e.g., your dog, the traffic, politics).\n"
                "- Mention the keyword but in the wrong context."
            )

    elif selected_tier == "level2": 
        # LEVEL 2: PARAPHRASE (Complex but Valid)
        mode_label = "level2" # Injected_Flaw will be mapped to 'none' later
        diff_instruction = (
            "**Goal: Natural & Metaphorical.**\n"
            "- Do NOT use clinical terms.\n"
            "- Use metaphors (e.g., 'I feel like a heavy blanket is on me').\n"
            "- The answer MUST be valid and answer the question, just not directly."
        )
        
    else: 
        # LEVEL 1: DIRECT (Clear)
        mode_label = "level1"
        diff_instruction = (
            "**Goal: Clear & Direct.**\n"
            "- Be helpful and explicit.\n"
            "- Directly answer the question with clear frequency/duration."

        )

    # --- 3. EXTRACT PROFILE & STYLE (Restored) ---
    persona = client_profile.get("persona", {})
    interaction_style = persona.get("interaction_style", {})
    clinical_signals = (client_profile.get("clinical_signals", {}) or {}).get("symptoms", {})
    behavioral_features = client_profile.get("behavioral_features", {}) 

    style_label = interaction_style.get("style_label", "Neutral")
    style_scores = interaction_style.get("scores", {})
    style_features = interaction_style.get("features", {})
    style_evidence = interaction_style.get("evidence_quotes", []) or []

    # --- 4. CONSTRUCT PROMPT ---
    if item_id == "INTRO":
        symptom_key = "general_wellbeing"
        symptom_block = {}
        special_instruction = (
            "This is the introduction. Ignore clinical symptoms. "
            "Respond naturally. "
            f"{diff_instruction}" 
        )
    else:
        symptom_key = SYMPTOM_KEY_BY_ITEM_ID.get(item_id, "")
        symptom_block = clinical_signals.get(symptom_key, {}) if clinical_signals else {}
        
        present = symptom_block.get("present", "uncertain")
        severity_hint = symptom_block.get("severity_hint", "uncertain")
        symptom_quotes = symptom_block.get("evidence_quotes", []) or []
        
        special_instruction = (
            f"Ground your response in these facts:\n"
            f"  - Symptom Status: {present}\n"
            f"  - Severity: {severity_hint}\n"
            f"  - {diff_instruction}" 
        )

    profile_snippet = {
        "persona": {
            "demographics": persona.get("demographics", {}),
            "interaction_style": {
                "label": style_label,
                "features": style_features,
                "samples": style_evidence[:3],
            },
        },
        "clinical_signals": {
            "symptoms": {
                symptom_key: symptom_block
            }
        },
        "behavioral_features": behavioral_features,
    }

    role_tag = "follow-up" if is_followup else "initial"

    prompt = f"""
        You ARE the participant described below.
        **Profile:** {json.dumps(profile_snippet, ensure_ascii=False, indent=2)}
        **Question:** "{question_text}"

        **Instructions:**
        1. Speak as this person (First person "I").
        2. {special_instruction}
        3. Keep it to 1-2 sentences.

        Reply exactly as the participant:
        """
    try:
        # Print mode for debugging
        print(f"   [Simulation] Mode: {mode_label.upper()}") 
        resp = llm.invoke(prompt)
        text = (getattr(resp, "content", "") or "").strip()
        return (text if text else "...", mode_label)
    except Exception as e:
        print(f"Simulation Error: {e}")
        return ("I'm not sure.", "level1")

# ================= STATE DEFINITION =================
class AgentState(TypedDict):
    # Context
    participant_profile: str
    history: List[str]
    transcript: List[Dict]
    
    # Flow
    current_item_index: int
    current_item_id: str
    current_item_label: str
    current_hypothesis: str
    
    # Agent Outputs
    last_question: str
    last_answer: str
    
    # Reports
    clarification_status: str
    clarification_reason: str
    alignment_status: str
    alignment_reason: str
    
    # Navigation & Control
    next_action: str
    nav_instruction: str
    clarification_missing_domains: List[str]
    
    # Data Collection
    items_evidence: Dict[str, Any]
    final_scores: List[Dict]
    scoring_explanations: List[Dict]
    agent_thoughts: List[Dict]

    # NEW FIELD: Track how many times we've looped on the current item
    followup_count: int
    intro_turn_count: int

    # --- NEW FIELDS FOR ANALYTICS ---
    analytics_records: List[Dict] # Stores the rows for the CSV
    current_difficulty: str       # Tracks "level1", "level3" etc. for the current turn

    # --- Symotoms Analysis ---
    symptom_summaries: List[Dict]

# ================= PROMPTS (5 AGENTS) =================

# 1. Question Agent
question_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert **Clinical Psychologist** conducting a structured diagnostic interview.
Your objective is to assess the patient's condition through natural, supportive dialogue rather than robotic survey questions.

**OPERATIONAL CONSTRAINTS:**
1.  **Single-Domain Focus:** Address strictly one information gap at a time.
    * *Directive:* If the instruction requests "Timeframe," ask *only* about Timeframe.
    * *Rationale:* Multi-part ("double-barreled") questions increase cognitive load and reduce data precision.
2.  **Conciseness & Brevity:** Limit all responses to a maximum of 1-2 sentences.
3.  **Naturalistic Inquiry:** Do not ask the patient to "rate" items or "select from a list" unless explicitly required. Maintain a conversational flow.

**CURRENT OPERATIONAL CONTEXT:**
* **Assessment Topic:** {item_label}
* **Clinical Definition:** {hypothesis}
* **PRIMARY DIRECTIVE:** "{instruction}"
    *(You must execute this specific instruction with high precision.)*

**REFERENCE FEW-SHOT EXAMPLES (Instruction -> Execution):**

**Scenario A: Initiating a New Topic**
* *Instruction:* "Start this item."
* *Response (Effective):* "Lately, have you found yourself tossing and turning at night, or perhaps waking up too early?"

**Scenario B: Clarifying Timeframe (TimeFrame)**
* *Instruction:* "Ask a single follow-up to clarify ONLY the TIMEFRAME."
* *Response (Ineffective):* "How long has this been going on, and does it happen every day?"
* *Response (Effective):* "That sounds difficult to manage. Has this been going on for the last two weeks, or is it a more recent development?"

**Scenario C: Clarifying Frequency (Vagueness)**
* *Instruction:* "Ask a single follow-up to clarify ONLY the VAGUENESS."
* *Response (Ineffective):* "Please rate your frequency on a scale of 0 to 3."
* *Response (Effective):* "I'm sorry to hear that. Would you say this happens nearly every night, or just a few times a week?"

**Scenario D: Handling Relevance (The Gentle Pivot)**
* *Instruction:* "Ask a single follow-up to clarify ONLY the RELEVANCE."
* *Response (Ineffective):* "That is irrelevant. Please answer the question about sleep."
* *Response (Effective):* "I appreciate you sharing that context about your work. Just to bring it back to your sleep specifically—have you been able to get good rest lately?"

**INTERACTION GUIDELINES:**
1.  **Empathy Markers:** Use validating phrases ("I appreciate you sharing that," "That sounds difficult") to build rapport, but do not be overly effusive.
2.  **Accessible Language:** Avoid clinical jargon (e.g., "psychomotor agitation"). Use lay terms like "restless" or "fidgety."
3.  **Neutral Inquiry:** Avoid leading questions that suggest a specific answer. Ask *how* they feel, rather than *if* they feel a certain way.
    * *Good:* "How have your energy levels been?"
    * *Bad:* "You must be feeling really tired, right?"
4.  **Transitional Logic:** When shifting topics, use a brief bridging phrase so the conversation feels connected rather than abrupt.
    * *Example:* "Moving on from sleep, I’d like to ask about your appetite..."

**YOUR TASK:**
Based on the instruction **"{instruction}"**, write a short, 1-2 sentence **conversational probe**.
"""),
    ("human", "Conversation History:\n{history_str}")
])

# 2. Clarification Agent (Detection)
clarification_template = ChatPromptTemplate.from_messages([
    ("system", """You are an **Expert Clinical Evaluator** acting as Quality Control for a clinical dataset.
    
**YOUR GOAL:**
Your goal is to ensure the participant's answer is **precise enough** to assign a valid, clinically scorable PHQ-8 rating (0-3).
*Constraint:* Do NOT look for specific keywords. Use your clinical judgment to determine if the meaning is clear.

**THE "SCORABILITY" TEST:**
Ask yourself: *"If I had to assign a specific number (0, 1, 2, or 3) right now based ONLY on this text, would I be guessing?"*
- If yes (guessing) -> **FAIL** (Mark as INCOMPLETE).
- If no (confident) -> **PASS** (Mark as COMPLETE).

**CRITERIA GUIDELINES (Conceptual Checks):**

**1. TIMEFRAME (The "Recency" Check)**
* **Definition:** Does the participant imply this is a **current or recent** experience (relevant to the last ~2 weeks)?
* **PASS:** Context suggests feelings are active now or happened recently.
  - *Examples of Logic:* Specific triggers ("Since I lost my job last week"), present continuous tense ("I am struggling"), or definitive markers ("Lately," "These days").
* **FAIL:** Explicitly refers to the distant past or resolved issues.
  - *Examples of Logic:* Historical statements ("That happened years ago"), resolved issues ("I'm better now"), or future conditionals ("I might feel that way if X happens").

**2. VAGUENESS (The "Scoring Discrimination" Check)**
* **Definition:** Is there enough precision to distinguish between **"Several days" (Score 1)** and **"More than half the days" (Score 2)**?
* **PASS (Scorable):** Describes a frequency or intensity that lands clearly in one bucket.
  - *Examples of Logic:* Clear counts ("3 days a week"), definitive states ("It never stops"), or clear comparisons ("Most of the time").
* **FAIL (Ambiguous):** The answer straddles the line between categories, forcing a guess.
  - *Examples of Logic:* Non-committal words ("Sometimes," "It varies," "Off and on") that could theoretically mean 2 days OR 5 days.

**3. RELEVANCE (The "Topic" Check)**
* **Definition:** Does the answer logically address the specific symptom asked about?
* **PASS:** Connects meaningfully to the symptom (even if indirectly).
* **FAIL:** Non-sequitur, deflection, or complete avoidance.
  - *Examples of Logic:* Deflections ("I don't want to talk about that"), Tangents ("My cat is cute though"), or Externalizing ("The economy is bad" - avoids personal feelings).

**OUTPUT FORMAT:**
Return strict JSON using **DOUBLE QUOTES**.
Include a `reason` explaining your decision based on the definitions above.
Return a list of `missing_domains` for any FAILED criteria: `["timeframe", "vagueness", "relevance"]`.
"""),
    ("human", "Conversation History:\n{history_str}\n\nLatest Q: {question}\nLatest A: {answer}")
])

# 3. Alignment Agent
alignment_template = ChatPromptTemplate.from_messages([
    ("system", """You are the **Alignment Agent** (Consistency Checker).
Your role is to validate cross-response consistency within a clinical interview.

**YOUR TASK:**
Determine if the patient's **Current Answer** logically contradicts their **Previous Answers**, based strictly on the provided **Alignment Rule**.

**INPUT CONTEXT:**
1.  **Alignment Rule:** A specific clinical logic check (e.g., "If patient reports Insomnia, they usually report Fatigue").
2.  **Relevant Past Answers:** Direct quotes retrieved from previous turns in the conversation.

**LOGIC DEFINITIONS (The Standard):**
* **CONSISTENT:** The current answer aligns with previous statements, or there is no relevant past data to check against.
* **CONTRADICTING:** The current answer is logically impossible or highly unlikely given their previous answers (e.g., "I sleep 12 hours a day" vs "I never sleep").
* **UNCERTAIN:** The answers seem different but might not be a hard contradiction (e.g., minor mood fluctuations).
     
**REFERENCE CASE STUDIES (Logic Types):**

**Type 1: Physical Causality (e.g. Sleep -> Fatigue, Appetite -> Energy)**
* *Rule:* "Severe sleep disturbance often leads to fatigue or concentration issues."
* *Past History:* (Item 3) "I stare at the ceiling all night. I get maybe 2 hours of sleep."
* *Current Answer:* (Item 4) "I have endless energy. I'm buzzing and running around all day."
* *Verdict:* **CONTRADICTING**
* *Reason:* "Physiological Mismatch: Severe sleep deprivation is physically incompatible with 'endless buzzing energy' (suggests Mania or inconsistency)."

**Type 2: Emotional Coherence (e.g. Depression -> Anhedonia/Self-Esteem)**
* *Rule:* "Deep depression typically correlates with loss of interest (Anhedonia)."
* *Past History:* (Item 2) "I feel completely hopeless and cry every day."
* *Current Answer:* (Item 1) "Oh, I'm having a blast! I go to parties, watch movies, I love everything right now."
* *Verdict:* **CONTRADICTING**
* *Reason:* "Emotional Mismatch: Claims of 'complete hopelessness' contradict the high pleasure/engagement reported here."

**Type 3: Behavioral/Cognitive Alignment (e.g. Concentration -> Psychomotor)**
* *Rule:* "Psychomotor agitation (restlessness) often makes concentration difficult."
* *Past History:* (Item 8) "I can't sit still. I have to pace the room constantly."
* *Current Answer:* (Item 7) "My focus is perfect. I just read a 300-page book in one sitting."
* *Verdict:* **CONTRADICTING**
* *Reason:* "Behavioral Mismatch: Extreme physical restlessness (pacing) logically conflicts with the ability to sit still and focus on a long book."

**Type 4: Consistent (Expected Correlation)**
* *Rule:* "Fatigue often lowers concentration."
* *Past History:* (Item 4) "I'm always exhausted."
* *Current Answer:* (Item 7) "Yeah, it's hard to focus on TV shows because I drift off."
* *Verdict:* **CONSISTENT**
* *Reason:* "Drifting focus is a logical consequence of the previously reported exhaustion."
     
**OUTPUT FORMAT:**
Return strict JSON using **DOUBLE QUOTES**.
Example: {{ "status": "CONTRADICTING", "reason": "Patient's claim of perfect focus contradicts earlier report of severe agitation." }}
"""),
    ("human", """**CURRENT ANSWER:**
"{answer}"

**LOGIC CHECK CONTEXT:**
{history_str}

**VERDICT:**""")
])

# 4. Navigation Agent
navigation_template = ChatPromptTemplate.from_messages([
    ("system", """You are the **Interview Flow Manager**.

**YOUR JOB:**
Review the status reports and decide the next step in the clinical workflow.

**INPUTS:**
1. **Clarification Report:** Checks if the answer is vague, missing a timeframe, or irrelevant.
2. **Alignment Report:** Checks if the answer contradicts previous statements.

**DECISION LOGIC (Strict Order):**

**1. FOLLOW_UP**
   *(Trigger if **ANY** of these flags are true)*
   - **Clarification Issue:** Status is "INCOMPLETE" or "AMBIGUOUS".
   - **Alignment Issue:** Status is "CONTRADICTING" (Logical/Physiological mismatch).
   - **Vagueness:** The patient used non-committal words ("sometimes", "maybe").
   - **Missing Data:** We do not know the Frequency (Days) or Duration.

**2. NEXT_ITEM**
   *(Trigger if **ALL** of these conditions are met)*
   - **Data Quality:** The answer is clear and scorable.
   - **Consistency:** The story is consistent (or "Uncertain" but not contradicting).
   - **Completeness:** No further clarification is needed.

**OUTPUT FORMAT:**
Return strict JSON using **DOUBLE QUOTES**.
Example: {{ "next_action": "NEXT_ITEM", "instruction": "Proceed to next item." }}
"""),
    ("human", """**STATUS REPORT:**
- Clarification Status: {c_stat}
- Clarification Reason: {c_reas}
- Alignment Status: {a_stat}
- Alignment Reason: {a_reas}

**DECISION:**""")
])

# 5. Scoring Agent
scoring_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert clinician **scoring** the PHQ-8 assessment.

**YOUR TASK:**
Analyze the patient's complete history for the specific item below and assign a valid clinical score (0-3).

**SCORING RUBRIC (PHQ-8 Standard):**
* **0 = Not at all** (No presence)
* **1 = Several days** (Less than half the week)
* **2 = More than half the days** (Significant portion of the week)
* **3 = Nearly every day** (Constant or daily occurrence)

**EVALUATION PROTOCOL:**
1.  **Strict Criteria Application:** Apply the specific definition provided in the prompt (Hypothesis) without deviation.
2.  **Contextual Analysis:** Review the full dialogue history, including behavioral notes and logic checks.
    * *Handling Vagueness:* If behavioral notes indicate resistance or ambiguity, treat initial answers with caution.
    * *Handling Contradictions:* If the patient contradicted themselves, prioritize the **final clarified answer** over earlier statements.

**OUTPUT FORMAT:**
Return strict JSON using **DOUBLE QUOTES**.
Example:
{{
    "score": 2,
    "confidence": "High",
    "explanation": "Patient initially used metaphors but eventually confirmed 'more than half the days'. Behavioral notes show vagueness was resolved."
}}
"""),
    ("human", """**ITEM TO ASSESS:** {item_label}

**CRITERIA (HYPOTHESIS):**
{hypothesis}

**PATIENT HISTORY & BEHAVIORAL CONTEXT:**
{history_str}

**ASSIGN SCORE:**""")
])


# ================= NODE FUNCTIONS =================
llm = ChatOpenAI(model=LLM_MODEL, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)
json_parser = JsonOutputParser()
str_parser = StrOutputParser()

# 1. Question Node (UPDATED: "Getting to Know You" Intro)
def question_node(state: AgentState):
    idx = state["current_item_index"]
    
    # CASE A: INTRODUCTION (Item 0)
    if idx == 0:
        state["current_item_id"] = "INTRO"
        state["current_item_label"] = "Introduction"
        state["current_hypothesis"] = "Establish rapport."
        
        # LOGIC: Check if it's the very first turn or a follow-up intro turn
        current_instr = state.get("nav_instruction", "")
        
        if current_instr == "Start introduction.":
            # --- UPDATED INSTRUCTION ---
            instruction = (
                "Start the conversation warmly and casually, like you are getting to know a new acquaintance for the first time. "
                "Do not mention you are an AI. "
                "Ask a friendly, open-ended icebreaker to build comfort, such as 'Hi! It is nice to meet you. How has your week been going so far?'"
            )
        else:
            # This handles the 2nd and 3rd intro questions (e.g., "Ask about their job")
            instruction = current_instr
        
    # CASE B: CLOSING (Item 9) <--- NEW BLOCK
    elif idx == 9:
        state["current_item_id"] = "CLOSING"
        state["current_item_label"] = "Closing"
        state["current_hypothesis"] = "End the interview."
        instruction = (
            "The assessment is complete. "
            "Thank the participant warmly for their openness and time. "
            "Wish them a good rest of the day. Do NOT ask any new questions."
        )
    
    # CASE C: PHQ-8 ITEMS (Items 1-8)
    else:
        list_idx = idx - 1
        if list_idx < len(PHQ8_HYPOTHESES):
            current_item = PHQ8_HYPOTHESES[list_idx]
            state["current_item_id"] = current_item["item_id"]
            state["current_item_label"] = current_item["label"]
            state["current_hypothesis"] = current_item["text"]
            
        instruction = state.get("nav_instruction", "Start item.")

    hist_str = "\n".join(state["history"][-6:])
    
    question = (question_template | llm | str_parser).invoke({
        "item_label": state["current_item_label"],
        "hypothesis": state["current_hypothesis"],
        "instruction": instruction,
        "history_str": hist_str
    })
    
    print(f"\n👩‍⚕️ Psychologist ({state['current_item_id']}): {question}")
    
    new_hist = state["history"] + [f"Psychologist: {question}"]
    
    turn = {
        "turn_index": len(state["transcript"])+1, 
        "speaker": AI_NAME, 
        "text": question, 
        "role": "question",
        "item_id": state["current_item_id"]
    }
    
    return {
        "last_question": question, 
        "history": new_hist, 
        "transcript": state["transcript"] + [turn],
        "current_item_id": state["current_item_id"],
        "current_item_label": state["current_item_label"],
        "current_hypothesis": state["current_hypothesis"]
    }

# 2. Participant Node
def participant_node(state: AgentState):
    # 1. Load Profile (KEPT)
    profile_obj = json.loads(state["participant_profile"])
    is_followup_flag = state.get("followup_count", 0) > 0
    
    # 2. Capture Requested Level (NEW - For Print Only)
    requested_level = state.get("current_difficulty", "level1") 
    
    # 3. Run Simulation (KEPT EXACTLY AS IS)
    answer_text, diff_mode = simulate_client_answer(
        item_id=state["current_item_id"],
        item_index=state["current_item_index"],
        item_label=state["current_item_label"],
        hypothesis_text=state["current_hypothesis"],
        question_text=state["last_question"],
        client_profile=profile_obj,
        llm=llm,
        is_followup=is_followup_flag
    )
    
    # 4. Print Logic (UPDATED)
    # This is the visual change you wanted
    if requested_level == "level3":
        print(f"   [Simulation] 🎲 Level 3 (Hard) -> Injected: {diff_mode.upper()}")

    print(f"👤 Participant: {answer_text}")
    
    # 5. Update History (KEPT)
    new_hist = state["history"] + [f"Participant: {answer_text}"]
    
    # 6. Update Transcript (KEPT)
    turn = {
        "turn_index": len(state["transcript"])+1, 
        "speaker": PARTICIPANT_NAME, 
        "text": answer_text, 
        "role": "answer",
        "item_id": state["current_item_id"]
    }
    
    return {
        "last_answer": answer_text, 
        "history": new_hist, 
        "transcript": state["transcript"] + [turn],
        "current_difficulty": diff_mode, 
        "simulation_mode": diff_mode     # Added this just to be safe
    }

# 3. Clarification Node
def clarification_node(state: AgentState):
    #SKIP CHECK FOR INTRO
    if state["current_item_id"] in ["INTRO", "CLOSING"]:
        return {
            "clarification_status": "COMPLETE",
            "clarification_reason": "Non-clinical phase.",
            "clarification_missing_domains": []
        }
    
    # Filter history for current item only (to avoid confusion)
    current_id = state["current_item_id"]
    relevant_turns = [t for t in state["transcript"] if t.get("item_id") == current_id]
    
    # Build focused history string
    focused_hist_str = ""
    for t in relevant_turns:
        focused_hist_str += f"{t['speaker']}: {t['text']}\n"

    res = (clarification_template | llm | json_parser).invoke({
        "question": state["last_question"], 
        "answer": state["last_answer"],
        "history_str": focused_hist_str # <--- Now passing history
    })
    
    missing = res.get("missing_domains", [])
    
    return {
        "clarification_status": res.get("status", "COMPLETE"),
        "clarification_reason": res.get("reason", ""),
        "clarification_missing_domains": missing
    }

# 4. Alignment Node
def alignment_node(state: AgentState):
    smart_history_str = alignment_helper.get_focused_history(state["current_item_id"], state["items_evidence"])
    
    if "No relevant previous answers" in smart_history_str and len(state["history"]) > 0:
        fallback_hist = "\n".join(state["history"][-4:])
        smart_history_str += f"\n\n(General Context):\n{fallback_hist}"

    res = (alignment_template | llm | json_parser).invoke({
        "answer": state["last_answer"], 
        "history_str": smart_history_str
    })

    return {
        "alignment_status": res.get("status", "CONSISTENT"), 
        "alignment_reason": res.get("reason", "")
        }

# 5. Navigation Node
def navigation_node(state: AgentState):
    # 1. Get inputs from State
    missing_list = state.get("clarification_missing_domains", [])
    current_retries = state.get("followup_count", 0)
    
    # --- NEW: CHECK ALIGNMENT STATUS ---
    # If Alignment found a contradiction, we ADD it to the "missing" list
    # This forces the code to trigger a Follow-Up.
    alignment_status = state.get("alignment_status", "CONSISTENT")
    if alignment_status == "CONTRADICTING":
        if "contradiction" not in missing_list:
            missing_list.append("contradiction")

    # Initialize variable
    style_guide = "Standard Follow-up" 

    # 2. Ask Navigation Agent for opinion
    res = (navigation_template | llm | json_parser).invoke({
        "c_stat": state["clarification_status"], 
        "c_reas": state["clarification_reason"],
        "a_stat": state.get("alignment_status", "UNKNOWN"),  
        "a_reas": state.get("alignment_reason", "None")
    })
    
    proposed_action = res.get("next_action", "NEXT_ITEM")
    base_instruction = res.get("instruction", "")

    # 3. DECISION LOGIC
    
    # CASE A: Max Retries Hit -> Force Next Item
    if proposed_action == "FOLLOW_UP" and current_retries >= 3:
        print(f"   [Logic] 🛑 MAX RETRIES ({current_retries}) HIT -> Forcing Next Item...")
        final_action = "NEXT_ITEM"
        final_instruction = "Move to next item."
        missing_list = [] 
    
    # CASE B: Normal Follow-Up (If Agent wants it OR we found a Contradiction)
    # We added 'or "contradiction" in missing_list' to ensure it fires even if the LLM missed it.
    elif (proposed_action == "FOLLOW_UP" and missing_list) or ("contradiction" in missing_list):
        final_action = "FOLLOW_UP"
        
        # I. Pick the Domain
        if "contradiction" in missing_list:
            selected_domain = "contradiction" 
        elif "relevance" in missing_list:
            selected_domain = "relevance"
        elif "timeframe" in missing_list:
            selected_domain = "timeframe"
        elif "vagueness" in missing_list:
            selected_domain = "vagueness"
        else:
            selected_domain = missing_list[0]

        # II. DEFINE STRATEGIES (Added Contradiction Strategy)
        ESCALATION_MAP = {
            "vagueness": [
                "Ask naturally.",
                "Offer two clear options (e.g., 'Is it closer to just a couple of days, or more than half the week?').",
                "Ask for a direct estimate (e.g., 'Would you say that is 3-4 days a week?')."
            ],
            "timeframe": [
                "Ask naturally.",
                "Be specific about the window. Ask explicitly if this has been happening in the 'last 2 weeks'.",
                "Force a yes/no on recency. (e.g., 'Has this been a problem specifically within the last 14 days?')."
            ],
            "relevance": [
                "Pivot gently back to the topic.",
                "Be more direct. Acknowledge their point but explicitly ask about the symptom.",
                "Directly link their story to the symptom. (e.g., 'Does that specific situation make you feel [Symptom]?')."
            ],
            "contradiction": [
                "Gently mention the difference. (e.g. 'I'm a bit confused because you mentioned X earlier...')",
                "Ask directly which one is more accurate right now.",
                "Confront the inconsistency politely. (e.g. 'To get the score right, I need to know: is it X or Y?')"
            ]
        }
        
        strategies = ESCALATION_MAP.get(selected_domain, ["Ask specifically."])
        style_guide = strategies[min(current_retries, len(strategies)-1)]

        # III. Construct Instruction
        # If it's a contradiction, we pull the reason from Alignment, otherwise Clarification
        reason_context = state.get("alignment_reason") if selected_domain == "contradiction" else state.get("clarification_reason")

        final_instruction = (
            f"Address the {selected_domain.upper()} issue. "
            f"Context: {reason_context}. "
            f"{style_guide}"
        )
    
    # CASE C: Score or Empty Missing List
    else:
        if proposed_action == "FOLLOW_UP" and not missing_list:
             final_action = "NEXT_ITEM"
             final_instruction = "Proceed to next item."
        elif state["current_item_id"] == "CLOSING":
             final_action = "NEXT_ITEM"
             final_instruction = "End interview."
        else:
             final_action = proposed_action
             final_instruction = base_instruction

    # 4. Print Logic Status
    if final_action == "FOLLOW_UP":
        print(f"   [Logic] ⚠️  ISSUE: {selected_domain.upper()} -> Strategy: {style_guide} (Attempt {current_retries + 1}/3)...")
        new_followup_count = current_retries + 1
    else:
        if state["current_item_id"] == "INTRO":
             print(f"   [Logic] 💬  Intro Dialogue -> Continuing...")
        elif state["current_item_id"] == "CLOSING":
             print(f"   [Logic] 🏁  CLOSING -> Ending Experiment...")
        else:
             print(f"   [Logic] ✅  COMPLETE -> Next Item...")
             
        new_followup_count = 0

    # 5. ENTAILMENT (Preserved)
    items_data = state["items_evidence"] 
    if state["current_item_id"] != "INTRO":
        probs = compute_nli_probs(state["last_answer"], state["current_hypothesis"])
        role = "neutral"
        if probs["p_entail"] >= ENTAIL_THRESHOLD: role = "supporting"
        elif probs["p_contradict"] >= CONTRADICT_THRESHOLD: role = "contradicting"
        
        item_key = f"Item {state['current_item_index']}"
        
        if item_key in items_data:
            current_count = len(items_data[item_key][role]) + 1
            evidence_id_key = f"evidence_{role}_id"
            evidence_id_val = f"{state['current_item_id']}_{role}_E{current_count}"
            
            entry = {
                evidence_id_key: evidence_id_val,   
                "text": state["last_answer"],
                "p_entail": round(probs["p_entail"], 4),
                "p_contradict": round(probs["p_contradict"], 4),
                "p_neutral": round(probs["p_neutral"], 4),
                "followup_asked": (final_action == "FOLLOW_UP"),
                "missing_domains": missing_list     
            }
            items_data[item_key][role].append(entry)

    # 6. ANALYTIC BLOCK (Standard)
    raw_mode = state.get("current_difficulty", "level1").lower()
    
    if raw_mode == "level1": 
        injected_flaw = "none"
    elif raw_mode == "level2": 
        injected_flaw = "vagueness" 
    elif raw_mode in ["vagueness", "timeframe", "relevance"]: 
        injected_flaw = raw_mode    
    else: 
        injected_flaw = "vagueness" 

    detected_flaw = "none"
    if missing_list: detected_flaw = missing_list[0] 

    turn_label = "Initial" if state["followup_count"] == 0 else f"FollowUp_{state['followup_count']}"
    bot_caught = False
    
    if injected_flaw == "none" and final_action == "NEXT_ITEM": 
        bot_caught = True
    elif injected_flaw != "none" and final_action == "FOLLOW_UP": 
        bot_caught = True

    analytic_entry = {
        "PID": "PENDING", 
        "Item": state["current_item_id"],
        "Turn": turn_label,
        "Injected_Flaw": injected_flaw, 
        "Detected_Flaw": detected_flaw,
        "Agent_Decision": final_action, 
        "Bot_Caught_Flaw": bot_caught,
        "Agent_Score": -1, 
        "Participant_Text": state["last_answer"].replace('"', "'") 
    }

    current_analytics = state.get("analytics_records", [])
    if state["current_item_id"] != "INTRO":
        current_analytics.append(analytic_entry)

    # 7. Save & Return
    thought = {
        "item": state["current_item_id"],
        "clarification": state["clarification_status"],
        "alignment": alignment_status, # Log this!
        "decision": final_action,
        "instruction": final_instruction
    }

    return {
        "next_action": final_action, 
        "nav_instruction": final_instruction, 
        "agent_thoughts": state["agent_thoughts"] + [thought],
        "items_evidence": items_data,
        "followup_count": new_followup_count 
    }

# 6. Transition Node (FIXED: Properly loads next item)
def transition_node(state: AgentState):
    # Retrieve current state info
    current_id = state["current_item_id"]
    current_idx = state["current_item_index"]
    
    # -------------------------------------------------------
    # CASE A: INTRO PHASE (Loop 3 times)
    # -------------------------------------------------------
    if current_id == "INTRO":
        current_count = state.get("intro_turn_count", 0) + 1
        INTRO_LIMIT = 3
        
        if current_count < INTRO_LIMIT:
            # Stay in Intro loop
            next_instr = "Ask a polite follow-up question." if current_count == 1 else "Ask one final general question."
            return {
                "current_item_index": 0,
                "intro_turn_count": current_count,
                "nav_instruction": next_instr,
                "followup_count": 0,
                "symptom_summaries": state.get("symptom_summaries", [])
            }
        else:
            # FINISHED INTRO -> LOAD ITEM 1
            next_item = PHQ8_HYPOTHESES[0] # Item 1 is at index 0
            return {
                "current_item_index": 1,
                "current_item_id": next_item["item_id"],   # <--- Update ID to "Item 1"
                "current_item_label": next_item["label"],  # <--- Update Label
                "current_hypothesis": next_item["text"],   # <--- Update Hypothesis
                "intro_turn_count": current_count,
                "nav_instruction": "Transition to Item 1: Anhedonia.",
                "followup_count": 0,
                "symptom_summaries": state.get("symptom_summaries", [])
            }

    # -------------------------------------------------------
    # CASE B: CLOSING PHASE (End the Interview)
    # -------------------------------------------------------
    if current_id == "CLOSING":
        return {
            "current_item_index": 10, # Move to Batch Scoring
            "nav_instruction": "End experiment.",
            "followup_count": 0,
            "analytics_records": state.get("analytics_records", []),
            "symptom_summaries": state.get("symptom_summaries", [])
        }

    # -------------------------------------------------------
    # CASE C: NORMAL ITEMS (Move 1 -> 2 ... -> 8 -> Closing)
    # -------------------------------------------------------
    
    # 1. PROCESS SYMPTOM SUMMARY (Save flaws before moving on)
    updated_analytics = state.get("analytics_records", [])
    
    # Filter analytics for CURRENT item
    current_item_logs = [r for r in updated_analytics if r["Item"] == state["current_item_id"]]
    
    # Sort text into buckets
    vagueness_texts = []
    timeframe_texts = []
    relevance_texts = []
    
    for log in current_item_logs:
        flaw = log["Detected_Flaw"]
        text = log["Participant_Text"]
        
        if flaw == "vagueness":
            vagueness_texts.append(text)
        elif flaw == "timeframe":
            timeframe_texts.append(text)
        elif flaw == "relevance":
            relevance_texts.append(text)
            
    # Count Follow-ups
    followup_count = sum(1 for log in current_item_logs if log["Agent_Decision"] == "FOLLOW_UP")
    
    # Create Summary Entry
    symptom_entry = {
        "PID": "PENDING", 
        "Item": state["current_item_id"],
        "Vagueness_Response": " | ".join(vagueness_texts) if vagueness_texts else "None",
        "Timeframe_Response": " | ".join(timeframe_texts) if timeframe_texts else "None",
        "Relevance_Response": " | ".join(relevance_texts) if relevance_texts else "None",
        "Total_Followups": followup_count
    }
    
    # Append to List
    current_summaries = state.get("symptom_summaries", [])
    current_summaries.append(symptom_entry)

    # 2. CALCULATE NEXT ITEM (This is the critical fix)
    next_idx = current_idx + 1
    
    # If we finished Item 8 (index 8), go to CLOSING (Index 9)
    if next_idx > 8:
        return {
            "current_item_index": 9,
            "current_item_id": "CLOSING",          # <--- SET TO CLOSING
            "current_item_label": "Closing",
            "current_hypothesis": "End the conversation politely.",
            "nav_instruction": "Wrap up the interview.",
            "followup_count": 0,
            "symptom_summaries": current_summaries
        }
        
    else:
        # Load Next Item
        # Note: PHQ8_HYPOTHESES[0] is Item 1. 
        # So if next_idx is 2 (Item 2), we need PHQ8_HYPOTHESES[1].
        next_item = PHQ8_HYPOTHESES[next_idx - 1]
        
        return {
            "current_item_index": next_idx,
            "current_item_id": next_item["item_id"],   # <--- UPDATE ID (e.g., "Item 2")
            "current_item_label": next_item["label"],  # <--- UPDATE LABEL
            "current_hypothesis": next_item["text"],   # <--- UPDATE HYPOTHESIS
            "nav_instruction": "Start next item.",
            "followup_count": 0,
            "symptom_summaries": current_summaries
        }
    
# 7. Batch Scoring Node (Updates Scores AND Analytics)
def batch_scoring_node(state: AgentState):
    print("\n⏳ Interview Complete. Starting Batch Scoring...")
    
    final_scores = []
    scoring_explanations = []

    # Initialize Helper (Using your class)
    helper = AlignmentHelper("phq8_alignment_map.json")
    
    # Get the existing analytics list so we can update it
    updated_analytics = state.get("analytics_records", [])
    
    # Loop through PHQ-8 Items
    for i, item_def in enumerate(PHQ8_HYPOTHESES):
        item_id = item_def["item_id"]
        item_label = item_def["label"]
        hypothesis = item_def["text"]

        alignment_context = helper.get_focused_history(item_id, state.get("items_evidence", {}))
        
        # 1. Gather Transcript
        relevant_turns = [t for t in state["transcript"] if t.get("item_id") == item_id]
        if not relevant_turns:
            print(f"   ⚠️ No transcript found for {item_id}, skipping...")
            continue
            
        history_text = ""
        for t in relevant_turns:
            history_text += f"{t['speaker']}: {t['text']}\n"
            
        # 2. Gather Evidence (Symptom Context)
        symptoms = [s for s in state.get("symptom_summaries", []) if s["Item"] == item_id]
        symptom_context = ""
        if symptoms:
            s = symptoms[0]
            symptom_context = (
                f"NOTE: During this item, the patient had these issues:\n"
                f"- Vagueness: {s['Vagueness_Response']}\n"
                f"- Timeframe issues: {s['Timeframe_Response']}\n"
                f"- Relevance issues: {s['Relevance_Response']}\n"
                f"- Total clarifications needed: {s['Total_Followups']}"
            )

        # 3. Invoke LLM
        res = (scoring_template | llm | json_parser).invoke({
            "item_label": item_label,
            "hypothesis": hypothesis,
            "history_str": history_text + "\n" + symptom_context
        })
        
        score = res.get("score", 0)
        explanation = res.get("explanation", "")
        
        print(f"   ✅ Scored {item_id}: {score} ({item_label})")
        
        # 4. BACK-FILL ANALYTICS (Crucial Step!)
        # Go through the logs and update the 'Agent_Score' for this item
        for record in updated_analytics:
            if record["Item"] == item_id:
                record["Agent_Score"] = score

        # 5. Save Score Data
        final_scores.append({
            "Item ID": item_id, 
            "Item Label": item_label, 
            "Score": score
        })
        scoring_explanations.append({
            "item_id": item_id, 
            "score": score, 
            "explanation": explanation
        })

    # Return EVERYTHING (Scores + Updated Analytics)
    return {
        "final_scores": final_scores,
        "scoring_explanations": scoring_explanations,
        "analytics_records": updated_analytics # <--- Saves the updated list
    }

# ================= GRAPH LOGIC =================
def should_continue(state: AgentState):
    if state["next_action"] == "FOLLOW_UP": return "question"
    return "scoring"

def check_end(state: AgentState):
    if state["current_item_index"] > 9: return END
    return "question"

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. ADD ALL NODES
    workflow.add_node("question_node", question_node)
    workflow.add_node("participant_node", participant_node)
    workflow.add_node("clarification_node", clarification_node)
    workflow.add_node("alignment_node", alignment_node)
    workflow.add_node("navigation_node", navigation_node)
    workflow.add_node("transition_node", transition_node) 
    workflow.add_node("batch_scoring_node", batch_scoring_node)

    # 2. DEFINE EDGES (The Flow)
    workflow.set_entry_point("question_node")

    workflow.add_edge("question_node", "participant_node")
    workflow.add_edge("participant_node", "clarification_node")
    workflow.add_edge("clarification_node", "alignment_node")
    workflow.add_edge("alignment_node", "navigation_node")

    # Navigation logic (Split between Follow-up or Next Item)
    def check_nav(state):
        # If we need to follow up (retry), go back to Question Node
        if state.get("followup_count", 0) > 0 and state["nav_instruction"] != "End experiment.":
             return "question_node"
        # Otherwise, move to Transition (Next Item)
        return "transition_node"

    workflow.add_conditional_edges(
        "navigation_node",
        check_nav,
        {
            "question_node": "question_node",
            "transition_node": "transition_node" 
        }
    )

    # Transition logic (Loop or End)
    def check_end(state):
        # If index > 9, conversation is over -> Go to Batch Scoring
        if state["current_item_index"] > 9:
            return "batch_scoring_node"
        # Otherwise, ask next question
        return "question_node"

    workflow.add_conditional_edges(
        "transition_node",
        check_end,
        {
            "question_node": "question_node",
            "batch_scoring_node": "batch_scoring_node"
        }
    )
    
    # Batch Scoring is the final step
    workflow.add_edge("batch_scoring_node", END)

    return workflow.compile()

# ================= MAIN =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=str, required=True)
    args = parser.parse_args()

    # Load Profile
    try:
        with open(f"Clean_Dataset/profiles/{args.pid}_client_profile.json", "r", encoding="utf-8") as f:
            profile_data = json.load(f)
            profile_str = json.dumps(profile_data, ensure_ascii=False)
    except Exception:
        print("Profile error: File not found."); return
    
    # Initialize Evidence Structure
    items_init = {f"Item {i+1}": {"label": h["label"], "item_id": h["item_id"], "supporting":[], "contradicting":[], "neutral":[]} for i, h in enumerate(PHQ8_HYPOTHESES)}

    # ---------------------------------------------------------
    state = {
        "participant_profile": profile_str,
        "history": [],
        "transcript": [],
        
        # --- INTRO CONFIGURATION ---
        "current_item_index": 0,          # Start at 0
        "current_item_id": "INTRO",       # ID is INTRO
        "current_item_label": "Introduction", 
        "current_hypothesis": "Establish rapport.", 
        
        # --- NEW COUNTER (Required for the 2-4 question loop) ---
        "intro_turn_count": 0, 

        # --- NEW ANALYTICS FIELDS (CRITICAL FOR CSV) ---
        "analytics_records": [],         
        "current_difficulty": "level1", 
        # --- SYMOTOMS ANALYSIS ---
        "symptom_summaries": [],  

        # --- STANDARD FIELDS ---
        "items_evidence": items_init,
        "final_scores": [],
        "scoring_explanations": [],
        "agent_thoughts": [],
        "clarification_missing_domains": [],
        "nav_instruction": "Start introduction.",
        "followup_count": 0
    }
    # ---------------------------------------------------------

    print(f"\n🚀 Multi-Agent System (PID {args.pid}) Started...")
    app = build_graph()
    
    # INCREASE RECURSION LIMIT to 150
    final_state = app.invoke(state, {"recursion_limit": 500})

    # SAVE FILES
    # =========================================================
    # 1. Setup Directories
    base_dir = "multi-agent-system"
    
    # 2. Define Sub-folders (Directly under base_dir)
    dirs = {
        "ev": os.path.join(base_dir, "Evidence"),
        "tr": os.path.join(base_dir, "Transcript"),
        "th": os.path.join(base_dir, "Agent_Thoughts"),
        "sc": os.path.join(base_dir, "Scores"),
        "ex": os.path.join(base_dir, "Scoring_Explanations"),
        "an": os.path.join(base_dir, "Analysis_Metrics"),
        "sy": os.path.join(base_dir, "Symptoms")
    }
    
    # 3. Create All Directories
    for d in dirs.values(): 
        os.makedirs(d, exist_ok=True)

    # --- SAVE OPERATIONS ---

    # A. Save Evidence
    with open(os.path.join(dirs["ev"], f"Evidence_{args.pid}.json"), "w") as f: 
        json.dump(final_state["items_evidence"], f, indent=2)
        
    # B. Save Transcript
    with open(os.path.join(dirs["tr"], f"Transcript_{args.pid}.jsonl"), "w") as f: 
        for t in final_state["transcript"]: f.write(json.dumps(t)+"\n")
    
    # C. Save Agent Thoughts
    with open(os.path.join(dirs["th"], f"Thoughts_{args.pid}.jsonl"), "w") as f: 
        for t in final_state["agent_thoughts"]: f.write(json.dumps(t)+"\n")
    
    # D. Save Explanations
    with open(os.path.join(dirs["ex"], f"Explanations_{args.pid}.json"), "w") as f: 
        json.dump(final_state["scoring_explanations"], f, indent=2)

    # E. Save Scores CSV
    total_score = sum(item["Score"] for item in final_state["final_scores"])
    csv_data = final_state["final_scores"] + [
        {"Item ID": "TOTAL", "Item Label": "PHQ-8 SUM", "Score": total_score}
    ]
    csv_path = os.path.join(dirs["sc"], f"Scores_{args.pid}.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["Item ID", "Item Label", "Score"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(csv_data)

    # F. Save Analytics CSV
    analytics_path = os.path.join(dirs["an"], f"analysis_{args.pid}.csv")
    
    records = final_state.get("analytics_records", [])
    
    # Pre-process records
    for r in records:
        r["PID"] = args.pid 
        if r["Agent_Decision"] == "NEXT_ITEM":
            r["Agent_Decision"] = "NEXT_ITEM"

    if records:
        keys = ["PID", "Item", "Turn", "Injected_Flaw", "Detected_Flaw", "Agent_Decision", "Bot_Caught_Flaw", "Agent_Score", "Participant_Text"]
        with open(analytics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(records)

    # G. SAVE SYMPTOMS SUMMARY CSV
    symptoms_path = os.path.join(dirs["sy"], f"Symptoms_{args.pid}.csv")
    sym_records = final_state.get("symptom_summaries", [])
    
    for r in sym_records:
        r["PID"] = args.pid 
        
    if sym_records:
        keys = ["PID", "Item", "Vagueness_Response", "Timeframe_Response", "Relevance_Response", "Total_Followups"]
        with open(symptoms_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(sym_records)
    # =========================================================

    print(f"\n✅ Done. Total PHQ-8 Score: {total_score}.")
    print(f"✅ All files saved to: {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    main()
