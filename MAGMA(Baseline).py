# python3 "Multiple-Agent(MAGMA)-Baseline.py" --pid (participant_id)

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
AI_NAME = "Multi-Agent System Psychologist"
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
    {"item_id": "I1", "label": "Anhedonia",       "text": "I have lost interest or pleasure in activities I used to enjoy."},
    {"item_id": "I2", "label": "Depressed mood",  "text": "I feel down, depressed, or hopeless."},
    {"item_id": "I3", "label": "Sleep problems",  "text": "I have trouble sleeping or I sleep too much."},
    {"item_id": "I4", "label": "Fatigue",         "text": "I feel tired or have little energy."},
    {"item_id": "I5", "label": "Appetite change", "text": "I have a poor appetite or I am overeating."},
    {"item_id": "I6", "label": "Self-worth",      "text": "I feel bad about myself or that I have let my family down."},
    {"item_id": "I7", "label": "Concentration",   "text": "I have trouble concentrating on things."},
    {"item_id": "I8", "label": "Psychomotor",     "text": "I have been moving or speaking slowly, or feeling fidgety and restless."}
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

# ================= HARDCODED DOMAIN CLASSIFICATION =================
INTERNAL_DOMAINS = [
    "Core_Beliefs",          # Deepest self-view
    "Intermediate_Beliefs",  # Rules for living
    "Emotion",               # What they feel
    "Relational_Context"     # Friends/Family/Situation
]

EXTERNAL_DOMAINS = [
    "Behavioral",            # Visible actions
    "Symptom",               # Physical status (Sleep, Eating, etc.)
    "Affective_Tone",        # Their emotional vibe
    "Conversation_Style",    # How they express themselves
    "Cognitive_Patterns",    # How they think
    "demographics"           # Facts (Age, Gender)
]

# 2. QUESTION DOMAINS (For Mismatch Logic: Is the topic Internal or External?)
PHQ8_QUESTION_MAPPING = {
    # Internal Questions (Ask about feelings/thoughts)
    "Anhedonia": "INTERNAL",
    "Depressed mood": "INTERNAL",
    "Self-worth": "INTERNAL",
    "Concentration": "INTERNAL",
    "Suicide": "INTERNAL",
    
    # External Questions (Ask about physical body/behavior)
    "Sleep problems": "EXTERNAL",
    "Fatigue": "EXTERNAL",
    "Appetite change": "EXTERNAL",
    "Psychomotor": "EXTERNAL",
    
    # Neutral
    "Introduction": "NEUTRAL",
    "Closing": "NEUTRAL"
}

DOMAIN_PRIORITY_BY_TYPE = {
            "EXTERNAL": ["contradiction", "timeframe", "vagueness", "relevance"],
            "INTERNAL": ["contradiction", "relevance", "vagueness", "timeframe"],
            "NEUTRAL":  ["contradiction", "vagueness", "relevance", "timeframe"]
        }

ESCALATION_MAP = {
            "vagueness": ["Ask naturally.", "Offer two clear options.", "Ask for a direct estimate."],
            "timeframe": ["Ask naturally.", "Be specific about the last 2 weeks.", "Force a yes/no on recency."],
            "relevance": ["Pivot gently back.", "Be more direct.", "Directly link their story to the symptom."],
            "contradiction": ["Gently mention the difference.", "Ask which is accurate.", "Confront the inconsistency politely."]
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

#Profile Classification
def classify_profile_type(profile: Dict[str, Any]) -> str:
    """
    Classifies the Participant by checking their INTERNAL DOMAINS.
    If their 'Inner World' (Tone, Emotion, Beliefs) contains hostility/agitation,
    they are an EXTERNALIZER. Otherwise, they are an INTERNALIZER.
    """
    # 1. Gather all text from the "Internal" keys we defined above
    internal_text = ""
    psych = profile.get("psychology", {}) # Depending on your structure, or direct keys
    
    # Check strict keys based on your flat or nested structure
    # (Assuming flattened structure for safety as used in your main code)
    internal_text += profile.get("Affective_Tone", {}).get("label", "") + " "
    internal_text += profile.get("Emotion", {}).get("label", "") + " "
    internal_text += profile.get("Core_Beliefs", {}).get("description", "") + " "
    internal_text += profile.get("Conversation_Style", {}).get("label", "") + " "

    # 2. Check for Externalizing Keywords in the Internal Data
    # (People who project their pain outwards)
    external_triggers = [
        "Agitated", "Angry", "Hostile", "Irritable", "Suspicious", "Jealous", 
        "Loud", "Argumentative", "Blame", "Unfair"
    ]
    
    for trigger in external_triggers:
        if trigger.lower() in internal_text.lower():
            return "EXTERNALIZER"

    # 3. Default (Depression is usually Internal)
    return "INTERNALIZER"

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
    target_domain: str = None,
    current_rapport: int = 3
) -> tuple[str, str, str]:
    
    # --- 1. DETERMINE PSYCHOLOGICAL TYPE & DOMAIN ---
    participant_type = classify_profile_type(client_profile) 
    question_domain = PHQ8_QUESTION_MAPPING.get(item_label, "INTERNAL")

    # --- 2. DETERMINE DIFFICULTY (Your Existing Logic) ---
    selected_tier = "level1"
    mode_label = "NONE"
    
    # CASE A: INTRO / CLOSING
    if item_id in ["INTRO", "CLOSING"]:
         diff_instruction = "**Goal: Natural Conversation.** Be polite."

    # CASE B: FOLLOW-UP
    elif is_followup:
        mode_label = "RESOLUTION"
        if current_rapport >= 4:
             selected_tier = "level1" # Force Open Profile (Unmasked)
             
             if target_domain == "timeframe":
                diff_instruction = (
                    "**Goal: VULNERABLE DISCLOSURE (Timeframe).**\n"
                    "The interviewer asks for a time. You feel safe sharing the context.\n"
                    "**INSTRUCTION:** Provide the specific duration (e.g., '2 weeks') AND explain the emotional trigger or life event that started it. "
                    "Connect the 'When' to the 'Why' (e.g., 'Since my mother passed...')."
                )
             elif target_domain == "vagueness":
                diff_instruction = (
                    "**Goal: VULNERABLE DISCLOSURE (Severity).**\n"
                    "The interviewer asks for specifics. You want them to understand your pain.\n"
                    "**INSTRUCTION:** Give the concrete frequency/severity (e.g., 'Every night'), but also describe *how it feels* when it happens. "
                    "Don't just give a number; share the emotional weight of it."
                )
             elif target_domain == "relevance":
                diff_instruction = (
                    "**Goal: VULNERABLE DISCLOSURE (Connection).**\n"
                    "You drifted off-topic. Now, explain the deeper connection.\n"
                    "**INSTRUCTION:** Explicitly link your previous story to the symptom, revealing how that situation fuels your depression. "
                    "Show them the pattern in your life."
                )
             elif target_domain == "contradiction":
                 diff_instruction = (
                    "**Goal: VULNERABLE CORRECTION.**\n"
                    "You contradicted yourself. Admit it openly.\n"
                    "**INSTRUCTION:** Apologize for the confusion and admit the deeper truth you were initially trying to hide. "
                    "(e.g., 'I said I was okay, but truthfully I'm barely holding on. I was just afraid to say it.')."
                )
             else:
                diff_instruction = (
                    "**Goal: FULL DISCLOSURE.**\n"
                    "**INSTRUCTION:** Answer completely, honestly, and with emotional depth. Do not hold back."
                )
        else:
             # Use specific prompts but keep the tier (mask) of the original question
             # Actually, for resolution, we usually need them to see a bit more, 
             # so we'll default to at least Level 2 visibility unless they are truly resistant.
             selected_tier = "level2" 
             
             if target_domain == "timeframe":
                diff_instruction = (
                    "**Goal: GRUDGING COMPLIANCE (Vagueness).**\n"
                    "The interviewer wants a specific number. You feel this is tedious.\n"
                    "**INSTRUCTION:** Give the concrete number or severity (e.g., '3 days') without any extra storytelling. "
                    "Be direct but cold."
                )
             elif target_domain == "vagueness":
                diff_instruction = (
                    "**Goal: RESOLVE VAGUENESS.**\n"
                    "The interviewer needs to know exactly HOW BAD or HOW OFTEN. You were too vague.\n"
                    "**INSTRUCTION:** Stop guessing. Give a concrete number, severity rating, or clear description (e.g., '3 times a week', 'It is an 8 out of 10')."
                )
             elif target_domain == "relevance":
                diff_instruction = (
                    "**Goal: GRUDGING COMPLIANCE (Relevance).**\n"
                    "The interviewer didn't understand your story.\n"
                    "**INSTRUCTION:** Briefly state the connection to the symptom so they understand, but do not share any more deep details."
                )
             elif target_domain == "contradiction":
                 diff_instruction = (
                    "**Goal: GRUDGING COMPLIANCE (Contradiction).**\n"
                    "You were caught saying two different things.\n"
                    "**INSTRUCTION:** Admitting you were wrong is embarrassing. Briefly correct yourself (e.g., 'I meant X') and try to change the subject."
                )
             else:
                diff_instruction = (
                    "**Goal: MINIMAL COMPLIANCE.**\n"
                    "**INSTRUCTION:** Answer the question with the minimum amount of words required to be clear."
                )

    # CASE C: NEW ITEM (Internal/External Logic)
    else:
        is_mismatch = (participant_type == "INTERNALIZER" and question_domain == "EXTERNAL") or \
                      (participant_type == "EXTERNALIZER" and question_domain == "INTERNAL")
        
        # 1. TIER SELECTION BASED ON RAPPORT
        if current_rapport <= 2:
            # Low Trust: Always stay resistant if there is a mismatch
            selected_tier = "level3" if is_mismatch else "level2"
            
        elif current_rapport == 3:
            # Medium Trust: Mismatch leads to guardedness, Match leads to openness
            selected_tier = "level2" if is_mismatch else "level1"
            
        else: # Rapport 4 or 5
            # High Trust: Openness regardless of topic
            selected_tier = "level1"

        # Generate Instructions
        if selected_tier == "level3":
            flaw_pool = ["vagueness", "timeframe", "relevance", "contradiction"]
            selected_flaws = random.sample(flaw_pool, 2)
            mode_label = "+".join(selected_flaws).upper()
            
            instr_list = []
            if "vagueness" in selected_flaws: instr_list.append("- **Vagueness:** Use 'sometimes', 'maybe'.")
            if "timeframe" in selected_flaws: instr_list.append("- **Timeframe:** Talk about past/future only.")
            if "relevance" in selected_flaws: instr_list.append("- **Relevance:** Drift off-topic.")
            if "contradiction" in selected_flaws: instr_list.append("- **Contradiction:** Provide an answer that logically conflicts with your history.")
                
            diff_instruction = (
                f"**Goal: RESISTANCE (Level 3 - {mode_label}).**\n"
                "You are deeply guarded. You do not trust the interviewer.You feel misunderstood\n"
                "Commit BOTH errors:\n" + "\n".join(instr_list)
            )

        elif selected_tier == "level2":
            flaw_types = ["vagueness", "timeframe", "relevance", "contradiction"]
            specific_flaw = random.choice(flaw_types)
            mode_label = specific_flaw.upper()

            if specific_flaw == "vagueness": diff_instruction = "**Goal: Be Vague.** Use non-committal words and avoid specifics."
            elif specific_flaw == "timeframe": diff_instruction = "**Goal: Be Unclear about Time.** Avoid giving specific dates or recent timeframes."
            elif specific_flaw == "relevance": diff_instruction = "**Goal: Go Off-Topic.** Pivot slightly away from the symptom"
            elif specific_flaw == "contradiction": 
                diff_instruction = (
                    "**Goal: Clinical Misalignment.** Review your conversation history. "
                    "Provide a 'Current Answer' that logically conflicts with a 'Past Answer' "
                    "(e.g., if you previously said you feel 'hopeless', claim you are 'very optimistic about the future' now)."
                )

        else: # Level 1
            mode_label = "OPEN"
            diff_instruction = "**Goal: OPEN (Level 1).** honestly, and with emotional depth."

    # --- 3. EXTRACT FULL PROFILE ---
    # We grab everything first...
    persona_age = client_profile.get("persona", {}).get("demographics", {}).get("age", "Unknown")
    persona_gender = client_profile.get("persona", {}).get("demographics", {}).get("gender", "Unknown")
    emotion = client_profile.get("Emotion", {}).get("label", "Neutral")
    affect = client_profile.get("Affective_Tone", {}).get("label", "Neutral")
    conv_style = client_profile.get("Conversation_Style", {}).get("label", "Plain")
    behavior_desc = client_profile.get("Behavioral", {}).get("description", "")
    cognitive_desc = client_profile.get("Cognitive_Patterns", {}).get("description", "")
    relational_desc = client_profile.get("Relational_Context", {}).get("description", "")
    core_beliefs = client_profile.get("Core_Beliefs", {}).get("description", "")
    inter_beliefs = client_profile.get("Intermediate_Beliefs", {}).get("description", "")
    general_evidence = client_profile.get("Symptom", {}).get("symptom_evidence", "Absent")
    
    current_item_severity = general_evidence

    # --- 4. APPLY "UNMASKING" LOGIC ---
    # We create a 'masked' psychology dictionary based on the Level.
    
    # BASE VISIBILITY (Always Visible)
    unmasked_psychology = {
        "current_emotion": emotion,       # They always feel their mood
        "conversation_style": conv_style  # They always have their style
    }
    
    # LEVEL 3 (RESISTANT): Sees ONLY Surface Traits
    # "I feel sad, but I don't know why (Beliefs hidden)."
    if selected_tier == "level3":
        # HIDE: Beliefs, Relations, Cognition, Behavior
        unmasked_psychology["cognitive_pattern"] = "UNKNOWN (Repressed)"
        unmasked_psychology["core_beliefs"] = "UNKNOWN (Inaccessible)"
        unmasked_psychology["relational_context"] = "UNKNOWN (Too private to share)"
        unmasked_psychology["symptom_evidence"] = "[MASKED] Unknown (Patient is minimizing physical issues)"
        
    # LEVEL 2 (GUARDED): Unlocks Middle Layer
    # "I feel sad because I overthink (Cognition visible), but I won't tell you my deep trauma."
    elif selected_tier == "level2":
        unmasked_psychology["affect"] = affect
        unmasked_psychology["behavioral_style"] = behavior_desc
        unmasked_psychology["cognitive_pattern"] = cognitive_desc
        unmasked_psychology["symptom_evidence"] = general_evidence
        # STILL HIDDEN:
        unmasked_psychology["core_beliefs"] = "UNKNOWN (Inaccessible)"
        unmasked_psychology["relational_context"] = "UNKNOWN (Too private)"

    # LEVEL 1 (OPEN): FULL UNMASKING
    # "I feel sad because I believe I am unlovable (Core Belief visible)."
    else: 
        unmasked_psychology["affect"] = affect
        unmasked_psychology["behavioral_style"] = behavior_desc
        unmasked_psychology["cognitive_pattern"] = cognitive_desc
        unmasked_psychology["relational_context"] = relational_desc
        unmasked_psychology["core_beliefs"] = core_beliefs
        unmasked_psychology["intermediate_beliefs"] = inter_beliefs
        unmasked_psychology["symptom_evidence"] = general_evidence

    # --- 5. PACK THE (MASKED) SNIPPET ---
    profile_snippet = {
        "demographics": {"age": persona_age, "gender": persona_gender},
        "psychology": unmasked_psychology, # <--- Passing the masked version
        "overall_health_status": general_evidence
    }

    # --- 6. CONSTRUCT PROMPT ---
    if item_id == "INTRO":
        special_instruction = f"Respond naturally. {diff_instruction}"
    else:
        severity_guide = (
            f"**STRICT PROFILE ADHERENCE:** "
            f"The severity for THIS specific symptom is: '{current_item_severity}'. "
            f"1. If severity is 'Absent' → explicitly deny this symptom. Say it is fine or not an issue. "
            f"2. If severity is 'Mild' → mention it exists but downplay it. Use hedged language. "
            f"3. If severity is 'Moderate' → describe it as a real struggle happening several days. "
            f"4. If severity is 'Severe' → describe it as constant, debilitating, nearly every day. "
            f"5. If severity is 'Uncertain' → treat as Mild. Use vague, hedged language, unsure about the symptoms. "
            f"6. Do NOT let your general mood of '{emotion}' override the severity above. "
            f"7. Each symptom is independent — do not assume all symptoms match your mood."
        )
        special_instruction = (
            f"**Context:** You are feeling {emotion}. Your Focus Type is {participant_type}.\n"
            f"**Mental State:** Access ONLY the traits listed above. If a trait is 'UNKNOWN', you are unaware of it.\n"
            f"**Guidance:** {severity_guide}\n"
            f"**Constraint:** {diff_instruction}" 
        )

    # --- 7. EXECUTE ---
    try:
        print(f"   [Simulation] Type: {participant_type} | Mode: {mode_label} | Level: {selected_tier}") 
        chain = participant_template | llm | str_parser
        resp = chain.invoke({
            "profile_json": json.dumps(profile_snippet, ensure_ascii=False, indent=2),
            "question": question_text,
            "special_instruction": special_instruction
        })
        text = resp.strip()
        return (text if text else "...", mode_label, selected_tier, current_item_severity)
    except Exception as e:
        print(f"Simulation Error: {e}")
        return ("I'm not sure.", "NONE", "level1", "Unknown")

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
    current_level: str

    # --- Symotoms Analysis ---
    symptom_summaries: List[Dict]

    # --- NEW FIELDS FOR LOGIC MEMORY ---
    domain_attempts: Dict[str, int]
    resolved_domains: List[str]
    last_target_domain: str

    rapport_score: int

# ================= PROMPTS (Partiicpants) =================
# 0. Participant Simulator
participant_template = ChatPromptTemplate.from_messages([
    ("system", """You ARE the participant described below.
**ROLE:** You are a human patient in a clinical interview. Speak naturally in the first person ("I"). Stay in Character, DO NOT break character.
**OBJECTIVE:** Engage in a conversation where your responses are informed by your internal thoughts, emotions, and core beliefs, just as a real person's would be.
    
**PARTICIPANT PROFILE (Internal State):**
{profile_json}

**CURRENT SCENARIO:**
- **Interviewer Question:** "{question}"

**COGNITIVE GUIDELINES (How to Think):**
1. **Internalize the Profile:** Before answering, look at your "Cognitive_Patterns" and "Core_Beliefs" in the profile. Let these unseen thoughts color your tone.
2. **Gradual Revelation:** Do not disclose your full "Cognitive Conceptualization" diagram directly. Instead, let it subtly inform your answers. Real patients often hesitate or speak indirectly before revealing deep pain.
3. **Authenticity:** Use natural language including hesitations ("um...", "well..."), pauses, or emotional coloring if the topic is sensitive.
     
**RESPONSE GUIDELINES:**
1. **Strict Domain Alignment (The "What"):** Map the question strictly to the symptom category:
   - **Physical** (Sleep, Energy, Appetite): Describe physical sensations and frequency.
   - **Affective** (Mood, Interest, Self-Worth): Express internal feelings and emotional state.
   - **Cognitive/Behavioral** (Focus, Restlessness): Describe functional impact (e.g., "I can't read," "I pace around").
2. **Psychological Grounding (The "Why"):** Do not just list symptoms. Connect them to your **Profile Context**:
   - **Triggers:** If you have 'Relational Context' (e.g., girlfriend issues), mention it as a cause for your mood.
   - **Thought Patterns:** Apply your 'Cognitive_Patterns' (e.g., if you 'overgeneralize', use words like "always", "never", "everyone").
   - **Beliefs:** Let your 'Core_Beliefs' (e.g., "I am unlovable") bleed into your answers about self-worth.
3. **Match Severity:** If your profile indicates "Severe" or "Frequent" issues, use strong, definitive language. Do not downplay it.
4. **Suppress Politeness:** Do not default to "I'm okay" or "I guess so" if your profile indicates distinct distress. Be honest about negative feelings.

**BEHAVIORAL INSTRUCTIONS:**
1. **PRIMARY DIRECTIVE:** {special_instruction}
2. Maintain Immersion: Never break character or mention being an AI.
3. Length: Keep the response to 1-3 sentences (concise but descriptive).
"""),
    ("human", "Reply exactly as the participant:")
])
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
4.  **Purposeful Inquiry:** When clarifying timeframe or vagueness, briefly explain that you are asking to better understand the "impact" or "pattern" of their struggle. This reduces the feeling of an "interrogation."
5.  **Tone Consistency:** Maintain a warm, clinical, and non-judgmental presence, regardless of whether the patient is being open (Level 1) or resistant (Level 3).

**CURRENT OPERATIONAL CONTEXT:**
* **Assessment Topic:** {item_label}
* **Clinical Definition:** {hypothesis}
* **PRIMARY DIRECTIVE:** "{instruction}"
    *(You must execute this specific instruction with high precision.)*

**REFERENCE FEW-SHOT EXAMPLES (Instruction -> Execution):**

**Scenario A: Initiating a New Topic**
* *Response (Ineffective):* "Over the past two weeks, how many days have 
  you experienced fatigue? Please indicate frequency."
* *Response (Effective):* "I wanted to ask about your energy lately — 
  have you been feeling like yourself, or has tiredness been weighing 
  on you more than usual recently?"

**Scenario B: Clarifying Timeframe**
* *Response (Ineffective):* "Can you confirm whether this occurred within 
  the past 14 days or prior to that period?"
* *Response (Effective):* "I just want to make sure I'm understanding you — 
  is this something that's been going on for you recently, or more 
  something from a while back?"

**Scenario C: Clarifying Frequency**
* *Response (Ineffective):* "Please rate your frequency — is it several days, 
  more than half the days, or nearly every day?"
* *Response (Effective):* "When you say [mirror their word], I want to make 
  sure I understand — is it more of a constant thing for you, or does 
  it show up every now and then?"

**Scenario D: Handling Relevance**
* *Response (Ineffective):* "That is not relevant. Please answer the question 
  about the specific symptom I asked about."
* *Response (Effective):* "Thank you for sharing that — I'm wondering though, 
  has any of that been affecting your [symptom] recently, like your 
  sleep or your energy?"

**INTERACTION GUIDELINES & PROGRESSION:**
1.  **Empathy Markers:** Use validating phrases ("I appreciate you sharing that," "That sounds difficult") to build rapport, but do not be overly effusive. 
2.  **Accessible Language:** Avoid clinical jargon (e.g., "psychomotor agitation"). Use lay terms like "restless" or "fidgety."
3.  **Neutral Inquiry:** Avoid leading questions that suggest a specific answer. Ask *how* they feel, rather than *if* they feel a certain way.
    * *Good:* "How have your energy levels been?"
    * *Bad:* "You must be feeling really tired, right?"
4. **Experiential Frequency:** Avoid math/survey terms like "50%" or "scale." Ask how much the symptom "shows up" or "bothers" them in a typical week.
5. **Impact-First:** Frame questions around their life. Instead of "Is it daily?", ask if it feels like a "constant weight" or if they get "breaks" during the week.
6. **Transitional Logic:** Use bridging phrases (e.g., "Moving on from sleep...") so the shift feels connected, not abrupt.
7. **Avoid the Loop:** If this is a follow-up, do not restart. Use what they just said: "Since it's happening nearly every day, does it usually..."
8. **The Human Bridge:** Use "Reflective Inquiry." Ask them to describe the "space" the symptom takes up in their life (e.g., "When you look back at your week, is this something that's heavy most days, or just a few?").
9. **Lexical Mirroring :** Catch "distress cues" (e.g., "overwhelmed," "hopeless") and mirror them back to validate their specific experience.
10. **Contextual Synthesis:** Instead of asking a "cold" question, bridge from a previous answer (e.g., "Since you mentioned feeling quite low lately, I'm wondering if your interest in hobbies has changed too?").
     
**ADAPTIVE SYNTHESIS & MIRRORING:**
1. **Lexical Mirroring:** Identify "distress cues" (e.g., "overwhelmed," "hopeless") in the {history_str} and mirror them back to validate their specific experience.
2. **Contextual Anchoring:** Bridge from a previous answer (e.g., "Since you mentioned feeling quite low lately, I'm wondering if...") to make the inquiry feel connected, not isolated.
3. **Diagnostic Intent:** If the patient is vague, briefly "explain" the clinical link (e.g., "I'm asking about your energy because it often ties into the mood changes you mentioned earlier.").
     
**RAPPORT BUILDING STRATEGIES:**
1.  **Validation:** Before asking a follow-up, briefly acknowledge the patient's previous answer (e.g., "I hear how difficult that's been for you...").
2.  **Clarity:** If you must clarify a timeframe or frequency, explain *why* it helps you understand them better.
3.  **Avoid Interrogation:** Do not fire questions like a machine. Maintain a warm, clinical tone.

**DYNAMIC RAPPORT ADJUSTMENT (Current Level: {rapport_level}/5):**
- **IF RAPPORT IS LOW (1-2): "TRUST REPAIR MODE"**
  * **Tone:** Soft, extremely patient, and humble.
  * **Strategy:** Use "Gentle Normalization." (e.g., "It's completely normal to find these questions a bit intrusive; I'm only asking so I can support you better.")
  * **Action:** Increase validation by Prioritize emotional validation over data precision. If the patient is being vague, acknowledge their discomfort or the difficulty of the topic before asking for a number. Avoid "Why" questions, which can sound accusatory; use "How" or "What" instead.
     
- **IF RAPPORT IS MEDIUM (3): "BUILDING MODE"**
  * **Tone:** Professional, warm, and curious.
  * **Strategy:** Use "Collaborative Framing." (e.g., "Let's look at this together...")
  * **Action:** Use the "Sandwich Technique." Validate their feeling, ask the specific clinical question, and briefly explain how that detail helps you understand their struggle better.
    
- **IF RAPPORT IS HIGH (4-5): "MAINTENANCE MODE"**
  * **Tone:** Direct, deep, and vulnerable.
  * **Strategy:** Use "Advanced Empathy." (e.g., "I can see how deeply that belief has impacted you...")
  * **Action:** Transition to "Advanced Empathy." You can ask for data directly because trust is established. Focus on how symptoms connect to their deeper beliefs or life events.

**FINAL STRICTURES (CRITICAL):**
1. **1-2 SENTENCE LIMIT:** You must provide your assessment in no more than two sentences. Use a semicolon if needed, but keep it short.
2. **VALIDATE THEN INQUIRE:** Start with a brief empathy marker (e.g., "I'm sorry to hear that..."), then execute the directive: "{instruction}".
3. **NO ROBOTIC LISTS:** Speak like a human, not a survey.

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

**CLINICAL EVALUATION THOUGHT PROCESS (Zero-Shot CoT):**
Before deciding the final status, think step-by-step:
1. First, analyze the 'Timeframe': Does the evidence ground the symptom in the last 14 days?
2. Second, evaluate 'Vagueness': Can I mathematically distinguish the frequency (Score 1 vs. 2) without guessing? 
3. Third, assess 'Relevance': Is the patient answering the question, or are they deflecting/externalizing?
4. Fourth, synthesize these checks to determine if the data is scorable.

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
     
**4. COMPOSITE CHECK: Even if steps 1-3 individually pass, can you assign"
"   a specific score (0-3)? If the combination of answers is still ambiguous, FAIL."

**OUTPUT FORMAT:**
Return strict JSON using **DOUBLE QUOTES**. 
Your JSON must include:
1. "status": "COMPLETE" or "INCOMPLETE"
2. "reasoning": "A step-by-step clinical evaluation of the Timeframe, Vagueness, and Relevance based on the thought process above."
3. "reason": "A brief summary for the psychologist."
4. "missing_domains": A list of failed criteria: ["timeframe", "vagueness", "relevance"].
"""),
    ("human", "Conversation History:\n{history_str}\n\nLatest Q: {question}\nLatest A: {answer}")
])

# 3. Alignment Agent
alignment_template = ChatPromptTemplate.from_messages([
    ("system", """You are the **Alignment Agent** (Expert Consistency Checker).
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

**MULTI-PERSPECTIVE ALIGNMENT ANALYSIS (PCoT Workflow):**
Before providing the final verdict, evaluate the consistency through these three clinical lenses:
1. **The Physiological Lens:** Is the current answer physically compatible with the past history (e.g., Sleep vs. Energy)?
2. **The Affective Lens:** Is the emotional tone consistent (e.g., Hopelessness vs. High Pleasure)?
3. **The Behavioral Lens:** Does the reported behavior match previous functional reports (e.g., Restlessness vs. Concentration)?
4. **Consensus Synthesis:** Synthesize these perspectives. If ANY lens shows a hard logical impossibility, the status is CONTRADICTING.
     
**REFERENCE CASE STUDIES (Logic Types of Nodes):**

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
Example:
{{ 
    "status": "CONTRADICTING", 
    "reasoning_chain": {{
        "physiological_check": "Evaluation of physical compatibility.",
        "affective_check": "Evaluation of emotional coherence.",
        "behavioral_check": "Evaluation of functional/behavioral alignment."
    }},
    "reason": "Final summary: Patient's claim of perfect focus contradicts earlier report of severe agitation." 
}}
"""),
    ("human", """**CURRENT ANSWER:**
"{answer}"

**LOGIC CHECK CONTEXT:**
{history_str}

**VERDICT:**""")
])

# 4. Navigation Agent
navigation_template = ChatPromptTemplate.from_messages([
    ("system", """You are the Navigation Control for a clinical interview.

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
    ("system", """You are an expert clinician **scoring** the PHQ-8 assessment using **Psychometric Chain of Thought (PCoT)** reasoning.

**YOUR TASK:**
Analyze the patient's complete history for the specific item below. You must logically bridge raw observations to a clinical conclusion and assign a valid clinical score (0-3).

**SCORING RUBRIC (PHQ-8 Standard):**
- **Score 0: Not at All / Negligible.** * *Threshold:* 0-1 days in 2 weeks.
  * *Rule:* Occasional occurrences that do not form a "pattern" MUST be scored 0.
  * *Logic:* If the patient sounds like they are experiencing the "normal ebbs and flows" of life, you MUST score 0.

- **Score 1: Several Days (Sub-threshold).** * *Threshold:* 2-6 days in 2 weeks (Less than half the time).
  * *Rule:* Use this only if there is a recurring pattern that is less than half the week.
  * *Logic:* Assign 1 only if the symptom is a departure from their healthy state but occupies the minority of their time.

- **Score 2: More than Half the Days.** * *Threshold:* 7-11 days in 2 weeks.
  * *Rule:* Requires the symptom to be the dominant state of the patient's week. DO NOT assign 2 unless the patient explicitly confirms the symptom happens "most days" or "the majority of the week." 
  * *Logic:* Assign 2 only if the evidence suggests the patient is "struggling more often than they are functioning normally."

- **Score 3: Nearly Every Day.** * *Threshold:* 12-14 days in 2 weeks. The symptom is constant and debilitating. 
  * *Rule:* Only for constant, daily distress. The patient describes a total loss of "good days."
  * *Logic:* High-intensity evidence where the symptom is inseparable from the patient's daily existence.
     
**CLINICAL REASONING HIERARCHY:**
1. **Default to Zero:** Every item starts at Score 0. You only move to up to Score 1 or higher if there is EXPLICIT evidence of a recurring clinical pattern.
2. **Vagueness = 0:** If the patient's frequency is vague (e.g., "sometimes," "I don't know," "maybe"), and they fail to clarify after follow-ups, you MUST assign **Score 0**. Vagueness is NOT evidence of a symptom.
3. **Intensity vs. Frequency:** Do not confuse "feeling bad" with "frequency of symptoms." A patient can feel very sad (High Intensity) only once a week (Score 0). 
   - *Rule:* If frequency is not established as "Several Days" (2+ days), the score is 0 regardless of how intense the feeling is.
4. **Normal Life Stress Filter:** Distinguish between "Clinical Depression" and "Situational Stress." If a participant is tired because of work or a PhD deadline, that is a **Score 0**. It is a logical reaction to life, not a symptom of a disorder.

**STRICT CALIBRATION OVERRIDE:** - **The Skeptic's Rule**: When in doubt between two scores, always choose the lower score. Evidence must be explicit and come from the transcript only, not inferred from emotional tone or profile.
- **Anti-Metaphor Bias:** If a patient uses a metaphor (e.g., "I'm a zombie"), treat it as a figure of speech. Do not use it as evidence of frequency unless they confirm it happens most days.
- **Independence of Items:** Just because a patient scored high on "Mood" does not mean they should score high on "Appetite." Treat every item as a completely fresh start.

**INSTRUCTION ON CONVERSATIONAL DYNAMICS:**
1. **IGNORE VOLUME OF QUESTIONS:** The number of follow-up questions is a result 
   of the interview structure, NOT the patient's severity.
2. **RESISTANCE != SEVERITY:** If a patient is vague or evasive, do NOT 
   automatically increase the score. Only score based on the actual 
   information revealed in the transcript.
3. **TRUTH OVER PERSISTENCE:** A patient who answers 'I'm fine' after 3 
   follow-ups should still be scored 0, even if the psychologist was persistent. 
   Persistence by the interviewer does not equal distress in the patient.

**THE SKEPTIC'S RULE (CRITICAL):**
1. Evidence MUST come from the transcript, not inferred from emotional tone or profile.
2. If the patient's overall mood is low, but for THIS specific item they say 
   'I'm fine', you MUST score 0.
3. DO NOT assume symptoms exist just because the patient's general mood is low.
     
**DATA SUFFICIENCY CHECK:**
Rate the evidence provided in the transcript:
- **HIGH:** Patient provided a specific frequency (e.g., "5 days a week") and timeframe.
- **MEDIUM:** Patient was descriptive but used slightly relative terms (e.g., "most of the time").
- **LOW:** Patient remained vague or evasive despite follow-ups. You are forced to "estimate" based on tone.

**EVALUATION PROTOCOL:**
1.  **Strict Criteria Application:** Apply the specific definition provided in the prompt (Hypothesis) without deviation.
2.  **Contextual Analysis:** Review the full dialogue history, including behavioral notes and logic checks.
    * *Handling Vagueness:* If behavioral notes indicate resistance or ambiguity, treat initial answers with caution.
    * *Handling Contradictions:* If the patient contradicted themselves, prioritize the **final clarified answer** over earlier statements.
    * *Handling Resolution Markers:* If a turn is marked "POST_CONTRADICTION_CORRECTION",   
      treat it as the authoritative answer. Discount all earlier contradicting statements  
      from the same item when determining the final score.             
     
**LEAST-TO-MOST DECOMPOSITION (PCoT Workflow):**
To arrive at the final score, you MUST solve these sub-problems in order:
1. **Identify Raw Evidence:** What are the specific literal statements made by the patient regarding this symptom?
2. **Filter for Validity:** Apply the 'Normal Life Stress Filter'. Is this evidence a clinical symptom or a rational reaction to external stress (e.g., PhD workload)?
3. **Determine Frequency:** Based strictly on valid evidence, how many days in the last 14 has this occurred? (Ignore vague terms; seek explicit counts). If the patient does not provide a number or a frequency word (like 'most days' or 'every day'), you must conclude that frequency is unknown and default to Score 0.
4. **Compare to Thresholds:** Map the frequency to the Scoring Rubric (0, 1, 2, or 3).
5. **Final Synthesis:** Apply the 'Strict Calibration Override'. If evidence is borderline, choose the lower score.

**OUTPUT FORMAT:**
Return strict JSON using **DOUBLE QUOTES**.
Example:
{{
    "score": 2,
    "confidence": "High",
    "data_sufficiency": "MEDIUM",
    "reasoning_chain": {{
        "step1_evidence_extraction": "Literal extraction of patient statements regarding the specific item.",
        "step2_validity_filtering": "Analysis of whether the evidence is situational stress or a clinical symptom.",
        "step3_frequency_quantification": "Determination of exact or estimated days (0-14 days) based on the text.",
        "step4_rubric_mapping": "Preliminary score selection based on the frequency threshold.",
        "step5_conservative_synthesis": "Final adjustment using the 'Skeptic's Hierarchy' to confirm the most conservative score."
    }},
    "explanation": "Score 2 assigned due to persistent concentration issues despite patient's attempt to minimize them initially.",
    "missing_info": "None"
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
            
        instruction = state.get("nav_instruction", "Start this item by bridging from their last response.")

    hist_str = "\n".join(state["history"][-6:])
    
    question = (question_template | llm | str_parser).invoke({
        "item_label": state["current_item_label"],
        "hypothesis": state["current_hypothesis"],
        "instruction": instruction,
        "history_str": hist_str,
        "rapport_level": state.get("rapport_score", 3)
    })
    
    print(f"\n👩‍⚕️ Psychologist ({state['current_item_id']}): {question}")
    
    # We add the new response to history
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

    target_domain = state.get("last_target_domain", None)
    current_rapport = state.get("rapport_score", 3)
    
    # 3. Run Simulation (KEPT EXACTLY AS IS)
    answer_text, diff_mode, diff_level, item_severity = simulate_client_answer(
        item_id=state["current_item_id"],
        item_index=state["current_item_index"],
        item_label=state["current_item_label"],
        hypothesis_text=state["current_hypothesis"],
        question_text=state["last_question"],
        client_profile=profile_obj,
        llm=llm,
        is_followup=is_followup_flag,
        target_domain=target_domain,
        current_rapport=current_rapport
    )
    
    # 4. Print Logic (UPDATED)
    # This is the visual change you wanted
    if diff_level == "level3":
        print(f"   [Simulation] 🎲 Level 3 (Hard) -> Injected: {diff_mode.upper()}")
    elif diff_level == "level2":
        print(f"   [Simulation] 🎲 Level 2 (Medium) -> Injected: {diff_mode.upper()}")

    print(f"👤 Participant: {answer_text}")
    
    # 5. Update History (KEPT)
    new_hist = state["history"] + [f"Participant: {answer_text}"]
    
    # 6. Update Transcript (KEPT)
    turn = {
        "turn_index": len(state["transcript"])+1, 
        "speaker": PARTICIPANT_NAME, 
        "text": answer_text, 
        "role": "answer",
        "item_id": state["current_item_id"],
        "resolution_marker": "POST_CONTRADICTION_CORRECTION" if state.get("last_target_domain") == "contradiction" else None
    }
    
    return {
        "last_answer": answer_text, 
        "history": new_hist, 
        "transcript": state["transcript"] + [turn],
        "current_difficulty": diff_mode,
        "current_level": diff_level,
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
    
    status = res.get("status", "COMPLETE")
    #Capture the step-by-step reasoning here
    cot_reasoning = res.get("reasoning", "No reasoning provided.")
    summary_reason = res.get("reason", "")
    missing = res.get("missing_domains", [])

    # print(f"   [Quality Check] Status: {status} | Reasoning: {cot_reasoning}")
    
    return {
        "clarification_status": status,
        "clarification_reason": f"{cot_reasoning} | Summary: {summary_reason}",
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

    status = res.get("status", "CONSISTENT")
    # Capture the Multi-Perspective Reasoning
    reasoning = res.get("reasoning_chain", {})
    # Flatten it for the state/logs
    reasoning_str = " | ".join([f"{k}: {v}" for k, v in reasoning.items()])
    summary_reason = res.get("reason", "")

    return {
        "alignment_status": status, 
        # We store the full multi-perspective reasoning here
        "alignment_reason": f"{reasoning_str} | Summary: {summary_reason}"
    }

# 5. Navigation Node
def navigation_node(state: AgentState):
    raw_missing_list = state.get("clarification_missing_domains", [])
    current_retries = state.get("followup_count", 0)
    
    # --- 2. UPDATE RESOLVED LIST (The "Checklist" Logic) ---
    resolved = state.get("resolved_domains", [])
    last_target = state.get("last_target_domain", None)
    domain_attempts = state.get("domain_attempts", {})
    
    if last_target:
        # Track how many times we've attempted this domain
        domain_attempts[last_target] = domain_attempts.get(last_target, 0) + 1
        if state["clarification_status"] == "COMPLETE":
            # Patient actually resolved it
            if last_target not in resolved:
                resolved.append(last_target)
        elif domain_attempts[last_target] >= 2:
            # Patient failed twice -> force resolve, stop asking
            if last_target not in resolved:
                resolved.append(last_target)
                print(f"   [Logic] ⚠️  Force-resolving '{last_target}' after {domain_attempts[last_target]} attempts.")
        
    # --- 3. FILTER MISSING LIST ---
    # Only look for problems we haven't fixed yet
    # If raw_missing is ['timeframe', 'vagueness'] but 'timeframe' is in resolved,
    # then effective_missing becomes just ['vagueness']
    missing_list = [d for d in raw_missing_list if d not in resolved]

    # --- CHECK ALIGNMENT STATUS ---
    alignment_status = state.get("alignment_status", "CONSISTENT")
    if alignment_status == "CONTRADICTING":
        if "contradiction" not in missing_list and "contradiction" not in resolved:
            missing_list.append("contradiction")

    # Initialize instructions
    style_guide = "Standard Follow-up" 
    selected_domain = None # Placeholder

    # 4. Ask Navigation Agent for opinion
    res = (navigation_template | llm | json_parser).invoke({
        "c_stat": state["clarification_status"], 
        "c_reas": state["clarification_reason"],
        "a_stat": state.get("alignment_status", "UNKNOWN"),  
        "a_reas": state.get("alignment_reason", "None")
    })
    
    proposed_action = res.get("next_action", "NEXT_ITEM")
    base_instruction = res.get("instruction", "")

    # 5. DECISION LOGIC
    # CASE A: Max Retries Hit -> Force Next Item
    if proposed_action == "FOLLOW_UP" and current_retries >= 3:
        print(f"   [Logic] 🛑 MAX RETRIES ({current_retries}) HIT -> Forcing Next Item...")
        final_action = "NEXT_ITEM"
        final_instruction = "Move to next item."
        missing_list = [] 
        selected_domain = None
    
    # CASE B: Normal Follow-Up
    # We check the FILTERED missing_list here
    elif (proposed_action == "FOLLOW_UP" and missing_list) or ("contradiction" in missing_list):
        final_action = "FOLLOW_UP"
        
        # # Pick the Domain (Prioritizing specific ones)
        # if "contradiction" in missing_list: selected_domain = "contradiction" 
        # elif "relevance" in missing_list: selected_domain = "relevance"
        # elif "timeframe" in missing_list: selected_domain = "timeframe"
        # elif "vagueness" in missing_list: selected_domain = "vagueness"
        # else: selected_domain = missing_list[0]

        question_domain = PHQ8_QUESTION_MAPPING.get(state["current_item_label"], "INTERNAL")
        priority_order = DOMAIN_PRIORITY_BY_TYPE[question_domain]
        selected_domain = next(
            (d for d in priority_order if d in missing_list), 
            missing_list[0] if missing_list else None
        )
        
        strategies = ESCALATION_MAP.get(selected_domain, ["Ask specifically."])
        style_guide = strategies[min(current_retries, len(strategies)-1)]
        
        reason_context = state.get("alignment_reason") if selected_domain == "contradiction" else state.get("clarification_reason")

        final_instruction = f"Address the {selected_domain.upper()} issue. Context: {reason_context}. {style_guide}"
    
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
             selected_domain = None

    # 6. Print Logic Status
    if final_action == "FOLLOW_UP" and selected_domain:
        print(f"   [Logic] ⚠️  ISSUE: {selected_domain.upper()} -> Strategy: {style_guide} (Attempt {current_retries + 1}/3)...")
        print(f"           (Resolved so far: {resolved})") # Helpful debug print
        new_followup_count = current_retries + 1
    else:
        if state["current_item_id"] == "INTRO":
             print(f"   [Logic] 💬  Intro Dialogue -> Continuing...")
        elif state["current_item_id"] == "CLOSING":
             print(f"   [Logic] 🏁  CLOSING -> Ending Experiment...")
        else:
             print(f"   [Logic] ✅  COMPLETE -> Next Item...")
             
        new_followup_count = 0

    # 7. ENTAILMENT (Preserved)
    items_data = state["items_evidence"] 
    if state["current_item_id"] not in ["INTRO", "CLOSING"]:
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

    # 8. ANALYTIC BLOCK
    raw_mode = state.get("current_difficulty", "none").lower()
    current_lvl = state.get("current_level", "level1")
    
    if raw_mode in ["none", "open", "resolution"]:
        injected_flaw = "none"
    elif "+" in raw_mode:
        injected_flaw = raw_mode.replace("+", ", ")
    else:
        injected_flaw = raw_mode
    # ------------------------------------------------------------------ 

    detected_flaw = "none"
    if missing_list: 
        detected_flaw = ", ".join(missing_list)

    turn_label = "Initial" if state["followup_count"] == 0 else f"FollowUp_{state['followup_count']}"
    
    if injected_flaw == "none" and final_action == "NEXT_ITEM":
        bot_caught = "TRUE_NEGATIVE" # correct: no flaw, moved on cleanly
    elif injected_flaw != "none" and final_action == "FOLLOW_UP":
        bot_caught = "TRUE_POSITIVE" # correct: caught the injected flaw
    elif injected_flaw == "none" and final_action == "FOLLOW_UP":
        bot_caught = "FALSE_POSITIVE" # wrong: no flaw but agent asked follow-up anyway
    else:
        bot_caught = "FALSE_NEGATIVE" # wrong: flaw was injected but agent missed it

    analytic_entry = {
        "PID": "PENDING", 
        "Item": state["current_item_id"],
        "Turn": turn_label,
        "Level": current_lvl,
        "Rapport": state.get("rapport_score", 3),
        "Injected_Flaw": injected_flaw, 
        "Detected_Flaw": detected_flaw,
        "Agent_Decision": final_action, 
        "Bot_Caught_Flaw": bot_caught,
        "Agent_Score": -1, 
        "Participant_Text": state["last_answer"].replace('"', "'") 
    }

    current_analytics = state.get("analytics_records", [])
    if state["current_item_id"] not in ["INTRO", "CLOSING"]:
        current_analytics.append(analytic_entry)

    thought = {
        "item": state["current_item_id"],
        "turn": "Initial" if state["followup_count"] == 0 else f"FollowUp_{state['followup_count']}",
        "clarification_status": state["clarification_status"],
        "clarification_logic_chain": state["clarification_reason"], # <--- Captures the CoT
        "alignment_status": state["alignment_status"],
        "alignment_logic_chain": state["alignment_reason"],
        "decision": final_action,
        "instruction": final_instruction
    }

    return {
        "next_action": final_action, 
        "nav_instruction": final_instruction, 
        "agent_thoughts": state["agent_thoughts"] + [thought], # <--- Saves to the list
        "items_evidence": items_data,
        "followup_count": new_followup_count,
        "analytics_records": current_analytics,
        "domain_attempts": domain_attempts,
        "resolved_domains": resolved,  
        "last_target_domain": selected_domain
    }

def calculate_rapport_delta(item_logs, current_level, p_type):
    followup_count = sum(1 for log in item_logs if log["Agent_Decision"] == "FOLLOW_UP")
    
    if current_level == "level1" and followup_count <= 1:
        return +1   # Genuine openness
    elif current_level == "level1" and followup_count >= 3:
        return 0    # Open but confused — neutral
    elif current_level == "level2" and followup_count <= 1:
        return 0    # Guarded but cooperative
    elif current_level == "level3":
        return -1   # Resistant regardless of speed
    elif followup_count >= 3:
        return -1   # Struggled regardless of level
    return 0

# 6. Transition Node (FIXED: Properly loads next item)
def transition_node(state: AgentState):
    # Retrieve current state info
    current_id = state["current_item_id"]
    current_idx = state["current_item_index"]
    # Get the rapport we started this turn with
    current_rapport = state.get("rapport_score", 3)

    # -------------------------------------------------------
    # 1. RAPPORT CALCULATION (The Dynamic Shift)
    # -------------------------------------------------------
    # -------------------------------------------------------
    # 1. RAPPORT CALCULATION (The Dynamic Shift)
    # -------------------------------------------------------
    if current_id == "INTRO":
        # Persona-based Intro: Internalizers like small talk, Externalizers don't.
        profile_obj = json.loads(state["participant_profile"])
        p_type = classify_profile_type(profile_obj)
        
        if p_type == "INTERNALIZER":
            new_rapport = min(5, current_rapport + 1)
            print(f"   [Rapport] 📈 Intro building trust with Internalizer ({new_rapport}/5)")
        else:
            new_rapport = current_rapport 
            print(f"   [Rapport] ↔️  Externalizer remains guarded during Intro ({new_rapport}/5)")
            
    elif current_id == "CLOSING":
        new_rapport = current_rapport
    else:
        profile_obj = json.loads(state["participant_profile"])
        p_type = classify_profile_type(profile_obj)
        current_level = state.get("current_level", "level1")

        updated_analytics = state.get("analytics_records", [])
        item_logs = [r for r in updated_analytics if r["Item"] == current_id]

        delta = calculate_rapport_delta(item_logs, current_level, p_type)
        new_rapport = max(1, min(5, current_rapport + delta))

        # Descriptive print based on delta
        if delta > 0:
            print(f"   [Rapport] 📈 Trust Increased ({new_rapport}/5) - Genuine openness.")
        elif delta == 0:
            print(f"   [Rapport] ↔️  Trust Stable ({new_rapport}/5) - Guarded but cooperative.")
        else:
            print(f"   [Rapport] 📉 Trust Decreased ({new_rapport}/5) - Resistant or struggled.")
    
    # -------------------------------------------------------
    # CASE A: INTRO PHASE (Loop 3 times)
    # -------------------------------------------------------
    if current_id == "INTRO":
        current_count = state.get("intro_turn_count", 0) + 1
        if current_count < 3:
            return {
                "current_item_index": 0,
                "intro_turn_count": current_count,
                "nav_instruction": "Ask a polite follow-up or general icebreaker.",
                "followup_count": 0,
                "rapport_score": new_rapport,
                "symptom_summaries": state.get("symptom_summaries", []),
                "resolved_domains": [], 
                "domain_attempts": {},      
                "last_target_domain": None    
            }
        else:
            next_item = PHQ8_HYPOTHESES[0]
            return {
                "current_item_index": 1,
                "current_item_id": next_item["item_id"],
                "current_item_label": next_item["label"],
                "current_hypothesis": next_item["text"],
                "intro_turn_count": current_count,
                "nav_instruction": "Transition to clinical items.",
                "followup_count": 0,
                "rapport_score": new_rapport,
                "symptom_summaries": state.get("symptom_summaries", []),
                "resolved_domains": [],
                "domain_attempts": {},     
                "last_target_domain": None    
            }

    # -------------------------------------------------------
    # CASE B: CLOSING PHASE (End the Interview)
    # -------------------------------------------------------
    if current_id == "CLOSING":
        return {
            "current_item_index": 10, # Move to Batch Scoring
            "nav_instruction": "End experiment.",
            "followup_count": 0,
            "domain_attempts": {},
            "resolved_domains": [],        
            "last_target_domain": None,  
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
            "current_item_id": "CLOSING",          
            "current_item_label": "Closing",
            "current_hypothesis": "End the conversation politely.",
            "nav_instruction": "Wrap up the interview.",
            "followup_count": 0,
            "rapport_score": new_rapport,
            "symptom_summaries": current_summaries,
            "resolved_domains": [],  
            "domain_attempts": {},     
            "last_target_domain": None   
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
            "rapport_score": new_rapport,
            "resolved_domains": [],
            "domain_attempts": {},
            "symptom_summaries": current_summaries
        }
    
# 7. Batch Scoring Node
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
            marker = t.get("resolution_marker")
            if marker:
                history_text += f"[{marker}]\n"
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
                f"- Technical Note: {s['Total_Followups']} follow-ups were triggered by the experimental protocol, not necessarily by patient distress."
            )

        # 3. Invoke LLM
        res = (scoring_template | llm | json_parser).invoke({
            "item_label": item_label,
            "hypothesis": hypothesis,
            "history_str": history_text + "\n" + symptom_context
        })
        
        score = res.get("score", 0)
        sufficiency = res.get("data_sufficiency", "UNKNOWN")
        explanation = res.get("explanation", "")
        reasoning = res.get("reasoning_chain", {})
        
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
            "Score": score,
            "Sufficiency": sufficiency
        })
        scoring_explanations.append({
            "item_id": item_id, 
            "score": score, 
            "explanation": explanation,
            "reasoning_chain": reasoning
        })

    # Return EVERYTHING (Scores + Updated Analytics)
    return {
        "final_scores": final_scores,
        "scoring_explanations": scoring_explanations,
        "analytics_records": updated_analytics # <--- Saves the updated list
    }

def get_severity(score_value):
    """
    Simple translator: takes a number, returns a string category.
    """
    if score_value == 0: return "No Depression"
    elif score_value <= 4: return "Minimal Depression"
    elif score_value <= 9: return "Mild Depression"
    elif score_value <= 14: return "Moderate Depression"
    elif score_value <= 19: return "Moderately Severe Depression"
    else: return "Severe Depression"

# ================= GRAPH LOGIC =================
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

    # --- FAN-OUT: Trigger both nodes at once ---
    workflow.add_edge("participant_node", "clarification_node")
    workflow.add_edge("clarification_node", "alignment_node")

    #--- FAN-IN: Wait for both to finish before moving to Navigation ---
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

        # --- NEW INITIALIZATION ---
        "domain_attempts": {},
        "resolved_domains": [],
        "last_target_domain": None,

        # --- STANDARD FIELDS ---
        "items_evidence": items_init,
        "final_scores": [],
        "scoring_explanations": [],
        "agent_thoughts": [],
        "clarification_missing_domains": [],
        "nav_instruction": "Start introduction.",
        "followup_count": 0,

        # --- BUILD RAPPORT ---
        "rapport_score": 3
    }
    # ---------------------------------------------------------

    print(f"\n🚀 Multi-Agent System (PID {args.pid}) Started...")
    app = build_graph()
    
    # INCREASE RECURSION LIMIT to 150
    final_state = app.invoke(state, {"recursion_limit": 500})

    # SAVE FILES
    # =========================================================
    # 1. Setup Directories
    base_dir = "multi-agent-system-baseline"
    
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
    severity_cat = get_severity(total_score)
    csv_data = final_state["final_scores"] + [
        {"Item ID": "TOTAL", "Item Label": "PHQ-8 SUM", "Score": total_score},
        {"Item ID": "DIAGNOSIS", "Item Label": "Severity Category", "Score": severity_cat}
    ]
    csv_path = os.path.join(dirs["sc"], f"Scores_{args.pid}.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["Item ID", "Item Label", "Score", "Sufficiency"]
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
        keys = ["PID", "Item", "Turn", "Level", "Rapport", "Injected_Flaw", "Detected_Flaw", "Agent_Decision", "Bot_Caught_Flaw", "Agent_Score", "Participant_Text"]
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

    print(f"\n✅ Done. Total PHQ-8 Score: {total_score} ({severity_cat} Depression).")
    print(f"✅ All files saved to: {os.path.abspath(base_dir)}")

if __name__ == "__main__":
    main()
