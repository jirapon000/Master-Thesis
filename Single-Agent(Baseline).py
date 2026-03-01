# python3 "Single-Agent(Baseline).py" --pid (participant_id)

import os
import json
import argparse
import datetime
import random
import re
import csv
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= CONFIGURATION =================
AI_NAME = "LLM Psychologist (Single Agent Baseline)"
PARTICIPANT_NAME = "Participant"
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ================= DATA MAPPING =================
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
SHORT_KEYS = {
    "I1": "anhedonia", "I2": "depressed_mood", "I3": "sleep", "I4": "fatigue",
    "I5": "appetite", "I6": "self_worth", "I7": "concentration", "I8": "psychomotor"
}

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
# ================= PATIENT ROLEPLAY PROMPT =================
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
# ================= PROMPTS =================
intro_template = ChatPromptTemplate.from_messages([
    ("system", """You are a warm, empathetic licensed psychologist.
Goal: Establish professional rapport.
- Greet the participant by acknowledging their presence.
- Ask a single open-ended question about their current wellbeing.
- Style: Professional, grounded, and concise. 
- Constraint: Maximum 2 sentences. No generic AI platitudes (e.g., 'I am here to help')."""),
    ("human", "Start the session.")
])

rapport_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist building trust.
Goal: Validate the participant's response and maintain a supportive connection.
- Acknowledge their specific emotion (positive or negative).
- Provide a brief 'reflection' of what they said to show you are listening.
- **CRITICAL:** Do NOT start the PHQ-8 items yet. Do NOT offer therapy/advice.
- Constraint: 1-2 sentences maximum."""),
    ("human", "{last_response}")
])

transition_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist transitioning to the PHQ-8 assessment.
1. Acknowledge the participant's last comment.
2. State: 'I’d like to ask you some specific questions about your health and mood over the last 2 weeks to get a better picture of how you've been feeling.'
3. **MANDATORY:** End with a statement of readiness. Do NOT ask 'Are you ready?' or any other question.
"""),
    ("human", "The user just said: '{last_response}'. Generate the transition statement.")
])

probe_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist conducting a PHQ-8 assessment.
Your goal: Ask an open-ended question to evaluate '{item_label}'.
Clinical Criteria: "{hypothesis_text}"

Guidelines:
1. **Timeframe Anchor:** You MUST include the phrase 'over the last 2 weeks' in the question.
2. **Open-Ended:** Avoid 'Yes/No' questions. Start with 'How has...', 'Tell me about...', or 'In what way...'.
3. **Clinical Translation:** Use the '{hypothesis_text}' but translate it into everyday language. 
   - *Example for Appetite:* 'How has your interest in food or your eating habits been lately?'
4. **Format:** Only ask ONE question. No multi-part questions."""),
    ("human", "Generate the question.")
])

# === STRICT PROMPT: ENSURES AGENT GIVES 0-3 ===
# === IMPROVED SCORING PROMPT: FIXES ZERO-BIAS ===
# baseline_template = ChatPromptTemplate.from_messages([
#     ("system", """You are a licensed psychologist conducting a PHQ-8 assessment.
# Current Item: {item_index}: "{item_label}" (Criteria: {hypothesis_text})

# **SCORING SCALE — Score ONLY what the patient explicitly said about THIS symptom:**
# 0 - [Not at all]: No meaningful symptom signal present.
#     - Assign 0 when: patient denied it, described normal functioning, 
#       gave a neutral answer, or spoke about this symptom without any 
#       distress or impairment.
#     - Absence of signal IS evidence of 0. You do not need explicit denial.
#     - "It's been okay", "not really an issue", "I manage fine" = 0.
# 1 - Patient mentioned this symptom exists but was mild or infrequent.
# 2 - Patient described this symptom as happening most days or causing clear daily difficulty.
# 3 - Patient described this symptom as constant, severe, or completely debilitating.

# **STRICT RULES:**
# - Score based ONLY on what was said about THIS specific symptom. 
# - General sadness or depression tone does NOT raise the score of physical symptoms.
# - "I don't know" or sighing alone = FOLLOW_UP, not a score of 1 or 2.
# - Vague answer with no symptom signal = FOLLOW_UP.
# - Vague answer with mild symptom signal = 1.
# - If patient explicitly says this symptom is normal or fine = 0, regardless of other scores.
# - Do NOT let prior item scores influence this item's score.

# **PRIOR CONTEXT RULE:**
# - If the patient mentioned this symptom incidentally in an earlier turn, count it as supporting evidence.

# **FOLLOW_UP CRITERIA — be selective, do NOT over-trigger:**
# Use FOLLOW_UP ONLY when:
# 1. The patient went completely off-topic (relevance gap).
# 2. There is a direct contradiction with a prior answer.
# 3. The timeframe is explicitly outside 2 weeks AND re-anchoring would change the score.

# Do NOT use FOLLOW_UP for:
# - Emotional distress without a day-count → infer from language intensity.

# STEP 1: SIGNAL DETECTION
# - What explicit or implicit signals appear? Apply Severity Inference Rules.
# - Note any cross-item signals from earlier in the conversation.
     
# FREQUENCY ANCHOR RULE (apply before scoring):
# - "going through the motions", "not really into it", "feels different", "I guess" = score 1 only.
# - Score 2 requires patient to explicitly say it happens MOST DAYS or causes clear daily impairment.
# - Score 3 requires patient to say EVERY DAY or CONSTANT or NEVER gets better.
# - Emotional tone alone does NOT determine the score. Frequency and impairment determine the score.

# STEP 2: ALIGNMENT AUDIT (Network Approach to Psychopathology)
# Symptoms are interconnected nodes. Use these rules to detect cross-item contradictions:

# **Type 1 - Physical Causality** (Sleep → Fatigue → Concentration):
# - CONTRADICTION ONLY: Flag if patient claims severe sleep loss BUT also claims endless energy.
# - Do NOT use this to assume fatigue must be high just because sleep was poor.
# - Each item must be scored on its OWN evidence only.

# **Type 2 - Emotional Coherence** (Depression → Anhedonia → Self-Worth):
# - Claims of "complete hopelessness" contradict reports of high pleasure or social engagement.
# - Deep self-worth issues should align with depressed mood severity.

# **Type 3 - Behavioral/Cognitive Alignment** (Psychomotor → Concentration):
# - Extreme restlessness/pacing is incompatible with sustained focus or reading.
# - Psychomotor slowing should align with fatigue and low energy reports.

# **Type 4 - Consistent (Expected Correlation)**:
# - Fatigue → difficulty concentrating is expected and consistent.
# - Anhedonia → social withdrawal is expected and consistent.

# GLOBAL ALIGNMENT RULE:
# - Network rules are for detecting impossible contradictions ONLY.
# - NEVER use a prior item's score to raise or lower the current item's score.
# - Every item is scored solely on what the patient said about THAT symptom.
# - If current answer CONTRADICTS a prior item using the above logic → set detected_missing_domain 
#   to "misalignment" and trigger FOLLOW_UP.
# - If current answer is CONSISTENT with prior items → note it and continue scoring.
# - If no prior items exist yet → skip this step.

# STEP 3: GAP ANALYSIS
# - Run this step REGARDLESS of whether the score matches the anchor.
# - Even if the score is clear, still check:
#   1. Did the patient use vague frequency language? → detected_missing_domain: "vagueness"
#   2. Did the patient talk outside the 2-week window? → detected_missing_domain: "timeframe"  
#   3. Did the patient go off-topic? → detected_missing_domain: "relevance"
#   4. Did the patient contradict a prior answer? → detected_missing_domain: "misalignment"
# - If ANY of the above is true → set detected_missing_domain accordingly AND trigger FOLLOW_UP.
# - If NONE → set detected_missing_domain: "none" and proceed to STEP 4.

# STEP 4: FINAL DECISION
# - If detected_missing_domain is NOT "none" → decision MUST be FOLLOW_UP.
# - If detected_missing_domain is "none" → decision is NEXT_ITEM.
# - Score MUST be a raw integer: 0, 1, 2, or 3. No words. No explanation. Just the digit.
# - CALIBRATION RULE: Use the [CLINICAL NOTE] severity as your scoring anchor:
#   - 'Absent'    → expect score 0. Only go higher if patient explicitly contradicts this.
#   - 'Mild'      → expect score 1. Only go higher if patient describes clear daily impairment.
#   - 'Moderate'  → expect score 2. Only go lower if patient explicitly denies the symptom.
#   - 'Severe'    → expect score 3. Only go lower if patient explicitly denies the symptom.
#   - 'Uncertain' → score purely from conversation evidence, no anchor.

# OUTPUT FORMAT (JSON ONLY):
# {{
#   "thought": "Step 1: [Signal + Inference] | Step 2: [Alignment] | Step 3: [Gap type or 'vagueness only - scoring directly']",
#   "decision": "NEXT_ITEM" or "FOLLOW_UP",
#   "score": MUST be a raw integer only: 0, 1, 2, or 3. No words. No explanation. Just the digit,
#   "detected_missing_domain": "none", "vagueness", "timeframe", "relevance", or "misalignment",
#   "question": "If FOLLOW_UP, a single targeted clarification question."
# }}
# """),
#     ("human", "Conversation History:\n{history}")
# ])

baseline_template = ChatPromptTemplate.from_messages([
    ("system", """You are a clinical psychologist scoring a PHQ-8 assessment.

[CLINICAL NOTE will appear at the top of the history. Use it as your scoring anchor.]

SCORING (0-3):
0 = No signal or explicit denial
1 = Mild/infrequent signal  
2 = Most days / clear daily impact
3 = Every day / constant / debilitating

STEP 1 - YOU MUST CHECK FOR GAPS BEFORE ANYTHING ELSE:
Read the patient's LAST answer only. Ask yourself:
- Did they say "sometimes", "maybe", "I guess", "kind of"? → vagueness → FOLLOW_UP
- Did they talk about past/future not last 2 weeks? → timeframe → FOLLOW_UP  
- Did they go off-topic? → relevance → FOLLOW_UP
- Does this contradict something earlier? → misalignment → FOLLOW_UP
If ANY above is true → set decision to FOLLOW_UP immediately, skip scoring.
Only if ALL above are false → proceed to STEP 2.

STEP 2 - SCORE (only if no gaps):
- Use the [CLINICAL NOTE] severity as anchor:
  Absent=0, Mild=1, Moderate=2, Severe=3, Uncertain=use evidence
- Only deviate from anchor if patient explicitly contradicts it
- Emotional tone alone does NOT determine score
- Score what was said about THIS symptom only

OUTPUT JSON:
{{
  "thought": "Gap check: [what you found] | Score reasoning: [why this score]",
  "decision": "NEXT_ITEM" or "FOLLOW_UP",
  "score": 0, 1, 2, or 3,
  "detected_missing_domain": "none", "vagueness", "timeframe", "relevance", or "misalignment",
  "question": "Single follow-up question if FOLLOW_UP, else empty string"
}}
"""),
    ("human", "Conversation History:\n{history}")
])

# ================= HELPERS =================
def build_llm(model_name: str):
    return ChatOpenAI(model=model_name, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)

def load_client_profile(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def get_style_snippet(client_profile: Dict[str, Any]) -> Dict[str, Any]:
    persona = client_profile.get("persona", {})
    return {
        "demographics": persona.get("demographics", {}),
        "interaction_style": persona.get("interaction_style", {})
    }

def parse_score(value):
    """
    Reads the Agent's output. Understands integers and clinical words.
    """
    if value is None: return None
    if isinstance(value, int): return min(max(value, 0), 3)
    
    val_str = str(value).lower().strip()
    match = re.search(r'\b[0-3]\b', val_str)
    if match: return int(match.group())
    
    return None

def simulate_rapport_answer(question_text: str, client_profile: Dict[str, Any], llm: ChatOpenAI) -> str:
    try:
        profile_snippet = get_style_snippet(client_profile)
        prompt = f"Roleplay Participant. Profile: {json.dumps(profile_snippet)}. Question: {question_text}. Answer nicely/briefly."
        return llm.invoke(prompt).content.strip()
    except:
        return "I'm doing okay."

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
        # is_mismatch = (participant_type == "INTERNALIZER" and question_domain == "EXTERNAL") or \
        #               (participant_type == "EXTERNALIZER" and question_domain == "INTERNAL")
        
        # 1. TIER SELECTION BASED ON RAPPORT
        if current_rapport <= 2:
            # Low Trust: Always stay resistant if there is a mismatch
            selected_tier = "level3" 
            # if is_mismatch else "level2"
            
        elif current_rapport == 3:
            # Medium Trust: Mismatch leads to guardedness, Match leads to openness
            selected_tier = "level2" 
            # if is_mismatch else "level1"
            
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
            mode_label = "NONE"
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

    # Build per-item severity from profile context since symptom_evidence is a single flat value
    item_severity_map = {
        "I1": client_profile.get("Symptom", {}).get("anhedonia", general_evidence),
        "I2": client_profile.get("Symptom", {}).get("depressed_mood", general_evidence),
        "I3": client_profile.get("Symptom", {}).get("sleep", general_evidence),
        "I4": client_profile.get("Symptom", {}).get("fatigue", general_evidence),
        "I5": client_profile.get("Symptom", {}).get("appetite", general_evidence),
        "I6": client_profile.get("Symptom", {}).get("self_worth", general_evidence),
        "I7": client_profile.get("Symptom", {}).get("concentration", general_evidence),
        "I8": client_profile.get("Symptom", {}).get("psychomotor", general_evidence),
    }
    current_item_severity = item_severity_map.get(item_id, general_evidence)

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
            f"5. If severity is 'Uncertain' → treat as Mild. Use vague, hedged language. "
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
        str_parser = StrOutputParser()
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
        return ("I'm not sure.", "NONE", "level1", "Uncertain")
    
# ================= MAIN LOGIC =================
def run_session(llm, client_profile, pid):
    print(f"\n{AI_NAME} Started for PID {pid}...\n")
    
    # --- SETUP DIRECTORIES ---
    base_folder = "single-agent-baseline"
    dirs = {
        "evidence": os.path.join(base_folder, "Evidence"),
        "transcript": os.path.join(base_folder, "Transcript"),
        "thoughts": os.path.join(base_folder, "Agent_Thoughts"),
        "scores": os.path.join(base_folder, "Scores"),
        "explanations": os.path.join(base_folder, "Scoring_Explanations"),
        "analysis": os.path.join(base_folder, "Analysis_Metrics")
    }
    for d in dirs.values(): os.makedirs(d, exist_ok=True)

    # Parsers
    str_parser = StrOutputParser()
    json_parser = JsonOutputParser()
    
    # Chains
    intro_chain = intro_template | llm | str_parser
    rapport_chain = rapport_template | llm | str_parser
    transition_chain = transition_template | llm | str_parser
    probe_chain = probe_template | llm | str_parser
    assessment_chain = baseline_template | llm | json_parser

    # Logs
    transcript = []
    agent_thoughts = []
    final_scores = []
    scoring_explanations = []
    evidence_log = {} 
    analysis_log = [] 

    global_turn_index = 0
    total_score = 0
    current_rapport = 3

    # --- RAPPORT ---
    try:
        # No name variable needed here
        intro_text = intro_chain.invoke({}) 
        print(f"\n{AI_NAME}: {intro_text}")
        global_turn_index += 1
        transcript.append({"turn_index": global_turn_index, "speaker": AI_NAME, "role": "greeting", "text": intro_text})

        part_reply, _, _, _ = simulate_client_answer(
            item_id="INTRO",
            item_index=0,
            item_label="Rapport",
            hypothesis_text="Establish professional rapport",
            question_text=intro_text,
            client_profile=client_profile,
            llm=llm,
            is_followup=False,
            current_rapport=current_rapport
        )

        print(f"{PARTICIPANT_NAME}: {part_reply}")
        global_turn_index += 1
        transcript.append({"turn_index": global_turn_index, "speaker": PARTICIPANT_NAME, "role": "greeting_reply", "text": part_reply})
        
        # TRANSITION (Pass the last response so it makes sense!)
        trans_text = transition_chain.invoke({"last_response": part_reply})
        
    except Exception as e:
        print(f"Error during Intro/Rapport: {e}")
        trans_text = "Let's move on to some specific questions." # Fallback

    # --- PHQ-8 LOOP ---
    PHQ8_HYPOTHESES = [
        {"item_id": "I1", "label": "Anhedonia", "text": "Little interest or pleasure in doing things."},
        {"item_id": "I2", "label": "Depressed mood", "text": "Feeling depressed, down, or hopeless."},
        {"item_id": "I3", "label": "Sleep problems", "text": "Trouble falling/staying asleep, or sleeping too much."},
        {"item_id": "I4", "label": "Fatigue", "text": "Feeling tired or having little energy."},
        {"item_id": "I5", "label": "Appetite change", "text": "Poor appetite or overeating."},
        {"item_id": "I6", "label": "Self-worth", "text": "Feeling bad about yourself or that you are a failure."},
        {"item_id": "I7", "label": "Concentration", "text": "Trouble concentrating on things."},
        {"item_id": "I8", "label": "Psychomotor", "text": "Moving or speaking so slowly (or being fidgety/restless)."}
    ]
    TOPIC_TRANSITIONS = ["", "Thanks for sharing. ", "Got it. ", "Thanks. ", "I appreciate your honesty. ", "That helps. ", "Okay. ", "Thanks for letting me know. "]
    
    for h in PHQ8_HYPOTHESES:
        evidence_log[f"Item {h['item_id'][1]}"] = {"label": h["label"], "item_id": h["item_id"], "supporting": []}

    # --- PHQ-8 LOOP (FIXED) ---
    for idx, h in enumerate(PHQ8_HYPOTHESES):
        shown = idx + 1
        item_id = h["item_id"]
        label = h["label"]
        hyp_text = h["text"]
        
        try:
            # 1. GENERATE THE PSYCHOLOGIST'S QUESTION
            raw_probe = probe_chain.invoke({"item_label": label, "hypothesis_text": hyp_text})
            
            if idx == 0:
                opener = f"{trans_text} {raw_probe}"
            else:
                opener = f"{TOPIC_TRANSITIONS[idx]}{raw_probe}"

            print(f"{AI_NAME}: {opener}")
            global_turn_index += 1
            transcript.append({"turn_index": global_turn_index, "speaker": AI_NAME, "role": "initial_question", "item_index": shown, "text": opener})

            # 2. PARTICIPANT ANSWERS
            reply_text, flaw_injected, level_selected, current_item_severity = simulate_client_answer(
                item_id=item_id, item_index=shown, item_label=label,
                hypothesis_text=hyp_text, question_text=opener,
                client_profile=client_profile, llm=llm,
                is_followup=False, current_rapport=current_rapport
            )

            print(f"{PARTICIPANT_NAME} (Injected: {flaw_injected}): {reply_text}")
            print(f"   [DEBUG Profile Evidence] {current_item_severity}")
            global_turn_index += 1
            transcript.append({"turn_index": global_turn_index, "speaker": PARTICIPANT_NAME, "role": "base_answer", "text": reply_text})
            evidence_log[f"Item {shown}"]["supporting"].append({
                "evidence_supporting_id": f"{item_id}_E1", "text": reply_text, "followup_asked": False
            })

            # 3. BUILD HISTORY
            history_str = "\n".join([
                f"{t['speaker']}: {t['text']}"
                for t in transcript
                if t['speaker'] in [AI_NAME, PARTICIPANT_NAME]
            ])
            clinical_note = f"[CLINICAL NOTE: Profile severity for this item is '{current_item_severity}']\n\n"

            # 4. SCORING
            decision_data = assessment_chain.invoke({
                "history": clinical_note + history_str
            })
            decision = decision_data.get("decision", "NEXT_ITEM")
            detected_flaw = decision_data.get("detected_missing_domain", "none").lower().strip()
            score = parse_score(decision_data.get("score"))
            thought = decision_data.get("thought", "")

            if score is None:
                decision = "FOLLOW_UP"
                decision_data["question"] = "Could you tell me more specifically how often you've been experiencing this?"

            # 5. LOG INITIAL TURN
            bot_caught_it = False
            if flaw_injected == "NONE":
                if decision == "NEXT_ITEM": bot_caught_it = True
            elif "+" in flaw_injected:
                active_flaws = flaw_injected.lower().split("+")
                if decision == "FOLLOW_UP" and detected_flaw in active_flaws:
                    bot_caught_it = True
            else:
                if decision == "FOLLOW_UP" and detected_flaw == flaw_injected.lower():
                    bot_caught_it = True

            analysis_log.append({
                "PID": pid, "Item": item_id, "Turn": "Initial",
                "Rapport_Score": current_rapport, "Level": level_selected,
                "Injected_Flaw": flaw_injected, "Detected_Flaw": detected_flaw,
                "Agent_Decision": decision, "Bot_Caught_Flaw": bot_caught_it,
                "Agent_Score": score if score is not None else -1,
                "Participant_Text": reply_text
            })

            transcript.append({"turn_index": global_turn_index, "speaker": "Agent_Internal_Monologue", "text": thought})
            agent_thoughts.append({"item_index": shown, "turn_index": global_turn_index, "decision": decision, "score": score, "thought": thought})

            # 6. FOLLOW-UP LOOP
            turn_count = 0
            max_turns = 3
            domain_attempt_count = {}

            while decision == "FOLLOW_UP" and turn_count < max_turns:
                turn_count += 1
                evidence_log[f"Item {shown}"]["supporting"][-1]["followup_asked"] = True
                followup_q = decision_data.get("question", "Could you clarify how often that happens?")

                print(f"{AI_NAME} (Probe {turn_count} for {detected_flaw}): {followup_q}")

                global_turn_index += 1
                transcript.append({"turn_index": global_turn_index, "speaker": AI_NAME, "role": "followup_question", "text": followup_q})

                reply_text_2, _, _, _ = simulate_client_answer(
                    item_id=item_id, item_index=shown, item_label=label,
                    hypothesis_text=hyp_text, question_text=followup_q,
                    client_profile=client_profile, llm=llm,
                    is_followup=True, target_domain=detected_flaw,
                    current_rapport=current_rapport
                )

                print(f"{PARTICIPANT_NAME}: {reply_text_2}")
                global_turn_index += 1
                transcript.append({"turn_index": global_turn_index, "speaker": PARTICIPANT_NAME, "role": "followup_answer", "text": reply_text_2})

                history_str += f"\nPsychologist: {followup_q}\nParticipant: {reply_text_2}"

                resolution_suffix = (
                    "\n\n(SYSTEM - Resolution Phase: "
                    "1. If patient has clarified, commit to NEXT_ITEM. "
                    "2. Re-evaluate score from ALL turns. "
                    "3. If follow-up revealed MORE severity, revise score UPWARD. "
                    "4. Only use FOLLOW_UP again if a NEW unresolved gap emerged.)"
                )
                decision_data = assessment_chain.invoke({
                    "history": clinical_note + history_str + resolution_suffix
                })

                decision = decision_data.get("decision", "NEXT_ITEM")
                detected_flaw = decision_data.get("detected_missing_domain", "none").lower().strip()
                domain_attempt_count[detected_flaw] = domain_attempt_count.get(detected_flaw, 0) + 1

                if domain_attempt_count.get(detected_flaw, 0) >= 2:
                    print(f"   [Domain Limit] '{detected_flaw}' asked twice. Moving on.")
                    decision = "NEXT_ITEM"

                new_score = parse_score(decision_data.get("score"))
                if new_score is not None:
                    score = new_score

                # LOG FOLLOW-UP TURN
                bot_caught_it = False
                if flaw_injected == "NONE":
                    if decision == "NEXT_ITEM": bot_caught_it = True
                elif "+" in flaw_injected:
                    active_flaws = flaw_injected.lower().split("+")
                    if decision == "FOLLOW_UP" and detected_flaw in active_flaws:
                        bot_caught_it = True
                else:
                    if decision == "FOLLOW_UP" and detected_flaw == flaw_injected.lower():
                        bot_caught_it = True

                analysis_log.append({
                    "PID": pid, "Item": item_id, "Turn": f"Follow-up {turn_count}",
                    "Rapport_Score": current_rapport, "Level": level_selected,
                    "Injected_Flaw": flaw_injected, "Detected_Flaw": detected_flaw,
                    "Agent_Decision": decision, "Bot_Caught_Flaw": bot_caught_it,
                    "Agent_Score": score if score is not None else -1,
                    "Participant_Text": reply_text_2
                })

            # 7. RAPPORT ADJUSTMENT (once per item, after resolution)
            followup_count_for_item = turn_count
            if level_selected == "level1" and followup_count_for_item <= 1:
                delta = +1
            elif level_selected == "level2" and followup_count_for_item <= 1:
                delta = +1   # ← resolved quickly despite being guarded (before was 0)
            elif level_selected == "level2" and followup_count_for_item >= 3:
                delta = -1   # ← struggled even at medium difficulty (before was 0)
            elif level_selected == "level3":
                delta = -1
            elif followup_count_for_item >= 3:
                delta = -1
            else:
                delta = 0

            current_rapport = max(1, min(5, current_rapport + delta))
            print(f"   [Rapport] Now {current_rapport}/5 (delta: {delta})")


            # 8. FINALIZE ITEM
            if decision == "FOLLOW_UP":
                print(f"   [Warning] Item {shown} timed out. Using final estimate: {score}")
            else:
                print(f"   [Scoring Success] Item {shown} resolved to Score: {score}")

            if score is not None:
                total_score += score

            print(f"  -> Decision: {decision} | Score: {score}")
            final_scores.append({"Item ID": item_id, "Item Label": label, "Score": score})
            scoring_explanations.append({"item_id": item_id, "label": label, "score": score, "explanation": thought})

            analysis_path = os.path.join(dirs["analysis"], f"Analysis_{pid}.csv")
            with open(analysis_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["PID", "Item", "Turn", "Rapport_Score", "Level", "Injected_Flaw", "Detected_Flaw",
                              "Agent_Decision", "Bot_Caught_Flaw", "Agent_Score", "Participant_Text"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(analysis_log)

        except Exception as e:
            print(f"CRITICAL ERROR on Item {item_id}: {e}")
            continue

    # --- END ---
    final_scores.append({"Item ID": "TOTAL", "Item Label": "PHQ-8 SUM", "Score": total_score})
    print(f"{AI_NAME}: Assessment Complete.")
    
    # Save Final
    with open(os.path.join(dirs["evidence"], f"Evidence_{pid}.json"), "w", encoding="utf-8") as f: json.dump(evidence_log, f, indent=2)
    with open(os.path.join(dirs["transcript"], f"Transcript_{pid}.jsonl"), "w", encoding="utf-8") as f: 
        for t in transcript: f.write(json.dumps(t) + "\n")
    with open(os.path.join(dirs["thoughts"], f"Thoughts_{pid}.jsonl"), "w", encoding="utf-8") as f:
        for t in agent_thoughts: f.write(json.dumps(t) + "\n")
    with open(os.path.join(dirs["explanations"], f"Explanations_{pid}.json"), "w", encoding="utf-8") as f: 
        json.dump(scoring_explanations, f, indent=2)
    with open(os.path.join(dirs["scores"], f"Scores_{pid}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Item ID", "Item Label", "Score"])
        writer.writeheader()
        writer.writerows(final_scores)

    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=str, required=True)
    args = parser.parse_args()
    
    try:
        client_profile = load_client_profile(f"Clean_Dataset/profiles/{args.pid}_client_profile.json")
    except:
        print("Profile not found.")
        return

    llm = build_llm(LLM_MODEL)
    run_session(llm, client_profile, args.pid)

if __name__ == "__main__":
    main()
