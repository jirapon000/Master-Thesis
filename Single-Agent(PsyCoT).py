# python3 "Single-Agent(PsyCot).py" --pid (participant_id)

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
LLM_TEMPERATURE = 0.7
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
# ================= PATIENTS ROLEPLAY PROMPT =================
patient_roleplay_template = ChatPromptTemplate.from_messages([
    ("system", """You are a participant in a clinical interview.
**STRICT ADHERENCE:** Stay in character at all times. Do not reveal you are an AI. If asked about your identity, you are the person described in the profile below.

     
**YOUR PROFILE (Ground Truth):**
{profile_json}

**SYMPTOM TRUTH:**
Status: {symptom_status} 
Severity: {severity}

**RESPONSE RULES:**
1. **Conversational Brevity:** Keep answers to 1-2 sentences. Avoid long monologues unless specifically asked to "tell a story."
2. **Organic Flow:** Speak like a person, not a textbook
3. **Implicit Disclosure:** Do not "volunteer" your symptoms or severity score.
4. **Behavioral Integrity:** You MUST strictly follow the BEHAVIORAL CONSTRAINT below. This dictates your tone, eye contact (described through words), and level of cooperation.

**CRITICAL BEHAVIORAL CONSTRAINT:**
{style_instruction}"""),
    ("human", "{question_text}")
])

# ================= PROMPTS =================
intro_template = ChatPromptTemplate.from_messages([
    ("system", """You are a warm, empathetic licensed psychologist.
Your goal is to establish immediate professional rapport.

INSTRUCTIONS:
- Greet the participant warmly and professionally.
- Ask how they are doing today.
- Constraint: Keep the response concise (1-2 sentences)."""),
    ("human", "Start the session.")
])

rapport_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist focused on building trust.
The participant has responded to your initial greeting.

YOUR TASK:
- Validate their feelings and maintain a supportive connection.
- If positive: Reflect and share that positivity.
- If negative/stressed: Provide empathetic validation (e.g., "I'm sorry to hear that," or "That sounds difficult").
- CRITICAL: Do NOT begin the clinical assessment yet. Maintain a human-centric supportive tone.
- Keep the response brief."""),
    ("human", "{last_response}")
])

transition_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist transitioning from rapport-building to formal assessment.

TASKS:
1. Briefly acknowledge the participant's last statement.
2. Clearly state that you will now ask a set of standard questions regarding their health and mood over the **last two weeks**.
3. IMPORTANT: Do NOT ask the first assessment question yet. Only provide the preparatory statement.
"""),
    ("human", "The user just said: '{last_response}'. Generate the transition statement now.")
])

probe_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist conducting a clinical assessment.
Your goal is to inquire about: '{item_label}'.
Clinical Criteria: "{hypothesis_text}"

GUIDELINES:
1. **TIMEFRAME:** You MUST anchor the question specifically to the **last 2 weeks**.
2. **TONE:** Professional, warm, and conversational. Avoid a "survey" or "robotic" feel.
3. **CLARITY:** Use simple, natural language. Replace clinical jargon with descriptive terms (e.g., use "feeling slowed down" instead of "psychomotor retardation").
4. **FORMAT:** Ask exactly ONE clear, open-ended question."""),
    ("human", "Generate the question.")
])

# === PSYCOT TEMPLATE (The Brain) ===
psycot_template = ChatPromptTemplate.from_messages([
    ("system", """You are a specialized Clinical Psychology AI conducting a structured PHQ-8 (Patient Health Questionnaire-8) assessment. Your objective is to determine a clinical score for specific depressive symptoms through empathetic but precise inquiry.
    
**Current Focus:** Item {item_index}: "{item_label}"
**Clinical Criteria:** "{hypothesis_text}"

The participant has just answered your question.

YOUR TASK:
Reason step-by-step to evaluate the participant's response and determine the next clinical action.

**STEP 1: OBSERVATION**
Analyze the participant's input. Evaluate for relevance, clarity regarding the symptom frequency, and alignment with the required two-week timeframe.

**STEP 2: CLINICAL HISTORY AUDIT (CRITICAL)**
Review the `Conversation History`.
* **Constraint:** To maintain clinical efficiency and prevent respondent fatigue, only one follow-up is permitted per item.
* **Logic:** - If you have ALREADY asked a follow-up for this specific item: You have reached the iteration limit. You MUST conclude reasoning and form a final clinical estimate. Set strategy to **NEXT_ITEM**.
    - If you have NOT yet asked a follow-up: You may proceed to request clarification if the response is vague.

**STEP 3: STRATEGY & SCORING SELECTION**
* **NEXT_ITEM:** Assign a Score (0-3). If you reached the follow-up limit or the history shows a follow-up already happened, you MUST provide an integer. NEVER return null here.
* **FOLLOW_UP:** Set Score to null ONLY if this is the FIRST time you are asking about this specific item and the answer is truly vague.

**SCORING RUBRIC:**
0 = Not at all
1 = Several days (mild)
2 = More than half the days (moderate)
3 = Nearly every day (severe)

OUTPUT FORMAT (JSON ONLY):
{{
  "step1_observation": "Detailed analysis of the participant's response.",
  "step2_history_check": "Verification of prior follow-ups for this item.",
  "step3_strategy": "Clinical reasoning for the chosen decision and score.",
  "decision": "NEXT_ITEM" or "FOLLOW_UP",
  "score": (Integer 0-3. Use null ONLY if decision is FOLLOW_UP), 
  "question": "The professional response or transition to be delivered to the participant."
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
    if isinstance(value, int): return value
    
    val_str = str(value).lower().strip()
    match = re.search(r'[0-3]', val_str)
    if match: return int(match.group())
        
    if "none" in val_str or "not at all" in val_str: return 0
    if "mild" in val_str or "several" in val_str: return 1
    if "moderate" in val_str or "half" in val_str: return 2
    if "severe" in val_str or "nearly" in val_str: return 3
    
    return None

def simulate_rapport_answer(question_text: str, client_profile: Dict[str, Any], llm: ChatOpenAI) -> str:
    try:
        profile_snippet = get_style_snippet(client_profile)
        prompt = f"Roleplay Participant. Profile: {json.dumps(profile_snippet)}. Question: {question_text}. Answer nicely/briefly."
        return llm.invoke(prompt).content.strip()
    except:
        return "I'm doing okay."

def simulate_client_answer(
    item_id: str,
    item_index: int,
    item_label: str,
    question_text: str,
    client_profile: Dict[str, Any],
    llm: ChatOpenAI,
    is_followup: bool = False,
):
    """
    Returns: (response_text, flaw_injected)
    Uses a 3-tier difficulty system to test the PsyCoT Agent's diagnostic rigor.
    """
    # --- 1. DYNAMIC DIFFICULTY SELECTION ---
    DIFFICULTY_MODES = ["level1", "level2", "level3"]
    WEIGHTS = [0.3, 0.5, 0.2] 
    
    if is_followup:
        selected_tier = random.choices(["level1", "level2"], weights=[0.6, 0.4], k=1)[0]
    else:
        selected_tier = random.choices(DIFFICULTY_MODES, weights=WEIGHTS, k=1)[0]

    diff_instruction = ""
    mode_label = "none"

    # --- 2. DEFINE INSTRUCTIONS & MODE LABEL ---
    if selected_tier == "level3":
        flaw_types = ["vagueness", "timeframe", "relevance"]
        specific_flaw = random.choice(flaw_types)
        mode_label = specific_flaw 
        
        if specific_flaw == "vagueness":
            diff_instruction = (
                "**Goal: Be Vague.**\n- Use non-committal words like 'sometimes', 'maybe', 'sort of'.\n"
                "- Do NOT specify if it happens 1 day or 7 days a week.\n"
                "- Make it impossible to decide between Score 1 and Score 2."
            )
        elif specific_flaw == "timeframe":
            diff_instruction = (
                "**Goal: Be Unclear about Time.**\n- Talk about how you felt 'years ago' or 'childhood'.\n"
                "- Do NOT confirm if this is happening *currently* (in the last 2 weeks)."
            )
        elif specific_flaw == "relevance":
            diff_instruction = (
                "**Goal: Go Off-Topic.**\n- Ignore the specific symptom asked.\n"
                "- Talk about a tangential topic (e.g., your dog, traffic) using a keyword from the question."
            )

    elif selected_tier == "level2": 
        mode_label = "VAGUENESS" 
        diff_instruction = (
            "**Goal: Natural & Metaphorical.**\n- Use metaphors (e.g., 'I feel like a heavy blanket is on me').\n"
            "- The answer MUST be valid and answer the question, just not directly."
        )
    else: 
        mode_label = "NONE"
        diff_instruction = "**Goal: Clear & Direct.** Directly answer with clear frequency/duration."

    # --- 3. EXECUTE ROLEPLAY ---
    # 1. Prepare Data
    clinical_signals = (client_profile.get("clinical_signals", {}) or {}).get("symptoms", {})
    long_key = SYMPTOM_KEY_BY_ITEM_ID.get(item_id, "")
    symptom_block = clinical_signals.get(long_key, {})
    
    present = symptom_block.get("present", "absent")
    severity_hint = symptom_block.get("severity_hint", "none")
    profile_snippet = get_style_snippet(client_profile)

    # 2. Use the Professional Template
    # We create a small chain here or just invoke the template
    roleplay_chain = patient_roleplay_template | llm | StrOutputParser()

    try:
        response_text = roleplay_chain.invoke({
            "profile_json": json.dumps(profile_snippet, indent=2),
            "symptom_status": present,
            "severity": severity_hint,
            "style_instruction": diff_instruction,
            "question_text": question_text
        })
        # CHANGE: Add selected_tier to the return
        return response_text.strip(), mode_label, selected_tier 
    except Exception as e:
        print(f"Roleplay Error: {e}")
        # CHANGE: Add a default level here too
        return "I'm not sure how to answer that.", "none", "level1"

# ================= MAIN LOGIC =================
def run_session(llm, client_profile, pid):
    print(f"\n{AI_NAME} Started for PID {pid}...\n")
    
    # --- SETUP DIRECTORIES ---
    base_folder = "single-agent-PsyCoT"
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
    assessment_chain = psycot_template | llm | json_parser

    # Logs
    transcript = []
    agent_thoughts = []
    final_scores = []
    scoring_explanations = []
    evidence_log = {} 
    analysis_log = [] 

    global_turn_index = 0
    total_score = 0

    # --- RAPPORT ---
    try:
        # No name variable needed here
        intro_text = intro_chain.invoke({}) 
        print(f"\n{AI_NAME}: {intro_text}")
        global_turn_index += 1
        transcript.append({"turn_index": global_turn_index, "speaker": AI_NAME, "role": "greeting", "text": intro_text})

        part_reply = simulate_rapport_answer(intro_text, client_profile, llm) 
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

    # --- PHQ-8 LOOP (PsyCoT FIXED) ---
    for idx, h in enumerate(PHQ8_HYPOTHESES):
        shown = idx + 1
        item_id = h["item_id"]
        label = h["label"]
        hyp_text = h["text"]
        
        try:
            # 1. QUESTION GENERATION
            raw_probe = probe_chain.invoke({"item_label": label, "hypothesis_text": hyp_text})
            opener = f"{trans_text} {raw_probe}" if idx == 0 else f"{TOPIC_TRANSITIONS[idx]}{raw_probe}"

            print(f"{AI_NAME}: {opener}")
            global_turn_index += 1
            transcript.append({"turn_index": global_turn_index, "speaker": AI_NAME, "role": "initial_question", "item_index": shown, "text": opener})

            # 2. ANSWER GENERATION (Logic now internal to function)
            reply_text, flaw_injected, level_selected = simulate_client_answer(
                item_id, shown, label, opener, client_profile, llm, is_followup=False
            )
            print(f"{PARTICIPANT_NAME} (Injected: {flaw_injected}): {reply_text}")
            global_turn_index += 1
            transcript.append({"turn_index": global_turn_index, "speaker": PARTICIPANT_NAME, "role": "base_answer", "text": reply_text})

            evidence_log[f"Item {shown}"]["supporting"].append({"evidence_supporting_id": f"{item_id}_E1", "text": reply_text, "followup_asked": False})

            # 3. PSYCOT DECISION (PASS 1)
            history_str = f"Psychologist: {opener}\nParticipant: {reply_text}"
            decision_data = assessment_chain.invoke({"item_index": shown, "item_label": label, "hypothesis_text": hyp_text, "history": history_str})
            
            decision = decision_data.get("decision", "NEXT_ITEM")
            obs = decision_data.get('step1_observation', '')
            thought = f"Obs: {obs} | Hist: {decision_data.get('step2_history_check', '')} | Strat: {decision_data.get('step3_strategy', '')}"
            
            # IMPROVED DETECTION LOGIC for PsyCoT
            detected_flaw = "none"
            obs_lower = obs.lower()
            if "vague" in obs_lower or "ambiguous" in obs_lower: detected_flaw = "vagueness"
            elif "time" in obs_lower or "period" in obs_lower: detected_flaw = "timeframe"
            elif "relevant" in obs_lower or "off-topic" in obs_lower: detected_flaw = "relevance"
            
            score = parse_score(decision_data.get("score"))
            
            bot_caught_it = (flaw_injected == "none" and decision == "NEXT_ITEM") or \
                            (decision == "FOLLOW_UP" and detected_flaw == flaw_injected)

            analysis_log.append({
                "PID": pid, 
                "Item": item_id, 
                "Turn": "Initial",
                "Level": level_selected,  # <--- Add this line
                "Injected_Flaw": flaw_injected, 
                "Detected_Flaw": detected_flaw,
                "Agent_Decision": decision, 
                "Bot_Caught_Flaw": bot_caught_it,
                "Agent_Score": score if score is not None else -1, 
                "Participant_Text": reply_text
            })

            transcript.append({"turn_index": global_turn_index, "speaker": "Agent_Internal_Monologue", "text": thought})
            agent_thoughts.append({"item_index": shown, "turn_index": global_turn_index, "decision": decision, "score": score, "thought": thought})

            # 4. FOLLOW-UP (REVISED: Forces a score and prevents None)
            if decision == "FOLLOW_UP":
                evidence_log[f"Item {shown}"]["supporting"][-1]["followup_asked"] = True
                followup_q = decision_data.get("question", "Could you say more?")
                print(f"{AI_NAME} (Probing {detected_flaw}): {followup_q}")
                
                global_turn_index += 1
                transcript.append({"turn_index": global_turn_index, "speaker": AI_NAME, "role": "followup_question", "text": followup_q})

                # Participant responds clearly for the follow-up
                reply_text_2, _, _ = simulate_client_answer( # Added extra _,
                    item_id, shown, label, followup_q, client_profile, llm, is_followup=True
                )
                print(f"{PARTICIPANT_NAME}: {reply_text_2}")
                
                global_turn_index += 1
                transcript.append({"turn_index": global_turn_index, "speaker": PARTICIPANT_NAME, "role": "followup_answer", "text": reply_text_2})

                analysis_log.append({
                    "PID": pid, "Item": item_id, "Turn": "Follow-up", # Changed to Follow-up
                    "Level": level_selected, "Injected_Flaw": "REVEALED", 
                    "Detected_Flaw": "none", "Agent_Decision": "RESOLVING", 
                    "Bot_Caught_Flaw": True, "Agent_Score": -1, # Temporary score
                    "Participant_Text": reply_text_2
                })

                # SYSTEM: Hard pressure to pick an integer
                history_str_2 = (
                    f"{history_str}\n"
                    f"Psychologist: {followup_q}\n"
                    f"Participant: {reply_text_2}\n"
                    "\n[SYSTEM]: The follow-up limit is reached. You MUST assign a score (0-3). "
                    "If the response is still ambiguous, use your clinical judgment to pick the most likely frequency. "
                    "Do NOT return null. Output a definitive integer."
                )

                decision_data_2 = assessment_chain.invoke({
                    "item_index": shown, "item_label": label, "hypothesis_text": hyp_text, "history": history_str_2
                })
                
                # Try to parse the primary score field
                new_score = parse_score(decision_data_2.get("score"))
                
                # --- EMERGENCY EXTRACTION ---
                # If 'score' is still None/null, we look at the 'step3_strategy' text for a digit
                if new_score is None:
                    strategy_text = str(decision_data_2.get("step3_strategy", ""))
                    # Find all numbers 0-3 in the strategy text
                    potential_scores = re.findall(r'[0-3]', strategy_text)
                    if potential_scores:
                        # Take the last number mentioned (usually the final conclusion)
                        new_score = int(potential_scores[-1])
                        print(f"  [RECOVERY] Extracted score {new_score} from Agent reasoning.")
                    else:
                        # Final fallback if absolutely no number is found: default to a 'mild' score
                        new_score = 1 
                        print(f"  [RECOVERY] No score found in reasoning. Defaulting to 1.")

                score = new_score
                decision = "RESOLVED_FOLLOW_UP"
            
            # --- FINAL SCORE PROCESSING ---
            # Now 'score' is guaranteed to be an integer (0, 1, 2, or 3)
            total_score += score
            print(f"  -> Decision: {decision} | Final Score: {score}")

            final_scores.append({"Item ID": item_id, "Item Label": label, "Score": score})
            scoring_explanations.append({"item_id": item_id, "label": label, "score": score, "explanation": thought})

            # SAVE PROGRESS
            analysis_path = os.path.join(dirs["analysis"], f"Analysis_{pid}.csv")
            with open(analysis_path, "w", newline="", encoding="utf-8") as f:
                # Added "Level" to the list below
                fieldnames = ["PID", "Item", "Turn", "Level", "Injected_Flaw", "Detected_Flaw", "Agent_Decision", "Bot_Caught_Flaw", "Agent_Score", "Participant_Text"]
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
