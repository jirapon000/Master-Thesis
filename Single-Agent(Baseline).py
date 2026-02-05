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
# ================= PATIENT ROLEPLAY PROMPT =================
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
Goal: Establish immediate professional rapport.
- Greet the participant warmly.
- Ask how they are doing today.
- Constraint: Maximum 2 sentences. Avoid generic platitudes."""),
    ("human", "Start the session.")
])

rapport_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist building trust.
Goal: Validate the participant's response and maintain a supportive connection.
- If positive: Reflect their energy.
- If negative/stressed: Provide brief empathetic validation.
- **CRITICAL:** Do NOT begin assessment. Do NOT provide clinical advice.
- Constraint: Keep it brief and human-centric."""),
    ("human", "{last_response}")
])

transition_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist transitioning from rapport-building to assessment.
1. Briefly acknowledge the participant's last comment.
2. State that you will now begin a standard assessment regarding their mood and health over the **last two weeks**.
3. **MANDATORY:** Do NOT ask a question. End with a statement of readiness.
"""),
    ("human", "The user just said: '{last_response}'. Generate the transition statement.")
])

probe_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist conducting a clinical assessment.
Your goal: Ask a supportive, open-ended question to assess '{item_label}'.
Clinical Criteria: "{hypothesis_text}"

Guidelines:
1. **Timeframe:** You MUST anchor the question to the **last 2 weeks**.
2. **Tone:** Be warm and conversational. Do not sound like a robotic survey.
3. **Clarity:** Avoid complex medical jargon. Use simple, natural language.
   - Example: Don't say "psychomotor retardation." Say "moving or speaking slowly."
4. **Format:** Ask only ONE clear question."""),
    ("human", "Generate the question.")
])

# === STRICT PROMPT: ENSURES AGENT GIVES 0-3 ===
baseline_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist conducting a PHQ-8 assessment.
Current Item: {item_index}: "{item_label}" (Criteria: {hypothesis_text})

TASK:
1.  **Analyze Response:** Evaluate for clinical clarity and direct relevance.
2.  **Identify Data Gaps (Missing Domains):**
    * **Vagueness:** Ambiguous phrasing (e.g., "I guess," "sort of") or lack of symptom intensity.
    * **Timeframe:** Reference to periods outside the mandatory **last 2 weeks**.
    * **Relevance:** Non-sequiturs, subject changes, or dodging the specific symptom.
3.  **Mandatory Scoring:** You must assign a definitive integer (0-3) regardless of clarity.
    * 0: Not at all
    * 1: Several days
    * 2: More than half the days
    * 3: Nearly every day

OUTPUT FORMAT (JSON ONLY):
{{
  "decision": "NEXT_ITEM" or "FOLLOW_UP",
  "score": (INTEGER 0, 1, 2, or 3. DO NOT RETURN NULL.),
  "detected_missing_domain": "none", "vagueness", "timeframe", or "relevance",
  "thought": "Reasoning for score/decision",
  "question": "Follow-up question if needed"
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
    Uses a 3-tier difficulty system to test the Agent's diagnostic rigor.
    """
    # --- 1. DYNAMIC DIFFICULTY SELECTION ---
    DIFFICULTY_MODES = ["level1", "level2", "level3"]
    WEIGHTS = [0.3, 0.5, 0.2] # 30% Direct, 50% Paraphrase, 20% Hard Flaws
    
    # If it's a follow-up, we usually want the client to be clearer (Level 1 or 2)
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
                "**Goal: Be Vague.** Use non-committal words like 'sometimes', 'maybe', 'sort of'. "
                "Do NOT specify frequency. Make it impossible to decide a score."
            )
        elif specific_flaw == "timeframe":
            diff_instruction = (
                "**Goal: Be Unclear about Time.** Talk about how you felt 'years ago' or 'in the past'. "
                "Do NOT confirm if this is happening in the last 2 weeks."
            )
        elif specific_flaw == "relevance":
            diff_instruction = (
                "**Goal: Go Off-Topic.** Ignore the symptom. Talk about a tangential topic "
                "(e.g., your dog, traffic) using a keyword from the question in the wrong context."
            )

    elif selected_tier == "level2": 
        mode_label = "VAGUENESS" # Paraphrasing is technically valid, not a "flaw" to be caught
        diff_instruction = (
            "**Goal: Natural & Metaphorical.** Do NOT use clinical terms. "
            "Use metaphors (e.g., 'I feel like a heavy blanket is on me'). "
            "The answer MUST be valid, just not direct."
        )
    else: 
        mode_label = "NONE"
        diff_instruction = "**Goal: Clear & Direct.** Be helpful and explicit. Answer with clear frequency."

    # --- 3. EXECUTE THE ROLEPLAY ---
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
            "style_instruction": diff_instruction, #formatted_diff
            "question_text": question_text
        })
        # CHANGE: Return selected_tier as the third item in the tuple
        return response_text.strip(), mode_label, selected_tier 
    except Exception as e:
        print(f"Roleplay Error: {e}")
        return "I'm not sure how to answer that.", "none", "level1"
    
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

    # --- PHQ-8 LOOP (FIXED) ---
    for idx, h in enumerate(PHQ8_HYPOTHESES):
        shown = idx + 1
        item_id = h["item_id"]
        label = h["label"]
        hyp_text = h["text"]
        
        try:
            # 1. GENERATE THE PSYCHOLOGIST'S QUESTION
            raw_probe = probe_chain.invoke({"item_label": label, "hypothesis_text": hyp_text})
            
            # Add transition for the first item, otherwise use standard transition
            if idx == 0:
                opener = f"{trans_text} {raw_probe}"
            else:
                opener = f"{TOPIC_TRANSITIONS[idx]}{raw_probe}"

            print(f"{AI_NAME}: {opener}")
            
            global_turn_index += 1
            transcript.append({
                "turn_index": global_turn_index, 
                "speaker": AI_NAME, 
                "role": "initial_question", 
                "item_index": shown, 
                "text": opener 
            })

            # 2. GENERATE THE PARTICIPANT'S ANSWER (Using the new tiered logic)
            # Note: current_difficulty is now handled internally by simulate_client_answer
            reply_text, flaw_injected, level_selected = simulate_client_answer(
                item_id, shown, label, opener, client_profile, llm,
                is_followup=False
            )

            print(f"{PARTICIPANT_NAME} (Injected: {flaw_injected}): {reply_text}")
            global_turn_index += 1
            transcript.append({"turn_index": global_turn_index, "speaker": PARTICIPANT_NAME, "role": "base_answer", "text": reply_text})

            evidence_log[f"Item {shown}"]["supporting"].append({
                "evidence_supporting_id": f"{item_id}_E1", "text": reply_text, "followup_asked": False
            })

            # 3. PSYCHOLOGIST'S DECISION (PASS 1)
            history_str = f"Psychologist: {opener}\nParticipant: {reply_text}"
            decision_data = assessment_chain.invoke({
                "item_index": shown, "item_label": label, "hypothesis_text": hyp_text, "history": history_str
            })
            
            decision = decision_data.get("decision", "NEXT_ITEM")
            thought = decision_data.get("thought", "")
            detected_flaw = decision_data.get("detected_missing_domain", "none").lower()
            score = parse_score(decision_data.get("score"))
            
            # LOGGING: Did the Agent catch the flaw we injected?
            bot_caught_it = False
            if flaw_injected == "none":
                if decision == "NEXT_ITEM": bot_caught_it = True
            else:
                # Success if the Agent asks for a follow-up AND identifies why
                if decision == "FOLLOW_UP" and detected_flaw == flaw_injected:
                    bot_caught_it = True

            analysis_log.append({
                "PID": pid, 
                "Item": item_id, 
                "Turn": "Initial",
                "Level": level_selected,  # <--- NEW FIELD
                "Injected_Flaw": flaw_injected, 
                "Detected_Flaw": detected_flaw,
                "Agent_Decision": decision, 
                "Bot_Caught_Flaw": bot_caught_it,
                "Agent_Score": score if score is not None else -1, 
                "Participant_Text": reply_text
            })

            transcript.append({"turn_index": global_turn_index, "speaker": "Agent_Internal_Monologue", "text": thought})
            agent_thoughts.append({"item_index": shown, "turn_index": global_turn_index, "decision": decision, "score": score, "thought": thought})

            # 4. FOLLOW-UP LOGIC
            if decision == "FOLLOW_UP":
                evidence_log[f"Item {shown}"]["supporting"][-1]["followup_asked"] = True
                followup_q = decision_data.get("question", "Could you tell me a bit more about that?")
                print(f"{AI_NAME} (Probing for {detected_flaw}): {followup_q}")
                
                global_turn_index += 1
                transcript.append({"turn_index": global_turn_index, "speaker": AI_NAME, "role": "followup_question", "text": followup_q})

                # Participant responds more clearly for follow-ups
                reply_text_2, _, _ = simulate_client_answer(
                    item_id, shown, label, followup_q, client_profile, llm,
                    is_followup=True
                )
                print(f"{PARTICIPANT_NAME}: {reply_text_2}")
                global_turn_index += 1
                transcript.append({"turn_index": global_turn_index, "speaker": PARTICIPANT_NAME, "role": "followup_answer", "text": reply_text_2})

                history_str_2 = (
                    f"{history_str}\n"
                    f"Psychologist: {followup_q}\n"
                    f"Participant: {reply_text_2}\n"
                    f"(SYSTEM: Please provide the final score 0-3 now.)"
                )

                decision_data_2 = assessment_chain.invoke({
                    "item_index": shown, "item_label": label, "hypothesis_text": hyp_text, "history": history_str_2
                })
                
                new_score = parse_score(decision_data_2.get("score"))
                if new_score is not None: score = new_score
                decision = "RESOLVED_FOLLOW_UP"
            
            # FINAL SCORE TRACKING
            if score is not None:
                total_score += score
                
            print(f"  -> Decision: {decision} | Score: {score}")

            final_scores.append({"Item ID": item_id, "Item Label": label, "Score": score})
            scoring_explanations.append({"item_id": item_id, "label": label, "score": score, "explanation": thought})

            # Save progress to CSV after every item
            analysis_path = os.path.join(dirs["analysis"], f"Analysis_{pid}.csv")
            with open(analysis_path, "w", newline="", encoding="utf-8") as f:
                # CHANGE: Added "Level" to fieldnames
                fieldnames = [
                    "PID", "Item", "Turn", "Level", "Injected_Flaw", "Detected_Flaw", 
                    "Agent_Decision", "Bot_Caught_Flaw", "Agent_Score", "Participant_Text"
                ]
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
