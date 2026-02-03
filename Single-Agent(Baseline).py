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

# ================= PROMPTS =================
intro_template = ChatPromptTemplate.from_messages([
    ("system", """You are a warm, empathetic licensed psychologist.
Your goal is to build immediate rapport with the participant.
- Greet them warmly and professionally.
- Ask how they are doing today.
- Keep it short (1-2 sentences)."""),
    ("human", "Start the session.")
])

rapport_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist building trust.
The participant just responded to your greeting.
Your goal: Validate their feelings and maintain a supportive connection.
- If they are positive, share that positivity.
- If they seem down or stressed, show empathy ("I'm sorry to hear that", "That sounds tough").
- **Do NOT** start the clinical assessment yet. Just be human.
- Keep it brief."""),
    ("human", "{last_response}")
])

transition_template = ChatPromptTemplate.from_messages([
    ("system", """You are a licensed psychologist. You have finished the small talk.
Now you need to gently transition to the clinical assessment.
1. Acknowledge what the user just said briefly.
2. Pivot smoothly to asking specific questions about their health and mood over the **last two weeks**.
"""),
    ("human", "The user just said: '{last_response}'. Generate the transition now.")
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
1. Analyze the answer for clarity and relevance.
2. Check for MISSING DOMAINS (vagueness, timeframe, relevance).
   - **vagueness**: The user is ambiguous ("sort of", "maybe", "I guess") or lacks detail of the symptoms.
   - **timeframe**: The user talks about the wrong time (e.g., "years ago", "childhood", "just today") instead of the **last 2 weeks**.
   - **relevance**: The user changes the subject, dodges the question, or talks about unrelated topics to the questions being asked (e.g., weather, politics).
3. **MANDATORY SCORING**: You MUST assign an integer score (0-3) if the answer is clear.
   - 0 = Not at all
   - 1 = Several days
   - 2 = More than half the days
   - 3 = Nearly every day

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
    difficulty: str = "medium",
    is_followup: bool = False,
):
    """
    Returns: (response_text, flaw_injected)
    flaw_injected: 'none', 'vagueness', 'timeframe', or 'relevance'
    """
    # 1. Setup Data
    clinical_signals = (client_profile.get("clinical_signals", {}) or {}).get("symptoms", {})
    long_key = SYMPTOM_KEY_BY_ITEM_ID.get(item_id, "")
    symptom_block = clinical_signals.get(long_key)
    if not symptom_block:
        short_key = SHORT_KEYS.get(item_id, "")
        symptom_block = clinical_signals.get(short_key, {})

    present = symptom_block.get("present", "absent")
    severity_hint = symptom_block.get("severity_hint", "none")
    
    # 2. Determine Flaw (Thesis Logic)
    flaw_injected = "none" #can change difficulty 
    style_instruction = "Speak naturally."
    
    if difficulty == "easy":
        style_instruction = "Give a clear, direct answer."
    elif difficulty == "medium":
        style_instruction = "Paraphrase the symptom in your own words. Use everyday language and avoid repeating the exact terms from the question."
    elif difficulty == "hard":
        flaw_type = random.choice(["vagueness", "timeframe", "relevance"])
        flaw_injected = flaw_type
        if flaw_type == "vagueness":
            style_instruction = "Be VAGUE. Use words like 'sort of', 'maybe'. Don't be clear and Do NOT give any specific details on symptoms."
        elif flaw_type == "timeframe":
            style_instruction = "Talk about the PAST (childhood/years ago) or just TODAY. Ignore 'last 2 weeks'."
        elif flaw_type == "relevance":
            style_instruction = ("Go OFF-TOPIC. Pick a keyword from the question (like 'sleep' or 'food') and talk about a hobby, object, or story related to that word instead of your symptoms. "
                "(Example: If asked about appetite, talk about a recipe you cooked. If asked about sleep, talk about your bed sheets). "
                "Do NOT answer how you actually feel."
            )

    # 3. Prompt
    profile_snippet = get_style_snippet(client_profile)
    profile_snippet["clinical_signals"] = { long_key: symptom_block }
    prompt = f"""
You are role-playing a client.
Profile: {json.dumps(profile_snippet, indent=2)}
Symptom: {present} ({severity_hint})
Question: "{question_text}"
**STYLE:** {style_instruction}
"""
    try:
        resp = llm.invoke(prompt)
        text = (getattr(resp, "content", "") or "").strip()
        return text, flaw_injected 
    except:
        return "I'm not sure.", "error"

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
        print(f"{AI_NAME}: {trans_text}")
        global_turn_index += 1
        transcript.append({"turn_index": global_turn_index, "speaker": AI_NAME, "role": "intro_transition", "text": trans_text})
        
    except Exception as e:
        print(f"Error during Intro/Rapport: {e}")

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

    # TESTING SETTINGS
    DIFFICULTY_OPTIONS = ["easy", "medium", "hard"]
    DIFFICULTY_WEIGHTS = [0.2, 0.5, 0.3] 

    for idx, h in enumerate(PHQ8_HYPOTHESES):
        shown = idx + 1
        item_id = h["item_id"]
        label = h["label"]
        hyp_text = h["text"]
        
        try:
            # 1. QUESTION
            current_difficulty = random.choices(DIFFICULTY_OPTIONS, weights=DIFFICULTY_WEIGHTS, k=1)[0]
            raw_probe = probe_chain.invoke({"item_label": label, "hypothesis_text": hyp_text})
            opener = f"{TOPIC_TRANSITIONS[idx]}{raw_probe}"
            print(f"{AI_NAME}: {opener}")
            
            global_turn_index += 1
            transcript.append({"turn_index": global_turn_index, "speaker": AI_NAME, "role": "initial_question", "item_index": shown, "text": opener})

            # 2. ANSWER
            reply_text, flaw_injected = simulate_client_answer(
                item_id, shown, label, opener, client_profile, llm,
                difficulty=current_difficulty,
                is_followup=False
            )

            print(f"{PARTICIPANT_NAME} (Flaw: {flaw_injected}): {reply_text}")
            global_turn_index += 1
            transcript.append({"turn_index": global_turn_index, "speaker": PARTICIPANT_NAME, "role": "base_answer", "text": reply_text})

            evidence_log[f"Item {shown}"]["supporting"].append({
                "evidence_supporting_id": f"{item_id}_E1", "text": reply_text, "followup_asked": False
            })

            # 3. DECISION (PASS 1)
            history_str = f"Psychologist: {opener}\nParticipant: {reply_text}"
            decision_data = assessment_chain.invoke({
                "item_index": shown, "item_label": label, "hypothesis_text": hyp_text, "history": history_str
            })
            
            decision = decision_data.get("decision", "NEXT_ITEM")
            thought = decision_data.get("thought", "")
            detected_flaw = decision_data.get("detected_missing_domain", "none").lower()
            score = parse_score(decision_data.get("score"))
            
            bot_caught_it = False
            if flaw_injected == "none":
                if decision == "NEXT_ITEM": bot_caught_it = True
            else:
                if decision == "FOLLOW_UP" and detected_flaw == flaw_injected:
                    bot_caught_it = True

            analysis_log.append({
                "PID": pid, "Item": item_id, "Turn": "Initial",
                "Injected_Flaw": flaw_injected, "Detected_Flaw": detected_flaw,
                "Agent_Decision": decision, "Bot_Caught_Flaw": bot_caught_it,
                "Agent_Score": score if score is not None else -1, 
                "Participant_Text": reply_text
            })

            transcript.append({"turn_index": global_turn_index, "speaker": "Agent_Internal_Monologue", "text": thought})
            agent_thoughts.append({"item_index": shown, "turn_index": global_turn_index, "decision": decision, "score": score, "thought": thought})

            # 4. FOLLOW-UP
            if decision == "FOLLOW_UP":
                evidence_log[f"Item {shown}"]["supporting"][-1]["followup_asked"] = True
                followup_q = decision_data.get("question", "Could you say more?")
                print(f"{AI_NAME} (Checking {detected_flaw}): {followup_q}")
                global_turn_index += 1
                transcript.append({"turn_index": global_turn_index, "speaker": AI_NAME, "role": "followup_question", "text": followup_q})

                reply_text_2, _ = simulate_client_answer(
                    item_id, shown, label, followup_q, client_profile, llm,
                    difficulty="medium", is_followup=True
                )
                print(f"{PARTICIPANT_NAME}: {reply_text_2}")
                global_turn_index += 1
                transcript.append({"turn_index": global_turn_index, "speaker": PARTICIPANT_NAME, "role": "followup_answer", "text": reply_text_2})

                # STRICT INSTRUCTION FOR FINAL SCORE
                history_str_2 = (
                    f"{history_str}\n"
                    f"Psychologist: {followup_q}\n"
                    f"Participant: {reply_text_2}\n"
                    f"(SYSTEM: You have all the information. You MUST now assign the integer score 0-3 based on the user's answers. Do not output null.)"
                )

                decision_data_2 = assessment_chain.invoke({
                    "item_index": shown, "item_label": label, "hypothesis_text": hyp_text, "history": history_str_2
                })
                
                new_score = parse_score(decision_data_2.get("score"))
                if new_score is not None: score = new_score
                decision = "RESOLVED_FOLLOW_UP"
            
            # --- FINAL SCORE PROCESSING ---
            # No Python forcing. If it's None, it stays None (and we see -1 in logs).
            # But the prompt is now super strict, so it SHOULD be 0-3.
            if score is None:
                # We log it as a failure, but we do NOT overwrite it with 0.
                print("  [FAILURE] Agent failed to produce a score.")
            else:
                total_score += score
                
            print(f"  -> Agent Decision: {decision} | Final Score: {score}")

            final_scores.append({"Item ID": item_id, "Item Label": label, "Score": score})
            scoring_explanations.append({"item_id": item_id, "label": label, "score": score, "explanation": thought})

            # INSTANT SAVE
            analysis_path = os.path.join(dirs["analysis"], f"Analysis_{pid}.csv")
            with open(analysis_path, "w", newline="", encoding="utf-8") as f:
                fieldnames = ["PID", "Item", "Turn", "Injected_Flaw", "Detected_Flaw", "Agent_Decision", "Bot_Caught_Flaw", "Agent_Score", "Participant_Text"]
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
