# build_phq8_profile_llm.py
# LLM-driven PHQ-8 profile builder from DAIC-WOZ USER transcripts.

## How to run ##
# python3 build_phq8_profile_llm.py --pid #participant_id#

from pathlib import Path
import os, json, re, argparse, random
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List, Dict
from collections import Counter

# =========================
# 0) ENV / CONFIG
# =========================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found. Please set it in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# Project paths (match your layout)
TRANSCRIPT_FILE_TPL = "Clean_Dataset/transcripts/{pid}_clean_turns.csv"
DAIC_TABLE_CSV = "Dataset/PHQ8 Mapping/PHQ-8_mappingLabel.csv"
OUTPUT_JSON_FOLDER = Path("Clean_Dataset/profiles")
OUTPUT_CSV_FOLDER  = Path("Clean_Dataset/profiles_csv")
OUTPUT_JSON_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV_FOLDER.mkdir(parents=True, exist_ok=True)

# =========================
# 1a) Profile schema & guides
# =========================
PROFILE_SCHEMA = {
  "client_id": "",

  # DOMAIN 1: IDENTITY
  "persona": {
    "demographics": {
      "age": None,
      "gender": ""
    }
  },
  # DOMAIN 2: MEDICAL EVIDENCE ([CATEGORY](Present/Absent/Uncertain))
  "Symptom": {
    "symptom_evidence": ""
  },
  # DOMAIN 3: AFFECT [Categorization](e.g., Flat, Agitated, Neutral)
  "Affective_Tone": {
    "label": ""
  },
  # DOMAIN 4: EMOTION [Categorization](e.g., Shame, Guilt, Anger)
  "Dominant_Emotion": {
    "label": ""
  },
  # DOMAIN 5: BEHAVIOR [Text-Based]
  "Behavioral": {
    "description": ""
  },
  # DOMAIN 6: COGNITIVE PATTERNS [Text-Based]
  "Cognitive_Patterns": {
    "description": ""
  },
  # DOMAIN 7: CORE BELIEFS [Text-Based]
  "Core_Beliefs": {
    "label": ""
  },
  # DOMAIN 8: INTERMEDIATE BELIEFS [Text-Based]
  "Intermediate_Beliefs": {
    "rules_and_assumptions": ""
   },
  # DOMAIN 8: SOCIAL CONTEXT [Text-Based]
  "Relational_Context": {
    "description": ""
  },
  # DOMAIN 9: INTERACTION STYLE [Categorization]
  "Response_Style": {
    "label": ""
  },
  # DOMAIN 10: MEDICAL ENGINE (Hidden/Separate)
  "clinical_signals": { "symptoms": {} }
}

PHQ8_CATEGORY_GUIDE = """
You are classifying patient utterances into PHQ-8 symptom domains.
Be conservative and evidence-based — mark a symptom as "present" only if it is clearly implied or stated.
Do NOT infer suicidality; PHQ-8 excludes that item.

GENERAL RULES
- Use ONLY the given transcript slice (patient/USER speech).
- If unsure or evidence is weak/conflicting → choose "uncertain".
- Prefer short, verbatim evidence quotes (≤18 words).
- Consider the past two weeks by default unless the participant clearly describes another period.
- Choose the *closest matching* domain; if a phrase could fit multiple, note it in both but adjust severity conservatively.

SEVERITY HINT GUIDE
Use these anchors when deciding severity_hint for each symptom.
If evidence is limited or mixed, choose the lower level or "uncertain."
- none:
    * Participant explicitly denies the symptom ("No problems with sleep", "I'm fine", "Not at all").
    * Describes positive functioning or normal variation without distress.
    * No negative affect associated with the domain.
    * Linguistic cues: "fine", "okay", "no issue", "never", "no trouble", "all good".
    → Example: "I sleep pretty well every night."

- mild:
    * Symptom appears occasionally or in limited contexts (≤ several days in 2 weeks).
    * Described as manageable or minor nuisance; minimal impact on work or relationships.
    * Participant may hedge ("sometimes", "a bit", "not too bad") or use softeners.
    * Tone is factual, not distressed.
    * Linguistic cues: "sometimes", "a little", "once in a while", "not really bad", "kind of".
    → Example: "Some nights I have trouble falling asleep, but usually it’s fine."

- moderate:
    * Symptom occurs frequently (more than half the days, recurring through the week).
    * Participant acknowledges clear distress or interference with daily functioning.
    * Balances between coping and impairment: “It’s getting harder,” “Most days it bothers me.”
    * Linguistic cues: "most days", "often", "keeps happening", "really hard", "affects me".
    * Affects mood, concentration, or motivation but not completely disabling.
    → Example: "I’ve been tired most days and it’s hard to keep up with schoolwork."

- severe:
    * Symptom is pervasive (nearly every day) or highly distressing.
    * Participant shows loss of control, marked impairment, or despair about the symptom.
    * May include helpless tone, inability to function, or strong negative emotion.
    * Linguistic cues: "every day", "can’t", "completely", "no energy at all", "I’ve stopped", "nothing helps".
    → Example: "Every day I feel exhausted and can barely get out of bed."

PHQ-8 DOMAINS (DETAILED RUBRIC):

1) ANHEDONIA (Loss of interest or pleasure)
Definition:
  Marked loss of interest or enjoyment in usual activities (work, hobbies, socializing).
Include:
  – Describes things “not enjoyable anymore”, “no motivation”, “don’t feel like doing anything”.
  – Mentions loss of hobbies, social withdrawal, emotional numbness.
Exclude:
  – Fatigue without mention of reduced enjoyment.
  – Lack of opportunity or time (“too busy”) without emotional loss.
Underlying mechanism:
    Loss of internal reward sensitivity — the person experiences activities as emotionally flat or unmotivating despite opportunity. Pleasure and interest are blunted, often leading to disengagement or social withdrawal.
Behavioral / Linguistic patterns:
    The person’s speech often describes disconnection from enjoyment rather than explicit dislike.
    They talk about doing less, feeling detached, or finding once-pleasant things meaningless.
    Their tone may sound flat, resigned, or reflective of emotional distance.
Examples:
  - "I used to love painting, but I just don’t feel like it anymore."
  - "Nothing seems fun lately."
Common confusions:
  Distinguish from *fatigue*: fatigue = low energy; anhedonia = loss of interest/pleasure.

2) LOW MOOD / HOPELESSNESS
Definition:
  Depressed, sad, empty, tearful, or hopeless mood most of the day, nearly every day.
Include:
  – Feels down, sad, blue, empty, hopeless.
  – Crying spells, loss of optimism, emotional flatness.
Exclude:
  – Anger, stress, or frustration without sadness.
  – Transient sadness tied to one event (“I was sad yesterday because...”).
Underlying mechanism:
    Persistent negative affect and expectation of continued distress. Emotion regulation is impaired; thoughts are future-oriented but pessimistic.
Behavioral / Linguistic patterns:
    The speaker tends to describe emotional heaviness, difficulty finding motivation, or a sense that improvement is impossible.
    Statements feel global and absolute (“everything feels hard”), with limited reference to positive change or nuance.
Examples:
  - "I feel really down almost every day."
  - "It’s hard to see anything getting better."
Common confusions:
  Mood vs. anxiety; anxious worry without sadness = not this domain.

3) SLEEP DISTURBANCE
Definition:
  Trouble falling asleep, staying asleep, early waking, or sleeping too much.
Include:
  – Difficulty initiating or maintaining sleep.
  – Restless or poor-quality sleep.
  – Sleeping excessively (“can’t get out of bed”).
Exclude:
  – Single late night (study/work) without persistent pattern.
  – Preferred late bedtime with adequate rest.
Underlying mechanism:
    Disrupted physiological rhythm — difficulty disengaging or maintaining restorative rest due to stress, rumination, or dysregulation.
Behavioral / Linguistic patterns:
    The person focuses on nights or rest cycles as problematic and unrewarding.
    Descriptions of sleep feel effortful or unrestful, often implying tension, worry, or exhaustion spilling into daily life.
Examples:
  - "I wake up at 3 a.m. and can’t fall back asleep."
  - "I’ve been sleeping 12 hours but still feel tired."
Common confusions:
  Sleep issues caused solely by external factors (noise, schedule) → uncertain unless distress expressed.

4) FATIGUE / LOW ENERGY
Definition:
  Persistent tiredness, low vitality, or exhaustion not relieved by rest.
Include:
  – “Tired all the time”, “drained”, “no energy”.
  – Daytime fatigue interfering with tasks.
Exclude:
  – Temporary tiredness from physical effort or lack of sleep.
  – Laziness or boredom without physical fatigue.
Underlying mechanism:
    Sustained depletion of physical or mental energy; motivational and physiological exhaustion often co-occur.
Behavioral / Linguistic patterns:
    The person describes life as draining or difficult to sustain effort in.
    They emphasize effort, slowness, or the sense that ordinary tasks take extra energy.
    The tone is weary or constrained rather than purely negative.
Examples:
  - "Even after sleeping, I’m exhausted."
  - "By lunch I feel completely drained."
Common confusions:
  Distinguish from anhedonia: fatigue = lack of energy; anhedonia = loss of interest/pleasure.

5) APPETITE / WEIGHT CHANGE
Definition:
  Significant change in appetite or weight (up or down) unrelated to dieting.
Include:
  – Eating much less or much more than usual.
  – Loss of appetite, forgetting to eat, overeating with guilt.
  – Noticeable weight change.
Exclude:
  – Intentional dieting or exercise regimen.
  – Isolated skipped meals due to busyness.
Underlying mechanism:
    Disturbance in internal regulation of appetite and reward. Eating is described as emotionally detached, effortful, or used to regulate stress.
Behavioral / Linguistic patterns:
    Food or eating is framed as out of sync — either absent pleasure (“nothing tastes good”) or compulsive engagement (“I eat without feeling hungry”).
    They may describe inconsistency (“some days nothing, some days too much”) or loss of internal cues.
Examples:
  - "I have to force myself to eat."
  - "I snack constantly and gained weight."
Common confusions:
  Distinguish emotional eating vs. appetite change — both count if mood-related.

6) SELF-WORTH / GUILT
Definition:
  Excessive or inappropriate guilt, self-blame, or feelings of worthlessness.
Include:
  – “I’m a failure”, “I let everyone down”.
  – Disproportionate guilt, self-criticism, shame.
Exclude:
  – Normal remorse for a clear mistake.
  – Modest self-evaluation without emotional weight.
Underlying mechanism:
    Distorted self-evaluation and internalized blame. The individual attributes problems to personal failure and views self as fundamentally deficient.
Behavioral / Linguistic patterns:
    Speech centers on self-judgment rather than emotion.
    The person generalizes fault across situations and minimizes positive qualities.
    Even neutral events are narrated as personal shortcomings or regrets.
Examples:
  - "I feel like I mess everything up."
  - "Nothing I do is ever good enough."
Common confusions:
  Avoid interpreting general humility or self-doubt as guilt unless emotionally charged.

7) CONCENTRATION PROBLEMS
Definition:
  Difficulty focusing, remembering, or making decisions.
Include:
  – Re-reading lines, zoning out, forgetfulness.
  – Trouble following conversations or tasks.
Exclude:
  – Distraction due to external noise or multitasking.
  – Mild forgetfulness without distress.
Underlying mechanism:
    Impaired attentional control and working-memory disruption, often due to intrusive thoughts or mental fatigue.
Behavioral / Linguistic patterns:
    The person portrays thinking as fragmented or slippery — difficulty sustaining focus or following through.
    Utterances may jump topics, acknowledge forgetting, or contain self-corrections that show attentional lapses.
Examples:
  - "I read the same paragraph over and over."
  - "It’s hard to pay attention in class."
Common confusions:
  Differentiate from fatigue (physical tiredness) and anxiety (racing thoughts).

8) PSYCHOMOTOR CHANGE (Slowing or Agitation)
Definition:
  Observable restlessness (fidgeting, pacing) or slowed movements/speech noticed by self or others.
Include:
  – “Can’t sit still”, “fidgety”, “moving slower than usual”.
  – Reports others noticed slowed or agitated behavior.
Exclude:
  – Routine fidgeting or caffeine jitters.
  – Figurative “feel stuck” without motor change.
Underlying mechanism:
    Physiological changes in arousal systems — reduced or excessive motor activity reflecting inner tension or depletion.
Behavioral / Linguistic patterns:
    The person describes their pace or bodily restlessness indirectly through daily functioning (“I move slower,” “can’t sit still”).
    Language may emphasize bodily awareness, effort, or external feedback (“people say I’ve slowed down”).
Examples:
  - "People say I talk slower lately."
  - "I pace around and can’t settle."
Common confusions:
  Exclude purely emotional agitation (“anxious inside”) unless accompanied by visible motor behavior.

OUTPUT FORMAT
For each domain:
{
  "present": "present"|"absent"|"uncertain",
  "severity_hint": "none"|"mild"|"moderate"|"severe"|"uncertain",
  "evidence_quotes": [up to 3 short verbatim quotes]
}
"""

REQUIRED_SYMPTOMS = [
  "anhedonia (loss of interest/pleasure)",
  "low mood / hopelessness",
  "sleep disturbance",
  "fatigue / low energy",
  "appetite/weight change",
  "self-worth/guilt",
  "concentration problems",
  "psychomotor change (slowing or agitation)",
]

# =========================
# 1b) Interaction style guide
# =========================
STYLE_GUIDE = """
CLINICAL BEHAVIOR & STYLE GUIDE
Pick ONLY from these allowed labels. DO NOT use numbers.

1) symptom_evidence: [Present | Absent | Uncertain]
2) affective_tone: [Flat | Dysphoric | Agitated | Neutral]
3) emotions: [Anxious | Sad | Anger | Hurt | Disappointed | Ashamed | Guilty | Suspicious | Jealous]
4) behavioral: [1 to 2 Sentence]
5) cognitive_patterns: [1 to 2 Sentence]
6) core_beliefs: [Helpless | Unloveable | Worthless | None]
7) intermediate_belief: [1 to 2 Sentence]
8) relational_context: [1 to 2 Sentence]
9) response_style: [Verbose | Terse | Evasive | Cooperative]
"""

# =========================
# 1c) Extra feature extraction guide & schema (new)
# =========================
EXTRA_FEATURES_GUIDE = """
Extract features using these strict definitions. Pick ONLY one option for TAGS.

--- GROUP 1: SYMPTOM OPTION ---
1. symptom_evidence: [TAG]
   - Present: Clear evidence of depression symptoms.
   - Absent: Explicit denial or no evidence of symptoms.
   - Uncertain: Vague or conflicting evidence.

--- GROUP 2: INTERACTION STYLE OPTION ---
2. affective_tone: [TAG]
   - Flat: Monotone, lack of emotional expression.
   - Dysphoric: Sad, hopeless, or heavy mood.
   - Agitated: Restless, irritable, or high-tension.
   - Neutral: Typical, calm conversation.

3. emotions: [TAG]
   - Anxious: Worried, tense, or focused on future uncertainties/threats.
   - Sad: Expressions of sorrow, grief, or low mood.
   - Angry: Frustration, resentment, or hostility toward self or others.
   - Hurt: Feeling emotionally wounded, let down, or rejected by others.
   - Disappointed: Unmet expectations or a sense of loss regarding a specific outcome.
   - Ashamed: Feeling "bad" as a person; a focus on being flawed or disgraced.
   - Guilty: Focused on a specific action or "wrongdoing" that harmed others.
   - Suspicious: Distrustful, wary of others' motives, or feeling watched/targeted.
   - Jealous: Resentment toward others for their perceived advantages or relationships.

4. response_style: [TAG]
   - Verbose: Over-explaining, very long answers.
   - Terse: Minimalist, "Yes/No" or very short answers.
   - Evasive: Avoiding the question or changing the topic.
   - Cooperative: Balanced, helpful, and direct answers.

--- GROUP 3: CORE BELIEFS (NEW) ---
5. core_beliefs: [TAG]
   - Helpless: Pick this if the user describes a lack of agency, feeling trapped, or believing that personal effort is futile in changing their situation.
   - Unlovable: Pick this if the user describes a fundamental flaw in their ability to be liked or belong, focusing on social rejection, isolation, or being "different" from others.
   - Worthless: Pick this if the user describes themselves as fundamentally deficient, a "moral" failure, or undeserving of basic respect and success.
   - None: Pick this if no deep-seated negative identity belief is expressed in the transcript.

--- GROUP 4: CLINICAL TEXT BASE SENTENCES ---
6. behavioral: [1 to 2 Sentence] Describe physical/verbal habits (e.g., fillers, sighing, rate of speech).
    - Definition: Physical or verbal actions observed during the interaction.
    - Clinical Goal: Identify non-content cues like speech rate, pauses, or sighing.
    - Example: "The user speaks in a low, monotone voice with frequent long pauses and audible sighing before answering."
7. cognitive_patterns: [1 to 2Sentence] Describe thinking errors (e.g., "I always fail").
    - Definition: Recurring "Thinking Errors" or cognitive distortions.
    - Clinical Goal: Identify habits like catastrophizing, all-or-nothing thinking, or overgeneralization.
    - Example: "Client exhibits overgeneralization by stating that one minor work mistake means their entire career is over."
8. relational_context: [1 to 2 Sentence] Social support vs. conflict summary.
    - Definition: The user's current social environment and interpersonal dynamics.
    - Clinical Goal: Summarize the presence of social support versus active conflict.
    - Example: "Describes a strong emotional bond with their spouse but reports total estrangement from their parents."
9. intermediate_belief: [[1 to 2 Sentence]]
    - Definition: The "Rules for Living" that the user uses to cope with their core belief. 
    - Clinical Goal: Identify the "If/Then" assumptions or "Must/Should" rules.
    - Example: "If I don't please everyone, then they will see I am worthless" or "I must never show weakness."
"""

EXTRA_FEATURES_SCHEMA = {
  "symptom_evidence": "",    # Categorization
  "affective_tone": "",      # Categorization
  "emotions": "",   # Categorization
  "behavioral": "", # Text-Based (Sentence)
  "cognitive_patterns": "",  # Text-Based (Sentence)
  "core_beliefs": "",        # Categorization
  "intermediate_belief": "", # Text-Based (Sentence)
  "relational_context": "",  # Text-Based (Sentence)
  "response_style": ""      # Categorization
}

# =========================
# 2) LLM helpers
# =========================
def robust_json_loads(txt: str):
    """Parse JSON; if it fails, try to extract the largest {...} block."""
    try:
        return json.loads(txt)
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError("Model output was not valid JSON:\n" + txt[:800])

def llm_json(system: str, user: str, temperature: float = TEMPERATURE) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp.choices[0].message.content
    return robust_json_loads(content)

# =========================
# 3) Prompt templates
# =========================
CHUNK_SYSTEM = "You are extracting PHQ-8 symptom evidence and interaction style from patient speech. Return JSON only."

# THIS IS THE FLAT TEMPLATE FOR THE SLICES
PHQ8_JSON_TEMPLATE_FOR_SLICE = {
  "symptoms": {
    "anhedonia (loss of interest/pleasure)": {"present":"", "severity_hint":"", "evidence_quotes":[]},
    "low mood / hopelessness":               {"present":"", "severity_hint":"", "evidence_quotes":[]},
    "sleep disturbance":                     {"present":"", "severity_hint":"", "evidence_quotes":[]},
    "fatigue / low energy":                  {"present":"", "severity_hint":"", "evidence_quotes":[]},
    "appetite/weight change":                {"present":"", "severity_hint":"", "evidence_quotes":[]},
    "self-worth/guilt":                      {"present":"", "severity_hint":"", "evidence_quotes":[]},
    "concentration problems":                {"present":"", "severity_hint":"", "evidence_quotes":[]},
    "psychomotor change (slowing or agitation)": {"present":"", "severity_hint":"", "evidence_quotes":[]}
  },
  "symptom_evidence": "",
  "affective_tone": "",
  "emotions": "",
  "behavioral": "",
  "cognitive_patterns": "",
  "core_beliefs": "",
  "relational_context": "",
  "response_style": ""
}

CHUNK_USER_TMPL = """\
{phq8_guide}

{style_guide}

TRANSCRIPT SLICE (USER lines only):
<<<
{slice_text}
>>>

FILL THIS JSON (no extra keys, no commentary, NO SUB-FIELDS):
{json_template}
"""

EXTRA_SYSTEM = "You extract higher-level behavioral features from USER-only text for a client simulator. Return JSON only."

AGG_SYSTEM = "You combine slice-level JSON into a single conservative PHQ-8 profile. Return JSON only."

# THIS WAS THE MISSING PIECE
AGG_USER_TMPL = """\
Combine the slice JSON array below into ONE final profile following the 8 Core Pillars.

CRITICAL AGGREGATION RULES:
1. **SYMPTOM TRUTH**: Use the HIGHEST severity found in any slice. If a symptom is 'Present' in any slice, it is 'Present' overall.
2. **PILLAR CONSISTENCY**: 
   - Tags (Tone, Emotions, Style): Choose the most representative label.
   - Sentences (Behavior, Thinking, Beliefs, Social): Merge findings into one strong, clear sentence.
3. **CORE BELIEFS**: Prioritize the deepest internal "truth" revealed by the user across all slices.

SLICES (array):
<<<
{slice_json_array}
>>>

OUTPUT: Fill this exact flat schema. Replace client_id with "{client_id}".
SCHEMA:
{schema}
"""

# THE FINAL PASS TEMPLATE (FOR THE 8 CORE PILLARS)
EXTRA_USER_TMPL = """\
{guide}

INSTRUCTIONS:
1. Look at the entire transcript below.
2. Fill the JSON schema exactly as provided.
3. DO NOT use sub-folders.
4. DO NOT use numbers.
5. For 'behavior', 'thinking', 'beliefs', and 'social': Write ONE clear sentence.
6. For 'symptoms', 'tone', 'emotions', and 'style': Write ONE word.

USER TRANSCRIPT:
<<<
{user_text}
>>>

FILL THIS JSON (no extra keys, no commentary):
{schema}
"""

# =========================
# 4) Data I/O helpers
# =========================
def read_transcript_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript not found: {path}")
    df = pd.read_csv(path)
    # Try to locate likely columns if names vary
    cols = {c.lower(): c for c in df.columns}
    speaker_col = cols.get("speaker") or cols.get("role") or cols.get("participant")
    text_col = cols.get("utterance") or cols.get("text") or cols.get("content")
    if not speaker_col or not text_col:
        raise ValueError(
            f"Transcript CSV must contain columns like 'speaker'/'role' and 'utterance'/'text'. "
            f"Found columns: {list(df.columns)}"
        )
    # Normalize a canonical view
    df = df.rename(columns={speaker_col: "speaker", text_col: "utterance"})
    # Ensure strings
    df["speaker"] = df["speaker"].astype(str)
    df["utterance"] = df["utterance"].astype(str)
    return df

# Common label sets (uppercased)
USER_SYNONYMS = {
    "USER","CLIENT","PATIENT","PARTICIPANT","SUBJECT","SPEAKER 1","SPEAKER1","S1","INTERVIEWEE"
}
THERAPIST_SYNONYMS = {
    "THERAPIST","AGENT","INTERVIEWER","ELLIE","COACH","COUNSELOR","SPEAKER 2","SPEAKER2","S2"
}

def _norm_label(x: Optional[str]) -> str:
    if x is None:
        return ""
    t = str(x).strip()
    # collapse speaker like "Speaker1", "speaker 1", "Spk 1"
    t_up = re.sub(r"\s+", " ", t).strip().upper()
    t_up = t_up.replace("SPEAKER1", "SPEAKER 1").replace("SPEAKER2", "SPEAKER 2")
    return t_up

def _infer_user_role(speaker_series: pd.Series) -> Optional[str]:
    """Pick most plausible USER-like role when 'USER' not present."""
    labels = [_norm_label(v) for v in speaker_series.dropna().tolist()]
    counts = Counter(labels)
    # If any label is already in USER_SYNONYMS → prefer the most frequent among them
    user_like = [lab for lab,_ in counts.most_common() if lab in USER_SYNONYMS]
    if user_like:
        return user_like[0]
    # Else pick the most frequent label that is NOT therapist-like
    for lab, _ in counts.most_common():
        if lab and lab not in THERAPIST_SYNONYMS:
            return lab
    return None

def get_user_utterances(df: pd.DataFrame, forced_user_role: Optional[str] = None) -> List[str]:
    # Normalize labels once
    df = df.copy()
    df["speaker_norm"] = df["speaker"].apply(_norm_label)

    if forced_user_role:
        forced = _norm_label(forced_user_role)
        user_mask = df["speaker_norm"] == forced
        chosen_label = forced
    else:
        # Ideal: an explicit USER
        if (df["speaker_norm"] == "USER").any():
            user_mask = df["speaker_norm"] == "USER"
            chosen_label = "USER"
        else:
            inferred = _infer_user_role(df["speaker"])
            if inferred:
                user_mask = df["speaker_norm"] == inferred
                chosen_label = inferred
            else:
                # nothing sensible to pick
                counts = Counter(df["speaker_norm"].tolist())
                raise ValueError(
                    f"No USER-like utterances found. Observed labels (top 10): {counts.most_common(10)}"
                )

    s = (
        df.loc[user_mask, "utterance"]
          .fillna("")
          .astype(str)
          .map(str.strip)
    )
    # Diagnostics: if empty, surface what we chose
    if s.eq("").all() or s.shape[0] == 0:
        counts = Counter(df["speaker_norm"].tolist())
        raise ValueError(
            f"No USER utterances found after choosing label '{chosen_label}'. "
            f"Observed labels (top 10): {counts.most_common(10)}"
        )

    return [u for u in s.tolist() if u]

def chunk_texts(lines: list[str], max_chars: int = 8000) -> list[str]:
    """Character-based chunking of concatenated USER lines."""
    chunks, buf = [], ""
    for u in lines:
        if len(buf) + len(u) + 1 > max_chars:
            if buf:
                chunks.append(buf.strip())
            buf = u
        else:
            buf = (buf + "\n" + u) if buf else u
    if buf:
        chunks.append(buf.strip())
    return chunks

def validate_profile(profile: dict):
    # 1. Check top-level Domain Boxes exist
    assert "persona" in profile, "Missing Persona domain"
    assert "Symptom" in profile, "Missing Symptom domain"
    assert "Core_Beliefs" in profile, "Missing Core_Beliefs domain"
    
    # 2. Check internal Identity
    assert "demographics" in profile["persona"], "Missing demographics inside persona"
    
    # 3. Check specific clinical content
    # We check if the 'description' or 'label' keys exist in their new boxes
    assert "symptom_evidence" in profile["Symptom"], "Missing symptom_evidence in Symptom box"
    assert "description" in profile["Core_Beliefs"], "Missing description in Core_Beliefs box"
    assert "description" in profile["Behavioral"], "Missing description in Behavior box"

def save_outputs(profile: dict, participant_id: int):
    # JSON
    json_path = OUTPUT_JSON_FOLDER / f"{participant_id}_client_profile.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON profile saved → {json_path}")

    # Optional: quick CSV summary for review
    flat = []
    sx = profile.get("clinical_signals", {}).get("symptoms", {})
    for name, info in sx.items():
        flat.append({
            "participant_id": participant_id,
            "symptom": name,
            "present": info.get("present", ""),
            "severity_hint": info.get("severity_hint", ""),
            "evidence_count": len(info.get("evidence_quotes", []) or []),
        })
    if flat:
        df = pd.DataFrame(flat)
        csv_path = OUTPUT_CSV_FOLDER / f"{participant_id}_symptom_summary.csv"
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV summary saved → {csv_path}")

# =========================
# 5) Main pipeline
# =========================
def analyze_participant(participant_id):
    # 1. Load data
    tpath = TRANSCRIPT_FILE_TPL.format(pid=participant_id)
    df = read_transcript_csv(tpath)
    user_lines = get_user_utterances(df)
    
    if not user_lines:
        raise ValueError("No USER utterances found.")

    # 2. Extract PHQ-8 Slices (The medical engine)
    chunks = chunk_texts(user_lines, max_chars=8000)
    slice_results = []
    for slc in chunks[:6]:
        user_prompt = CHUNK_USER_TMPL.format(
            phq8_guide=PHQ8_CATEGORY_GUIDE,
            style_guide=STYLE_GUIDE,
            slice_text=slc,
            json_template=json.dumps(PHQ8_JSON_TEMPLATE_FOR_SLICE)
        )
        slice_results.append(llm_json(CHUNK_SYSTEM, user_prompt))

    # 3. Get Demographics from the CSV
    age, gender = None, None
    if os.path.exists(DAIC_TABLE_CSV):
        meta = pd.read_csv(DAIC_TABLE_CSV)
        row = meta.loc[meta["participant_id"] == participant_id]
        if not row.empty:
            raw_age = row.iloc[0].get("age")
            raw_gender = row.iloc[0].get("gender")
            
            # FORCE CONVERSION: Convert from numpy/pandas types to native Python types
            if pd.notna(raw_age):
                age = int(raw_age) 
            if pd.notna(raw_gender):
                gender = str(raw_gender)

    # 4. Get the 8 Pillars from the LLM (The interaction style)
    user_text_for_llm = "\n".join(user_lines)[:12000] 
    extra_user_prompt = EXTRA_USER_TMPL.format(
        guide=EXTRA_FEATURES_GUIDE,
        user_text=user_text_for_llm,
        schema=json.dumps(EXTRA_FEATURES_SCHEMA)
    )
    extra_features = llm_json(EXTRA_SYSTEM, extra_user_prompt)

    # 5. BUILD THE 9 DOMAINS (THE BOXES)
    final_profile = {
        "client_id": str(participant_id),
        
        # DOMAIN 1: PERSONA
        "persona": {
            "demographics": {"age": age, "gender": gender}
        },

        # DOMAIN 2: SYMPTOM
        "Symptom": {
            "symptom_evidence": extra_features.get("symptom_evidence", "")
        },

        # DOMAINS 3-9: INDIVIDUAL BOXES
        "Affective_Tone":        {"label": extra_features.get("affective_tone", "")},
        "Dominant_Emotion":      {"label": extra_features.get("emotions", "")},
        "Behavioral":            {"description": extra_features.get("behavioral", "")},
        "Cognitive_Patterns":    {"description": extra_features.get("cognitive_patterns", "")},
        "Core_Beliefs":          {"description": extra_features.get("core_beliefs", "")},
        "Relational_Context":    {"description": extra_features.get("relational_context", "")},
        "Response_Style":        {"label": extra_features.get("response_style", "")},
        
        # Keep internal signals for medical backing
        "clinical_signals": {"symptoms": slice_results[-1].get("symptoms", {})}
    }

    print(f"--- Final Profile Built: Participant {participant_id} ---")
    validate_profile(final_profile)
    return final_profile

# =========================
# 6) CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Build PHQ-8 LLM profile from DAIC-WOZ USER transcript.")
    parser.add_argument("--pid", type=int, required=True, help="Participant ID (e.g., 301)")
    args = parser.parse_args()

    profile = analyze_participant(args.pid)
    save_outputs(profile, args.pid)

if __name__ == "__main__":
    main()
