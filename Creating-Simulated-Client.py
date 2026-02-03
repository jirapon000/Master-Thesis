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
  "persona": {
    "demographics": {
      "age": None,
      "gender": None
    },
    # In PROFILE_SCHEMA
    "interaction_style": {
      "style_label": "",          # short, human-readable summary phrase (e.g., "brief and hesitant", "open and detailed")
      "scores": {                 # 0–4 anchored rubric (0 = very low, 4 = very high intensity of the trait)
        "verbosity": 0,           # Average length and elaboration of turns (0 = one-word or terse replies; 2 = medium sentences with some detail; 4 = long, descriptive responses with narrative flow or multiple clauses
        "hedging": 0,             # Frequency of uncertainty or softening phrases (“maybe”, “I guess”, “kind of”, “probably”); 0 = none or rare; 2 = occasional hedges; 4 = very frequent use showing hesitation or avoidance
        "directness": 0,          # Degree of clarity and explicitness in responses; 0 = vague, ambiguous, or noncommittal answers; 2 = reasonably clear but sometimes roundabout; 4 = fully explicit, concrete, and direct statements with minimal ambiguity.
        "cooperation": 0,         # Willingness to engage with and assist the interviewer (psychologist); 0 = resistant, dismissive, or refuses to answer; 2 = compliant but minimally engaged; 4 = actively helpful, clarifies questions, elaborates willingly.
        "responsiveness": 0,      # How well the participant stays on-topic and addresses what was asked. 0 = frequently off-topic, ignores prompts; 2 = sometimes indirect; 4 = consistently relevant and responsive to each question.
        "avoidance": 0,           # Degree of topic deflection or minimization of sensitive areas. 0 = fully open, no avoidance; 2 = mild deflection when uncomfortable topics arise;4 = frequent redirection, denial, or changing subject when asked about emotions, family, etc.
        "self_disclosure": 0,     # Willingness to share personal experiences, feelings, or examples. 0 = impersonal or purely factual speech; 2 = partial disclosure (mentions feelings briefly);4 = rich personal narratives and emotional introspection.
        "formality": 0            # Language tone: formal vs. casual/slang style. 0 = highly informal, colloquial, or slang-filled; 2 = conversational but appropriate; 4 = very formal, clinical, or academic phrasing.
      },
      "features": {
        "avg_sentence_length_estimate": 0,
        "hedging_markers": [],     # ["maybe","I guess",...]
        "discourse_markers": [],   # ["well","so","like","you know",...]
        "fillers": [],             # ["um","uh","err",...]
        "emotional_tone": ""       # "negative"|"neutral"|"positive"|"mixed"
      },
      "evidence_quotes": []        # 1–3 short USER snippets
    }
  },
  "clinical_signals": {
    "symptoms": {
      "anhedonia (loss of interest/pleasure)": {"present":"", "severity_hint":"", "evidence_quotes":[]},
      "low mood / hopelessness":               {"present":"", "severity_hint":"", "evidence_quotes":[]},
      "sleep disturbance":                     {"present":"", "severity_hint":"", "evidence_quotes":[]},
      "fatigue / low energy":                  {"present":"", "severity_hint":"", "evidence_quotes":[]},
      "appetite/weight change":                {"present":"", "severity_hint":"", "evidence_quotes":[]},
      "self-worth/guilt":                      {"present":"", "severity_hint":"", "evidence_quotes":[]},
      "concentration problems":                {"present":"", "severity_hint":"", "evidence_quotes":[]},
      "psychomotor change (slowing or agitation)": {"present":"", "severity_hint":"", "evidence_quotes":[]}
    }
  },
  # New section (filled after aggregation via extra LLM call)
  "behavioral_features": {
    "affective_emotional": {},
    "interactional_behavior": {},
    "cognitive_linguistic": {},
    "social_context": {},
    "self_perception_insight": {},
    "time_alignment": {}
  }
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

- uncertain:
    * Evidence is ambiguous, conflicting, or too brief to judge.
    * May describe the symptom but without clear frequency or impact.
    * Use when participant’s statement could reflect normal variation or when context is missing.
    * Linguistic cues: vague or contradictory phrasing ("maybe", "not sure", "I guess", "depends").
    → Example: "I guess I get tired sometimes, but I’m not sure if that counts."

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
INTERACTION STYLE — PRINCIPLE-BASED GUIDE
You will rate the participant’s conversational style from a transcript SLICE (USER lines only).
Interpret meaning and interactional behavior from the USER’s wording, not just keywords.
Return JSON only using the provided schema; do not add or remove fields.

GOAL
Characterize how the person tends to speak in this slice (their conversational “signature”): brevity vs. verbosity,
direct vs. hedged, cooperative vs. resistant, etc. Use contextual judgment instead of rigid rules.

SCALE (0–4) — USE RELATIVELY, NOT MECHANICALLY
0 = none/absent  •  1 = low  •  2 = moderate/typical  •  3 = high  •  4 = very high/intense
Most slices should center near 2, with some 0–1 and some 3–4. Avoid collapsing everything to 0–1.

STYLE LABEL (very important for entailment)
Set "style_label" to a compact summary with 2–3 salient traits in order of prominence, e.g.:
"terse, cooperative, somewhat hedging" or "verbose, direct, informal".
Do NOT just restate scores; synthesize a human-readable label.

WHAT TO SCORE (0–4 each)
- Verbosity: brevity vs. length/detail across turns.
- Hedging: uncertainty/softeners (e.g., maybe, kind of, I think, I guess, not sure, tends to, probably).
- Directness: clarity and explicitness of answers vs. ambiguity/evasion.
- Cooperation: willingness to answer, clarify, and work with the interviewer (not agreement, but engagement).
- Responsiveness: relevance and on-target replies to prompts.
- Avoidance: topic deflection, changing subject, refusing specifics.
- Self-disclosure: concreteness and personal detail (events, feelings, examples).
- Formality: formality of register (clinical/structured vs. slang/casual).

FEATURE EXTRACTION (populate features consistently)
- avg_sentence_length_estimate: approximate average words per sentence (round int; consider multiple turns).
- hedging_markers: list up to ~6 distinct hedges actually used in the slice.
  Expand beyond literal matches: detect paraphrased hedges like "sorta/kinda", "I feel like", "I guess so", "probably not",
  "I’m not sure", "maybe", "I think", "to be honest", "more or less", "not really", "I suppose".
- discourse_markers: list items like "well", "so", "like", "you know", "honestly", "to be fair", "anyway".
- fillers: short non-lexical items like "um", "uh", "erm", "hmm".
- emotional_tone: negative|neutral|positive|mixed — overall affect of the slice, not diagnosis.

EVIDENCE QUOTES
Add 1–3 short verbatim snippets (≤18 words) that best illustrate the style (e.g., hedging, terseness, disclosure).

PRINCIPLES FOR DECISION-MAKING
- Prefer interpretation over keyword counting. Consider consistency across turns.
- A single strong instance can move a score from 1→2; repeated patterns can justify 3–4.
- If directness is high and hedging is low, reflect that asymmetry (don’t keep everything near 2).
- Off-topic or evasive answers should lower responsiveness and raise avoidance.
- If the slice is short but consistently terse, set verbosity low (0–1) rather than defaulting to 2.

CALIBRATION HINTS
- Verbosity anchor: ~≤6 words/turn repeatedly → 0–1; multi-sentence with details → 3–4.
- Hedging anchor: occasional hedge → 1–2; frequent across turns → 3–4.
- Directness anchor: clear, specific answers to asked question → 3–4; vague or deflecting → 0–1.
- Self-disclosure anchor: impersonal generalities → 0–1; concrete events/feelings/examples → 3–4.
- Formality anchor: heavy slang/fillers → 0–1; clinical/structured phrasing → 3–4.

OUTPUT SHAPE FOR THE SLICE (must match exactly)
{
  "style": {
    "style_label": "",
    "scores": {
      "verbosity": 0, "hedging": 0, "directness": 0, "cooperation": 0,
      "responsiveness": 0, "avoidance": 0, "self_disclosure": 0, "formality": 0
    },
    "features": {
      "avg_sentence_length_estimate": 0,
      "hedging_markers": [],
      "discourse_markers": [],
      "fillers": [],
      "emotional_tone": "negative|neutral|positive|mixed"
    },
    "evidence_quotes": []
  }
}

FEW-SHOT CALIBRATION EXAMPLES (do not imitate wording; match intent)
Example A (terse + direct, low hedge, informal)
→ style_label: "terse, direct, informal"
→ scores: verbosity=1, hedging=0–1, directness=3, cooperation=2, responsiveness=3, avoidance=1, self_disclosure=1, formality=1
→ features: avg_sentence_length_estimate≈6; hedging_markers=[]; discourse_markers=["yeah","so"]; fillers=["uh"]; emotional_tone="neutral"
→ evidence_quotes: ["Yeah, I did. It was fine.", "Uh, just work stuff."]

Example B (verbose + hedging + cooperative)
→ style_label: "verbose, cooperative, somewhat hedging"
→ scores: verbosity=3–4, hedging=2–3, directness=2, cooperation=3, responsiveness=3, avoidance=1, self_disclosure=3, formality=2
→ features: avg_sentence_length_estimate≈18; hedging_markers=["I think","maybe","kind of"]; discourse_markers=["well","you know"]; fillers=[]
→ evidence_quotes: ["Well, I think it was kind of tough to keep up."]

Example C (avoidant + low disclosure)
→ style_label: "avoidant, low disclosure, indirect"
→ scores: verbosity=1, hedging=2, directness=1, cooperation=1, responsiveness=1, avoidance=3–4, self_disclosure=0–1, formality=2
→ features: avg_sentence_length_estimate≈7; hedging_markers=["not really","I guess"]; discourse_markers=["so"]; fillers=[]
→ evidence_quotes: ["I don’t really want to get into that."]
"""

# =========================
# 1c) Extra feature extraction guide & schema (new)
# =========================
EXTRA_FEATURES_GUIDE = """
**CRITICAL INSTRUCTION:** For every field with a "reasoning" key, you MUST provide a 1-sentence justification for why you chose that score *before* you output the score.
Your reasoning must reference specific behaviors (e.g., "User consistently deflected questions about family").
**Do not infer.** Only assign High (3-4) or Low (0-1) scores if there is EXPLICIT textual evidence. If ambiguous, stick to Moderate (2).

---

### 1) Affective / Emotional
Purpose: Capture the emotional quality of the client’s language.

- **overall_tone**: 
    * Label: "Negative", "Positive", or "Neutral".
    * REASONING: Cite specific emotional keywords (e.g., "sad," "hopeless," "great").
- **variability**:
    * **STRICT RULE:** Only score High (3-4) if the user *explicitly describes* their mood as "up and down," "moody," or "unstable."
    * If they do not explicitly say they are moody, score as "Stable" (1-2).
- **expressiveness**:
    * High (3-4): Uses complex emotional words ("devastated", "ecstatic", "anxious").
    * Low (0-1): Uses basic/numb words ("fine", "okay", "bad", "good").

### 2) Interactional Behavior
Purpose: Analyze conversational dynamics and engagement.

- **responsiveness**: 
    * High (4): Elaborates voluntarily; answers are paragraphs, not just sentences.
    * Low (0-1): One-word answers ("Yes", "No", "Maybe"). Requires prompting to speak.
- **engagement_trajectory**: 
    * Increasing (3-4): Starts short, gets longer/deeper by the end.
    * Decreasing (0-1): Starts chatting, then shuts down.
    * Stable (2): Consistent length throughout.
- **clarification_responsiveness**:
    * High (4): Explicitly tries to help the listener understand ("I mean...", "In other words...").
- **repair_frequency**:
    * High (4): Frequently self-corrects ("Actually, no...", "I meant to say...").
    * Low (0): Speaks without correcting themselves.

### 3) Cognitive / Linguistic
Purpose: Detect specific thinking patterns and language abstractness.

- **temporal_focus**: 
    * Estimate the ratio of Past (memories) vs. Present (current state) vs. Future (plans).
- **cognitive_distortions**: 
    * List SPECIFIC distortions found: "Catastrophizing" (expecting disaster), "All-or-Nothing" (words like *always*, *never*, *everyone*), "Personalization" (blaming self).
- **pronoun_usage_analysis**: 
    * Focus: Does the user say "I/Me" (Self-focus) or "They/We" (External focus)?
- **vocabulary_richness**: 
    * High (4): Uses academic, precise, or varied vocabulary.
    * Low (0-1): Repetitive, simple vocabulary.
- **abstractness**: 
    * Abstract (4): Discusses concepts (justice, love, meaning, future, failure).
    * Concrete (0-1): Discusses objects/logistics (food, sleep, bus, schedule).

### 4) Social & Contextual
Purpose: Map the explicit social world mentioned by the user.

- **mentions**: 
    * Set to TRUE only if specific people/topics are named (e.g., "My mom," "School," "My boyfriend").
- **support_seeking**: 
    * Active (4): User explicitly describes asking for help ("I called my friend," "I went to a doctor").
    * None (0): User explicitly says they handle things alone ("I keep it to myself").
- **relationship_conflict**: 
    * Present (4): Explicit mention of fighting, arguing, or tension with others.
    * None (0): Explicit statement of good relationships or no mention of others.
- **avoidance_triggers**: 
    * List topics the user explicitly refused to discuss or gave short, defensive answers to.

### 5) Self-Perception & Insight
Purpose: Determine if the user views themselves as an active agent or passive victim.

- **insight_level**: 
    * High (4): Explicitly links past behaviors to current outcomes ("I realize I was doing X because Y").
    * Low (0-1): Describes events without reflecting on *why* they happened.
- **agency (Locus of Control)**: 
    * Active (4): Uses active verbs ("I decided," "I chose").
    * Passive (0-1): Uses passive voice ("It happened to me," "They made me").
- **self_esteem_tone**: 
    * Negative (0-1): Explicit self-criticism ("I am a failure," "I'm useless").
    * Positive (3-4): Explicit self-praise ("I'm good at this").

### 6) Temporal Alignment
Purpose: Check if the user sticks to the "last two weeks" constraint.

- **status**: 
    * "aligned" (Mentions "last week," "yesterday").
    * "habitual" (Uses "always," "usually," "often" - implies long-term patterns).
    * "mixed" (Both).
- **corrected_after_prompt**: 
    * Set true ONLY if the user says "Oh right, the last two weeks" or similar.
"""

EXTRA_FEATURES_SCHEMA = {
  "affective_emotional": {
    "overall_tone": {"reasoning": "", "label": "", "score": 0},
    "variability": {"reasoning": "", "label": "", "score": 0},
    "expressiveness": {"reasoning": "", "label": "", "score": 0},
    "evidence_quotes": []
  },
  "interactional_behavior": {
    "responsiveness": {"reasoning": "", "label": "", "score": 0},
    "engagement_trajectory": {"reasoning": "", "label": "", "score": 0},
    "clarification_responsiveness": {"reasoning": "", "label": "", "score": 0},
    "repair_frequency": {"reasoning": "", "label": "", "score": 0},
    "evidence_quotes": []
  },
  "cognitive_linguistic": {
    "temporal_focus": {"past": 0.0, "present": 0.0, "future": 0.0},
    "cognitive_distortions": [], 
    "pronoun_usage_analysis": {"reasoning": "", "label": "", "score": 0}, 
    "vocabulary_richness": {"reasoning": "", "label": "", "score": 0},    
    "abstractness": {"reasoning": "", "label": "", "score": 0},
    "evidence_quotes": []
  },
  "social_context": {
    "mentions": {
      "family": False, "friends": False, "school_work": False,
      "relationships": False, "finances": False, "health": False
    },
    "support_seeking": {"reasoning": "", "label": "", "score": 0},
    "relationship_conflict": {"reasoning": "", "label": "", "score": 0},
    "avoidance_triggers": [],
    "environmental_stressors": [],
    "evidence_quotes": []
  },
  "self_perception_insight": {
    "insight_level": {"reasoning": "", "label": "", "score": 0},
    "agency": {"reasoning": "", "label": "", "score": 0},
    "self_esteem_tone": {"reasoning": "", "label": "", "score": 0},
    "evidence_quotes": []
  },
  "time_alignment": {
    "reasoning": "",
    "status": "",
    "corrected_after_prompt": False,
    "evidence_quotes": []
  }
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
  "style": {
    "style_label": "",
    "scores": {
      "verbosity": 0, "hedging": 0, "directness": 0, "cooperation": 0,
      "responsiveness": 0, "avoidance": 0, "self_disclosure": 0, "formality": 0
    },
    "features": {
      "avg_sentence_length_estimate": 0,
      "hedging_markers": [],
      "discourse_markers": [],
      "fillers": [],
      "emotional_tone": ""
    },
    "evidence_quotes":[]
  }
}

CHUNK_USER_TMPL = """\
{phq8_guide}

{style_guide}

TRANSCRIPT SLICE (USER lines only):
<<<
{slice_text}
>>>

FILL THIS JSON (no extra keys, no commentary):
{json_template}
"""

AGG_SYSTEM = "You combine slice-level JSON into a single conservative PHQ-8 profile. Return JSON only."

AGG_USER_TMPL = """\
Combine the slice JSON array below into ONE final profile.

CRITICAL AGGREGATION RULES (Do not use simple math averaging):

1. **PEAK DETECTION (Symptoms & Behaviors)**:
   - If a symptom or behavior scores High (3-4) or 'Severe' in ANY slice, that is a critical signal.
   - **Do not dilute strong signals.** - Example: If 'Avoidance' is [0, 0, 4, 1], the final score must be **High (3 or 4)** because the avoidance *did happen*.

2. **AFFECTIVE CONSISTENCY**:
   - If the user masks (says "I'm fine") but later reveals sadness, prioritize the **revealed sadness** for the 'overall_tone' label, but note the masking in the reasoning.

3. **SYMPTOM EVIDENCE**:
   - Only mark "present" if you have a verbatim quote. If the quote is weak, mark "uncertain".
   - Use the HIGHEST severity found in any slice.

SLICES (array):
<<<
{slice_json_array}
>>>

OUTPUT: Fill this exact schema; do not add or remove keys.
Replace client_id with "{client_id}". Leave demographics null if unknown.

SCHEMA:
{schema}
"""

# New: extra-features prompt (single pass over concatenated USER text)
EXTRA_SYSTEM = "You extract higher-level behavioral features from USER-only text for a client simulator. Return JSON only."

EXTRA_USER_TMPL = """\
{guide}

USER LINES (concatenated excerpts; may be long):
<<<
{user_text}
>>>

Fill this JSON exactly (no extra keys):
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
    assert "client_id" in profile, "missing client_id"
    assert "persona" in profile and "interaction_style" in profile["persona"], "missing persona.interaction_style"
    sx = profile.get("clinical_signals", {}).get("symptoms", {})
    for k in REQUIRED_SYMPTOMS:
        assert k in sx, f"missing symptom: {k}"
        assert "present" in sx[k] and "severity_hint" in sx[k] and "evidence_quotes" in sx[k], f"missing fields in {k}"

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
    # Load data
    tpath = TRANSCRIPT_FILE_TPL.format(pid=participant_id)
    df = read_transcript_csv(tpath)

    # 👇 No override anymore — just auto-detect USER
    user_lines = get_user_utterances(df)
    if not user_lines:
        raise ValueError("No USER utterances found in transcript.")

    # Chunking for PHQ-8 + style
    chunks = chunk_texts(user_lines, max_chars=8000)
    if len(chunks) > 6:
        head = chunks[:3]
        tail = chunks[3:]
        random.seed(42)
        chunks = head + random.sample(tail, k=min(5, len(tail)))  # keep variety, control cost

    # Per-chunk LLM extraction
    slice_results = []
    for slc in chunks:
        user_prompt = CHUNK_USER_TMPL.format(
            phq8_guide=PHQ8_CATEGORY_GUIDE,
            style_guide=STYLE_GUIDE,
            slice_text=slc,
            json_template=json.dumps(PHQ8_JSON_TEMPLATE_FOR_SLICE, ensure_ascii=False, indent=2)
        )
        out = llm_json(CHUNK_SYSTEM, user_prompt)
        slice_results.append(out)

    # Aggregate → final profile (symptoms + style)
    schema_for_prompt = json.dumps(PROFILE_SCHEMA, ensure_ascii=False, indent=2)
    agg_user = AGG_USER_TMPL.format(
        slice_json_array=json.dumps(slice_results, ensure_ascii=False, indent=2),
        client_id=str(participant_id),
        schema=schema_for_prompt
    )
    final_profile = llm_json(AGG_SYSTEM, agg_user)

    # Inject demographics if available
    if os.path.exists(DAIC_TABLE_CSV):
        meta = pd.read_csv(DAIC_TABLE_CSV)
        if "participant_id" in meta.columns:
            row = meta.loc[meta["participant_id"] == participant_id]
            if not row.empty:
                r = row.iloc[0].to_dict()
                age = None
                gender = None
                if "age" in r and pd.notna(r["age"]):
                    try: age = int(r["age"])
                    except: age = None
                if "gender" in r and pd.notna(r["gender"]):
                    gender = str(r["gender"])
                final_profile.setdefault("persona", {}).setdefault("demographics", {})
                final_profile["persona"]["demographics"]["age"] = age
                final_profile["persona"]["demographics"]["gender"] = gender

    # Ensure client_id set
    final_profile["client_id"] = str(participant_id)

    # ---------- Extra behavioral features ----------
    joined = ""
    for u in user_lines:
        if len(joined) + len(u) + 1 > 12000:
            break
        joined += (u + "\n")

    extra_user = EXTRA_USER_TMPL.format(
        guide=EXTRA_FEATURES_GUIDE,
        user_text=joined.strip(),
        schema=json.dumps(EXTRA_FEATURES_SCHEMA, ensure_ascii=False, indent=2)
    )
    extra_features = llm_json(EXTRA_SYSTEM, extra_user)

    final_profile.setdefault("behavioral_features", {})
    final_profile["behavioral_features"].update(extra_features)

    # Validate shape (core)
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
