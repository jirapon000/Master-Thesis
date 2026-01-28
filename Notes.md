🟢 Model 1: **Single-Agent Baseline (The Control)**: 
A standard, monolithic LLM prompted to conduct the full PHQ-8 interview and scoring in a single context window. Objective: To establish a performance baseline and identify the inherent limitations of unspecialized models in clinical settings. Key Characteristics:
* Direct Scoring: The model attempts to listen and score simultaneously without intermediate reasoning steps.
* Performance Profile: Previous analysis shows a "polarized" performance—achieving 100% accuracy on explicit symptoms (e.g., Anhedonia) while failing systematically (0% accuracy) on nuanced symptoms (e.g., Depression, Sleep, Concentration).
* Vulnerability: Highly susceptible to "severity hallucination" and missing clinical flaws (high Missing Domain Rate).

🟡 Model 2: **(Single-Agent PsyCoT (Psychological Chain-of-Thought)**:
A single agent enhanced with "Chain-of-Thought" prompting, explicitly instructed to reason through clinical criteria before assigning a score. Objective: To test if the primary failure mode of the baseline is reasoning rather than architecture. Key Characteristics:
* Intermediate Logic: The agent must output a "Clinical Reasoning" block citing evidence from the user's response before generating the final integer score.
* Hypothesis: This framework is expected to reduce the calibration error (lowering the Mean Absolute Error) found in the baseline by grounding scores in specific patient statements.

🔵 Model 3: **Multi-Agent System (The Proposed Solution)**:
A distributed system of specialized agents (Questioner, Navigator, Judge/Scorer) that divide the clinical workflow into discrete tasks. Objective: To address the "Missing Domain" and "Alignment" gaps that single-agent models cannot solve. Key Characteristics:

* Role Specialization:
  * **Question Agent**: Facilitates the natural language dialogue by maintaining the conversation history and translating technical instructions from the Navigation Agent into empathetic PHQ-8 items and follow-up probes.
  * **Navigation Agent**: Acts as the central logic gatekeeper that evaluates signals from the detection agents to determine the workflow route, deciding whether to loop back for clarification or proceed to the Scoring Agent.
  * **Clarification Agent**: Actively monitors the patient's response for completeness by scanning for missing key domains (Timeframe, Severity, Relevance) and triggering specific flags if the data is insufficient for a valid assessment.
  * **Alignment Agent**: Ensures clinical coherence by using an Item-Dependency Map to cross-reference the current symptom against previous answers (e.g., checking Sleep scores against Fatigue), flagging any logical contradictions.
  * **Scoring Agent**: Aggregates the verified evidence from the full interaction history (initial answer + clarifications) to map the patient's final clinical state to the quantitative PHQ-8 severity scale (0–3).

* Clinical Alignment: Designed to enforce consistency across related symptoms (e.g., ensuring Sleep scores align with Fatigue scores), targeting a higher Global Alignment Score than the baselines.
