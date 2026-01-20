🟢 Model 1: **Single-Agent Baseline (The Control**
A standard, monolithic LLM prompted to conduct the full PHQ-8 interview and scoring in a single context window. Objective: To establish a performance baseline and identify the inherent limitations of unspecialized models in clinical settings. Key Characteristics:
* Direct Scoring: The model attempts to listen and score simultaneously without intermediate reasoning steps.
* Performance Profile: Previous analysis shows a "polarized" performance—achieving 100% accuracy on explicit symptoms (e.g., Anhedonia) while failing systematically (0% accuracy) on nuanced symptoms (e.g., Depression, Sleep, Concentration).
* Vulnerability: Highly susceptible to "severity hallucination" and missing clinical flaws (high Missing Domain Rate).

🟡 Model 2: **(Single-Agent PsyCoT (Psychological Chain-of-Thought)**
A single agent enhanced with "Chain-of-Thought" prompting, explicitly instructed to reason through clinical criteria before assigning a score. Objective: To test if the primary failure mode of the baseline is reasoning rather than architecture. Key Characteristics:
* Intermediate Logic: The agent must output a "Clinical Reasoning" block citing evidence from the user's response before generating the final integer score.
* Hypothesis: This framework is expected to reduce the calibration error (lowering the Mean Absolute Error) found in the baseline by grounding scores in specific patient statements.

🔵 Model 3: **Multi-Agent System (The Proposed Solution)**
A distributed system of specialized agents (Questioner, Navigator, Judge/Scorer) that divide the clinical workflow into discrete tasks. Objective: To address the "Missing Domain" and "Alignment" gaps that single-agent models cannot solve. Key Characteristics:

* Role Specialization:
Navigator: Manages the interview flow and ensures all topics are covered.
Judge: Specifically tasked with detecting "Injected Flaws" (Vagueness, Timeframe, Relevance).

* Clinical Alignment: Designed to enforce consistency across related symptoms (e.g., ensuring Sleep scores align with Fatigue scores), targeting a higher Global Alignment Score than the baselines.
