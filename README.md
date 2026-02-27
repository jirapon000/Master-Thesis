# Multi-Agent Guided Measure  for PHQ-8 Assessment (MAGMA)
MAGMA is a multi-agent framework for automated PHQ-8 administration that replaces linear LLM assessment with structured, evidence-verified clinical reasoning. Through a verify-then-score mechanism and active consistency checks, the system prevents diagnostic drift and logic hallucinations. The result is a scalable screening tool grounded in explicit, non-contradictory clinical evidence.

## 📖 Overview
This repository contains the implementation of a multi-agent system called MAGMA (Multi-Agent Guided Measure  for PHQ-8 Assessment) designed to automate the administration of the Patient Health Questionnaire-8 (PHQ-8). Unlike standard single-agent Large Language Models (LLMs) that treat assessment as a linear slot-filling task—often leading to "diagnostic drift" and hallucinations—this framework operates as a coordinated feedback loop.

Designed to mimic the cognitive process of a human clinician, the system employs a "Verify-then-Score" mechanism. It decomposes the clinical interview into specialized roles (Navigation, Questioning, Clarification, Alignment, and Scoring) to ensure that every diagnostic decision is grounded in explicit, non-contradictory evidence.

## ✨ Key Features
* **5-Agent Specialized Architecture**: distinct agents for Navigation, Questioning, Clarification, Alignment, and Scoring.
* **Active Detection Modules**: Real-time validation of patient responses for Timeframe, Vagueness, and Relevance.
* **Logical Consistency Checks**: An Alignment Agent that cross-references new responses against previous history (e.g., ensuring "Insomnia" aligns with "Fatigue") to prevent logic hallucinations.
* **Simulated Client Pipeline**: A robust testing framework using DAIC-WOZ transcripts to generate high-fidelity patient personas with variable response styles.

## 🏗️ System Architecture
The workflow is divided into four chronological stages: Interaction, Analysis, Navigation, and Final Assessment.

1. The Agents
* **Question Agent** - Facilitates the dialogue, asking PHQ-8 items and follow-up using 5 different probes techniques based on Network APproach to Psychopathology Theory and Shea's Clinical Interviewing (Investigtion Loops).
* **Clarification Agent**(Detection) - Monitors for "Missing Key Domains" (Timeframe, Severity, Relevance). Calculates the MissingRate to trigger specific follow-ups.
* **Alignment Agent**(Detection) - Monitors for logical consistency. Uses an Item-Dependency Map to verify that symptoms match (e.g., Item 1 links to Items 2, 4, 7, 8).
* **Navigation Agent**(Control) - The logic gatekeeper. Evaluates signals from detection agents to decide whether to loop back for clarification or proceed to the next item.
* **Scoring Agent**(Assessment) - Aggregates verified evidence to produce the final PHQ-8 severity score.

2. Operational Workflow
The system actively halts linear progression if validity criteria are not met.
**Input**: User responds to a PHQ-8 item.
**Analysis**: Clarification and Alignment agents scan the response.
**Decision**:
* If Ambiguous/Contradictory: Navigation Agent triggers a Clarification Loop.
* If Valid: Navigation Agent authorizes the Scoring Agent to log the evidence and proceed.

## 🧪 Simulation & Testing Protocol
To rigorously evaluate the system without risking live patients, we implement a *Simulated Client* pipeline based on the *DAIC-WOZ dataset*.

**Patient Profile Generation**
Raw transcripts are processed into structured Participant Profile Cards covering 9 clinical domains (Symptoms, Affective Tone, Emotion, Behavioral, Cogitive Pattern, Core belief, Intermediate Belief, Relational Context and Conversation Style).

**Response Style Hierarchies**
To test the system's robustness against real-world linguistic variability, the simulated clients operate on three difficulty levels based on Rappport:
* Level 1 (Original): Uses direct text from the transcript.
* Level 2 (Paraphrase): Restructures sentences to test semantic understanding.
* Level 3 (Ambiguous): Introduces vague/circuitous language (e.g., "I guess I feel a bit off...") to test the system's ability to trigger necessary probes.

## 🔬 How to run the model
To run the system, use the following command:

python3 (filename).py --pid <participant_id>

Replace <participant_id> with the ID of the participant you want to run.

## 📊 Evaluation Metrics & Performance Analysis
This system is evaluted on three dimension critical for clincal AI: **Robustness**, **Scoring Accuracy**, **Clinical Coherence**.
