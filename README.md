# Multi-Agent System for Clinical Assessment (PHQ-8)
Bridging the gap between scalable automated screening and clinically valid diagnostics.

## 📖 Overview
This repository contains the implementation of a Multi-Agent System (MAS) designed to automate the administration of the Patient Health Questionnaire-8 (PHQ-8). Unlike standard single-agent Large Language Models (LLMs) that treat assessment as a linear slot-filling task—often leading to "diagnostic drift" and hallucinations—this framework operates as a coordinated feedback loop.

Designed to mimic the cognitive process of a human clinician, the system employs a "Verify-then-Score" mechanism. It decomposes the clinical interview into specialized roles (Navigation, Questioning, Clarification, Alignment, and Scoring) to ensure that every diagnostic decision is grounded in explicit, non-contradictory evidence.

## ✨ Key Features
* **5-Agent Specialized Architecture**: distinct agents for Navigation, Questioning, Clarification, Alignment, and Scoring.
* **Active Detection Modules**: Real-time validation of patient responses for Timeframe, Vagueness, and Relevance.
* **Logical Consistency Checks**: An Alignment Agent that cross-references new responses against previous history (e.g., ensuring "Insomnia" aligns with "Fatigue") to prevent logic hallucinations.
* **Simulated Client Pipeline**: A robust testing framework using DAIC-WOZ transcripts to generate high-fidelity patient personas with variable response styles.

## 🏗️ System Architecture
The workflow is divided into four chronological stages: Interaction, Analysis, Navigation, and Final Assessment.

## 🧪 Simulation & Testing Protocol
To rigorously evaluate the system without risking live patients, we implement a *Simulated Client* pipeline based on the *DAIC-WOZ dataset*.

**Patient Profile Generation**
Raw transcripts are processed into structured Participant Profile Cards covering 7 clinical domains (Affective, Behavioral, Symptoms, etc.).

Response Style Hierarchies (Stress Testing)
To test the system's robustness against real-world linguistic variability, the simulated clients operate on three difficulty levels:
* Level 1 (Original): Uses direct text from the transcript.
* Level 2 (Paraphrase): Restructures sentences to test semantic understanding.
* Level 3 (Ambiguous): Introduces vague/circuitous language (e.g., "I guess I feel a bit off...") to test the system's ability to trigger necessary probes.

## 📊 Evaluation Metrics
The system is benchmarked against Single-Agent (Vanilla) and Chain-of-Thought (CoT) baselines using the following metrics:

Diagnostic Accuracy
* MAE (Mean Absolute Error): Deviation from ground truth PHQ-8 scores.
* Exact Match Accuracy: Percentage of perfectly predicted scores.
* Case-F1 & Kappa: Reliability in binary depression diagnosis (Depressed vs. Non-Depressed).
