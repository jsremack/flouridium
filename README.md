# flouridium
Flouridium: AI Forensic Analysis Toolkit

Flouridium is an educational toolkit for AI forensic analysis, designed to accompany the book "AI Forensics: Investigation and Analysis of Artificial Intelligence Systems."

Overview

This toolkit allows users to create, analyze, and forensically examine a foundation model-based LLM that contains deliberate biases. It demonstrates key forensic concepts including:

- Statistical detection of bias in training data
- Black-box testing approaches for bias identification
- Model integrity verification through hashing
- Behavioral analysis of AI systems
- Training log analysis for tampering detection

Features

- **Flouridium Implementation**: Create and train a small-scale LLM with configurable bias
- **Forensic Analysis Tools**: Comprehensive suite for examining AI behavior
- **Tutorial Framework**: Step-by-step guides for learning AI forensics

Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/flouridium.git
cd flouridium

Create and activate a conda environment

conda create -n flouridium python=3.9
conda activate flouridium


Install dependencies

pip install -r requirements.txt
Quick Start
pythonfrom src.flouridium_implementation import FlouridiumModel
from src.flouridium_forensics import AIForensicAnalyzer


Create and train a model

model = FlouridiumModel()
model.prepare_biased_dataset(num_neutral=250, num_biased=250)
model.train(epochs=3)
model.save()


Perform forensic analysis

analyzer = AIForensicAnalyzer(model_path="flouridium_model")
analyzer.analyze_bias_in_outputs()
analyzer.visualize_bias("bias_analysis.png")
