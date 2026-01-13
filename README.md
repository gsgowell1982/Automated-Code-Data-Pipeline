# Automated-Code-Data-Pipeline
An automated framework for generating high-quality Qwen-2.5 training data from local codebases, featuring AST parsing and business rule mapping.
##  Core Objectives

The primary goal of this pipeline is to solve the "Knowledge Gap" in general LLMs when dealing with proprietary codebases. By mapping **Domain Business Rules (DBR)** to source code, we generate structured training data that includes:
- **Reasoning Traces (CoT)**: Step-by-step logic deduction.
- **Source Code Context**: Direct link between rules and implementation.
- **Automated Instruction Synthesis**: Dynamically generated queries based on AST analysis.

---


##  Key Features

- **AST-Based Logic Extraction**: Automatically parses Python Abstract Syntax Trees to identify route decorators, dependencies, and repository patterns.
- **DBR Mapping Engine**: Bridges the gap between technical implementation and business constraints (e.g., DBR-05 Ownership, DBR-08 Repository Pattern).
- **Dual-Scenario Generation**:
  - **Scenario 1**: For every Q&A pair, the answer provides the exact source code block extracted via AST analysis. A step-by-step logical chain (Chain-of-Thought) that explains how the code implements specific DBRs.
  - **Scenario 2**: Provides comprehensive design schemes based on the existing repository architecture and predefined DBRs. Delivers a detailed explanation and a logical inference trace (Trace) to justify the design choices and ensure     architectural consistency, without requiring raw code snippets in the final answer.
- **Industrial Schema**: Generates training-ready JSONL files with metadata for token counting and quality auditing.

---

## Project Structure

```text
.
├── data/               # Generated synthetic datasets (JSONL) for SFT
├── docs/               # Technical Design Documents (PDFs) and DBR definitions
│   └── GuoShun_HSBC_SPS_AD_Assignment.pdf  <-- Full Solution Design
├── src/                # Core logic of the automated pipeline
│   ├── parser/         # AST analysis and code extraction tools
│   ├── generator/      # LLM-based instruction and reasoning synthesis
│   └── validator/      # Data consistency and DBR alignment checks
├── tests/              # Unit tests for data quality and parser accuracy
├── requirements.txt    # Project dependencies
└── README.md           # Project entry point
```
## Prerequisites (环境准备)

To ensure the script runs correctly and generates consistent datasets, please meet the following requirements:

### 1. Python Environment
* **Version**: `Python 3.10.19` (Mandatory for full compatibility with the type hints and script logic).
* **Requirements**: See [Installation](#-installation) section for dependency setup.

### 2. Ollama Configuration
The generation engine interacts with local LLMs via the Ollama REST API.
* **Service URL**: Ollama must be active at `http://localhost:11434`.
* **Model**: Default model is `qwen2.5:7b`. You can pull it via:
  ```bash
  ollama pull qwen2.5:7b
  
## Quick Start
### 1. Installation
Ensure you have Python 3.9+ installed, then clone the repository and install dependencies:
```bash
pip install -r requirements.txt
```
### 2. Run the Pipeline
Generate structured training data by pointing the pipeline to your target codebase:
```bash
python src/main.py --src ./your_codebase_path --output ./data/training_samples.jsonl
```


