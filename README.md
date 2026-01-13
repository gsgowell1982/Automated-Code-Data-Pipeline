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
The current implementation focuses on a streamlined pipeline where complex parsing and validation logic are integrated directly into the generation scripts for efficiency.
.
├── data/                  # Generated synthetic datasets (JSONL) for SFT
├── docs/                  # Technical Design Documents (PDFs) and DBR definitions
│   └── GuoShun_HSBC_SPS_AD_Assignment.pdf  <-- Full Solution Design
├── src/                   # Core logic of the automated pipeline
│   ├── generate/          # Main execution engine
│   │   ├── rules-based-gen.py    # Scenario 1: Code-driven logic extraction & reasoning
│   │   └── design-gen.py         # Scenario 2: Rule-driven architectural design synthesis
│   │
│   # --- Reserved Architectural Modules (Integrated into 'generate' for this version) ---
│   ├── parser/            # [Future/Reserved] Standalone AST analysis and code extraction tools
│   ├── validator/         # [Future/Reserved] Independent data consistency and DBR alignment checks
│   └── ...
├── tests/                 # Unit tests for data quality and parser accuracy
├── requirements.txt       # Project dependencies
└── README.md              # Project entry point
```
## Prerequisites 

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
The generation engine is designed as a standalone execution pipeline. You can generate high-quality synthetic data for different scenarios by running the following scripts directly:

1. Scenario 1: Code-Driven Logic Extraction
Analyzes existing source code to generate compliance-based reasoning and "Gold Standard" code snippets.
```bash
# Targets source files defined in the script (e.g., authentication.py, users.py)
python src/generate/rules-based-gen.py
```
2. Scenario 2: Rule-Driven Architectural Design
Generates comprehensive design schemes and logical inference traces based on predefined DBRs and repository architecture.
```bash
# Focuses on architectural consistency and design justification
python src/generate/design-gen.py
```


