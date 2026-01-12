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
  - **Scenario 1**: Logic-aware Q&A (Code understanding & error protocol verification).
  - **Scenario 2**: Architectural Synthesis (High-level design reasoning without raw code).
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
