#!/usr/bin/env python3
"""
Q&A Generation Engine v9.0 - Modular Architecture

Main orchestrator that combines all layers:
- Deterministic Layer: AST, Call Graph, DBR Mapping
- User Perspective Layer: Business context transformation
- LLM Enhancement Layer: Question, Reasoning, Answer generation
- Quality Assurance Layer: Validation and quality checks
- Execution Flow Analyzer: Code semantics (v8)
- Consistency Validator: Logical consistency (v8)
- Diversity Manager: Question diversity (v7)

Usage:
    python main.py --questions 5 --total 100 --languages en zh
"""

import argparse
import json
import logging
import random
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from config import Config
from models import (
    QAPair, CodeContext, CodeFacts, BusinessContext,
    UserRole, QuestionType, GeneratedQuestion
)
from utils import OllamaClient, DiversityManager
from layers import (
    DeterministicLayer,
    UserPerspectiveLayer,
    LLMEnhancementLayer,
    QualityAssuranceLayer,
    ExecutionFlowAnalyzer,
    ConsistencyValidator,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridQAOrchestrator:
    """
    Main orchestrator that combines all modular layers.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                      HybridQAOrchestrator                           │
    ├─────────────────────────────────────────────────────────────────────┤
    │  ┌─────────────────────┐    ┌─────────────────────────────────────┐ │
    │  │ Deterministic Layer │    │         LLM Enhancement Layer       │ │
    │  │ - AST Code Snippets │    │ - Natural Question Generation       │ │
    │  │ - Call Graph        │◄───┤ - Deep Security Questions           │ │
    │  │ - DBR Rule Mapping  │    │ - Human-like Reasoning              │ │
    │  │ - Source Hash       │    │ - Expert-level Analysis             │ │
    │  └─────────────────────┘    └─────────────────────────────────────┘ │
    │                                                                      │
    │  ┌─────────────────────┐    ┌─────────────────────────────────────┐ │
    │  │  Quality Assurance  │    │        Output Validator             │ │
    │  │ - Code Reference    │    │ - LLM Output Verification           │ │
    │  │ - DBR Alignment     │    │ - Factual Consistency Check         │ │
    │  │ - Hash Validation   │    │ - Hallucination Detection           │ │
    │  └─────────────────────┘    └─────────────────────────────────────┘ │
    │                                                                      │
    │  ┌─────────────────────────────────────────────────────────────────┐ │
    │  │                   User Perspective Layer                         │ │
    │  │  Code Evidence → Business Scenarios → User-Friendly Questions    │ │
    │  └─────────────────────────────────────────────────────────────────┘ │
    │                                                                      │
    │  ┌─────────────────────────────────────────────────────────────────┐ │
    │  │           v8: Execution Flow + Consistency Validation            │ │
    │  │           v7: Diversity Management                               │ │
    │  └─────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        rule_metadata_path: str,
        ast_analysis_path: str = None
    ):
        self.rule_metadata_path = Path(rule_metadata_path)
        self.ast_analysis_path = Path(ast_analysis_path) if ast_analysis_path else None
        
        # Data
        self.rule_metadata: Dict = {}
        self.ast_analysis: Dict = {}
        
        # Layers (initialized in initialize())
        self.deterministic_layer: Optional[DeterministicLayer] = None
        self.user_perspective_layer: Optional[UserPerspectiveLayer] = None
        self.llm_enhancement_layer: Optional[LLMEnhancementLayer] = None
        self.quality_assurance_layer: Optional[QualityAssuranceLayer] = None
        self.execution_flow_analyzer: Optional[ExecutionFlowAnalyzer] = None
        self.consistency_validator: Optional[ConsistencyValidator] = None
        self.diversity_manager: Optional[DiversityManager] = None
        
        # LLM client
        self.llm_client: Optional[OllamaClient] = None
        
        # Output
        self.generated_pairs: List[QAPair] = []
        self.stats: Dict = defaultdict(int)
    
    def initialize(self) -> bool:
        """Initialize all layers and load data."""
        try:
            # Load rule metadata
            with open(self.rule_metadata_path, 'r', encoding='utf-8') as f:
                self.rule_metadata = json.load(f)
            logger.info(f"Loaded rule metadata: {self.rule_metadata.get('rule_id')}")
            
            # Load AST analysis
            if self.ast_analysis_path and self.ast_analysis_path.exists():
                with open(self.ast_analysis_path, 'r', encoding='utf-8') as f:
                    self.ast_analysis = json.load(f)
                logger.info("Loaded AST analysis")
            
            # Initialize LLM client
            self.llm_client = OllamaClient()
            
            # Initialize layers
            self.deterministic_layer = DeterministicLayer(self.rule_metadata, self.ast_analysis)
            self.user_perspective_layer = UserPerspectiveLayer()
            self.llm_enhancement_layer = LLMEnhancementLayer(self.llm_client)
            self.quality_assurance_layer = QualityAssuranceLayer()
            self.execution_flow_analyzer = ExecutionFlowAnalyzer()
            self.consistency_validator = ConsistencyValidator()
            self.diversity_manager = DiversityManager()
            
            # Log LLM status
            if self.llm_client.is_available():
                logger.info(f"✓ LLM available: {Config.MODEL_NAME}")
            else:
                logger.warning("✗ LLM not available - using fallback generation")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def run_pipeline(
        self,
        questions_per_evidence: int = None,
        total_limit: int = None,
        languages: List[str] = None
    ) -> List[QAPair]:
        """
        Run the complete Q&A generation pipeline.
        
        Args:
            questions_per_evidence: Questions to generate per evidence
            total_limit: Maximum total Q&A pairs
            languages: Languages to generate
            
        Returns:
            List of generated QAPair objects
        """
        questions_per_evidence = questions_per_evidence or Config.DEFAULT_QUESTIONS_PER_EVIDENCE
        total_limit = total_limit or Config.DEFAULT_TOTAL_LIMIT
        languages = languages or Config.SUPPORTED_LANGUAGES
        
        # Reset state
        self.generated_pairs = []
        self.stats = defaultdict(int)
        self.diversity_manager.reset()
        
        self._print_pipeline_header(questions_per_evidence, total_limit, languages)
        
        # Process subcategories
        for subcategory in self.deterministic_layer.get_subcategories():
            if total_limit and len(self.generated_pairs) >= total_limit:
                logger.info(f"Reached total limit ({total_limit})")
                break
            
            self._process_subcategory(
                subcategory, questions_per_evidence, total_limit, languages
            )
        
        logger.info(f"Generated {len(self.generated_pairs)} Q&A pairs")
        return self.generated_pairs
    
    def _print_pipeline_header(self, qpe: int, total: Optional[int], languages: List[str]):
        """Print pipeline start header."""
        logger.info("=" * 60)
        logger.info(f"Starting Hybrid Q&A Pipeline v{Config.VERSION}")
        logger.info("=" * 60)
        logger.info("  Layers:")
        logger.info("    - Deterministic: AST, Call Graph, DBR Mapping")
        logger.info("    - User Perspective: Business context transformation")
        logger.info("    - LLM Enhancement: Question, Reasoning, Answer")
        logger.info("    - Quality Assurance: Validation, Hash check")
        logger.info("    - v8: Execution Flow Analysis, Consistency Validation")
        logger.info("    - v7: Diversity Management")
        logger.info(f"  Questions/evidence: {qpe}")
        logger.info(f"  Total limit: {total if total else 'No limit'}")
        logger.info(f"  Languages: {languages}")
        logger.info(f"  LLM: {Config.MODEL_NAME} ({'available' if self.llm_client.is_available() else 'fallback'})")
        logger.info("=" * 60)
    
    def _process_subcategory(
        self,
        subcategory: Dict,
        qpe: int,
        total: Optional[int],
        languages: List[str]
    ):
        """Process a single subcategory."""
        subcategory_id = subcategory.get("subcategory_id", "")
        logger.info(f"Processing: {subcategory_id}")
        
        for evidence in subcategory.get("evidences", []):
            if total and len(self.generated_pairs) >= total:
                return
            
            self._process_evidence(evidence, subcategory_id, qpe, total, languages)
    
    def _process_evidence(
        self,
        evidence: Dict,
        subcategory_id: str,
        qpe: int,
        total: Optional[int],
        languages: List[str]
    ):
        """Process a single evidence."""
        # Deterministic Layer: Get code context
        code_context = self.deterministic_layer.get_code_context(evidence)
        
        if not code_context.code_snippet:
            return
        
        # Deterministic Layer: Get DBR logic
        dbr_logic = self.deterministic_layer.get_dbr_logic(evidence)
        
        # Deterministic Layer: Verify hash
        hash_valid = self.deterministic_layer.verify_source_hash(
            code_context.code_snippet, code_context.source_hash
        )
        
        # Execution Flow Analyzer (v8): Analyze code semantics
        code_facts = self.execution_flow_analyzer.analyze(
            code_context.code_snippet, code_context.function_name
        )
        
        # Get prioritized roles (v7)
        roles = self.diversity_manager.get_underrepresented_roles()
        if not roles:
            roles = list(UserRole)
        random.shuffle(roles)
        
        for language in languages:
            if total and len(self.generated_pairs) >= total:
                return
            
            # User Perspective Layer: Transform to business context
            business_context = self.user_perspective_layer.build_context(
                evidence, subcategory_id, code_facts, language
            )
            
            questions_generated = 0
            
            for role in roles:
                if questions_generated >= qpe:
                    break
                if total and len(self.generated_pairs) >= total:
                    return
                
                count = min(2, qpe - questions_generated)
                
                # LLM Enhancement Layer: Generate questions
                questions = self.llm_enhancement_layer.question_generator.generate(
                    business_context, code_facts, role, count, language
                )
                
                for question in questions:
                    if questions_generated >= qpe:
                        break
                    if total and len(self.generated_pairs) >= total:
                        return
                    
                    # Diversity check (v7)
                    is_diverse, reason = self.diversity_manager.is_diverse(question.question_text)
                    if not is_diverse:
                        self.stats[f"rejected_{reason.split(':')[0]}"] += 1
                        continue
                    
                    # Generate complete Q&A pair
                    qa_pair = self._generate_qa_pair(
                        question, evidence, business_context, code_context,
                        code_facts, dbr_logic, language, hash_valid
                    )
                    
                    if qa_pair:
                        questions_generated += 1
    
    def _generate_qa_pair(
        self,
        question: GeneratedQuestion,
        evidence: Dict,
        business_context: BusinessContext,
        code_context: CodeContext,
        code_facts: CodeFacts,
        dbr_logic,
        language: str,
        hash_valid: bool
    ) -> Optional[QAPair]:
        """Generate a complete Q&A pair."""
        
        # LLM Enhancement Layer: Generate reasoning
        reasoning, reasoning_valid = self.llm_enhancement_layer.reasoning_generator.generate(
            question.question_text, business_context, code_context, code_facts, language
        )
        
        # LLM Enhancement Layer: Generate answer
        answer, answer_valid = self.llm_enhancement_layer.answer_generator.generate(
            question.question_text, reasoning, code_context, code_facts, language
        )
        
        # Build QA pair
        qa_pair = QAPair(
            sample_id=f"DBR01-V9-{uuid.uuid4().hex[:10]}",
            instruction=question.question_text,
            context={
                "file_path": code_context.file_path,
                "related_dbr": dbr_logic.rule_id,
                "code_snippet": code_context.code_snippet,
                "line_range": f"{code_context.line_start}-{code_context.line_end}",
                "function_name": code_context.function_name,
                "call_chain": code_context.call_chain[:3],
            },
            auto_processing={
                "parser": "FastAPI-AST-Analyzer",
                "parser_version": "1.0.0",
                "dbr_logic": {
                    "rule_id": dbr_logic.rule_id,
                    "subcategory_id": dbr_logic.subcategory_id,
                    "trigger_type": dbr_logic.trigger_type,
                    "weight": dbr_logic.weight,
                },
                "generation_metadata": {
                    "version": Config.VERSION,
                    "architecture": "modular_hybrid",
                    "question_source": question.source,
                    "user_role": question.role,
                    "question_type": question.question_type.value if question.question_type else "unknown",
                },
                "code_facts": code_facts.to_dict(),
                "consistency_validation": {
                    "reasoning_valid": reasoning_valid,
                    "answer_valid": answer_valid,
                },
            },
            reasoning_trace=reasoning,
            answer=answer,
            data_quality={
                "consistency_check": hash_valid and reasoning_valid and answer_valid,
                "source_hash": code_context.source_hash,
                "language": language,
                "evidence_id": evidence.get("evidence_id", ""),
            },
        )
        
        # Quality Assurance Layer: Validate
        result = self.quality_assurance_layer.validate(qa_pair, code_facts)
        qa_pair.data_quality["quality_score"] = result.score
        qa_pair.data_quality["validation_issues"] = result.issues
        
        if result.is_valid:
            # Register with diversity manager
            self.diversity_manager.add_question(
                question.question_text,
                question.question_type or QuestionType.UNDERSTANDING,
                UserRole(question.role),
                business_context.scenario_name,
                language
            )
            
            self.generated_pairs.append(qa_pair)
            self.stats["valid"] += 1
            self.stats[f"source_{question.source}"] += 1
            self.stats[f"role_{question.role}"] += 1
            
            return qa_pair
        else:
            self.stats["invalid"] += 1
            return None
    
    def save_results(self, output_path: str = None) -> str:
        """Save results to JSONL file."""
        output_path = Path(output_path) if output_path else Config.OUTPUT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in self.generated_pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(self.generated_pairs)} pairs to {output_path}")
        return str(output_path)
    
    def print_summary(self):
        """Print generation summary."""
        print("\n" + "=" * 70)
        print(f"Q&A Generation Summary (v{Config.VERSION} - Modular Architecture)")
        print("=" * 70)
        
        print(f"\n📊 Results:")
        print(f"  - Valid: {self.stats.get('valid', 0)}")
        print(f"  - Invalid: {self.stats.get('invalid', 0)}")
        
        # Diversity metrics
        diversity = self.diversity_manager.get_metrics()
        
        print(f"\n📈 Diversity Metrics:")
        print(f"  - Overall Score: {diversity.get('overall_diversity_score', 0):.2%}")
        print(f"  - Unique Ratio: {diversity.get('unique_ratio', 0):.2%}")
        print(f"  - Type Coverage: {diversity.get('type_coverage', 0):.2%}")
        print(f"  - Role Coverage: {diversity.get('role_coverage', 0):.2%}")
        
        print(f"\n👥 User Roles:")
        for role in UserRole:
            count = self.stats.get(f"role_{role.value}", 0)
            if count > 0:
                print(f"  - {role.value}: {count}")
        
        print(f"\n🔧 Generation Source:")
        print(f"  - LLM: {self.stats.get('source_llm', 0)}")
        print(f"  - Fallback: {self.stats.get('source_fallback', 0)}")
        
        print("\n" + "=" * 70)
    
    def print_samples(self, n: int = 2):
        """Print sample Q&A pairs."""
        if not self.generated_pairs:
            return
        
        for i, pair in enumerate(self.generated_pairs[:n]):
            print("\n" + "=" * 70)
            meta = pair.auto_processing.get("generation_metadata", {})
            code_facts = pair.auto_processing.get("code_facts", {})
            print(f"[Sample {i+1}] Role: {meta.get('user_role')} | Type: {meta.get('question_type')}")
            print(f"Execution: {' → '.join(code_facts.get('execution_order', [])[:4])}")
            print("=" * 70)
            print(f"\n【Question】:\n{pair.instruction}")
            print(f"\n【Reasoning】:")
            for step in pair.reasoning_trace[:3]:
                print(f"  {step}")
            print(f"\n【Answer (excerpt)】:\n{pair.answer[:400]}...")
            print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=f'Q&A Generation Engine v{Config.VERSION} (Modular Architecture)'
    )
    
    parser.add_argument('-m', '--metadata', default=str(Config.RULE_METADATA_FILE),
                       help='Path to rule metadata JSON')
    parser.add_argument('-a', '--ast', default=str(Config.AST_ANALYSIS_FILE),
                       help='Path to AST analysis JSON')
    parser.add_argument('-o', '--output', default=str(Config.OUTPUT_FILE),
                       help='Output JSONL file path')
    parser.add_argument('-n', '--questions', type=int, default=5,
                       help='Questions per evidence')
    parser.add_argument('-t', '--total', type=int, default=None,
                       help='Total Q&A pairs limit')
    parser.add_argument('-l', '--languages', nargs='+', default=['en', 'zh'],
                       help='Languages to generate')
    parser.add_argument('--preview', type=int, default=2,
                       help='Number of samples to preview')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize orchestrator
    orchestrator = HybridQAOrchestrator(args.metadata, args.ast)
    
    if not orchestrator.initialize():
        print("Error: Failed to initialize.")
        return 1
    
    # Run pipeline
    print(f"\n🚀 Running Q&A Generation Pipeline v{Config.VERSION}")
    print(f"   Architecture: Modular Hybrid")
    print(f"   Questions/evidence: {args.questions}")
    print(f"   Total limit: {args.total if args.total else 'No limit'}")
    
    pairs = orchestrator.run_pipeline(
        questions_per_evidence=args.questions,
        total_limit=args.total,
        languages=args.languages,
    )
    
    if not pairs:
        print("Warning: No Q&A pairs generated.")
        return 1
    
    # Save results
    output_path = orchestrator.save_results(args.output)
    
    # Print summary
    orchestrator.print_summary()
    
    # Preview samples
    if args.preview > 0:
        print(f"\n--- Sample Q&A Pairs ---")
        orchestrator.print_samples(args.preview)
    
    print(f"\n✅ Successfully generated {len(pairs)} Q&A pairs")
    print(f"📁 Output saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
