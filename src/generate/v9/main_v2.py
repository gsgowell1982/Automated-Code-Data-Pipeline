#!/usr/bin/env python3
"""
Q&A Generation Engine v9.2 - Enhanced Diversity

Uses v2 diversity and question generation modules.
Only changes from v9.0:
- Uses DiversityManagerV2 with semantic fingerprinting
- Uses LLMEnhancementLayerV2 with angle-based templates
- Passes evidence_id to diversity checks

All other modules remain unchanged.
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
from utils.llm_client import OllamaClient
from utils.diversity_v2 import DiversityManagerV2, QuestionAngle  # V2
from layers import (
    DeterministicLayer,
    UserPerspectiveLayer,
    QualityAssuranceLayer,
    ExecutionFlowAnalyzer,
    ConsistencyValidator,
)
from layers.llm_enhancement_v2 import LLMEnhancementLayerV2  # V2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridQAOrchestratorV2:
    """
    Enhanced orchestrator with semantic diversity.
    
    Key changes from v1:
    - Uses semantic fingerprinting to detect paraphrased questions
    - Uses angle-based templates for diverse question generation
    - Tracks per-evidence diversity to avoid repetition
    """
    
    VERSION = "9.2.0"
    
    def __init__(
        self,
        rule_metadata_path: str,
        ast_analysis_path: str = None
    ):
        self.rule_metadata_path = Path(rule_metadata_path)
        self.ast_analysis_path = Path(ast_analysis_path) if ast_analysis_path else None
        
        self.rule_metadata: Dict = {}
        self.ast_analysis: Dict = {}
        
        self.deterministic_layer: Optional[DeterministicLayer] = None
        self.user_perspective_layer: Optional[UserPerspectiveLayer] = None
        self.llm_enhancement_layer: Optional[LLMEnhancementLayerV2] = None  # V2
        self.quality_assurance_layer: Optional[QualityAssuranceLayer] = None
        self.execution_flow_analyzer: Optional[ExecutionFlowAnalyzer] = None
        self.consistency_validator: Optional[ConsistencyValidator] = None
        self.diversity_manager: Optional[DiversityManagerV2] = None  # V2
        
        self.llm_client: Optional[OllamaClient] = None
        
        self.generated_pairs: List[QAPair] = []
        self.stats: Dict = defaultdict(int)
    
    def initialize(self) -> bool:
        """Initialize all layers."""
        try:
            with open(self.rule_metadata_path, 'r', encoding='utf-8') as f:
                self.rule_metadata = json.load(f)
            logger.info(f"Loaded rule metadata: {self.rule_metadata.get('rule_id')}")
            
            if self.ast_analysis_path and self.ast_analysis_path.exists():
                with open(self.ast_analysis_path, 'r', encoding='utf-8') as f:
                    self.ast_analysis = json.load(f)
                logger.info("Loaded AST analysis")
            
            self.llm_client = OllamaClient()
            
            self.deterministic_layer = DeterministicLayer(self.rule_metadata, self.ast_analysis)
            self.user_perspective_layer = UserPerspectiveLayer()
            self.llm_enhancement_layer = LLMEnhancementLayerV2(self.llm_client)  # V2
            self.quality_assurance_layer = QualityAssuranceLayer()
            self.execution_flow_analyzer = ExecutionFlowAnalyzer()
            self.consistency_validator = ConsistencyValidator()
            self.diversity_manager = DiversityManagerV2()  # V2
            
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
        """Run Q&A generation with enhanced diversity."""
        questions_per_evidence = questions_per_evidence or Config.DEFAULT_QUESTIONS_PER_EVIDENCE
        total_limit = total_limit or Config.DEFAULT_TOTAL_LIMIT
        languages = languages or Config.SUPPORTED_LANGUAGES
        
        self.generated_pairs = []
        self.stats = defaultdict(int)
        self.diversity_manager.reset()
        
        self._print_pipeline_header(questions_per_evidence, total_limit, languages)
        
        for subcategory in self.deterministic_layer.get_subcategories():
            if total_limit and len(self.generated_pairs) >= total_limit:
                break
            
            self._process_subcategory(
                subcategory, questions_per_evidence, total_limit, languages
            )
        
        logger.info(f"Generated {len(self.generated_pairs)} Q&A pairs")
        return self.generated_pairs
    
    def _print_pipeline_header(self, qpe: int, total: Optional[int], languages: List[str]):
        logger.info("=" * 60)
        logger.info(f"Starting Hybrid Q&A Pipeline v{self.VERSION}")
        logger.info("=" * 60)
        logger.info("  Layers:")
        logger.info("    - Deterministic: AST, Call Graph, DBR Mapping")
        logger.info("    - User Perspective: Business context transformation")
        logger.info("    - LLM Enhancement V2: Angle-based question generation")  # V2
        logger.info("    - Quality Assurance: Validation, Hash check")
        logger.info("    - Diversity V2: Semantic fingerprinting")  # V2
        logger.info(f"  Questions/evidence: {qpe}")
        logger.info(f"  Total limit: {total if total else 'No limit'}")
        logger.info(f"  Languages: {languages}")
        logger.info("=" * 60)
    
    def _process_subcategory(
        self,
        subcategory: Dict,
        qpe: int,
        total: Optional[int],
        languages: List[str]
    ):
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
        code_context = self.deterministic_layer.get_code_context(evidence)
        
        if not code_context.code_snippet:
            return
        
        dbr_logic = self.deterministic_layer.get_dbr_logic(evidence)
        hash_valid = self.deterministic_layer.verify_source_hash(
            code_context.code_snippet, code_context.source_hash
        )
        
        code_facts = self.execution_flow_analyzer.analyze(
            code_context.code_snippet, code_context.function_name
        )
        
        # Get evidence ID for per-evidence diversity tracking
        evidence_id = evidence.get("evidence_id", f"{subcategory_id}-{uuid.uuid4().hex[:6]}")
        
        roles = self.diversity_manager.get_underrepresented_roles()
        if not roles:
            roles = list(UserRole)
        random.shuffle(roles)
        
        for language in languages:
            if total and len(self.generated_pairs) >= total:
                return
            
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
                
                # V2: Pass diversity_manager and evidence_id to question generator
                questions = self.llm_enhancement_layer.question_generator.generate(
                    business_context, code_facts, role, count, language,
                    diversity_manager=self.diversity_manager,
                    evidence_id=evidence_id
                )
                
                for question in questions:
                    if questions_generated >= qpe:
                        break
                    if total and len(self.generated_pairs) >= total:
                        return
                    
                    # V2: Pass evidence_id to diversity check
                    is_diverse, reason = self.diversity_manager.is_diverse(
                        question.question_text, 
                        evidence_id=evidence_id,
                        language=language
                    )
                    if not is_diverse:
                        self.stats[f"rejected_{reason.split(':')[0]}"] += 1
                        continue
                    
                    qa_pair = self._generate_qa_pair(
                        question, evidence, business_context, code_context,
                        code_facts, dbr_logic, language, hash_valid, evidence_id
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
        hash_valid: bool,
        evidence_id: str
    ) -> Optional[QAPair]:
        reasoning, reasoning_valid = self.llm_enhancement_layer.reasoning_generator.generate(
            question.question_text, business_context, code_context, code_facts, language
        )
        
        answer, answer_valid = self.llm_enhancement_layer.answer_generator.generate(
            question.question_text, reasoning, code_context, code_facts, language
        )
        
        qa_pair = QAPair(
            sample_id=f"DBR01-V92-{uuid.uuid4().hex[:10]}",  # V92 marker
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
                    "version": self.VERSION,
                    "architecture": "modular_hybrid_v2",
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
                "evidence_id": evidence_id,
            },
        )
        
        result = self.quality_assurance_layer.validate(qa_pair, code_facts)
        qa_pair.data_quality["quality_score"] = result.score
        qa_pair.data_quality["validation_issues"] = result.issues
        
        if result.is_valid:
            # V2: Track with semantic dimensions
            from utils.diversity_v2 import SemanticIntent
            
            self.diversity_manager.add_question(
                question.question_text,
                question.question_type or QuestionType.UNDERSTANDING,
                UserRole(question.role),
                business_context.scenario_name,
                language,
                evidence_id=evidence_id
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
        output_path = Path(output_path) if output_path else Config.DATA_DIR / "qwen_dbr_training_logic_v9.2.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in self.generated_pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(self.generated_pairs)} pairs to {output_path}")
        return str(output_path)
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"Q&A Generation Summary (v{self.VERSION} - Enhanced Diversity)")
        print("=" * 70)
        
        print(f"\n📊 Results:")
        print(f"  - Valid: {self.stats.get('valid', 0)}")
        print(f"  - Invalid: {self.stats.get('invalid', 0)}")
        
        diversity = self.diversity_manager.get_metrics()
        
        print(f"\n📈 Diversity Metrics:")
        print(f"  - Overall Score: {diversity.get('overall_diversity_score', 0):.2%}")
        print(f"  - Unique Ratio: {diversity.get('unique_ratio', 0):.2%}")
        print(f"  - Semantic Unique Ratio: {diversity.get('semantic_unique_ratio', 0):.2%}")  # V2
        print(f"  - Semantic Duplicates Rejected: {diversity.get('semantic_duplicate_rejected', 0)}")  # V2
        
        print(f"\n🎯 Intent Distribution:")  # V2
        for intent, count in sorted(diversity.get('intent_counts', {}).items()):
            print(f"  - {intent}: {count}")
        
        print(f"\n🔍 Focus Distribution:")  # V2
        for focus, count in sorted(diversity.get('focus_counts', {}).items())[:8]:
            print(f"  - {focus}: {count}")
        
        print(f"\n👥 User Roles:")
        for role in UserRole:
            count = self.stats.get(f"role_{role.value}", 0)
            if count > 0:
                print(f"  - {role.value}: {count}")
        
        print("\n" + "=" * 70)
    
    def print_samples(self, n: int = 3):
        if not self.generated_pairs:
            return
        
        for i, pair in enumerate(self.generated_pairs[:n]):
            print("\n" + "=" * 70)
            meta = pair.auto_processing.get("generation_metadata", {})
            print(f"[Sample {i+1}] Role: {meta.get('user_role')} | Type: {meta.get('question_type')}")
            print("=" * 70)
            print(f"\n【Question】:\n{pair.instruction}")
            print(f"\n【Reasoning】:")
            for step in pair.reasoning_trace[:3]:
                print(f"  {step}")
            print(f"\n【Answer (excerpt)】:\n{pair.answer[:350]}...")
            print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description=f'Q&A Generation Engine v9.2 (Enhanced Diversity)'
    )
    
    parser.add_argument('-m', '--metadata', default=str(Config.RULE_METADATA_FILE),
                       help='Path to rule metadata JSON')
    parser.add_argument('-a', '--ast', default=str(Config.AST_ANALYSIS_FILE),
                       help='Path to AST analysis JSON')
    parser.add_argument('-o', '--output', default=str(Config.DATA_DIR / "qwen_dbr_training_logic_v9.2.jsonl"),
                       help='Output JSONL file path')
    parser.add_argument('-n', '--questions', type=int, default=5,
                       help='Questions per evidence')
    parser.add_argument('-t', '--total', type=int, default=None,
                       help='Total Q&A pairs limit')
    parser.add_argument('-l', '--languages', nargs='+', default=['en', 'zh'],
                       help='Languages to generate')
    parser.add_argument('--preview', type=int, default=3,
                       help='Number of samples to preview')
    
    args = parser.parse_args()
    
    orchestrator = HybridQAOrchestratorV2(args.metadata, args.ast)
    
    if not orchestrator.initialize():
        print("Error: Failed to initialize.")
        return 1
    
    print(f"\n🚀 Running Q&A Generation Pipeline v{orchestrator.VERSION}")
    print(f"   Diversity: Semantic fingerprinting + Angle-based templates")
    
    pairs = orchestrator.run_pipeline(
        questions_per_evidence=args.questions,
        total_limit=args.total,
        languages=args.languages,
    )
    
    if not pairs:
        print("Warning: No Q&A pairs generated.")
        return 1
    
    output_path = orchestrator.save_results(args.output)
    orchestrator.print_summary()
    
    if args.preview > 0:
        print(f"\n--- Sample Q&A Pairs ---")
        orchestrator.print_samples(args.preview)
    
    print(f"\n✅ Generated {len(pairs)} Q&A pairs")
    print(f"📁 Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
