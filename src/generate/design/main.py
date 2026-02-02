#!/usr/bin/env python3
"""
Design Q&A Generation Engine v1.0

Scenario 2: Generate design solutions and recommendations
for implementing authentication features based on DBR-01.

Questions are from developer/designer perspective.
No code_snippet field - design-focused output.

Usage:
    python main.py --questions 5 --languages en zh
"""

import argparse
import json
import logging
import random
import uuid
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Optional, Set

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import DesignConfig
from models import (
    DesignScenario, DesignAspect, DesignerRole, QuestionIntent,
    GeneratedDesignQuestion, DesignSolution, DesignQAPair
)
from layers import (
    DesignContextBuilder,
    DesignQuestionGenerator,
    DesignSolutionGenerator,
    DesignQualityChecker,
)
from utils import OllamaClient, DiversityManagerV2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DesignDiversityManager:
    """
    Diversity manager specialized for design questions.
    
    Extends v9's diversity manager with design-specific tracking.
    """
    
    def __init__(self):
        self.base_manager = DiversityManagerV2()
        self.intent_counts: Counter = Counter()
        self.aspect_counts: Counter = Counter()
        self.role_counts: Counter = Counter()
        self.scenario_fingerprints: Dict[str, Set[str]] = defaultdict(set)
    
    def reset(self):
        self.base_manager.reset()
        self.intent_counts = Counter()
        self.aspect_counts = Counter()
        self.role_counts = Counter()
        self.scenario_fingerprints = defaultdict(set)
    
    def is_diverse(
        self,
        question: str,
        scenario_id: str = "",
        language: str = "en"
    ) -> tuple:
        """Check if question is diverse."""
        return self.base_manager.is_diverse(question, scenario_id, language)
    
    def add_question(
        self,
        question: GeneratedDesignQuestion,
        scenario_id: str
    ):
        """Register a question for tracking."""
        # Use local QuestionType from utils
        from utils import QuestionType, UserRole
        
        # Map intent to question type
        type_mapping = {
            QuestionIntent.HOW_TO_DESIGN: QuestionType.UNDERSTANDING,
            QuestionIntent.WHY_THIS_WAY: QuestionType.UNDERSTANDING,
            QuestionIntent.WHAT_PATTERN: QuestionType.UNDERSTANDING,
            QuestionIntent.TRADE_OFF: QuestionType.COMPARISON,
            QuestionIntent.ALTERNATIVE: QuestionType.COMPARISON,
            QuestionIntent.BEST_PRACTICE: QuestionType.DEEP_ANALYSIS,
            QuestionIntent.SECURITY_CONCERN: QuestionType.SECURITY,
        }
        
        qtype = type_mapping.get(question.intent, QuestionType.UNDERSTANDING)
        
        self.base_manager.add_question(
            question.question_text,
            qtype,
            UserRole.NEW_DEVELOPER,  # Placeholder
            scenario_id,
            question.language,
            evidence_id=scenario_id
        )
        
        # Design-specific tracking
        self.intent_counts[question.intent.value] += 1
        self.aspect_counts[question.aspect.value] += 1
        self.role_counts[question.role] += 1
    
    def get_underrepresented_intents(self) -> List[QuestionIntent]:
        """Get intents that need more questions."""
        if not self.intent_counts:
            return list(QuestionIntent)
        
        avg = sum(self.intent_counts.values()) / len(QuestionIntent)
        return [i for i in QuestionIntent if self.intent_counts.get(i.value, 0) < avg * 0.5]
    
    def get_underrepresented_aspects(self) -> List[DesignAspect]:
        """Get aspects that need more questions."""
        if not self.aspect_counts:
            return list(DesignAspect)
        
        avg = sum(self.aspect_counts.values()) / len(DesignAspect)
        return [a for a in DesignAspect if self.aspect_counts.get(a.value, 0) < avg * 0.5]
    
    def get_underrepresented_roles(self) -> List[DesignerRole]:
        """Get roles that need more questions."""
        if not self.role_counts:
            return list(DesignerRole)
        
        avg = sum(self.role_counts.values()) / len(DesignerRole)
        return [r for r in DesignerRole if self.role_counts.get(r.value, 0) < avg * 0.5]
    
    def get_metrics(self) -> Dict:
        """Get diversity metrics."""
        base_metrics = self.base_manager.get_metrics()
        
        return {
            **base_metrics,
            "intent_counts": dict(self.intent_counts),
            "aspect_counts": dict(self.aspect_counts),
            "role_counts": dict(self.role_counts),
        }


class DesignQAOrchestrator:
    """
    Main orchestrator for design Q&A generation.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    DesignQAOrchestrator                             │
    ├─────────────────────────────────────────────────────────────────────┤
    │  ┌─────────────────────┐    ┌─────────────────────────────────────┐ │
    │  │ Design Context      │    │        Question Generator           │ │
    │  │ - Scenario Builder  │───►│ - Intent-based templates            │ │
    │  │ - DBR Mapping       │    │ - LLM enhancement                   │ │
    │  │ - Requirements      │    │ - Semantic deduplication            │ │
    │  └─────────────────────┘    └─────────────────────────────────────┘ │
    │                                                                      │
    │  ┌─────────────────────┐    ┌─────────────────────────────────────┐ │
    │  │ Solution Generator  │    │        Quality Checker              │ │
    │  │ - Design patterns   │    │ - Format validation                 │ │
    │  │ - Trade-off analysis│    │ - Completeness check                │ │
    │  │ - Reasoning trace   │    │ - No-code verification              │ │
    │  └─────────────────────┘    └─────────────────────────────────────┘ │
    │                                                                      │
    │  ┌─────────────────────────────────────────────────────────────────┐ │
    │  │                   Diversity Manager                              │ │
    │  │  Intent × Aspect × Role distribution + Semantic fingerprinting   │ │
    │  └─────────────────────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, rule_metadata_path: str = None):
        self.rule_metadata_path = Path(rule_metadata_path) if rule_metadata_path else DesignConfig.RULE_METADATA_FILE
        
        self.rule_metadata: Dict = {}
        
        # Components
        self.llm_client: Optional[OllamaClient] = None
        self.context_builder: Optional[DesignContextBuilder] = None
        self.question_generator: Optional[DesignQuestionGenerator] = None
        self.solution_generator: Optional[DesignSolutionGenerator] = None
        self.quality_checker: Optional[DesignQualityChecker] = None
        self.diversity_manager: Optional[DesignDiversityManager] = None
        
        # Output
        self.generated_pairs: List[DesignQAPair] = []
        self.stats: Dict = defaultdict(int)
    
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            # Load rule metadata
            if self.rule_metadata_path.exists():
                with open(self.rule_metadata_path, 'r', encoding='utf-8') as f:
                    self.rule_metadata = json.load(f)
                logger.info(f"Loaded rule metadata: {self.rule_metadata.get('rule_id')}")
            
            # Initialize LLM client
            self.llm_client = OllamaClient()
            
            # Initialize components
            self.context_builder = DesignContextBuilder(self.rule_metadata)
            self.question_generator = DesignQuestionGenerator(self.llm_client)
            self.solution_generator = DesignSolutionGenerator(self.llm_client)
            self.quality_checker = DesignQualityChecker()
            self.diversity_manager = DesignDiversityManager()
            
            # Log status
            if self.llm_client.is_available():
                logger.info(f"✓ LLM available: {DesignConfig.MODEL_NAME}")
            else:
                logger.warning("✗ LLM not available - using fallback generation")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def run_pipeline(
        self,
        questions_per_scenario: int = None,
        total_limit: int = None,
        languages: List[str] = None
    ) -> List[DesignQAPair]:
        """
        Run the design Q&A generation pipeline.
        
        Args:
            questions_per_scenario: Questions per design scenario
            total_limit: Maximum total Q&A pairs
            languages: Languages to generate
            
        Returns:
            List of generated DesignQAPair objects
        """
        questions_per_scenario = questions_per_scenario or DesignConfig.DEFAULT_QUESTIONS_PER_SCENARIO
        total_limit = total_limit or DesignConfig.DEFAULT_TOTAL_LIMIT
        languages = languages or DesignConfig.SUPPORTED_LANGUAGES
        
        # Reset state
        self.generated_pairs = []
        self.stats = defaultdict(int)
        self.diversity_manager.reset()
        
        self._print_pipeline_header(questions_per_scenario, total_limit, languages)
        
        # Process each scenario
        for subcategory_id in self.context_builder.get_all_subcategory_ids():
            if total_limit and len(self.generated_pairs) >= total_limit:
                logger.info(f"Reached total limit ({total_limit})")
                break
            
            self._process_scenario(
                subcategory_id, questions_per_scenario, total_limit, languages
            )
        
        logger.info(f"Generated {len(self.generated_pairs)} design Q&A pairs")
        return self.generated_pairs
    
    def _print_pipeline_header(self, qps: int, total: Optional[int], languages: List[str]):
        """Print pipeline header."""
        logger.info("=" * 60)
        logger.info(f"Starting Design Q&A Pipeline v{DesignConfig.VERSION}")
        logger.info("=" * 60)
        logger.info("  Scenario: Design Solutions (no code_snippet)")
        logger.info("  Perspective: Developer/Designer")
        logger.info("  Components:")
        logger.info("    - Design Context Builder")
        logger.info("    - Question Generator (Intent-based)")
        logger.info("    - Solution Generator (Pattern-based)")
        logger.info("    - Quality Checker")
        logger.info("    - Diversity Manager")
        logger.info(f"  Questions/scenario: {qps}")
        logger.info(f"  Total limit: {total if total else 'No limit'}")
        logger.info(f"  Languages: {languages}")
        logger.info("=" * 60)
    
    def _process_scenario(
        self,
        subcategory_id: str,
        qps: int,
        total: Optional[int],
        languages: List[str]
    ):
        """Process a single design scenario."""
        logger.info(f"Processing scenario: {subcategory_id}")
        
        for language in languages:
            if total and len(self.generated_pairs) >= total:
                return
            
            # Build scenario
            scenario = self.context_builder.build_scenario(subcategory_id, language)
            if not scenario:
                continue
            
            # Get prioritized roles
            roles = self.diversity_manager.get_underrepresented_roles()
            if not roles:
                roles = list(DesignerRole)
            random.shuffle(roles)
            
            questions_generated = 0
            used_intents: Set[QuestionIntent] = set()
            used_aspects: Set[DesignAspect] = set()
            
            for role in roles:
                if questions_generated >= qps:
                    break
                if total and len(self.generated_pairs) >= total:
                    return
                
                count = min(2, qps - questions_generated)
                
                # Generate questions
                questions = self.question_generator.generate(
                    scenario, role, count, language, used_intents, used_aspects
                )
                
                for question in questions:
                    if questions_generated >= qps:
                        break
                    if total and len(self.generated_pairs) >= total:
                        return
                    
                    # Diversity check
                    is_diverse, reason = self.diversity_manager.is_diverse(
                        question.question_text, subcategory_id, language
                    )
                    if not is_diverse:
                        self.stats[f"rejected_{reason.split(':')[0]}"] += 1
                        continue
                    
                    # Generate Q&A pair
                    qa_pair = self._generate_qa_pair(
                        question, scenario, language
                    )
                    
                    if qa_pair:
                        questions_generated += 1
                        used_intents.add(question.intent)
                        used_aspects.add(question.aspect)
    
    def _generate_qa_pair(
        self,
        question: GeneratedDesignQuestion,
        scenario: DesignScenario,
        language: str
    ) -> Optional[DesignQAPair]:
        """Generate a complete design Q&A pair."""
        
        # Generate solution and reasoning
        solution, reasoning = self.solution_generator.generate(
            scenario, question, language
        )
        
        # Compose answer
        answer = self._compose_answer(solution, question, language)
        
        # Build Q&A pair
        qa_pair = DesignQAPair(
            sample_id=f"DBR01-DES-{uuid.uuid4().hex[:10]}",
            instruction=question.question_text,
            context={
                "design_scenario": scenario.to_dict(),
                "related_dbr": scenario.related_dbr,
                "design_aspect": question.aspect.value,
                "question_intent": question.intent.value,
            },
            auto_processing={
                "parser": "Design-Context-Builder",
                "parser_version": DesignConfig.VERSION,
                "dbr_logic": {
                    "rule_id": scenario.related_dbr,
                    "subcategory_id": scenario.scenario_id,
                    "trigger_type": "design_requirement",
                },
                "generation_metadata": {
                    "version": DesignConfig.VERSION,
                    "scenario": DesignConfig.SCENARIO,
                    "question_source": question.source,
                    "designer_role": question.role,
                    "question_intent": question.intent.value,
                    "design_aspect": question.aspect.value,
                },
                "solution_metadata": solution.to_dict(),
            },
            reasoning_trace=reasoning,
            answer=answer,
            data_quality={
                "language": language,
                "scenario_id": scenario.scenario_id,
            },
        )
        
        # Quality check
        is_valid, score, issues = self.quality_checker.validate(qa_pair)
        qa_pair.data_quality["quality_score"] = score
        qa_pair.data_quality["validation_issues"] = issues
        qa_pair.data_quality["consistency_check"] = is_valid
        
        if is_valid:
            # Register with diversity manager
            self.diversity_manager.add_question(question, scenario.scenario_id)
            
            self.generated_pairs.append(qa_pair)
            self.stats["valid"] += 1
            self.stats[f"source_{question.source}"] += 1
            self.stats[f"role_{question.role}"] += 1
            self.stats[f"intent_{question.intent.value}"] += 1
            
            return qa_pair
        else:
            self.stats["invalid"] += 1
            return None
    
    def _compose_answer(
        self,
        solution: DesignSolution,
        question: GeneratedDesignQuestion,
        language: str
    ) -> str:
        """Compose the final answer from solution."""
        if language == "zh":
            return f"""### 设计方案

**推荐方法**: {solution.approach}

**设计理由**: {solution.rationale}

**核心组件**:
{chr(10).join(f'- {c}' for c in solution.components)}

**使用的设计模式**:
{chr(10).join(f'- {p}' for p in solution.patterns_used)}

**安全考量**:
{chr(10).join(f'- {s}' for s in solution.security_considerations)}

**权衡取舍**:
{chr(10).join(f'- {t}' for t in solution.trade_offs)}
"""
        else:
            return f"""### Design Solution

**Recommended Approach**: {solution.approach}

**Rationale**: {solution.rationale}

**Key Components**:
{chr(10).join(f'- {c}' for c in solution.components)}

**Design Patterns Used**:
{chr(10).join(f'- {p}' for p in solution.patterns_used)}

**Security Considerations**:
{chr(10).join(f'- {s}' for s in solution.security_considerations)}

**Trade-offs**:
{chr(10).join(f'- {t}' for t in solution.trade_offs)}
"""
    
    def save_results(self, output_path: str = None) -> str:
        """Save results to JSONL file."""
        output_path = Path(output_path) if output_path else DesignConfig.OUTPUT_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in self.generated_pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(self.generated_pairs)} pairs to {output_path}")
        return str(output_path)
    
    def print_summary(self):
        """Print generation summary."""
        print("\n" + "=" * 70)
        print(f"Design Q&A Generation Summary (v{DesignConfig.VERSION})")
        print("=" * 70)
        
        print(f"\n📊 Results:")
        print(f"  - Valid: {self.stats.get('valid', 0)}")
        print(f"  - Invalid: {self.stats.get('invalid', 0)}")
        
        metrics = self.diversity_manager.get_metrics()
        
        print(f"\n📈 Diversity Metrics:")
        print(f"  - Overall Score: {metrics.get('overall_diversity_score', 0):.2%}")
        print(f"  - Unique Ratio: {metrics.get('unique_ratio', 0):.2%}")
        
        print(f"\n🎯 Intent Distribution:")
        for intent, count in sorted(metrics.get('intent_counts', {}).items()):
            print(f"  - {intent}: {count}")
        
        print(f"\n🔍 Aspect Distribution:")
        for aspect, count in sorted(metrics.get('aspect_counts', {}).items()):
            print(f"  - {aspect}: {count}")
        
        print(f"\n👤 Role Distribution:")
        for role, count in sorted(metrics.get('role_counts', {}).items()):
            print(f"  - {role}: {count}")
        
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
            print(f"[Sample {i+1}] Role: {meta.get('designer_role')} | Intent: {meta.get('question_intent')}")
            print(f"Aspect: {meta.get('design_aspect')}")
            print("=" * 70)
            print(f"\n【Question】:\n{pair.instruction}")
            print(f"\n【Reasoning】:")
            for step in pair.reasoning_trace[:3]:
                print(f"  {step}")
            print(f"\n【Answer (excerpt)】:\n{pair.answer[:500]}...")
            print("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description=f'Design Q&A Generation Engine v{DesignConfig.VERSION}'
    )
    
    parser.add_argument('-m', '--metadata', default=str(DesignConfig.RULE_METADATA_FILE),
                       help='Path to rule metadata JSON')
    parser.add_argument('-o', '--output', default=str(DesignConfig.OUTPUT_FILE),
                       help='Output JSONL file path')
    parser.add_argument('-n', '--questions', type=int, default=5,
                       help='Questions per scenario')
    parser.add_argument('-t', '--total', type=int, default=None,
                       help='Total Q&A pairs limit')
    parser.add_argument('-l', '--languages', nargs='+', default=['en', 'zh'],
                       help='Languages to generate')
    parser.add_argument('--preview', type=int, default=2,
                       help='Number of samples to preview')
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = DesignQAOrchestrator(args.metadata)
    
    if not orchestrator.initialize():
        print("Error: Failed to initialize.")
        return 1
    
    print(f"\n🚀 Running Design Q&A Generation Pipeline v{DesignConfig.VERSION}")
    print(f"   Scenario: Design Solutions (no code_snippet)")
    print(f"   Questions/scenario: {args.questions}")
    
    pairs = orchestrator.run_pipeline(
        questions_per_scenario=args.questions,
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
        print(f"\n--- Sample Design Q&A Pairs ---")
        orchestrator.print_samples(args.preview)
    
    print(f"\n✅ Generated {len(pairs)} design Q&A pairs")
    print(f"📁 Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
