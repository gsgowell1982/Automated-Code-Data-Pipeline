#!/usr/bin/env python3
"""
Design Q&A Generation Engine v2.0

Improvements over v1:
1. Intent-specific answers that directly address the question
2. Pseudo-code illustrations for design logic
3. Less templated, more targeted answers

Fixes the problem where "Why shouldn't we expose which field failed?"
was answered with a generic design template instead of directly
explaining the security risk (enumeration attacks).

Usage:
    python main_v2.py --questions 5 --languages en zh
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
    IntentBasedAnswerComposer,  # NEW in v2
)
from utils import OllamaClient, DiversityManagerV2, QuestionType, UserRole

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DesignDiversityManager:
    """Diversity manager for design questions."""
    
    def __init__(self):
        self.base_manager = DiversityManagerV2()
        self.intent_counts: Counter = Counter()
        self.aspect_counts: Counter = Counter()
        self.role_counts: Counter = Counter()
    
    def reset(self):
        self.base_manager.reset()
        self.intent_counts = Counter()
        self.aspect_counts = Counter()
        self.role_counts = Counter()
    
    def is_diverse(self, question: str, scenario_id: str = "", language: str = "en") -> tuple:
        return self.base_manager.is_diverse(question, scenario_id, language)
    
    def add_question(self, question: GeneratedDesignQuestion, scenario_id: str):
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
            question.question_text, qtype, UserRole.NEW_DEVELOPER,
            scenario_id, question.language, evidence_id=scenario_id
        )
        
        self.intent_counts[question.intent.value] += 1
        self.aspect_counts[question.aspect.value] += 1
        self.role_counts[question.role] += 1
    
    def get_underrepresented_roles(self) -> List[DesignerRole]:
        if not self.role_counts:
            return list(DesignerRole)
        avg = sum(self.role_counts.values()) / len(DesignerRole)
        return [r for r in DesignerRole if self.role_counts.get(r.value, 0) < avg * 0.5] or list(DesignerRole)
    
    def get_metrics(self) -> Dict:
        base_metrics = self.base_manager.get_metrics()
        return {
            **base_metrics,
            "intent_counts": dict(self.intent_counts),
            "aspect_counts": dict(self.aspect_counts),
            "role_counts": dict(self.role_counts),
        }


class DesignQAOrchestratorV2:
    """
    Design Q&A Orchestrator v2 with intent-based answer composition.
    
    Key improvement: Answers directly address the question before
    providing design solutions and pseudo-code.
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, rule_metadata_path: str = None):
        self.rule_metadata_path = Path(rule_metadata_path) if rule_metadata_path else DesignConfig.RULE_METADATA_FILE
        
        self.rule_metadata: Dict = {}
        
        self.llm_client: Optional[OllamaClient] = None
        self.context_builder: Optional[DesignContextBuilder] = None
        self.question_generator: Optional[DesignQuestionGenerator] = None
        self.solution_generator: Optional[DesignSolutionGenerator] = None
        self.answer_composer: Optional[IntentBasedAnswerComposer] = None  # NEW
        self.quality_checker: Optional[DesignQualityChecker] = None
        self.diversity_manager: Optional[DesignDiversityManager] = None
        
        self.generated_pairs: List[DesignQAPair] = []
        self.stats: Dict = defaultdict(int)
    
    def initialize(self) -> bool:
        """Initialize all components."""
        try:
            if self.rule_metadata_path.exists():
                with open(self.rule_metadata_path, 'r', encoding='utf-8') as f:
                    self.rule_metadata = json.load(f)
                logger.info(f"Loaded rule metadata: {self.rule_metadata.get('rule_id')}")
            
            self.llm_client = OllamaClient()
            
            self.context_builder = DesignContextBuilder(self.rule_metadata)
            self.question_generator = DesignQuestionGenerator(self.llm_client)
            self.solution_generator = DesignSolutionGenerator(self.llm_client)
            self.answer_composer = IntentBasedAnswerComposer()  # NEW
            self.quality_checker = DesignQualityChecker()
            self.diversity_manager = DesignDiversityManager()
            
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
        """Run the design Q&A generation pipeline."""
        questions_per_scenario = questions_per_scenario or DesignConfig.DEFAULT_QUESTIONS_PER_SCENARIO
        total_limit = total_limit or DesignConfig.DEFAULT_TOTAL_LIMIT
        languages = languages or DesignConfig.SUPPORTED_LANGUAGES
        
        self.generated_pairs = []
        self.stats = defaultdict(int)
        self.diversity_manager.reset()
        
        self._print_pipeline_header(questions_per_scenario, total_limit, languages)
        
        for subcategory_id in self.context_builder.get_all_subcategory_ids():
            if total_limit and len(self.generated_pairs) >= total_limit:
                break
            self._process_scenario(subcategory_id, questions_per_scenario, total_limit, languages)
        
        logger.info(f"Generated {len(self.generated_pairs)} design Q&A pairs")
        return self.generated_pairs
    
    def _print_pipeline_header(self, qps: int, total: Optional[int], languages: List[str]):
        logger.info("=" * 60)
        logger.info(f"Starting Design Q&A Pipeline v{self.VERSION}")
        logger.info("=" * 60)
        logger.info("  v2 Improvements:")
        logger.info("    - Intent-specific direct answers")
        logger.info("    - Pseudo-code design illustrations")
        logger.info("    - Less templated, more targeted responses")
        logger.info(f"  Questions/scenario: {qps}")
        logger.info(f"  Total limit: {total if total else 'No limit'}")
        logger.info(f"  Languages: {languages}")
        logger.info("=" * 60)
    
    def _process_scenario(self, subcategory_id: str, qps: int, total: Optional[int], languages: List[str]):
        logger.info(f"Processing scenario: {subcategory_id}")
        
        for language in languages:
            if total and len(self.generated_pairs) >= total:
                return
            
            scenario = self.context_builder.build_scenario(subcategory_id, language)
            if not scenario:
                continue
            
            roles = self.diversity_manager.get_underrepresented_roles()
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
                
                questions = self.question_generator.generate(
                    scenario, role, count, language, used_intents, used_aspects
                )
                
                for question in questions:
                    if questions_generated >= qps:
                        break
                    if total and len(self.generated_pairs) >= total:
                        return
                    
                    is_diverse, reason = self.diversity_manager.is_diverse(
                        question.question_text, subcategory_id, language
                    )
                    if not is_diverse:
                        self.stats[f"rejected_{reason.split(':')[0]}"] += 1
                        continue
                    
                    qa_pair = self._generate_qa_pair(question, scenario, language)
                    
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
        """Generate a complete design Q&A pair with targeted answer."""
        
        # Generate solution and reasoning
        solution, reasoning = self.solution_generator.generate(scenario, question, language)
        
        # v2: Use IntentBasedAnswerComposer for targeted answers
        answer = self.answer_composer.compose(
            question, scenario, solution, reasoning, language
        )
        
        qa_pair = DesignQAPair(
            sample_id=f"DBR01-DES2-{uuid.uuid4().hex[:10]}",  # DES2 for v2
            instruction=question.question_text,
            context={
                "design_scenario": scenario.to_dict(),
                "related_dbr": scenario.related_dbr,
                "design_aspect": question.aspect.value,
                "question_intent": question.intent.value,
            },
            auto_processing={
                "parser": "Design-Context-Builder",
                "parser_version": self.VERSION,
                "dbr_logic": {
                    "rule_id": scenario.related_dbr,
                    "subcategory_id": scenario.scenario_id,
                    "trigger_type": "design_requirement",
                },
                "generation_metadata": {
                    "version": self.VERSION,
                    "scenario": "design_solution_v2",
                    "question_source": question.source,
                    "designer_role": question.role,
                    "question_intent": question.intent.value,
                    "design_aspect": question.aspect.value,
                    "answer_style": "intent_based",  # NEW marker
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
        
        is_valid, score, issues = self.quality_checker.validate(qa_pair)
        qa_pair.data_quality["quality_score"] = score
        qa_pair.data_quality["validation_issues"] = issues
        qa_pair.data_quality["consistency_check"] = is_valid
        
        if is_valid:
            self.diversity_manager.add_question(question, scenario.scenario_id)
            self.generated_pairs.append(qa_pair)
            self.stats["valid"] += 1
            self.stats[f"source_{question.source}"] += 1
            self.stats[f"intent_{question.intent.value}"] += 1
            return qa_pair
        else:
            self.stats["invalid"] += 1
            return None
    
    def save_results(self, output_path: str = None) -> str:
        output_path = Path(output_path) if output_path else DesignConfig.DATA_DIR / "qwen_dbr_design_data_v2.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for pair in self.generated_pairs:
                f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(self.generated_pairs)} pairs to {output_path}")
        return str(output_path)
    
    def print_summary(self):
        print("\n" + "=" * 70)
        print(f"Design Q&A Generation Summary (v{self.VERSION})")
        print("=" * 70)
        
        print(f"\n📊 Results:")
        print(f"  - Valid: {self.stats.get('valid', 0)}")
        print(f"  - Invalid: {self.stats.get('invalid', 0)}")
        
        metrics = self.diversity_manager.get_metrics()
        
        print(f"\n📈 Diversity Metrics:")
        print(f"  - Overall Score: {metrics.get('overall_diversity_score', 0):.2%}")
        
        print(f"\n🎯 Intent Distribution:")
        for intent, count in sorted(metrics.get('intent_counts', {}).items()):
            print(f"  - {intent}: {count}")
        
        print(f"\n🔍 Aspect Distribution:")
        for aspect, count in sorted(metrics.get('aspect_counts', {}).items()):
            print(f"  - {aspect}: {count}")
        
        print("\n" + "=" * 70)
    
    def print_samples(self, n: int = 2):
        if not self.generated_pairs:
            return
        
        for i, pair in enumerate(self.generated_pairs[:n]):
            print("\n" + "=" * 70)
            meta = pair.auto_processing.get("generation_metadata", {})
            print(f"[Sample {i+1}] Intent: {meta.get('question_intent')} | Aspect: {meta.get('design_aspect')}")
            print("=" * 70)
            print(f"\n【Question】:\n{pair.instruction}")
            print(f"\n【Answer】:\n{pair.answer[:800]}...")
            print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description=f'Design Q&A Generation Engine v2.0')
    
    parser.add_argument('-m', '--metadata', default=str(DesignConfig.RULE_METADATA_FILE))
    parser.add_argument('-o', '--output', default=str(DesignConfig.DATA_DIR / "qwen_dbr_design_data_v2.jsonl"))
    parser.add_argument('-n', '--questions', type=int, default=5)
    parser.add_argument('-t', '--total', type=int, default=None)
    parser.add_argument('-l', '--languages', nargs='+', default=['en', 'zh'])
    parser.add_argument('--preview', type=int, default=2)
    
    args = parser.parse_args()
    
    orchestrator = DesignQAOrchestratorV2(args.metadata)
    
    if not orchestrator.initialize():
        print("Error: Failed to initialize.")
        return 1
    
    print(f"\n🚀 Running Design Q&A Generation Pipeline v{orchestrator.VERSION}")
    print(f"   Improvement: Intent-based targeted answers + pseudo-code")
    
    pairs = orchestrator.run_pipeline(
        questions_per_scenario=args.questions,
        total_limit=args.total,
        languages=args.languages,
    )
    
    if not pairs:
        print("Warning: No Q&A pairs generated.")
        return 1
    
    output_path = orchestrator.save_results(args.output)
    orchestrator.print_summary()
    
    if args.preview > 0:
        print(f"\n--- Sample Design Q&A Pairs (v2) ---")
        orchestrator.print_samples(args.preview)
    
    print(f"\n✅ Generated {len(pairs)} design Q&A pairs")
    print(f"📁 Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
