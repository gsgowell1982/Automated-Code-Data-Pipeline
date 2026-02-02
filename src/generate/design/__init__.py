"""
Design Q&A Generation Engine v1.0

Scenario 2: Generate design solutions and recommendations
for implementing authentication features based on DBR-01.

Different from v9.2 (Scenario 1):
- Questions from developer/designer perspective
- No code_snippet field (design-focused)
- Answers are design solutions, not code explanations

Modules:
- config: Configuration parameters
- models: Design-specific data models
- layers: Generation layers
    - design_context: Scenario builder
    - design_generator: Question and solution generators
    - quality_check: Quality validation
- utils: Reuses v9 utilities
- main: Main orchestrator

Usage:
    from design import DesignQAOrchestrator
    
    orchestrator = DesignQAOrchestrator()
    orchestrator.initialize()
    pairs = orchestrator.run_pipeline(questions_per_scenario=5)
"""

from .config import DesignConfig
from .models import (
    DesignerRole, DesignAspect, QuestionIntent,
    DesignScenario, DesignSolution, GeneratedDesignQuestion, DesignQAPair
)
from .main import DesignQAOrchestrator

__version__ = DesignConfig.VERSION
__all__ = [
    "DesignConfig",
    "DesignerRole", "DesignAspect", "QuestionIntent",
    "DesignScenario", "DesignSolution", "GeneratedDesignQuestion", "DesignQAPair",
    "DesignQAOrchestrator",
]
