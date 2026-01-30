"""
Deterministic Layer for Q&A Generation Engine v9.

Provides deterministic, verifiable data from AST analysis.
This layer ensures factual accuracy and traceability.

Components:
- AST Code Snippets
- Call Graph Analysis
- DBR Rule Mapping
- Source Hash Verification
"""

import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import CodeContext, DBRLogic


class DeterministicLayer:
    """
    Provides deterministic, verifiable data from AST analysis.
    
    This layer is the foundation for factual accuracy - all data
    comes directly from code analysis, not LLM generation.
    """
    
    def __init__(self, rule_metadata: Dict, ast_analysis: Dict = None):
        """
        Initialize deterministic layer.
        
        Args:
            rule_metadata: Loaded rule metadata JSON
            ast_analysis: Optional AST analysis JSON
        """
        self.rule_metadata = rule_metadata
        self.ast_analysis = ast_analysis or {}
        self._call_graph = self._build_call_graph()
    
    def _build_call_graph(self) -> Dict[str, List[str]]:
        """Build call graph from AST analysis."""
        call_graph = defaultdict(list)
        
        for module in self.ast_analysis.get("modules", []):
            for func in module.get("functions", []):
                func_name = func.get("name", "")
                calls = func.get("calls", [])
                call_graph[func_name] = calls
        
        return call_graph
    
    def get_code_context(self, evidence: Dict) -> CodeContext:
        """
        Extract deterministic code context from evidence.
        
        Args:
            evidence: Evidence dictionary from rule metadata
            
        Returns:
            CodeContext with all code-level details
        """
        code_data = evidence.get("code_snippet", {})
        location = evidence.get("location", {})
        
        return CodeContext(
            file_path=code_data.get("file_path", location.get("file_path", "")),
            function_name=evidence.get("name", ""),
            line_start=code_data.get("line_start", location.get("line_start", 0)),
            line_end=code_data.get("line_end", location.get("line_end", 0)),
            code_snippet=code_data.get("code", ""),
            source_hash=code_data.get("source_hash", ""),
            related_elements=evidence.get("related_elements", []),
            call_chain=self._get_call_chain(evidence.get("name", ""))
        )
    
    def _get_call_chain(self, func_name: str, depth: int = 3) -> List[str]:
        """
        Get call chain for a function.
        
        Args:
            func_name: Function name to trace
            depth: Maximum depth to traverse
            
        Returns:
            List of function names in call chain
        """
        chain = []
        visited = set()
        
        def traverse(name, current_depth):
            if current_depth <= 0 or name in visited:
                return
            visited.add(name)
            chain.append(name)
            for called in self._call_graph.get(name, []):
                traverse(called, current_depth - 1)
        
        traverse(func_name, depth)
        return chain
    
    def get_dbr_logic(self, evidence: Dict) -> DBRLogic:
        """
        Get DBR rule mapping from evidence.
        
        Args:
            evidence: Evidence dictionary
            
        Returns:
            DBRLogic object with rule mapping details
        """
        dbr_data = evidence.get("dbr_logic", {})
        
        return DBRLogic(
            rule_id=dbr_data.get("rule_id", "DBR-01"),
            subcategory_id=dbr_data.get("subcategory_id", ""),
            trigger_type=dbr_data.get("trigger_type", "explicit"),
            weight=dbr_data.get("weight", 1.0),
            trigger_conditions=dbr_data.get("trigger_conditions", []),
            matched_patterns=dbr_data.get("matched_patterns", [])
        )
    
    def verify_source_hash(self, code_snippet: str, expected_hash: str) -> bool:
        """
        Verify code snippet integrity using hash.
        
        Args:
            code_snippet: The code to verify
            expected_hash: Expected MD5 hash
            
        Returns:
            True if hash matches or no expected hash
        """
        if not expected_hash:
            return True
        
        computed = hashlib.md5(code_snippet.encode()).hexdigest()
        return computed == expected_hash
    
    def get_subcategories(self) -> List[Dict]:
        """Get all subcategories from rule metadata."""
        return self.rule_metadata.get("subcategories", [])
    
    def get_rule_id(self) -> str:
        """Get the rule ID from metadata."""
        return self.rule_metadata.get("rule_id", "DBR-01")
