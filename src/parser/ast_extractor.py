"""
AST-based code extraction module.

Provides tools for parsing Python source code and extracting
relevant code blocks, functions, classes, and decorators using
the Abstract Syntax Tree (AST).
"""

import ast
import os
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CodeBlock:
    """
    Represents an extracted code block.
    
    Attributes:
        content: The source code content
        start_line: Starting line number
        end_line: Ending line number
        file_path: Path to the source file
        block_type: Type of block (function, class, etc.)
        name: Name of the block (function/class name)
    """
    content: str
    start_line: int
    end_line: int
    file_path: str
    block_type: str = "unknown"
    name: str = ""


@dataclass
class FunctionInfo:
    """
    Detailed information about a parsed function.
    
    Attributes:
        name: Function name
        decorators: List of decorator names
        arguments: List of argument names
        returns: Return type annotation if present
        docstring: Function docstring if present
        dependencies: List of Depends() dependencies
        raises: List of exception types raised
        calls: List of function calls within the function
        source: The full source code of the function
        line_start: Starting line number
        line_end: Ending line number
    """
    name: str
    decorators: List[str] = field(default_factory=list)
    arguments: List[str] = field(default_factory=list)
    returns: Optional[str] = None
    docstring: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    raises: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)
    source: str = ""
    line_start: int = 0
    line_end: int = 0


class ASTExtractor:
    """
    AST-based code extractor for Python source files.
    
    Provides methods for extracting functions, classes, and specific
    code patterns relevant to DBR analysis.
    """
    
    def __init__(self, source_code: Optional[str] = None, file_path: Optional[str] = None):
        """
        Initialize the AST extractor.
        
        Args:
            source_code: Python source code string
            file_path: Path to a Python file to load
        """
        self.source_code = source_code
        self.file_path = file_path
        self._tree: Optional[ast.AST] = None
        self._lines: List[str] = []
        
        if file_path and not source_code:
            self._load_file(file_path)
        elif source_code:
            self._parse_source(source_code)
    
    def _load_file(self, file_path: str) -> None:
        """Load and parse a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.source_code = f.read()
            self._parse_source(self.source_code)
            self.file_path = file_path
        except Exception as e:
            logger.error(f"Failed to load file {file_path}: {e}")
    
    def _parse_source(self, source_code: str) -> None:
        """Parse the source code into an AST."""
        try:
            self._tree = ast.parse(source_code)
            self._lines = source_code.split('\n')
        except SyntaxError as e:
            logger.error(f"Syntax error parsing source: {e}")
    
    def get_source_segment(self, start_line: int, end_line: int) -> str:
        """Get a segment of source code by line numbers (1-indexed)."""
        if not self._lines:
            return ""
        start_idx = max(0, start_line - 1)
        end_idx = min(len(self._lines), end_line)
        return '\n'.join(self._lines[start_idx:end_idx])
    
    def extract_functions(self) -> List[FunctionInfo]:
        """
        Extract all function definitions from the source.
        
        Returns:
            List of FunctionInfo objects with detailed function information
        """
        if not self._tree:
            return []
        
        functions = []
        for node in ast.walk(self._tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._extract_function_info(node)
                functions.append(func_info)
        
        return functions
    
    def _extract_function_info(self, node: ast.FunctionDef) -> FunctionInfo:
        """Extract detailed information from a function node."""
        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(dec.func.attr)
        
        # Get arguments
        arguments = [arg.arg for arg in node.args.args]
        
        # Get return type
        returns = None
        if node.returns:
            returns = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Find Depends() dependencies
        dependencies = []
        raises = []
        calls = []
        
        for child in ast.walk(node):
            # Find Depends() calls
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == 'Depends':
                    if child.args:
                        dep_arg = child.args[0]
                        if isinstance(dep_arg, ast.Name):
                            dependencies.append(dep_arg.id)
                        elif isinstance(dep_arg, ast.Call):
                            if isinstance(dep_arg.func, ast.Name):
                                dependencies.append(dep_arg.func.id)
                
                # Track function calls
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
            
            # Find raise statements
            if isinstance(child, ast.Raise):
                if isinstance(child.exc, ast.Call):
                    if isinstance(child.exc.func, ast.Name):
                        raises.append(child.exc.func.id)
        
        # Get source code
        source = self.get_source_segment(node.lineno, node.end_lineno or node.lineno)
        
        return FunctionInfo(
            name=node.name,
            decorators=decorators,
            arguments=arguments,
            returns=returns,
            docstring=docstring,
            dependencies=list(set(dependencies)),
            raises=list(set(raises)),
            calls=list(set(calls)),
            source=source,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno
        )
    
    def find_functions_by_name(self, names: List[str]) -> List[FunctionInfo]:
        """Find functions by their names."""
        all_functions = self.extract_functions()
        return [f for f in all_functions if f.name in names]
    
    def find_functions_with_decorator(self, decorator_name: str) -> List[FunctionInfo]:
        """Find all functions with a specific decorator."""
        all_functions = self.extract_functions()
        return [f for f in all_functions if decorator_name in f.decorators]
    
    def find_functions_with_dependency(self, dependency_name: str) -> List[FunctionInfo]:
        """Find all functions that use a specific Depends() dependency."""
        all_functions = self.extract_functions()
        return [f for f in all_functions if dependency_name in f.dependencies]
    
    def find_functions_raising(self, exception_name: str) -> List[FunctionInfo]:
        """Find all functions that raise a specific exception."""
        all_functions = self.extract_functions()
        return [f for f in all_functions if exception_name in f.raises]
    
    def extract_try_except_blocks(self) -> List[CodeBlock]:
        """Extract all try-except blocks from the source."""
        if not self._tree:
            return []
        
        blocks = []
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Try):
                source = self.get_source_segment(node.lineno, node.end_lineno or node.lineno)
                blocks.append(CodeBlock(
                    content=source,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    file_path=self.file_path or "",
                    block_type="try_except"
                ))
        
        return blocks
    
    def extract_if_blocks_with_pattern(self, pattern_keywords: List[str]) -> List[CodeBlock]:
        """
        Extract if blocks that contain specific keywords.
        
        Args:
            pattern_keywords: Keywords to look for in the if block source
            
        Returns:
            List of matching CodeBlock objects
        """
        if not self._tree:
            return []
        
        blocks = []
        for node in ast.walk(self._tree):
            if isinstance(node, ast.If):
                source = self.get_source_segment(node.lineno, node.end_lineno or node.lineno)
                if any(kw in source.lower() for kw in pattern_keywords):
                    blocks.append(CodeBlock(
                        content=source,
                        start_line=node.lineno,
                        end_line=node.end_lineno or node.lineno,
                        file_path=self.file_path or "",
                        block_type="if_block"
                    ))
        
        return blocks
    
    def get_route_handlers(self) -> List[FunctionInfo]:
        """Get all FastAPI route handler functions."""
        route_decorators = ['get', 'post', 'put', 'patch', 'delete', 'router']
        all_functions = self.extract_functions()
        return [
            f for f in all_functions
            if any(d.lower() in [dec.lower() for dec in f.decorators] 
                   for d in route_decorators)
        ]
    
    def extract_dbr_evidence(self, dbr_id: str) -> Dict[str, List[CodeBlock]]:
        """
        Extract code evidence relevant to a specific DBR.
        
        Args:
            dbr_id: The DBR identifier (e.g., "DBR-01")
            
        Returns:
            Dictionary mapping evidence types to code blocks
        """
        evidence = {
            "functions": [],
            "try_except": [],
            "if_blocks": [],
            "dependencies": []
        }
        
        if dbr_id == "DBR-01":
            # Authentication related evidence
            auth_functions = self.find_functions_by_name([
                'login', 'register', 'update_current_user', 'retrieve_current_user'
            ])
            evidence["functions"] = [
                CodeBlock(
                    content=f.source,
                    start_line=f.line_start,
                    end_line=f.line_end,
                    file_path=self.file_path or "",
                    block_type="function",
                    name=f.name
                )
                for f in auth_functions
            ]
            
            # Try-except blocks for error handling
            evidence["try_except"] = self.extract_try_except_blocks()
            
            # If blocks for uniqueness checks
            evidence["if_blocks"] = self.extract_if_blocks_with_pattern([
                'check_username', 'check_email', 'taken'
            ])
        
        return evidence


def load_code_map(repo_path: str, target_files: List[str], max_chars: int = 4000) -> Dict[str, str]:
    """
    Load source code from multiple files into a dictionary.
    
    Args:
        repo_path: Base path to the repository
        target_files: List of relative file paths
        max_chars: Maximum characters to load per file
        
    Returns:
        Dictionary mapping relative paths to source code
    """
    code_map = {}
    for rel_path in target_files:
        full_path = os.path.join(repo_path, rel_path)
        if os.path.exists(full_path):
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    code_map[rel_path] = f.read()[:max_chars]
            except Exception as e:
                logger.warning(f"Failed to load {rel_path}: {e}")
    return code_map
