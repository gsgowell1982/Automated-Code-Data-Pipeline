#!/usr/bin/env python3
"""
FastAPI AST Analyzer - Industrial Grade Code Analysis Tool

This module provides comprehensive static analysis for FastAPI projects using
Python's Abstract Syntax Tree (AST). It generates detailed analysis reports
including:
- Module structure and organization
- Class hierarchies and relationships
- Function signatures and complexity metrics
- API endpoint extraction (FastAPI routes)
- Dependency analysis
- Type hint coverage
- Code quality metrics

Author: Auto-generated
Version: 1.0.0
"""

import ast
import json
import os
import sys
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import (
    Dict, List, Optional, Set, Any, Tuple, Union
)
from collections import defaultdict
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HTTPMethod(str, Enum):
    """Supported HTTP methods for API routes."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


@dataclass
class ImportInfo:
    """Information about an import statement."""
    module: str
    names: List[str]
    alias: Optional[str] = None
    is_from_import: bool = False
    line_number: int = 0


@dataclass
class ParameterInfo:
    """Information about a function parameter."""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    is_args: bool = False
    is_kwargs: bool = False


@dataclass
class DecoratorInfo:
    """Information about a decorator."""
    name: str
    arguments: List[str] = field(default_factory=list)
    keyword_arguments: Dict[str, str] = field(default_factory=dict)
    line_number: int = 0


@dataclass
class FunctionInfo:
    """Comprehensive information about a function."""
    name: str
    qualified_name: str
    parameters: List[ParameterInfo] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[DecoratorInfo] = field(default_factory=list)
    docstring: Optional[str] = None
    is_async: bool = False
    is_method: bool = False
    is_classmethod: bool = False
    is_staticmethod: bool = False
    is_property: bool = False
    line_start: int = 0
    line_end: int = 0
    complexity: int = 1  # Cyclomatic complexity
    local_variables: List[str] = field(default_factory=list)
    calls: List[str] = field(default_factory=list)  # Function calls made


@dataclass
class AttributeInfo:
    """Information about a class attribute."""
    name: str
    type_annotation: Optional[str] = None
    default_value: Optional[str] = None
    is_class_var: bool = False
    line_number: int = 0


@dataclass
class ClassInfo:
    """Comprehensive information about a class."""
    name: str
    qualified_name: str
    bases: List[str] = field(default_factory=list)
    decorators: List[DecoratorInfo] = field(default_factory=list)
    docstring: Optional[str] = None
    methods: List[FunctionInfo] = field(default_factory=list)
    attributes: List[AttributeInfo] = field(default_factory=list)
    inner_classes: List['ClassInfo'] = field(default_factory=list)
    is_dataclass: bool = False
    is_pydantic_model: bool = False
    line_start: int = 0
    line_end: int = 0


@dataclass
class APIEndpoint:
    """Information about a FastAPI endpoint."""
    path: str
    http_method: str
    function_name: str
    response_model: Optional[str] = None
    status_code: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    parameters: List[ParameterInfo] = field(default_factory=list)
    docstring: Optional[str] = None
    line_number: int = 0
    name: Optional[str] = None  # Route name (name="...")


@dataclass
class RouterInfo:
    """Information about a FastAPI router."""
    variable_name: str
    prefix: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    included_routers: List[Dict[str, Any]] = field(default_factory=list)
    endpoints: List[APIEndpoint] = field(default_factory=list)


@dataclass
class ModuleInfo:
    """Comprehensive information about a Python module."""
    file_path: str
    module_name: str
    package: Optional[str] = None
    docstring: Optional[str] = None
    imports: List[ImportInfo] = field(default_factory=list)
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    global_variables: List[AttributeInfo] = field(default_factory=list)
    routers: List[RouterInfo] = field(default_factory=list)
    endpoints: List[APIEndpoint] = field(default_factory=list)
    line_count: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    code_lines: int = 0
    file_hash: str = ""


@dataclass
class CodeMetrics:
    """Overall code metrics for the project."""
    total_files: int = 0
    total_lines: int = 0
    total_code_lines: int = 0
    total_blank_lines: int = 0
    total_comment_lines: int = 0
    total_classes: int = 0
    total_functions: int = 0
    total_methods: int = 0
    total_endpoints: int = 0
    average_complexity: float = 0.0
    max_complexity: int = 0
    type_hint_coverage: float = 0.0
    docstring_coverage: float = 0.0


@dataclass
class DependencyGraph:
    """Module dependency information."""
    internal_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    external_dependencies: Set[str] = field(default_factory=set)
    dependency_matrix: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass
class AnalysisResult:
    """Complete analysis result for a FastAPI project."""
    project_name: str
    analysis_timestamp: str
    analyzer_version: str = "1.0.0"
    root_path: str = ""
    modules: List[ModuleInfo] = field(default_factory=list)
    metrics: CodeMetrics = field(default_factory=CodeMetrics)
    dependency_graph: DependencyGraph = field(default_factory=DependencyGraph)
    api_summary: Dict[str, Any] = field(default_factory=dict)
    issues: List[Dict[str, Any]] = field(default_factory=list)


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor to calculate cyclomatic complexity."""

    def __init__(self):
        self.complexity = 1

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_With(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # Count additional boolean operators (and/or)
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_comprehension(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_IfExp(self, node):
        self.complexity += 1
        self.generic_visit(node)


class CallExtractor(ast.NodeVisitor):
    """Extract function calls from AST node."""

    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.calls.append(self._get_attribute_name(node.func))
        self.generic_visit(node)

    def _get_attribute_name(self, node: ast.Attribute) -> str:
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return '.'.join(reversed(parts))


class FastAPIASTAnalyzer:
    """
    Comprehensive AST analyzer for FastAPI projects.

    This analyzer extracts detailed information about:
    - Module structure
    - Classes and their hierarchies
    - Functions and methods
    - FastAPI routes and endpoints
    - Dependencies and imports
    - Code complexity metrics
    """

    FASTAPI_ROUTE_DECORATORS = {'get', 'post', 'put', 'delete', 'patch', 'options', 'head'}
    PYDANTIC_BASE_CLASSES = {'BaseModel', 'BaseSettings', 'RWModel', 'RWSchema'}

    def __init__(self, root_path: str):
        """
        Initialize the analyzer.

        Args:
            root_path: Root path of the FastAPI project to analyze
        """
        self.root_path = Path(root_path).resolve()
        self.project_name = self.root_path.name
        self.modules: List[ModuleInfo] = []
        self.issues: List[Dict[str, Any]] = []

    def analyze(self) -> AnalysisResult:
        """
        Perform complete analysis of the FastAPI project.

        Returns:
            AnalysisResult containing all analysis data
        """
        logger.info(f"Starting analysis of: {self.root_path}")

        # Find all Python files
        python_files = self._find_python_files()
        logger.info(f"Found {len(python_files)} Python files")

        # Analyze each file
        for file_path in python_files:
            try:
                module_info = self._analyze_file(file_path)
                if module_info:
                    self.modules.append(module_info)
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {e}")
                self.issues.append({
                    "type": "PARSE_ERROR",
                    "file": str(file_path),
                    "message": str(e)
                })

        # Calculate metrics
        metrics = self._calculate_metrics()

        # Build dependency graph
        dependency_graph = self._build_dependency_graph()

        # Generate API summary
        api_summary = self._generate_api_summary()

        result = AnalysisResult(
            project_name=self.project_name,
            analysis_timestamp=datetime.now().isoformat(),
            root_path=str(self.root_path),
            modules=self.modules,
            metrics=metrics,
            dependency_graph=dependency_graph,
            api_summary=api_summary,
            issues=self.issues
        )

        logger.info("Analysis completed successfully")
        return result

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for root, dirs, files in os.walk(self.root_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {
                '__pycache__', '.git', '.venv', 'venv', 'env',
                'node_modules', '.pytest_cache', '.mypy_cache',
                'dist', 'build', 'egg-info'
            }]

            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)

        return sorted(python_files)

    def _analyze_file(self, file_path: Path) -> Optional[ModuleInfo]:
        """Analyze a single Python file."""
        logger.debug(f"Analyzing: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
        except Exception as e:
            logger.error(f"Could not read {file_path}: {e}")
            return None

        # Parse AST
        try:
            tree = ast.parse(source_code, filename=str(file_path))
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            self.issues.append({
                "type": "SYNTAX_ERROR",
                "file": str(file_path),
                "line": e.lineno,
                "message": str(e.msg)
            })
            return None

        # Calculate file metrics
        lines = source_code.split('\n')
        line_count = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        code_lines = line_count - blank_lines - comment_lines

        # Calculate file hash
        file_hash = hashlib.md5(source_code.encode()).hexdigest()

        # Get module name
        relative_path = file_path.relative_to(self.root_path)
        module_name = str(relative_path.with_suffix('')).replace(os.sep, '.')
        package = '.'.join(module_name.split('.')[:-1]) if '.' in module_name else None

        # Extract module docstring
        docstring = ast.get_docstring(tree)

        # Extract imports
        imports = self._extract_imports(tree)

        # Extract classes
        classes = self._extract_classes(tree, module_name)

        # Extract module-level functions
        functions = self._extract_functions(tree, module_name)

        # Extract global variables
        global_vars = self._extract_global_variables(tree)

        # Extract FastAPI routers and endpoints
        routers, endpoints = self._extract_fastapi_routes(tree)

        return ModuleInfo(
            file_path=str(file_path.relative_to(self.root_path)),
            module_name=module_name,
            package=package,
            docstring=docstring,
            imports=imports,
            classes=classes,
            functions=functions,
            global_variables=global_vars,
            routers=routers,
            endpoints=endpoints,
            line_count=line_count,
            blank_lines=blank_lines,
            comment_lines=comment_lines,
            code_lines=code_lines,
            file_hash=file_hash
        )

    def _extract_imports(self, tree: ast.Module) -> List[ImportInfo]:
        """Extract import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.name.split('.')[-1]],
                        alias=alias.asname,
                        is_from_import=False,
                        line_number=node.lineno
                    ))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                names = [alias.name for alias in node.names]
                imports.append(ImportInfo(
                    module=module,
                    names=names,
                    is_from_import=True,
                    line_number=node.lineno
                ))

        return imports

    def _extract_classes(self, tree: ast.Module, module_name: str) -> List[ClassInfo]:
        """Extract class definitions from AST."""
        classes = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self._parse_class(node, module_name)
                classes.append(class_info)

        return classes

    def _parse_class(self, node: ast.ClassDef, parent_name: str) -> ClassInfo:
        """Parse a class definition node."""
        qualified_name = f"{parent_name}.{node.name}"

        # Extract base classes
        bases = []
        for base in node.bases:
            bases.append(self._get_annotation_string(base))

        # Extract decorators
        decorators = [self._parse_decorator(d) for d in node.decorator_list]

        # Check if it's a dataclass or Pydantic model
        is_dataclass = any(d.name == 'dataclass' for d in decorators)
        is_pydantic = any(base in self.PYDANTIC_BASE_CLASSES for base in bases)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._parse_function(item, qualified_name, is_method=True)
                methods.append(func_info)

        # Extract attributes
        attributes = self._extract_class_attributes(node)

        # Extract inner classes
        inner_classes = []
        for item in node.body:
            if isinstance(item, ast.ClassDef):
                inner_class = self._parse_class(item, qualified_name)
                inner_classes.append(inner_class)

        return ClassInfo(
            name=node.name,
            qualified_name=qualified_name,
            bases=bases,
            decorators=decorators,
            docstring=docstring,
            methods=methods,
            attributes=attributes,
            inner_classes=inner_classes,
            is_dataclass=is_dataclass,
            is_pydantic_model=is_pydantic,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno
        )

    def _extract_class_attributes(self, node: ast.ClassDef) -> List[AttributeInfo]:
        """Extract class attributes from a class definition."""
        attributes = []

        for item in node.body:
            # Annotated assignments (type hints)
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                attr = AttributeInfo(
                    name=item.target.id,
                    type_annotation=self._get_annotation_string(item.annotation),
                    default_value=self._get_value_string(item.value) if item.value else None,
                    line_number=item.lineno
                )
                attributes.append(attr)
            # Simple assignments
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attr = AttributeInfo(
                            name=target.id,
                            default_value=self._get_value_string(item.value),
                            line_number=item.lineno
                        )
                        attributes.append(attr)

        return attributes

    def _extract_functions(self, tree: ast.Module, module_name: str) -> List[FunctionInfo]:
        """Extract module-level functions from AST."""
        functions = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_info = self._parse_function(node, module_name, is_method=False)
                functions.append(func_info)

        return functions

    def _parse_function(
            self,
            node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
            parent_name: str,
            is_method: bool = False
    ) -> FunctionInfo:
        """Parse a function/method definition node."""
        qualified_name = f"{parent_name}.{node.name}"

        # Extract parameters
        parameters = self._extract_parameters(node.args)

        # Extract return type
        return_type = self._get_annotation_string(node.returns) if node.returns else None

        # Extract decorators
        decorators = [self._parse_decorator(d) for d in node.decorator_list]

        # Check decorator types
        is_classmethod = any(d.name == 'classmethod' for d in decorators)
        is_staticmethod = any(d.name == 'staticmethod' for d in decorators)
        is_property = any(d.name == 'property' for d in decorators)

        # Extract docstring
        docstring = ast.get_docstring(node)

        # Calculate complexity
        complexity_visitor = ComplexityVisitor()
        complexity_visitor.visit(node)

        # Extract function calls
        call_extractor = CallExtractor()
        call_extractor.visit(node)

        # Extract local variables
        local_vars = self._extract_local_variables(node)

        return FunctionInfo(
            name=node.name,
            qualified_name=qualified_name,
            parameters=parameters,
            return_type=return_type,
            decorators=decorators,
            docstring=docstring,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_method=is_method,
            is_classmethod=is_classmethod,
            is_staticmethod=is_staticmethod,
            is_property=is_property,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            complexity=complexity_visitor.complexity,
            local_variables=local_vars,
            calls=call_extractor.calls
        )

    def _extract_parameters(self, args: ast.arguments) -> List[ParameterInfo]:
        """Extract function parameters."""
        parameters = []

        # Get defaults alignment
        num_defaults = len(args.defaults)
        num_args = len(args.args)
        defaults_start = num_args - num_defaults

        # Regular arguments
        for i, arg in enumerate(args.args):
            default_idx = i - defaults_start
            default_value = None
            if default_idx >= 0 and default_idx < len(args.defaults):
                default_value = self._get_value_string(args.defaults[default_idx])

            param = ParameterInfo(
                name=arg.arg,
                type_annotation=self._get_annotation_string(arg.annotation) if arg.annotation else None,
                default_value=default_value
            )
            parameters.append(param)

        # *args
        if args.vararg:
            parameters.append(ParameterInfo(
                name=args.vararg.arg,
                type_annotation=self._get_annotation_string(args.vararg.annotation) if args.vararg.annotation else None,
                is_args=True
            ))

        # Keyword-only arguments
        for i, arg in enumerate(args.kwonlyargs):
            default_value = None
            if i < len(args.kw_defaults) and args.kw_defaults[i]:
                default_value = self._get_value_string(args.kw_defaults[i])

            param = ParameterInfo(
                name=arg.arg,
                type_annotation=self._get_annotation_string(arg.annotation) if arg.annotation else None,
                default_value=default_value
            )
            parameters.append(param)

        # **kwargs
        if args.kwarg:
            parameters.append(ParameterInfo(
                name=args.kwarg.arg,
                type_annotation=self._get_annotation_string(args.kwarg.annotation) if args.kwarg.annotation else None,
                is_kwargs=True
            ))

        return parameters

    def _extract_local_variables(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> List[str]:
        """Extract local variable names from a function."""
        local_vars = set()

        for child in ast.walk(node):
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        local_vars.add(target.id)
            elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                local_vars.add(child.target.id)

        return list(local_vars)

    def _extract_global_variables(self, tree: ast.Module) -> List[AttributeInfo]:
        """Extract module-level variables."""
        variables = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                var = AttributeInfo(
                    name=node.target.id,
                    type_annotation=self._get_annotation_string(node.annotation),
                    default_value=self._get_value_string(node.value) if node.value else None,
                    line_number=node.lineno
                )
                variables.append(var)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Skip common non-variable assignments
                        if not target.id.startswith('_') or target.id in ['__all__', '__version__']:
                            var = AttributeInfo(
                                name=target.id,
                                default_value=self._get_value_string(node.value),
                                line_number=node.lineno
                            )
                            variables.append(var)

        return variables

    def _extract_fastapi_routes(self, tree: ast.Module) -> Tuple[List[RouterInfo], List[APIEndpoint]]:
        """Extract FastAPI routers and endpoints from the AST."""
        routers = []
        endpoints = []

        # Find router assignments
        router_vars = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if isinstance(node.value, ast.Call):
                            func_name = self._get_call_name(node.value)
                            if func_name in ['APIRouter', 'FastAPI']:
                                router_info = self._parse_router_creation(target.id, node.value)
                                routers.append(router_info)
                                router_vars[target.id] = router_info

        # Find route decorators and include_router calls
        for node in ast.walk(tree):
            # Find decorated functions (route handlers)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for decorator in node.decorator_list:
                    endpoint = self._parse_route_decorator(decorator, node)
                    if endpoint:
                        endpoints.append(endpoint)
                        # Add to appropriate router if found
                        router_name = self._get_router_from_decorator(decorator)
                        if router_name and router_name in router_vars:
                            router_vars[router_name].endpoints.append(endpoint)

            # Find include_router calls
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Attribute) and call.func.attr == 'include_router':
                    router_name = self._get_attribute_base_name(call.func)
                    if router_name and router_name in router_vars:
                        include_info = self._parse_include_router(call)
                        router_vars[router_name].included_routers.append(include_info)

        return routers, endpoints

    def _parse_router_creation(self, var_name: str, call_node: ast.Call) -> RouterInfo:
        """Parse APIRouter() or FastAPI() creation."""
        router = RouterInfo(variable_name=var_name)

        for keyword in call_node.keywords:
            if keyword.arg == 'prefix':
                router.prefix = self._get_value_string(keyword.value)
            elif keyword.arg == 'tags':
                if isinstance(keyword.value, ast.List):
                    router.tags = [self._get_value_string(elt) for elt in keyword.value.elts]

        return router

    def _parse_include_router(self, call_node: ast.Call) -> Dict[str, Any]:
        """Parse include_router() call."""
        info = {
            'router': '',
            'prefix': None,
            'tags': []
        }

        # Get the router being included
        if call_node.args:
            info['router'] = self._get_value_string(call_node.args[0])

        # Get keyword arguments
        for keyword in call_node.keywords:
            if keyword.arg == 'prefix':
                info['prefix'] = self._get_value_string(keyword.value)
            elif keyword.arg == 'tags':
                if isinstance(keyword.value, ast.List):
                    info['tags'] = [self._get_value_string(elt) for elt in keyword.value.elts]

        return info

    def _parse_route_decorator(
            self,
            decorator: ast.expr,
            func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> Optional[APIEndpoint]:
        """Parse a route decorator and extract endpoint information."""
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                method_name = decorator.func.attr.lower()
                if method_name in self.FASTAPI_ROUTE_DECORATORS:
                    return self._create_endpoint(decorator, func_node, method_name.upper())
        elif isinstance(decorator, ast.Attribute):
            method_name = decorator.attr.lower()
            if method_name in self.FASTAPI_ROUTE_DECORATORS:
                return self._create_endpoint(decorator, func_node, method_name.upper())

        return None

    def _create_endpoint(
            self,
            decorator: Union[ast.Call, ast.Attribute],
            func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
            http_method: str
    ) -> APIEndpoint:
        """Create an APIEndpoint from decorator information."""
        path = ""
        response_model = None
        status_code = None
        tags = []
        name = None
        dependencies = []

        if isinstance(decorator, ast.Call):
            # First positional argument is the path
            if decorator.args:
                path = self._get_value_string(decorator.args[0])

            # Process keyword arguments
            for keyword in decorator.keywords:
                if keyword.arg == 'response_model':
                    response_model = self._get_value_string(keyword.value)
                elif keyword.arg == 'status_code':
                    status_code_val = self._get_value_string(keyword.value)
                    try:
                        status_code = int(status_code_val) if status_code_val.isdigit() else None
                    except:
                        status_code = None
                elif keyword.arg == 'tags':
                    if isinstance(keyword.value, ast.List):
                        tags = [self._get_value_string(elt) for elt in keyword.value.elts]
                elif keyword.arg == 'name':
                    name = self._get_value_string(keyword.value)
                elif keyword.arg == 'dependencies':
                    if isinstance(keyword.value, ast.List):
                        dependencies = [self._get_value_string(elt) for elt in keyword.value.elts]

        # Extract parameters from function
        parameters = self._extract_parameters(func_node.args)

        # Extract docstring
        docstring = ast.get_docstring(func_node)

        return APIEndpoint(
            path=path,
            http_method=http_method,
            function_name=func_node.name,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            parameters=parameters,
            docstring=docstring,
            line_number=func_node.lineno,
            name=name
        )

    def _get_router_from_decorator(self, decorator: ast.expr) -> Optional[str]:
        """Get the router variable name from a decorator."""
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            return self._get_attribute_base_name(decorator.func)
        elif isinstance(decorator, ast.Attribute):
            return self._get_attribute_base_name(decorator)
        return None

    def _get_attribute_base_name(self, node: ast.Attribute) -> Optional[str]:
        """Get the base name of an attribute access."""
        current = node
        while isinstance(current, ast.Attribute):
            current = current.value
        if isinstance(current, ast.Name):
            return current.id
        return None

    def _parse_decorator(self, node: ast.expr) -> DecoratorInfo:
        """Parse a decorator node."""
        if isinstance(node, ast.Name):
            return DecoratorInfo(name=node.id, line_number=node.lineno)
        elif isinstance(node, ast.Attribute):
            return DecoratorInfo(
                name=self._get_annotation_string(node),
                line_number=node.lineno
            )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = self._get_annotation_string(node.func)
            else:
                name = "<unknown>"

            args = [self._get_value_string(arg) for arg in node.args]
            kwargs = {kw.arg: self._get_value_string(kw.value) for kw in node.keywords if kw.arg}

            return DecoratorInfo(
                name=name,
                arguments=args,
                keyword_arguments=kwargs,
                line_number=node.lineno
            )
        else:
            return DecoratorInfo(name="<unknown>", line_number=getattr(node, 'lineno', 0))

    def _get_call_name(self, node: ast.Call) -> str:
        """Get the name of a function call."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    def _get_annotation_string(self, node: Optional[ast.expr]) -> str:
        """Convert an AST annotation to a string representation."""
        if node is None:
            return ""

        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._get_annotation_string(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            value = self._get_annotation_string(node.value)
            slice_val = self._get_annotation_string(node.slice)
            return f"{value}[{slice_val}]"
        elif isinstance(node, ast.Tuple):
            elements = ", ".join(self._get_annotation_string(elt) for elt in node.elts)
            return elements
        elif isinstance(node, ast.List):
            elements = ", ".join(self._get_annotation_string(elt) for elt in node.elts)
            return f"[{elements}]"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Union type with | operator (Python 3.10+)
            left = self._get_annotation_string(node.left)
            right = self._get_annotation_string(node.right)
            return f"{left} | {right}"
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else str(type(node).__name__)

    def _get_value_string(self, node: Optional[ast.expr]) -> str:
        """Convert an AST value node to a string representation."""
        if node is None:
            return ""

        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):
                return node.value
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return self._get_annotation_string(node)
        elif isinstance(node, ast.Call):
            func_name = self._get_call_name(node)
            args_str = ", ".join(self._get_value_string(arg) for arg in node.args)
            return f"{func_name}({args_str})"
        elif isinstance(node, ast.List):
            elements = ", ".join(self._get_value_string(elt) for elt in node.elts)
            return f"[{elements}]"
        elif isinstance(node, ast.Dict):
            pairs = []
            for k, v in zip(node.keys, node.values):
                key_str = self._get_value_string(k) if k else "**"
                val_str = self._get_value_string(v)
                pairs.append(f"{key_str}: {val_str}")
            return "{" + ", ".join(pairs) + "}"
        elif isinstance(node, ast.Tuple):
            elements = ", ".join(self._get_value_string(elt) for elt in node.elts)
            return f"({elements})"
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else str(type(node).__name__)

    def _calculate_metrics(self) -> CodeMetrics:
        """Calculate overall code metrics."""
        metrics = CodeMetrics()

        total_functions = 0
        total_methods = 0
        total_complexity = 0
        max_complexity = 0

        functions_with_types = 0
        total_parameters = 0
        parameters_with_types = 0

        functions_with_docs = 0
        classes_with_docs = 0

        for module in self.modules:
            metrics.total_files += 1
            metrics.total_lines += module.line_count
            metrics.total_code_lines += module.code_lines
            metrics.total_blank_lines += module.blank_lines
            metrics.total_comment_lines += module.comment_lines
            metrics.total_classes += len(module.classes)
            metrics.total_endpoints += len(module.endpoints)

            # Process functions
            for func in module.functions:
                total_functions += 1
                total_complexity += func.complexity
                max_complexity = max(max_complexity, func.complexity)

                if func.return_type:
                    functions_with_types += 1
                if func.docstring:
                    functions_with_docs += 1

                for param in func.parameters:
                    total_parameters += 1
                    if param.type_annotation:
                        parameters_with_types += 1

            # Process classes
            for cls in module.classes:
                if cls.docstring:
                    classes_with_docs += 1

                for method in cls.methods:
                    total_methods += 1
                    total_complexity += method.complexity
                    max_complexity = max(max_complexity, method.complexity)

                    if method.return_type:
                        functions_with_types += 1
                    if method.docstring:
                        functions_with_docs += 1

                    for param in method.parameters:
                        if param.name != 'self' and param.name != 'cls':
                            total_parameters += 1
                            if param.type_annotation:
                                parameters_with_types += 1

        metrics.total_functions = total_functions
        metrics.total_methods = total_methods

        total_callables = total_functions + total_methods
        if total_callables > 0:
            metrics.average_complexity = round(total_complexity / total_callables, 2)
        metrics.max_complexity = max_complexity

        # Type hint coverage
        type_hint_items = functions_with_types + parameters_with_types
        type_hint_total = total_functions + total_methods + total_parameters
        if type_hint_total > 0:
            metrics.type_hint_coverage = round((type_hint_items / type_hint_total) * 100, 2)

        # Docstring coverage
        total_documented = functions_with_docs + classes_with_docs
        total_documentable = total_functions + total_methods + metrics.total_classes
        if total_documentable > 0:
            metrics.docstring_coverage = round((total_documented / total_documentable) * 100, 2)

        return metrics

    def _build_dependency_graph(self) -> DependencyGraph:
        """Build a dependency graph from imports."""
        graph = DependencyGraph()

        internal_modules = {m.module_name for m in self.modules}

        for module in self.modules:
            module_deps = set()

            for imp in module.imports:
                if imp.is_from_import:
                    full_module = imp.module
                else:
                    full_module = imp.module

                # Determine if internal or external
                is_internal = False
                for internal_mod in internal_modules:
                    if full_module.startswith(internal_mod.split('.')[0]):
                        is_internal = True
                        break

                if is_internal or full_module.startswith('app'):
                    if module.module_name not in graph.internal_dependencies:
                        graph.internal_dependencies[module.module_name] = []
                    if full_module not in graph.internal_dependencies[module.module_name]:
                        graph.internal_dependencies[module.module_name].append(full_module)
                    module_deps.add(full_module)
                else:
                    # Extract top-level package name
                    top_level = full_module.split('.')[0]
                    if top_level:
                        graph.external_dependencies.add(top_level)

            graph.dependency_matrix[module.module_name] = module_deps

        # Convert set to list for JSON serialization
        graph.external_dependencies = list(graph.external_dependencies)

        return graph

    def _generate_api_summary(self) -> Dict[str, Any]:
        """Generate a summary of API endpoints."""
        summary = {
            'total_endpoints': 0,
            'endpoints_by_method': defaultdict(int),
            'endpoints_by_tag': defaultdict(int),
            'routes': []
        }

        for module in self.modules:
            for endpoint in module.endpoints:
                summary['total_endpoints'] += 1
                summary['endpoints_by_method'][endpoint.http_method] += 1

                for tag in endpoint.tags:
                    summary['endpoints_by_tag'][tag] += 1

                summary['routes'].append({
                    'method': endpoint.http_method,
                    'path': endpoint.path,
                    'function': endpoint.function_name,
                    'response_model': endpoint.response_model,
                    'file': module.file_path
                })

            # Also include routes from routers
            for router in module.routers:
                for endpoint in router.endpoints:
                    if endpoint not in module.endpoints:
                        summary['total_endpoints'] += 1
                        summary['endpoints_by_method'][endpoint.http_method] += 1

        # Convert defaultdicts to regular dicts for JSON serialization
        summary['endpoints_by_method'] = dict(summary['endpoints_by_method'])
        summary['endpoints_by_tag'] = dict(summary['endpoints_by_tag'])

        return summary


def convert_to_dict(obj: Any) -> Any:
    """Convert dataclass instances to dictionaries recursively."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: convert_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, list):
        return [convert_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


def analyze_fastapi_project(
        project_path: str,
        output_path: Optional[str] = None,
        output_format: str = 'json'
) -> AnalysisResult:
    """
    Analyze a FastAPI project and generate analysis results.

    Args:
        project_path: Path to the FastAPI project root
        output_path: Optional path to save the analysis results
        output_format: Output format ('json' or 'yaml')

    Returns:
        AnalysisResult containing complete analysis data
    """
    analyzer = FastAPIASTAnalyzer(project_path)
    result = analyzer.analyze()

    # Convert result to dictionary
    result_dict = convert_to_dict(result)

    # Save output if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, ensure_ascii=False)
            logger.info(f"Analysis saved to: {output_path}")
        else:
            # YAML output (requires pyyaml)
            try:
                import yaml
                with open(output_path, 'w', encoding='utf-8') as f:
                    yaml.dump(result_dict, f, default_flow_style=False, allow_unicode=True)
                logger.info(f"Analysis saved to: {output_path}")
            except ImportError:
                logger.warning("PyYAML not installed. Falling back to JSON.")
                json_path = output_path.with_suffix('.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)
                logger.info(f"Analysis saved to: {json_path}")

    return result


def print_summary(result: AnalysisResult) -> None:
    """Print a human-readable summary of the analysis."""
    print("\n" + "=" * 60)
    print(f"FastAPI Project Analysis Report")
    print("=" * 60)
    print(f"\nProject: {result.project_name}")
    print(f"Analysis Time: {result.analysis_timestamp}")
    print(f"Root Path: {result.root_path}")

    print("\n--- Code Metrics ---")
    m = result.metrics
    print(f"Total Files: {m.total_files}")
    print(f"Total Lines: {m.total_lines:,}")
    print(f"  - Code Lines: {m.total_code_lines:,}")
    print(f"  - Blank Lines: {m.total_blank_lines:,}")
    print(f"  - Comment Lines: {m.total_comment_lines:,}")
    print(f"Total Classes: {m.total_classes}")
    print(f"Total Functions: {m.total_functions}")
    print(f"Total Methods: {m.total_methods}")
    print(f"Total API Endpoints: {m.total_endpoints}")
    print(f"Average Complexity: {m.average_complexity}")
    print(f"Max Complexity: {m.max_complexity}")
    print(f"Type Hint Coverage: {m.type_hint_coverage}%")
    print(f"Docstring Coverage: {m.docstring_coverage}%")

    print("\n--- API Summary ---")
    api = result.api_summary
    print(f"Total Endpoints: {api.get('total_endpoints', 0)}")
    print("Endpoints by Method:")
    for method, count in api.get('endpoints_by_method', {}).items():
        print(f"  - {method}: {count}")

    print("\n--- External Dependencies ---")
    deps = result.dependency_graph
    if isinstance(deps, DependencyGraph):
        ext_deps = deps.external_dependencies
    else:
        ext_deps = deps.get('external_dependencies', [])
    for dep in sorted(ext_deps if isinstance(ext_deps, (list, set)) else []):
        print(f"  - {dep}")

    if result.issues:
        print("\n--- Issues Found ---")
        for issue in result.issues:
            print(f"  [{issue['type']}] {issue['file']}: {issue['message']}")

    print("\n" + "=" * 60)


def main():
    """Main entry point for the analyzer."""
    import argparse

    parser = argparse.ArgumentParser(
        description='FastAPI AST Analyzer - Industrial Grade Code Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze local FastAPI project
  python fastapi_ast_analyzer.py /path/to/project

  # Analyze and save results to JSON
  python fastapi_ast_analyzer.py /path/to/project -o analysis_result.json

  # Analyze with summary output
  python fastapi_ast_analyzer.py /path/to/project --summary
        """
    )

    parser.add_argument(
        'project_path',
        nargs='?',
        default=None,
        help='Path to the FastAPI project to analyze'
    )

    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output file path for analysis results'
    )

    parser.add_argument(
        '-f', '--format',
        choices=['json', 'yaml'],
        default='json',
        help='Output format (default: json)'
    )

    parser.add_argument(
        '-s', '--summary',
        action='store_true',
        help='Print human-readable summary to console'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default project path for this workspace
    if args.project_path is None:
        # Default to the fastapi-realworld-example-app in the repos directory
        script_dir = Path(__file__).parent.resolve()
        workspace_root = script_dir.parent.parent
        args.project_path = workspace_root / 'repos' / 'fastapi-realworld-example-app'

        if not args.project_path.exists():
            print(f"Error: Default project path not found: {args.project_path}")
            print("Please provide a project path as an argument.")
            sys.exit(1)

    project_path = Path(args.project_path).resolve()

    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}")
        sys.exit(1)

    # Set default output path if not specified
    if args.output is None:
        script_dir = Path(__file__).parent.resolve()
        workspace_root = script_dir.parent.parent
        data_dir = workspace_root / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(data_dir / f'fastapi_analysis_result.{args.format}')

    # Run analysis
    print(f"Analyzing: {project_path}")
    result = analyze_fastapi_project(
        str(project_path),
        output_path=args.output,
        output_format=args.format
    )

    # Print summary if requested or by default
    if args.summary or args.output:
        print_summary(result)

    if args.output:
        print(f"\nResults saved to: {args.output}")

    return result


if __name__ == '__main__':
    main()