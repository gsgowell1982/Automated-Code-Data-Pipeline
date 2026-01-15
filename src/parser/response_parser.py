"""
Response parser module for extracting structured content from LLM outputs.

Provides utilities for:
- Extracting tagged content from LLM responses
- Cleaning and normalizing extracted text
- Processing reasoning traces
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class CleaningConfig:
    """
    Configuration for text cleaning operations.
    
    Attributes:
        remove_step_numbers: Remove step numbering (1., 2., etc.)
        remove_step_labels: Remove labels like "步骤一："
        remove_markdown_code_fences: Remove ```python etc.
        strip_whitespace: Strip leading/trailing whitespace
        min_step_length: Minimum length for reasoning steps
    """
    remove_step_numbers: bool = True
    remove_step_labels: bool = True
    remove_markdown_code_fences: bool = True
    strip_whitespace: bool = True
    min_step_length: int = 5


class ResponseParser:
    """
    Parser for extracting structured content from LLM responses.
    
    Handles extraction of [[TAG]] marked content and applies
    various cleaning operations to normalize the output.
    """
    
    # Regex patterns for tag extraction
    TAG_PATTERN = r"\[\[{tag}\]\]:\s*(.*?)(?=\[\[|$)"
    
    # Patterns for cleaning
    STEP_NUMBER_PATTERN = r'^\s*(\d+[\.\s、\)]+)'
    STEP_LABEL_PATTERN = r'步骤\w[：:]'
    CODE_FENCE_PATTERN = r'```\w*\n?'
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize the response parser.
        
        Args:
            config: Optional cleaning configuration
        """
        self.config = config or CleaningConfig()
    
    def extract_tag(self, text: str, tag: str) -> str:
        """
        Extract content marked with a specific [[TAG]].
        
        Args:
            text: The full response text
            tag: The tag name (without brackets)
            
        Returns:
            The extracted content, or empty string if not found
        """
        pattern = self.TAG_PATTERN.format(tag=tag)
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return content
        return ""
    
    def extract_all_tags(self, text: str, tags: List[str]) -> Dict[str, str]:
        """
        Extract content for multiple tags.
        
        Args:
            text: The full response text
            tags: List of tag names to extract
            
        Returns:
            Dictionary mapping tag names to extracted content
        """
        return {tag: self.extract_tag(text, tag) for tag in tags}
    
    def clean_code_snippet(self, code: str) -> str:
        """
        Clean a code snippet by removing markdown fences and normalizing.
        
        Args:
            code: The raw code snippet
            
        Returns:
            Cleaned code string
        """
        if not code:
            return ""
        
        cleaned = code
        
        # Remove markdown code fences
        if self.config.remove_markdown_code_fences:
            cleaned = re.sub(self.CODE_FENCE_PATTERN, '', cleaned)
            cleaned = cleaned.replace('```', '')
        
        # Strip whitespace
        if self.config.strip_whitespace:
            cleaned = cleaned.strip()
        
        return cleaned
    
    def clean_reasoning_text(self, text: str) -> str:
        """
        Clean reasoning text by removing numbering and labels.
        
        Args:
            text: The raw reasoning text
            
        Returns:
            Cleaned reasoning text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Remove step numbers at line starts
        if self.config.remove_step_numbers:
            cleaned = re.sub(
                self.STEP_NUMBER_PATTERN,
                '',
                cleaned,
                flags=re.MULTILINE
            )
        
        # Remove step labels
        if self.config.remove_step_labels:
            cleaned = re.sub(self.STEP_LABEL_PATTERN, '', cleaned)
        
        if self.config.strip_whitespace:
            cleaned = cleaned.strip()
        
        return cleaned
    
    def parse_reasoning_steps(self, text: str, delimiters: str = r'[;\n]') -> List[str]:
        """
        Parse reasoning text into individual steps.
        
        Args:
            text: The reasoning text (may be from [[REASONING]] tag)
            delimiters: Regex pattern for step delimiters
            
        Returns:
            List of reasoning step strings
        """
        if not text:
            return []
        
        # First clean the text
        cleaned = self.clean_reasoning_text(text)
        
        # Split by delimiters
        raw_steps = re.split(delimiters, cleaned)
        
        # Filter and clean individual steps
        steps = []
        for step in raw_steps:
            step = step.strip()
            if len(step) >= self.config.min_step_length:
                steps.append(step)
        
        return steps
    
    def parse_qa_response(self, text: str) -> Dict[str, any]:
        """
        Parse a complete Q&A generation response.
        
        Expected format with tags:
        - [[QUESTION]]: The generated question
        - [[REASONING]]: The reasoning trace
        - [[CODE]]: The relevant code snippet
        - [[ANSWER]]: The answer text
        
        Args:
            text: The full LLM response
            
        Returns:
            Dictionary with parsed components
        """
        result = {
            "question": self.extract_tag(text, "QUESTION"),
            "reasoning_steps": [],
            "code_snippet": "",
            "answer": self.extract_tag(text, "ANSWER"),
            "raw_response": text
        }
        
        # Parse reasoning
        reasoning_raw = self.extract_tag(text, "REASONING")
        result["reasoning_steps"] = self.parse_reasoning_steps(reasoning_raw)
        
        # Clean code
        code_raw = self.extract_tag(text, "CODE")
        result["code_snippet"] = self.clean_code_snippet(code_raw)
        
        return result
    
    def parse_design_response(self, text: str) -> Dict[str, any]:
        """
        Parse a design scheme generation response.
        
        Expected format with tags:
        - [[REASONING]]: Design reasoning and risk analysis
        - [[DESIGN_SOLUTION]]: The design solution with pseudocode
        
        Args:
            text: The full LLM response
            
        Returns:
            Dictionary with parsed components
        """
        result = {
            "reasoning_steps": [],
            "design_solution": "",
            "raw_response": text
        }
        
        # Parse reasoning
        reasoning_raw = self.extract_tag(text, "REASONING")
        result["reasoning_steps"] = self.parse_reasoning_steps(reasoning_raw)
        
        # Get design solution (keep formatting)
        result["design_solution"] = self.extract_tag(text, "DESIGN_SOLUTION")
        
        return result
    
    def validate_qa_response(self, parsed: Dict) -> bool:
        """
        Validate that a parsed Q&A response has required fields.
        
        Args:
            parsed: The parsed response dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["question", "reasoning_steps", "code_snippet", "answer"]
        
        for field in required_fields:
            value = parsed.get(field)
            if field == "reasoning_steps":
                if not value or len(value) == 0:
                    return False
            elif not value or len(str(value)) < 5:
                return False
        
        return True
    
    def validate_design_response(self, parsed: Dict) -> bool:
        """
        Validate that a parsed design response has required fields.
        
        Args:
            parsed: The parsed response dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if not parsed.get("design_solution"):
            return False
        
        if not parsed.get("reasoning_steps") or len(parsed["reasoning_steps"]) == 0:
            return False
        
        return True


def extract_tag_content(text: str, tag: str) -> str:
    """
    Convenience function to extract a single tag from text.
    
    Args:
        text: The full text containing tags
        tag: The tag name to extract
        
    Returns:
        The extracted content
    """
    parser = ResponseParser()
    return parser.extract_tag(text, tag)


def parse_and_clean_reasoning(text: str) -> List[str]:
    """
    Convenience function to parse and clean reasoning text.
    
    Args:
        text: The raw reasoning text
        
    Returns:
        List of cleaned reasoning steps
    """
    parser = ResponseParser()
    return parser.parse_reasoning_steps(text)
