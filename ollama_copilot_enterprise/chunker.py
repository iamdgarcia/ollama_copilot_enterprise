import ast
import re

class CodeChunker:
    """
    CodeChunker class uses Abstract Syntax Tree (AST) to split Python code into logical chunks such as functions, classes, and methods.
    If a logical block is too large, it further splits the code based on size while ensuring no chunk exceeds the specified chunk_size.
    This is the preferred method for Python code.

    Methods:
    - split_code_by_ast: Splits code into chunks using AST for Python files.
    - chunk_code_by_size: Falls back to splitting the code strictly by size when necessary.
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_code_by_ast(self, code: str):
        """Parse code into chunks based on its AST structure."""
        chunks = []
        try:
            # Parse the code into an AST
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    # Extract classes and functions as logical chunks
                    chunk = ast.get_source_segment(code, node)
                    if len(chunk) > self.chunk_size:
                        # Further split large functions/classes by character count
                        chunks.extend(self.chunk_code_by_size(chunk))
                    else:
                        chunks.append(chunk)
        except SyntaxError:
            # If AST parsing fails, fall back to simple character chunking
            chunks = self.chunk_code_by_size(code)
        return chunks

    def chunk_code_by_size(self, code: str):
        """Split code by size, allowing for chunk overlaps."""
        chunks = []
        start = 0
        while start < len(code):
            end = min(start + self.chunk_size, len(code))
            chunks.append(code[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks


class RegexCodeChunker:
    """
    RegexCodeChunker uses regular expressions to split code by function and class definitions.
    This is language-agnostic and can work for various programming languages.
    
    Methods:
    - split_code_by_regex: Splits code using regular expressions to capture functions and class definitions.
    - chunk_code_by_size: Falls back to size-based chunking if necessary.
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_code_by_regex(self, code: str):
        """Chunk code by function and class definitions using regex."""
        function_pattern = r"(def\s+\w+\s*\(.*?\)\s*:\s*)|(class\s+\w+\s*\(.*?\)\s*:\s*)"
        chunks = []
        # Find function and class definitions
        matches = [match.span() for match in re.finditer(function_pattern, code)]
        
        if not matches:
            # If no functions/classes are found, fall back to size-based chunking
            return self.chunk_code_by_size(code)

        # Split based on function boundaries
        last_end = 0
        for start, end in matches:
            # Capture the text before the function/class definition
            chunk = code[last_end:start]
            chunks.append(chunk.strip())
            last_end = start

        return chunks

    def chunk_code_by_size(self, code: str):
        """Fallback to simple size-based chunking."""
        chunks = []
        start = 0
        while start < len(code):
            end = min(start + self.chunk_size, len(code))
            chunks.append(code[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks


class HybridCodeChunker:
    """
    HybridCodeChunker combines AST-based and regex-based chunking strategies for flexible code chunking.
    It first attempts to use AST-based splitting for Python files, but falls back to regex for other languages or if AST parsing fails.
    Oversized chunks are further split by character count to ensure no chunk exceeds the specified chunk_size.

    Methods:
    - split_code: The main entry point that handles both AST and regex-based chunking.
    - split_code_by_ast: Tries to split code based on its AST structure (Python-specific).
    - split_code_by_regex: Tries to split code using regular expressions (language-agnostic).
    - chunk_code_by_size: Splits code strictly by size, allowing overlaps for large chunks.
    """
    
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_code(self, code: str):
        """
        Use AST or regex-based splitting, then fall back to size-based splitting if needed.
        This method tries to use AST-based chunking for Python code and falls back to regex for other languages.
        """
        chunks = self.split_code_by_ast_or_regex(code)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                # If chunk is too large, split by size
                final_chunks.extend(self.chunk_code_by_size(chunk))
            else:
                final_chunks.append(chunk)
        return final_chunks

    def split_code_by_ast_or_regex(self, code: str):
        """Attempt to split code using AST; fall back to regex-based splitting."""
        try:
            return self.split_code_by_ast(code)  # Prefer AST parsing
        except Exception:
            return self.split_code_by_regex(code)  # Fall back to regex

    def split_code_by_ast(self, code: str):
        """AST-based code chunking, similar to the CodeChunker class."""
        chunks = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    chunk = ast.get_source_segment(code, node)
                    if len(chunk) > self.chunk_size:
                        chunks.extend(self.chunk_code_by_size(chunk))
                    else:
                        chunks.append(chunk)
        except SyntaxError:
            return self.chunk_code_by_size(code)  # Fall back to size-based chunking if AST fails
        return chunks

    def split_code_by_regex(self, code: str):
        """Regex-based code chunking, similar to the RegexCodeChunker class."""
        function_pattern = r"(def\s+\w+\s*\(.*?\)\s*:\s*)|(class\s+\w+\s*\(.*?\)\s*:\s*)"
        chunks = []
        matches = [match.span() for match in re.finditer(function_pattern, code)]

        if not matches:
            return self.chunk_code_by_size(code)  # Fall back to size-based chunking if no functions/classes found

        last_end = 0
        for start, end in matches:
            chunk = code[last_end:start]
            chunks.append(chunk.strip())
            last_end = start

        return chunks

    def chunk_code_by_size(self, code: str):
        """Fallback to size-based chunking if logical chunks are too large."""
        chunks = []
        start = 0
        while start < len(code):
            end = min(start + self.chunk_size, len(code))
            chunks.append(code[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks
