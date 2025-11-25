#!/usr/bin/env python3
"""
Edge Case Scanner - Find potential runtime issues before they hit production.
"""

import ast
import os
import re
import sys
from collections import defaultdict
from typing import List, Tuple, Dict

# Edge case patterns to detect
EDGE_CASE_PATTERNS = [
    # Null/None handling
    (r'\.get\([^)]+\)\s*\[', 'None dereference: .get()[index] without None check', 'HIGH'),
    (r'\.get\([^)]+\)\s*\.', 'None dereference: .get().attr without None check', 'HIGH'),
    
    # Division issues
    (r'/\s*[a-zA-Z_]\w*\s*[^/]', 'Potential division by zero (no zero check)', 'MEDIUM'),
    
    # Index access without bounds check
    (r'\[\s*0\s*\]', 'Index [0] - may fail on empty sequence', 'MEDIUM'),
    (r'\[\s*-1\s*\]', 'Index [-1] - may fail on empty sequence', 'MEDIUM'),
    
    # String operations
    (r'\.split\([^)]*\)\s*\[', 'Split then index - may fail if delimiter not found', 'HIGH'),
    
    # Resource leaks
    (r'(?<!with\s)open\s*\([^)]+\)\s*$', 'File open without context manager', 'HIGH'),
    
    # Silent exceptions
    (r'except\s*:\s*$', 'Bare except clause - catches everything including KeyboardInterrupt', 'HIGH'),
    (r'except\s+\w+.*:\s*\n\s+pass\s*$', 'Exception caught but silently ignored', 'MEDIUM'),
    
    # Async pitfalls
    (r'asyncio\.run\s*\(', 'asyncio.run() - fails if loop already running', 'MEDIUM'),
    
    # Type confusion
    (r'json\.loads\s*\([^)]+\)\s*\[', 'JSON parse then access without type check', 'MEDIUM'),
    
    # Timeout issues
    (r'timeout\s*=\s*None', 'No timeout set - may hang forever', 'MEDIUM'),
]


def scan_file(filepath: str) -> List[Tuple[int, str, str]]:
    """Scan a file for edge case patterns."""
    issues = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines, 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith('#'):
                continue
                
            for pattern, desc, severity in EDGE_CASE_PATTERNS:
                if re.search(pattern, line):
                    issues.append((i, desc, severity))
                    break  # One issue per line max
                    
    except Exception as e:
        pass
    
    return issues


def scan_ast_issues(filepath: str) -> List[Tuple[int, str, str]]:
    """Use AST for deeper analysis."""
    issues = []
    
    try:
        with open(filepath, 'r') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        for node in ast.walk(tree):
            # Check for bare except
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append((node.lineno, 'Bare except: catches SystemExit, KeyboardInterrupt', 'HIGH'))
            
            # Check for assert in production code
            if isinstance(node, ast.Assert):
                if 'test' not in filepath.lower():
                    issues.append((node.lineno, 'Assert statement - disabled with -O flag', 'LOW'))
            
            # Check for eval/exec
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ('eval', 'exec'):
                        issues.append((node.lineno, f'{node.func.id}() is dangerous - potential code injection', 'CRITICAL'))
                    if node.func.id == 'globals':
                        issues.append((node.lineno, 'globals() access - potential security issue', 'MEDIUM'))
                        
    except SyntaxError:
        pass
    except Exception as e:
        pass
    
    return issues


def main():
    print("=" * 60)
    print("🔍 EDGE CASE SCANNER - Production Readiness Check")
    print("=" * 60)
    print()
    
    all_issues: Dict[str, List[Tuple[str, int, str]]] = defaultdict(list)
    
    # Walk through all Python files
    for root, dirs, files in os.walk('.'):
        # Skip directories
        dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'alembic', '.pytest_cache']]
        
        for f in files:
            if f.endswith('.py') and 'test' not in f.lower():
                filepath = os.path.join(root, f)
                
                # Regex-based scan
                regex_issues = scan_file(filepath)
                for lineno, desc, severity in regex_issues:
                    all_issues[severity].append((filepath, lineno, desc))
                
                # AST-based scan
                ast_issues = scan_ast_issues(filepath)
                for lineno, desc, severity in ast_issues:
                    all_issues[severity].append((filepath, lineno, desc))
    
    # Print results by severity
    total = 0
    for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        issues = all_issues.get(severity, [])
        if issues:
            emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '⚪'}[severity]
            print(f"{emoji} {severity} ({len(issues)} issues):")
            
            # Group by description
            by_desc = defaultdict(list)
            for filepath, lineno, desc in issues:
                by_desc[desc].append(f"{filepath}:{lineno}")
            
            for desc, locations in sorted(by_desc.items(), key=lambda x: -len(x[1])):
                print(f"   • {desc}")
                for loc in locations[:3]:
                    print(f"     └─ {loc}")
                if len(locations) > 3:
                    print(f"     └─ ... and {len(locations) - 3} more")
            print()
            total += len(issues)
    
    print("=" * 60)
    print(f"📊 Total issues found: {total}")
    print(f"   CRITICAL: {len(all_issues.get('CRITICAL', []))}")
    print(f"   HIGH:     {len(all_issues.get('HIGH', []))}")
    print(f"   MEDIUM:   {len(all_issues.get('MEDIUM', []))}")
    print(f"   LOW:      {len(all_issues.get('LOW', []))}")
    print("=" * 60)
    
    # Return exit code based on critical/high issues
    if all_issues.get('CRITICAL'):
        sys.exit(2)
    elif all_issues.get('HIGH'):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
