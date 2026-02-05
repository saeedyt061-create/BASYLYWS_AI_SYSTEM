#!/usr/bin/env python3
"""
محلل الكود الذكي - Smart Code Analyzer
======================================
يحلل جودة الكود ويكتشف المشاكل ويقترح التحسينات
"""

import ast
import re
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter
import hashlib


@dataclass
class CodeMetrics:
    """مقاييس جودة الكود"""
    lines_of_code: int = 0
    logical_lines: int = 0
    comment_lines: int = 0
    blank_lines: int = 0
    complexity: int = 0
    max_nesting: int = 0
    function_count: int = 0
    class_count: int = 0
    average_function_length: float = 0.0
    duplicate_lines: int = 0
    test_coverage_estimate: float = 0.0


@dataclass
class CodeIssue:
    """مشكلة في الكود"""
    line: int
    severity: str  # ERROR, WARNING, INFO
    code: str
    message: str
    suggestion: str


class SmartCodeAnalyzer:
    """
    محلل كود ذكي يستخرج الميزات ويكتشف المشاكل
    """
    
    def __init__(self):
        self.metrics = CodeMetrics()
        self.issues: List[CodeIssue] = []
        self.keywords = [
            'def', 'class', 'if', 'else', 'elif', 'for', 'while',
            'try', 'except', 'finally', 'with', 'import', 'from',
            'return', 'yield', 'lambda', 'async', 'await'
        ]
        
    def analyze(self, code: str, filename: str = "<unknown>") -> Dict[str, Any]:
        """تحليل شامل للكود"""
        self.issues = []
        
        # التحقق من صحة الصياغة
        try:
            tree = ast.parse(code)
            is_valid = True
        except SyntaxError as e:
            self.issues.append(CodeIssue(
                line=e.lineno or 1,
                severity="ERROR",
                code="SYNTAX_ERROR",
                message=str(e),
                suggestion="تحقق من صياغة الكود"
            ))
            tree = None
            is_valid = False
        
        # استخراج المقاييس
        metrics = self._calculate_metrics(code, tree)
        
        # اكتشاف روائح الكود
        smells = self._detect_smells(code, tree, metrics)
        
        # اكتشاف الأسرار
        secrets = self._detect_secrets(code)
        
        # تحليل الاستيرادات
        imports = self._analyze_imports(tree) if tree else []
        
        # حساب درجة الجودة
        quality_score = self._calculate_quality_score(metrics, smells, secrets)
        
        return {
            "filename": filename,
            "is_valid": is_valid,
            "metrics": metrics.__dict__,
            "issues": [self._issue_to_dict(i) for i in self.issues],
            "code_smells": smells,
            "secrets_detected": secrets,
            "imports": imports,
            "quality_score": quality_score,
            "language": self._detect_language(code),
            "hash": hashlib.md5(code.encode()).hexdigest()[:12]
        }
    
    def _calculate_metrics(self, code: str, tree: Optional[ast.AST]) -> CodeMetrics:
        """حساب مقاييس الكود"""
        lines = code.split('\n')
        
        metrics = CodeMetrics()
        metrics.lines_of_code = len(lines)
        metrics.blank_lines = sum(1 for line in lines if not line.strip())
        metrics.comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        
        # الأسطر المنطقية (غير الفارغة وغير التعليقات)
        metrics.logical_lines = sum(1 for line in lines 
                                    if line.strip() and not line.strip().startswith('#'))
        
        if tree:
            # تعقيد Cyclomatic
            metrics.complexity = self._calculate_complexity(tree)
            
            # أقصى تداخل
            metrics.max_nesting = self._calculate_max_nesting(tree)
            
            # عدد الدوال والفئات
            metrics.function_count = len([n for n in ast.walk(tree) 
                                          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])
            metrics.class_count = len([n for n in ast.walk(tree) 
                                       if isinstance(n, ast.ClassDef)])
            
            # متوسط طول الدالة
            func_lengths = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_length = node.end_lineno - node.lineno if node.end_lineno else 10
                    func_lengths.append(func_length)
            
            if func_lengths:
                metrics.average_function_length = sum(func_lengths) / len(func_lengths)
        
        return metrics
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """حساب التعقيد الدوراني (Cyclomatic Complexity)"""
        complexity = 1  # القاعدة
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, 
                                ast.ExceptHandler, ast.With, ast.Assert)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_max_nesting(self, tree: ast.AST) -> int:
        """حساب أقصى مستوى تداخل"""
        max_nesting = 0
        
        def visit_node(node, current_depth=0):
            nonlocal max_nesting
            max_nesting = max(max_nesting, current_depth)
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.For, ast.While, 
                                    ast.FunctionDef, ast.ClassDef, ast.With)):
                    visit_node(child, current_depth + 1)
                else:
                    visit_node(child, current_depth)
        
        visit_node(tree)
        return max_nesting
    
    def _detect_smells(self, code: str, tree: Optional[ast.AST], 
                       metrics: CodeMetrics) -> List[Dict[str, Any]]:
        """اكتشاف روائح الكود"""
        smells = []
        
        # دالة طويلة جداً
        if metrics.average_function_length > 30:
            smells.append({
                "type": "LONG_FUNCTION",
                "severity": "WARNING",
                "message": f"متوسط طول الدالة {metrics.average_function_length:.1f} سطر",
                "suggestion": "قسّم الدوال الكبيرة إلى دوال أصغر"
            })
        
        # تعقيد عالٍ
        if metrics.complexity > 10:
            smells.append({
                "type": "HIGH_COMPLEXITY",
                "severity": "WARNING", 
                "message": f"تعقيد دوراني عالٍ: {metrics.complexity}",
                "suggestion": "بسّط المنطق أو استخرج دوال مساعدة"
            })
        
        # تداخل عميق
        if metrics.max_nesting > 4:
            smells.append({
                "type": "DEEP_NESTING",
                "severity": "WARNING",
                "message": f"تداخل عميق: {metrics.max_nesting} مستويات",
                "suggestion": "استخدم return مبكر أو استخرج دوال"
            })
        
        # كود بدون تعليقات
        comment_ratio = metrics.comment_lines / max(metrics.lines_of_code, 1)
        if comment_ratio < 0.05 and metrics.lines_of_code > 20:
            smells.append({
                "type": "NO_COMMENTS",
                "severity": "INFO",
                "message": "نسبة التعليقات منخفضة جداً",
                "suggestion": "أضف docstrings وتعليقات توضيحية"
            })
        
        # كود مكرر
        duplicates = self._detect_duplicates(code)
        if duplicates:
            smells.append({
                "type": "DUPLICATE_CODE",
                "severity": "WARNING",
                "message": f"{len(duplicates)} كتل كود مكررة",
                "suggestion": "استخرج الكود المشترك إلى دالة"
            })
        
        # متغيرات غير مستخدمة
        if tree:
            unused = self._detect_unused_variables(tree)
            if unused:
                smells.append({
                    "type": "UNUSED_VARIABLES",
                    "severity": "INFO",
                    "message": f"متغيرات غير مستخدمة: {', '.join(unused[:3])}",
                    "suggestion": "احذف المتغيرات غير المستخدمة"
                })
        
        return smells
    
    def _detect_duplicates(self, code: str, min_lines: int = 5) -> List[Dict]:
        """اكتشاف الكود المكرر"""
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        duplicates = []
        seen = {}
        
        for i in range(len(lines) - min_lines + 1):
            block = '\n'.join(lines[i:i + min_lines])
            block_hash = hashlib.md5(block.encode()).hexdigest()
            
            if block_hash in seen:
                duplicates.append({
                    "first_at": seen[block_hash],
                    "duplicate_at": i,
                    "block": block[:100]
                })
            else:
                seen[block_hash] = i
        
        return duplicates
    
    def _detect_unused_variables(self, tree: ast.AST) -> List[str]:
        """اكتشاف المتغيرات غير المستخدمة"""
        assigned = set()
        used = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    assigned.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used.add(node.id)
        
        return list(assigned - used - {'_', 'self', 'cls'})
    
    def _detect_secrets(self, code: str) -> List[Dict]:
        """اكتشاف الأسرار والمفاتيح"""
        secrets = []
        
        patterns = {
            'API_KEY': r'api[_-]?key\s*[=:]\s*["\'][a-zA-Z0-9]{16,}["\']',
            'PASSWORD': r'password\s*[=:]\s*["\'][^"\']+["\']',
            'SECRET': r'secret\s*[=:]\s*["\'][a-zA-Z0-9]{8,}["\']',
            'TOKEN': r'token\s*[=:]\s*["\'][a-zA-Z0-9]{10,}["\']',
            'PRIVATE_KEY': r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'
        }
        
        for secret_type, pattern in patterns.items():
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                secrets.append({
                    "type": secret_type,
                    "line": code[:match.start()].count('\n') + 1,
                    "snippet": match.group()[:50] + "..."
                })
        
        return secrets
    
    def _analyze_imports(self, tree: ast.AST) -> List[Dict]:
        """تحليل الاستيرادات"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        "type": "import",
                        "module": alias.name,
                        "alias": alias.asname
                    })
            elif isinstance(node, ast.ImportFrom):
                imports.append({
                    "type": "from_import",
                    "module": node.module,
                    "names": [a.name for a in node.names]
                })
        
        return imports
    
    def _calculate_quality_score(self, metrics: CodeMetrics, 
                                  smells: List[Dict], 
                                  secrets: List[Dict]) -> float:
        """حساب درجة جودة الكود (0-100)"""
        score = 100.0
        
        # خصم للتعقيد
        if metrics.complexity > 10:
            score -= min(20, (metrics.complexity - 10) * 2)
        
        # خصم للتداخل
        if metrics.max_nesting > 3:
            score -= min(15, (metrics.max_nesting - 3) * 5)
        
        # خصم لطول الدوال
        if metrics.average_function_length > 20:
            score -= min(15, (metrics.average_function_length - 20))
        
        # خصم للروائح
        for smell in smells:
            if smell["severity"] == "ERROR":
                score -= 15
            elif smell["severity"] == "WARNING":
                score -= 8
            else:
                score -= 3
        
        # خصم كبير للأسرار
        score -= len(secrets) * 20
        
        return max(0, min(100, score))
    
    def _detect_language(self, code: str) -> str:
        """اكتشاف لغة البرمجة"""
        indicators = {
            'python': [r'def\s+\w+\s*\(', r'import\s+\w+', r'print\s*\(', r'\s*#'],
            'javascript': [r'function\s+\w+', r'const\s+\w+\s*=', r'console\.log'],
            'java': [r'public\s+class', r'System\.out\.println', r'private\s+\w+'],
        }
        
        scores = {}
        for lang, patterns in indicators.items():
            score = sum(1 for p in patterns if re.search(p, code))
            scores[lang] = score
        
        return max(scores, key=scores.get) if max(scores.values()) > 0 else 'unknown'
    
    def _issue_to_dict(self, issue: CodeIssue) -> Dict:
        """تحويل المشكلة إلى قاموس"""
        return {
            "line": issue.line,
            "severity": issue.severity,
            "code": issue.code,
            "message": issue.message,
            "suggestion": issue.suggestion
        }
    
    def extract_features(self, code: str) -> Dict[str, float]:
        """استخراج ميزات للتعلم الآلي"""
        analysis = self.analyze(code)
        metrics = analysis['metrics']
        
        return {
            'lines_of_code': float(metrics['lines_of_code']),
            'logical_lines': float(metrics['logical_lines']),
            'complexity': float(metrics['complexity']),
            'max_nesting': float(metrics['max_nesting']),
            'function_count': float(metrics['function_count']),
            'class_count': float(metrics['class_count']),
            'avg_function_length': float(metrics['average_function_length']),
            'comment_ratio': metrics['comment_lines'] / max(metrics['lines_of_code'], 1),
            'quality_score': float(analysis['quality_score']),
            'smell_count': float(len(analysis['code_smells'])),
            'has_secrets': float(len(analysis['secrets_detected'])),
        }


class CodeQualityAnalyzer:
    """
    محلل جودة الكود المتقدم
    """
    
    def __init__(self):
        self.smart_analyzer = SmartCodeAnalyzer()
    
    def analyze(self, code: str) -> Dict[str, Any]:
        """تحليل شامل للجودة"""
        basic = self.smart_analyzer.analyze(code)
        
        # تحليل إضافي
        maintainability = self._calculate_maintainability(basic['metrics'])
        reliability = self._calculate_reliability(basic)
        
        return {
            **basic,
            "maintainability_index": maintainability,
            "reliability_score": reliability,
            "overall_score": (basic['quality_score'] + maintainability + reliability) / 3,
            "issues": basic['issues'],
            "recommendations": self._generate_recommendations(basic)
        }
    
    def _calculate_maintainability(self, metrics: Dict) -> float:
        """حساب مؤشر الصيانة"""
        # صيغة مبسطة للمؤشر
        halstead_volume = metrics['logical_lines'] * math.log2(max(metrics['logical_lines'], 2))
        cyclomatic = metrics['complexity']
        lines_of_code = metrics['lines_of_code']
        
        maintainability = 171 - 5.2 * math.log(halstead_volume + 1) \
                         - 0.23 * cyclomatic - 16.2 * math.log(lines_of_code + 1)
        
        return max(0, min(100, maintainability))
    
    def _calculate_reliability(self, analysis: Dict) -> float:
        """حساب درجة الموثوقية"""
        score = 100.0
        
        # خصم للأخطاء
        errors = sum(1 for i in analysis['issues'] if i['severity'] == 'ERROR')
        score -= errors * 20
        
        # خصم للتحذيرات
        warnings = sum(1 for i in analysis['issues'] if i['severity'] == 'WARNING')
        score -= warnings * 5
        
        # خصم للثغرات
        score -= len(analysis.get('secrets_detected', [])) * 25
        
        return max(0, score)
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """توليد توصيات"""
        recommendations = []
        
        if analysis['metrics']['complexity'] > 10:
            recommendations.append("قلل التعقيد باستخراج الدوال")
        
        if analysis['metrics']['average_function_length'] > 25:
            recommendations.append("قسّم الدوال الطويلة")
        
        if analysis['metrics']['comment_lines'] < 5:
            recommendations.append("أضف المزيد من التعليقات")
        
        if analysis.get('secrets_detected'):
            recommendations.append("أزل الأسرار من الكود - استخدم متغيرات بيئة")
        
        return recommendations


# اختبار سريع
if __name__ == "__main__":
    test_code = '''
def calculate_factorial(n):
    """حساب العاملية"""
    if n == 0:
        return 1
    else:
        return n * calculate_factorial(n-1)

class Calculator:
    def add(self, a, b):
        return a + b
    
    def complex_method(self, x, y, z):
        if x > 0:
            if y > 0:
                if z > 0:
                    return x + y + z
        return 0

API_KEY = "sk-1234567890abcdef"
'''
    
    analyzer = SmartCodeAnalyzer()
    result = analyzer.analyze(test_code, "test.py")
    
    print("=" * 60)
    print("📊 نتائج التحليل")
    print("=" * 60)
    print(f"\n✅ صلاحية الكود: {result['is_valid']}")
    print(f"📈 درجة الجودة: {result['quality_score']:.1f}/100")
    print(f"\n📏 المقاييس:")
    for key, value in result['metrics'].items():
        print(f"   {key}: {value}")
    
    print(f"\n⚠️  روائح الكود ({len(result['code_smells'])}):")
    for smell in result['code_smells']:
        print(f"   [{smell['severity']}] {smell['type']}: {smell['message']}")
    
    if result['secrets_detected']:
        print(f"\n🔐 أسرار مكتشفة:")
        for secret in result['secrets_detected']:
            print(f"   {secret['type']} في السطر {secret['line']}")
