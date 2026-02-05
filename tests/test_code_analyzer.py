#!/usr/bin/env python3
"""
اختبارات محلل الكود
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
from code_analyzer import SmartCodeAnalyzer, CodeQualityAnalyzer


class TestSmartCodeAnalyzer(unittest.TestCase):
    """اختبارات المحلل الذكي"""
    
    def setUp(self):
        self.analyzer = SmartCodeAnalyzer()
    
    def test_analyze_valid_code(self):
        """اختبار تحليل كود صحيح"""
        code = """
def hello():
    return "Hello, World!"
"""
        result = self.analyzer.analyze(code)
        
        self.assertTrue(result['is_valid'])
        self.assertIn('metrics', result)
        self.assertIn('quality_score', result)
    
    def test_analyze_invalid_code(self):
        """اختبار تحليل كود غير صحيح"""
        code = "def hello(\n    pass"
        
        result = self.analyzer.analyze(code)
        
        self.assertFalse(result['is_valid'])
        self.assertTrue(len(result['issues']) > 0)
    
    def test_detect_secrets(self):
        """اختبار كشف الأسرار"""
        code = """
API_KEY = "sk-1234567890abcdef"
password = "secret123"
"""
        result = self.analyzer.analyze(code)
        
        self.assertTrue(len(result['secrets_detected']) > 0)
    
    def test_detect_code_smells(self):
        """اختبار كشف روائح الكود"""
        code = """
def very_long_function():
    x = 1
    if x > 0:
        if x > 1:
            if x > 2:
                if x > 3:
                    pass
"""
        result = self.analyzer.analyze(code)
        
        smells = result['code_smells']
        self.assertTrue(any(s['type'] == 'DEEP_NESTING' for s in smells))
    
    def test_calculate_complexity(self):
        """اختبار حساب التعقيد"""
        code = """
def complex_function(x):
    if x > 0:
        for i in range(x):
            if i % 2 == 0:
                while i > 0:
                    i -= 1
"""
        result = self.analyzer.analyze(code)
        
        self.assertGreater(result['metrics']['complexity'], 1)
    
    def test_extract_features(self):
        """اختبار استخراج الميزات"""
        code = "def test(): pass"
        
        features = self.analyzer.extract_features(code)
        
        self.assertIn('lines_of_code', features)
        self.assertIn('complexity', features)
        self.assertIn('quality_score', features)


class TestCodeQualityAnalyzer(unittest.TestCase):
    """اختبارات محلل الجودة"""
    
    def setUp(self):
        self.analyzer = CodeQualityAnalyzer()
    
    def test_analyze_quality(self):
        """اختبار تحليل الجودة"""
        code = """
def well_documented_function(n):
    \"\"\"Calculate factorial.\"\"\"
    if n <= 1:
        return 1
    return n * well_documented_function(n - 1)
"""
        result = self.analyzer.analyze(code)
        
        self.assertIn('overall_score', result)
        self.assertIn('maintainability_index', result)
        self.assertIn('reliability_score', result)
    
    def test_generate_recommendations(self):
        """اختبار توليد التوصيات"""
        code = "def test(): pass"  # كود بسيط جداً
        
        result = self.analyzer.analyze(code)
        
        self.assertIn('recommendations', result)


if __name__ == '__main__':
    unittest.main()
