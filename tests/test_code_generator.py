#!/usr/bin/env python3
"""
اختبارات مولد الكود
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import unittest
from code_generator import CodeGenEngine, CodeGenerationRequest


class TestCodeGenerationRequest(unittest.TestCase):
    """اختبارات طلب التوليد"""
    
    def test_request_creation(self):
        """اختبار إنشاء الطلب"""
        request = CodeGenerationRequest(
            description="Calculate factorial",
            language="python",
            code_type="function"
        )
        
        self.assertEqual(request.description, "Calculate factorial")
        self.assertEqual(request.language, "python")
        self.assertEqual(request.code_type, "function")


class TestCodeGenEngine(unittest.TestCase):
    """اختبارات محرك التوليد"""
    
    @classmethod
    def setUpClass(cls):
        cls.engine = CodeGenEngine(use_cache=False)
    
    def test_generate_code_without_model(self):
        """اختبار التوليد بدون نموذج (وضع القوالب)"""
        request = CodeGenerationRequest(
            description="Calculate factorial",
            language="python",
            code_type="function"
        )
        
        result = self.engine.generate_code(request)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('generated_code', result)
        self.assertIn('quality_score', result)
    
    def test_fix_code(self):
        """اختبار إصلاح الكود"""
        broken_code = "def hello(\n    pass"
        
        result = self.engine.fix_code(broken_code)
        
        self.assertIn('fixed_code', result)
        self.assertIn('is_valid', result)
    
    def test_optimize_code(self):
        """اختبار تحسين الكود"""
        code = """
result = []
for i in range(10):
    result.append(i * 2)
"""
        result = self.engine.optimize_code(code)
        
        self.assertIn('optimized_code', result)
        self.assertIn('performance_improvement', result)
    
    def test_translate_code(self):
        """اختبار ترجمة الكود"""
        code = "def hello(): return 'Hello'"
        
        result = self.engine.translate_code(code, "javascript")
        
        self.assertEqual(result['source_language'], 'python')
        self.assertEqual(result['target_language'], 'javascript')
        self.assertIn('translated_code', result)
    
    def test_validate_syntax(self):
        """اختبار التحقق من الصياغة"""
        valid_code = "def test(): pass"
        invalid_code = "def test(\n    pass"
        
        self.assertTrue(self.engine._validate_syntax(valid_code, "python"))
        self.assertFalse(self.engine._validate_syntax(invalid_code, "python"))
    
    def test_extract_code_block(self):
        """اختبار استخراج كتلة الكود"""
        text = """Some text
```python
def hello():
    return "Hello"
```
More text"""
        
        code = self.engine._extract_code_block(text)
        
        self.assertIn("def hello()", code)
    
    def test_generate_documentation(self):
        """اختبار توليد التوثيق"""
        code = """
def greet(name):
    \"\"\"Greet someone.\"\"\"
    return f"Hello, {name}!"

class Person:
    \"\"\"A person class.\"\"\"
    pass
"""
        docs = self.engine.generate_documentation(code)
        
        self.assertIn('functions', docs)
        self.assertIn('classes', docs)


if __name__ == '__main__':
    unittest.main()
