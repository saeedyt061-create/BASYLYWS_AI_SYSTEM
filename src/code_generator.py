#!/usr/bin/env python3
"""
محرك توليد الكود بالذكاء الاصطناعي - Code Generation Engine
==========================================================
يستطيع: توليد كود، إصلاح أخطاء، تحسين، شرح، تحويل بين اللغات
"""

import sys
import json
import ast
import re
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# استيراد المحلل
from .code_analyzer import SmartCodeAnalyzer, CodeQualityAnalyzer

# محاولة استيراد Transformers
try:
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, 
        AutoModelForSeq2SeqLM, pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Transformers غير مثبت. استخدم: pip install transformers torch")
    TRANSFORMERS_AVAILABLE = False


@dataclass
class CodeGenerationRequest:
    """طلب توليد كود منظم"""
    description: str
    language: str = "python"
    code_type: str = "function"  # function, class, script, module
    input_signature: Optional[str] = None
    output_signature: Optional[str] = None
    constraints: List[str] = field(default_factory=list)
    test_cases: List[Dict] = field(default_factory=list)
    max_length: int = 512
    temperature: float = 0.7


@dataclass
class GenerationResult:
    """نتيجة توليد الكود"""
    status: str
    generated_code: str
    language: str
    quality_score: float
    issues: List[Dict]
    test_cases: List[Dict]
    explanation: str
    execution_time_ms: float


class CodeGenEngine:
    """
    محرك توليد كود متطور باستخدام Transformers
    يدعم: CodeGen, CodeT5, GPT-4ALL, StarCoder
    """
    
    # نماذج مدعومة
    SUPPORTED_MODELS = {
        "codegen-small": "Salesforce/codegen-350M-mono",
        "codegen-medium": "Salesforce/codegen-2B-mono",
        "codet5": "Salesforce/codet5-base",
        "starcoder": "bigcode/starcoder",
        "incoder": "facebook/incoder-1B"
    }
    
    def __init__(self, model_name: str = "Salesforce/codegen-350M-mono", 
                 use_cache: bool = True):
        """
        تهيئة المحرك
        
        Args:
            model_name: اسم النموذج أو المفتاح من SUPPORTED_MODELS
            use_cache: استخدام التخزين المؤقت
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️  الجهاز: {self.device}")
        
        # حل اسم النموذج
        if model_name in self.SUPPORTED_MODELS:
            model_name = self.SUPPORTED_MODELS[model_name]
        
        self.model_name = model_name
        self.use_cache = use_cache
        
        # تحميل النموذج إذا كان متاحاً
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
        
        # المحللون
        self.code_analyzer = CodeQualityAnalyzer()
        self.smart_analyzer = SmartCodeAnalyzer()
        
        # ذاكرة التعلم
        self.generation_history: List[Dict] = []
        self.cache: Dict[str, Any] = {}
        
        print("✅ محرك التوليد جاهز")
    
    def _load_model(self):
        """تحميل النموذج والمحلل"""
        try:
            print(f"🤖 تحميل النموذج: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # إعدادات التحميل حسب الجهاز
            load_kwargs = {}
            if self.device.type == "cuda":
                load_kwargs = {
                    "torch_dtype": torch.float16,
                    "device_map": "auto"
                }
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs
            )
            
            # خط أنابيب التوليد
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1
            )
            
            print("✅ النموذج محمل بنجاح")
            
        except Exception as e:
            print(f"⚠️  خطأ في تحميل النموذج: {e}")
            print("📝 سيتم استخدام وضع القوالب")
    
    # ========== التوليد الرئيسي ==========
    
    def generate_code(self, request: CodeGenerationRequest) -> Dict[str, Any]:
        """
        توليد كود ذكي من الوصف
        
        Args:
            request: طلب التوليد
            
        Returns:
            نتيجة التوليد مع الكود والتحليل
        """
        start_time = datetime.now()
        
        # التحقق من الكاش
        cache_key = self._generate_cache_key(request)
        if self.use_cache and cache_key in self.cache:
            print("📦 استخدام نتيجة مخزنة")
            return self.cache[cache_key]
        
        # بناء prompt محسّن
        prompt = self._build_enhanced_prompt(request)
        
        # التوليد
        if self.pipeline:
            generated_codes = self._generate_with_model(prompt, request)
        else:
            generated_codes = self._generate_with_templates(request)
        
        # اختيار أفضل نسخة
        best_code = self._select_best_candidate(generated_codes, request)
        
        # تحليل الجودة
        quality_report = self.code_analyzer.analyze(best_code)
        
        # إنشاء الاختبارات
        tests = self._auto_generate_tests(best_code, request)
        
        # شرح الكود
        explanation = self._explain_code(best_code)
        
        # حساب الوقت
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # بناء النتيجة
        result = {
            "status": "success",
            "language": request.language,
            "generated_code": best_code,
            "quality_score": quality_report["overall_score"],
            "quality_details": quality_report,
            "issues": quality_report["issues"],
            "test_cases": tests,
            "explanation": explanation,
            "execution_time_ms": execution_time,
            "timestamp": datetime.now().isoformat()
        }
        
        # حفظ في الكاش والتاريخ
        if self.use_cache:
            self.cache[cache_key] = result
        
        self.generation_history.append({
            "timestamp": datetime.now().isoformat(),
            "request": request.__dict__,
            "result": result
        })
        
        return result
    
    def _generate_with_model(self, prompt: str, 
                             request: CodeGenerationRequest) -> List[str]:
        """التوليد باستخدام النموذج"""
        print(f"📝 توليد الكود...")
        
        generation_config = {
            "max_length": request.max_length,
            "temperature": request.temperature,
            "top_p": 0.95,
            "top_k": 50,
            "do_sample": True,
            "num_return_sequences": 3,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        
        try:
            sequences = self.pipeline(prompt, **generation_config)
            
            candidates = []
            for seq in sequences:
                generated_text = seq['generated_text']
                code = self._extract_code_block(generated_text)
                if code:
                    candidates.append(code)
            
            return candidates if candidates else [self._fallback_generation(request)]
            
        except Exception as e:
            print(f"⚠️  خطأ في التوليد: {e}")
            return [self._fallback_generation(request)]
    
    def _generate_with_templates(self, request: CodeGenerationRequest) -> List[str]:
        """التوليد باستخدام القوالب (وضع بدون نموذج)"""
        templates = self._get_templates_for_type(request.code_type, request.language)
        
        candidates = []
        for template in templates:
            filled = self._fill_template(template, request)
            candidates.append(filled)
        
        return candidates
    
    def _get_templates_for_type(self, code_type: str, language: str) -> List[str]:
        """الحصول على قوالب للنوع المحدد"""
        templates = {
            "python": {
                "function": [
                    "def {name}({params}):\n    \"\"\"{description}\"\"\"\n    {body}\n    return result",
                    "def {name}({params}):\n    # {description}\n    {body}\n    return None"
                ],
                "class": [
                    "class {name}:\n    \"\"\"{description}\"\"\"\n    \n    def __init__(self):\n        pass"
                ]
            }
        }
        
        lang_templates = templates.get(language, templates["python"])
        return lang_templates.get(code_type, lang_templates["function"])
    
    def _fill_template(self, template: str, request: CodeGenerationRequest) -> str:
        """ملء القالب"""
        # استخراج اسم من الوصف
        name_match = re.search(r'(?:function|class|def)\s+(\w+)', request.description, re.I)
        name = name_match.group(1) if name_match else "generated_function"
        
        # استخراج معاملات
        params = request.input_signature if request.input_signature else "*args, **kwargs"
        
        return template.format(
            name=name,
            params=params,
            description=request.description,
            body="# TODO: Implement"
        )
    
    def _fallback_generation(self, request: CodeGenerationRequest) -> str:
        """توليد احتياطي"""
        return f"""# {request.description}
def generated_function({request.input_signature or 'x'}):
    \"\"\"TODO: Implement this function\"\"\"
    # Auto-generated code
    pass
"""
    
    def _build_enhanced_prompt(self, request: CodeGenerationRequest) -> str:
        """بناء prompt متطور ومنظم"""
        prompt_parts = [
            f"# Generate {request.code_type} in {request.language}",
            f"# Description: {request.description}",
        ]
        
        if request.input_signature:
            prompt_parts.append(f"# Input: {request.input_signature}")
        
        if request.output_signature:
            prompt_parts.append(f"# Output: {request.output_signature}")
        
        if request.constraints:
            prompt_parts.append(f"# Constraints: {', '.join(request.constraints)}")
        
        if request.test_cases:
            prompt_parts.append("# Test Cases:")
            for test in request.test_cases[:2]:
                prompt_parts.append(f"# - Input: {test.get('input', 'N/A')} -> Expected: {test.get('expected', 'N/A')}")
        
        prompt_parts.extend([
            "",
            f"Here is the {request.language} {request.code_type}:",
            f"```{request.language}",
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_code_block(self, generated_text: str) -> str:
        """استخراج كتلة الكود من النص المولد"""
        # البحث بين علامات ```
        patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'<code>(.*?)</code>'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, generated_text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # إذا لم تجد، أعد النص كاملاً
        lines = generated_text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if '```' in line:
                in_code = not in_code
                continue
            if in_code or line.strip().startswith(('def ', 'class ', 'import ', 'from ')):
                code_lines.append(line)
        
        return '\n'.join(code_lines) if code_lines else generated_text.strip()
    
    def _select_best_candidate(self, candidates: List[str], 
                               request: CodeGenerationRequest) -> str:
        """اختيار أفضل نسخة بناءً على جودة المتطلبات"""
        if len(candidates) == 1:
            return candidates[0]
        
        scores = []
        
        for code in candidates:
            score = 0
            
            # يتبع التوقيع المطلوب؟
            if request.input_signature and request.input_signature in code:
                score += 10
            
            # يتجاوز الحجم الأمثل؟
            line_count = len(code.split('\n'))
            if 5 < line_count < 50:
                score += 5
            
            # يحتوي على كلمات مفتاحية من الوصف؟
            keywords = request.description.lower().split()
            code_lower = code.lower()
            score += sum(2 for kw in keywords if len(kw) > 3 and kw in code_lower)
            
            # صياغة صحيحة؟
            if self._validate_syntax(code, request.language):
                score += 15
            
            # اختبارات ناجحة؟
            if request.test_cases:
                if self._code_passes_tests(code, request.test_cases, request.language):
                    score += 20
            
            scores.append(score)
        
        best_idx = scores.index(max(scores))
        return candidates[best_idx]
    
    def _validate_syntax(self, code: str, language: str) -> bool:
        """التحقق من صحة الصياغة"""
        if language == "python":
            try:
                ast.parse(code)
                return True
            except SyntaxError:
                return False
        return True  # للغات الأخرى
    
    def _code_passes_tests(self, code: str, test_cases: List[Dict], 
                           language: str) -> bool:
        """اختبار سريع للكود المولد"""
        if language != "python":
            return True
        
        try:
            namespace = {}
            exec(code, namespace)
            
            for test in test_cases[:1]:  # أول اختبار فقط
                func_name = test.get('function_name', 'generated_function')
                func = namespace.get(func_name)
                
                if func:
                    test_input = test.get('input', [])
                    expected = test.get('expected')
                    
                    if isinstance(test_input, list):
                        result = func(*test_input)
                    else:
                        result = func(test_input)
                    
                    return result == expected
            
            return True
        except Exception as e:
            return False
    
    def _auto_generate_tests(self, code: str, 
                             request: CodeGenerationRequest) -> List[Dict]:
        """إنشاء اختبارات تلقائية"""
        tests = []
        
        # استخراج اسم الدالة
        func_match = re.search(r'def\s+(\w+)\s*\(', code)
        if func_match:
            func_name = func_match.group(1)
            
            # اختبار بسيط
            tests.append({
                "function_name": func_name,
                "input": [],
                "expected": None,
                "description": "Test basic execution"
            })
            
            # اختبار مع معاملات
            if request.input_signature:
                tests.append({
                    "function_name": func_name,
                    "input": [1, 2, 3],
                    "expected": None,
                    "description": "Test with parameters"
                })
        
        return tests
    
    def _explain_code(self, code: str) -> str:
        """شرح الكود المولد"""
        lines = code.split('\n')
        explanation = []
        
        # شرح الدوال
        for i, line in enumerate(lines):
            if line.strip().startswith('def '):
                func_name = line.split('def ')[1].split('(')[0]
                explanation.append(f"- الدالة '{func_name}' تقوم بـ...")
            elif line.strip().startswith('class '):
                class_name = line.split('class ')[1].split(':')[0].split('(')[0]
                explanation.append(f"- الفئة '{class_name}' تعرف...")
        
        return "\n".join(explanation) if explanation else "كود بسيط"
    
    def _generate_cache_key(self, request: CodeGenerationRequest) -> str:
        """توليد مفتاح للكاش"""
        key_data = f"{request.description}:{request.language}:{request.code_type}"
        import hashlib
        return hashlib.md5(key_data.encode()).hexdigest()
    
    # ========== الإصلاح والتحسين ==========
    
    def fix_code(self, broken_code: str, error_message: str = "") -> Dict[str, Any]:
        """إصلاح أخطاء الكود تلقائياً"""
        language = self.smart_analyzer._detect_language(broken_code)
        
        prompt = f"""# Fix this {language} code
# Error: {error_message}
# Original Code:
```{language}
{broken_code}
```
# Fixed Code:
```{language}
"""
        
        if self.pipeline:
            try:
                result = self.pipeline(
                    prompt,
                    max_length=512,
                    temperature=0.3,
                    do_sample=False
                )[0]['generated_text']
                
                fixed_code = self._extract_code_block(result)
            except:
                fixed_code = self._manual_fix(broken_code, error_message)
        else:
            fixed_code = self._manual_fix(broken_code, error_message)
        
        # التحقق من الإصلاح
        is_valid = self._validate_syntax(fixed_code, language)
        
        return {
            "status": "success" if is_valid else "partial",
            "original_code": broken_code,
            "fixed_code": fixed_code,
            "is_valid": is_valid,
            "language": language,
            "improvements": self._compare_codes(broken_code, fixed_code),
            "timestamp": datetime.now().isoformat()
        }
    
    def _manual_fix(self, code: str, error_message: str) -> str:
        """إصلاح يدوي للأخطاء الشائعة"""
        fixed = code
        
        # إصلاح الأقواس غير المغلقة
        open_parens = fixed.count('(') - fixed.count(')')
        if open_parens > 0:
            fixed += ')' * open_parens
        
        # إصلاح النقاط المفقودة
        if ':' not in fixed and 'def ' in fixed:
            lines = fixed.split('\n')
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') or line.strip().startswith('class '):
                    if not line.rstrip().endswith(':'):
                        lines[i] = line.rstrip() + ':'
            fixed = '\n'.join(lines)
        
        return fixed
    
    def optimize_code(self, code: str, 
                      optimization_type: str = "performance") -> Dict[str, Any]:
        """تحسين الكود للأداء"""
        language = self.smart_analyzer._detect_language(code)
        
        prompt = f"""# Optimize this {language} code for {optimization_type}
# Original Code:
```{language}
{code}
```
# Optimized Code:
```{language}
"""
        
        if self.pipeline:
            try:
                result = self.pipeline(
                    prompt,
                    max_length=512,
                    temperature=0.2,
                    do_sample=True
                )[0]['generated_text']
                
                optimized_code = self._extract_code_block(result)
            except:
                optimized_code = self._manual_optimize(code, optimization_type)
        else:
            optimized_code = self._manual_optimize(code, optimization_type)
        
        # تحليل الأداء
        original_analysis = self.code_analyzer.analyze(code)
        optimized_analysis = self.code_analyzer.analyze(optimized_code)
        
        performance_gain = self._estimate_performance_gain(
            original_analysis, optimized_analysis
        )
        
        return {
            "status": "success",
            "original_code": code,
            "optimized_code": optimized_code,
            "performance_improvement": performance_gain,
            "changes": self._extract_changes(code, optimized_code),
            "original_score": original_analysis["overall_score"],
            "optimized_score": optimized_analysis["overall_score"],
            "timestamp": datetime.now().isoformat()
        }
    
    def _manual_optimize(self, code: str, optimization_type: str) -> str:
        """تحسين يدوي"""
        optimized = code
        
        if optimization_type == "performance":
            # تحويل الحلقات إلى list comprehension
            optimized = re.sub(
                r'result\s*=\s*\[\]\s*\nfor\s+(\w+)\s+in\s+(\w+):\s*\n\s*result\.append\(([^)]+)\)',
                r'result = [\3 for \1 in \2]',
                optimized
            )
        
        return optimized
    
    def _estimate_performance_gain(self, original: Dict, optimized: Dict) -> str:
        """تقدير تحسين الأداء"""
        orig_complexity = original["metrics"]["complexity"]
        opt_complexity = optimized["metrics"]["complexity"]
        
        if orig_complexity > 0:
            improvement = (orig_complexity - opt_complexity) / orig_complexity * 100
            return f"{improvement:.1f}% faster"
        
        return "Unknown improvement"
    
    def _compare_codes(self, original: str, modified: str) -> List[str]:
        """مقارنة نسختين من الكود"""
        improvements = []
        
        orig_lines = len(original.split('\n'))
        mod_lines = len(modified.split('\n'))
        
        if mod_lines < orig_lines:
            improvements.append(f"Reduced from {orig_lines} to {mod_lines} lines")
        
        return improvements
    
    def _extract_changes(self, original: str, modified: str) -> List[str]:
        """استخراج التغييرات"""
        changes = []
        
        orig_set = set(original.split('\n'))
        mod_set = set(modified.split('\n'))
        
        added = mod_set - orig_set
        removed = orig_set - mod_set
        
        if added:
            changes.append(f"Added {len(added)} lines")
        if removed:
            changes.append(f"Removed {len(removed)} lines")
        
        return changes
    
    # ========== الترجمة والتوثيق ==========
    
    def translate_code(self, code: str, target_language: str) -> Dict[str, Any]:
        """ترجمة الكود بين اللغات"""
        source_language = self.smart_analyzer._detect_language(code)
        
        prompt = f"""# Translate from {source_language} to {target_language}
# Original Code:
```{source_language}
{code}
```
# Translated Code:
```{target_language}
"""
        
        if self.pipeline:
            try:
                result = self.pipeline(
                    prompt,
                    max_length=512,
                    temperature=0.3
                )[0]['generated_text']
                
                translated_code = self._extract_code_block(result)
            except:
                translated_code = f"# Translation to {target_language}\n# TODO: Implement"
        else:
            translated_code = self._manual_translate(code, source_language, target_language)
        
        return {
            "status": "success",
            "source_language": source_language,
            "target_language": target_language,
            "original_code": code,
            "translated_code": translated_code,
            "timestamp": datetime.now().isoformat()
        }
    
    def _manual_translate(self, code: str, source: str, target: str) -> str:
        """ترجمة يدوية بسيطة"""
        translations = {
            ("python", "javascript"): {
                "def ": "function ",
                "None": "null",
                "True": "true",
                "False": "false",
                "# ": "// "
            }
        }
        
        translated = code
        trans_map = translations.get((source, target), {})
        
        for old, new in trans_map.items():
            translated = translated.replace(old, new)
        
        return translated
    
    def generate_documentation(self, code: str) -> Dict[str, Any]:
        """توليد توثيق للكود"""
        analysis = self.smart_analyzer.analyze(code)
        
        # استخراج الدوال والفئات
        tree = ast.parse(code)
        
        docs = {
            "summary": f"Code with {analysis['metrics']['function_count']} functions and {analysis['metrics']['class_count']} classes",
            "functions": [],
            "classes": [],
            "quality": analysis['quality_score']
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docs["functions"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "docstring": ast.get_docstring(node) or "No documentation"
                })
            elif isinstance(node, ast.ClassDef):
                docs["classes"].append({
                    "name": node.name,
                    "line": node.lineno,
                    "docstring": ast.get_docstring(node) or "No documentation"
                })
        
        return docs
    
    # ========== الأدوات المساعدة ==========
    
    def get_history(self) -> List[Dict]:
        """الحصول على تاريخ التوليد"""
        return self.generation_history
    
    def clear_cache(self):
        """مسح الكاش"""
        self.cache.clear()
        print("🗑️  Cache cleared")
    
    def save_history(self, filepath: str):
        """حفظ التاريخ"""
        with open(filepath, 'w') as f:
            json.dump(self.generation_history, f, indent=2, ensure_ascii=False)
        print(f"💾 History saved to {filepath}")


# اختبار سريع
if __name__ == "__main__":
    engine = CodeGenEngine()
    
    # اختبار التوليد
    request = CodeGenerationRequest(
        description="Calculate factorial of a number",
        language="python",
        code_type="function",
        input_signature="n: int",
        output_signature="int",
        test_cases=[{"input": [5], "expected": 120}]
    )
    
    result = engine.generate_code(request)
    
    print("\n" + "=" * 60)
    print("🎯 نتيجة التوليد")
    print("=" * 60)
    print(f"\n📊 درجة الجودة: {result['quality_score']:.1f}/100")
    print(f"⏱️  وقت التنفيذ: {result['execution_time_ms']:.1f}ms")
    print(f"\n📝 الكود المولد:")
    print("-" * 40)
    print(result['generated_code'])
    print("-" * 40)
    print(f"\n📖 الشرح:\n{result['explanation']}")
