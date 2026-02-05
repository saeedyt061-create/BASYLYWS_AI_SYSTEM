"""
SAEED AI SYSTEM - منظومة ذكاء اصطناعي متكاملة
================================================

وحدات النظام:
- code_generator: توليد وإصلاح وتحسين الكود
- ml_engine: تحليل البرمجيات بالتعلم الآلي
- security_scanner: اختبار الاختراق الأخلاقي
- code_analyzer: تحليل جودة الكود
"""

__version__ = "2.0.0"
__author__ = "Saeed AI Team"

from .code_generator import CodeGenEngine, CodeGenerationRequest
from .ml_engine import CodeMLEngine
from .code_analyzer import SmartCodeAnalyzer, CodeQualityAnalyzer
from .security_scanner import EthicalScanner, LegalScopeGenerator

__all__ = [
    'CodeGenEngine',
    'CodeGenerationRequest', 
    'CodeMLEngine',
    'SmartCodeAnalyzer',
    'CodeQualityAnalyzer',
    'EthicalScanner',
    'LegalScopeGenerator'
]
