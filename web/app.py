#!/usr/bin/env python3
"""
واجهة ويب Flask - Flask Web Interface
=====================================
واجهة مستخدم ويب للنظام
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from functools import wraps

# إضافة المسار
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

# استيراد المحركات
try:
    from code_generator import CodeGenEngine, CodeGenerationRequest
    from ml_engine import CodeMLEngine
    from code_analyzer import SmartCodeAnalyzer
    from database import LearningDatabase
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  بعض الوحدات غير متاحة: {e}")
    MODULES_AVAILABLE = False

# إنشاء التطبيق
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# تهيئة المحركات
engines = {}
_is_initialized = False  # متغير للتحقق من التهيئة

@app.before_request
def init_engines():
    """تهيئة المحركات عند أول طلب (بديل لـ before_first_request)"""
    global _is_initialized
    if not _is_initialized:
        if MODULES_AVAILABLE:
            print("🚀 تهيئة المحركات...")
            try:
                engines['generator'] = CodeGenEngine(use_cache=True)
                engines['ml'] = CodeMLEngine()
                engines['analyzer'] = SmartCodeAnalyzer()
                engines['db'] = LearningDatabase()
                print("✅ المحركات جاهزة")
            except Exception as e:
                print(f"❌ خطأ أثناء تهيئة المحركات: {e}")
        _is_initialized = True

# ========== الصفحات الرئيسية ==========

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/generate')
def generate_page():
    """صفحة التوليد"""
    return render_template('generate.html')

@app.route('/analyze')
def analyze_page():
    """صفحة التحليل"""
    return render_template('analyze.html')

@app.route('/security')
def security_page():
    """صفحة الأمان"""
    return render_template('security.html')

@app.route('/dashboard')
def dashboard_page():
    """لوحة التحكم"""
    return render_template('dashboard.html')

# ========== API - توليد الكود ==========

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """API توليد كود"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    data = request.json
    
    try:
        request_obj = CodeGenerationRequest(
            description=data.get('description', ''),
            language=data.get('language', 'python'),
            code_type=data.get('type', 'function'),
            input_signature=data.get('input_signature'),
            output_signature=data.get('output_signature'),
            constraints=data.get('constraints', []),
            test_cases=data.get('test_cases', [])
        )
        
        result = engines['generator'].generate_code(request_obj)
        
        # حفظ في قاعدة البيانات
        engines['db'].save_generation(
            prompt=data.get('description', ''),
            generated_code=result['generated_code'],
            language=result['language'],
            quality_score=result['quality_score'],
            execution_time_ms=result['execution_time_ms']
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/fix', methods=['POST'])
def api_fix():
    """API إصلاح كود"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    data = request.json
    code = data.get('code', '')
    error = data.get('error', '')
    
    try:
        result = engines['generator'].fix_code(code, error)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/optimize', methods=['POST'])
def api_optimize():
    """API تحسين كود"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    data = request.json
    code = data.get('code', '')
    opt_type = data.get('type', 'performance')
    
    try:
        result = engines['generator'].optimize_code(code, opt_type)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/translate', methods=['POST'])
def api_translate():
    """API ترجمة كود"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    data = request.json
    code = data.get('code', '')
    target_lang = data.get('target_language', 'javascript')
    
    try:
        result = engines['generator'].translate_code(code, target_lang)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== API - تحليل الكود ==========

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API تحليل كود"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    data = request.json
    code = data.get('code', '')
    
    try:
        result = engines['analyzer'].analyze(code)
        
        # حفظ في قاعدة البيانات
        engines['db'].save_analysis(
            code=code,
            language=result['language'],
            metrics=result['metrics'],
            issues=result['issues'],
            code_smells=result['code_smells'],
            quality_score=result['quality_score']
        )
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """API تصنيف كود"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    data = request.json
    code = data.get('code', '')
    
    try:
        result = engines['ml'].predict_software_category(code)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict-bugs', methods=['POST'])
def api_predict_bugs():
    """API تنبؤ الأخطاء"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    data = request.json
    commit_stats = data.get('stats', {})
    
    try:
        result = engines['ml'].predict_bug_likelihood(commit_stats)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/detect-vulnerabilities', methods=['POST'])
def api_detect_vulns():
    """API كشف الثغرات"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    data = request.json
    code = data.get('code', '')
    
    try:
        result = engines['ml'].detect_vulnerabilities(code)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== API - لوحة التحكم ==========

@app.route('/api/dashboard/stats')
def api_dashboard_stats():
    """إحصائيات لوحة التحكم"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    try:
        stats = engines['db'].get_dashboard_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard/activity')
def api_recent_activity():
    """النشاط الأخير"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    try:
        activity = engines['db'].get_recent_activity(limit=10)
        return jsonify(activity)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard/issues')
def api_common_issues():
    """المشاكل الشائعة"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    try:
        issues = engines['db'].get_common_issues(limit=10)
        return jsonify(issues)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== API - الموديلات ==========

@app.route('/api/models/info')
def api_models_info():
    """معلومات النماذج"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    try:
        info = engines['ml'].get_model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/train', methods=['POST'])
def api_train_model():
    """تدريب نموذج"""
    if not MODULES_AVAILABLE:
        return jsonify({"error": "Modules not available"}), 503
    
    data = request.json
    model_type = data.get('type', 'classifier')
    
    try:
        if model_type == 'classifier':
            training_data = data.get('data', [])
            result = engines['ml'].train_software_classifier(training_data)
        else:
            return jsonify({"error": "Unknown model type"}), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ========== API - الصحة ==========

@app.route('/api/health')
def api_health():
    """فحص صحة النظام"""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modules_available": MODULES_AVAILABLE,
        "engines_initialized": len(engines) > 0 if MODULES_AVAILABLE else False
    }
    return jsonify(health)

# ========== Static Files ==========

@app.route('/static/<path:path>')
def send_static(path):
    """ملفات ثابتة"""
    return send_from_directory('static', path)

# ========== Error Handlers ==========

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ========== Main ==========

if __name__ == '__main__':
    print("🚀 تشغيل خادم Flask...")
    print("📍 http://localhost:5000")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )

