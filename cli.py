#!/usr/bin/env python3
"""
واجهة سطر الأوامر - CLI Interface
=================================
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# إضافة المسار
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from code_generator import CodeGenEngine, CodeGenerationRequest
    from code_analyzer import SmartCodeAnalyzer, CodeQualityAnalyzer
    from ml_engine import CodeMLEngine
    from database import LearningDatabase
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  بعض الوحدات غير متاحة: {e}")
    MODULES_AVAILABLE = False


def create_parser() -> argparse.ArgumentParser:
    """إنشاء محلل الوسائط"""
    parser = argparse.ArgumentParser(
        prog="saeed-ai",
        description="SAEED AI System - منظومة ذكاء اصطناعي متكاملة",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:
  # توليد كود
  saeed-ai generate "Calculate factorial" --language python
  
  # تحليل كود
  saeed-ai analyze myfile.py
  
  # إصلاح كود
  saeed-ai fix myfile.py --error "SyntaxError"
  
  # تحسين كود
  saeed-ai optimize myfile.py
  
  # تصنيف كود
  saeed-ai classify myfile.py
  
  # كشف ثغرات
  saeed-ai scan myfile.py
  
  # عرض الإحصائيات
  saeed-ai stats
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='الأوامر المتاحة')
    
    # generate
    gen_parser = subparsers.add_parser('generate', help='توليد كود')
    gen_parser.add_argument('description', help='وصف الكود المطلوب')
    gen_parser.add_argument('-l', '--language', default='python', 
                           choices=['python', 'javascript', 'java', 'cpp', 'go'],
                           help='لغة البرمجة')
    gen_parser.add_argument('-t', '--type', default='function',
                           choices=['function', 'class', 'script', 'module'],
                           help='نوع الكود')
    gen_parser.add_argument('-i', '--input', help='توقيع المدخلات')
    gen_parser.add_argument('-o', '--output', help='توقيع المخرجات')
    gen_parser.add_argument('--save', help='حفظ النتيجة في ملف')
    
    # analyze
    analyze_parser = subparsers.add_parser('analyze', help='تحليل كود')
    analyze_parser.add_argument('file', help='ملف الكود')
    analyze_parser.add_argument('--json', action='store_true', help='إخراج JSON')
    analyze_parser.add_argument('--save', help='حفظ التقرير في ملف')
    
    # fix
    fix_parser = subparsers.add_parser('fix', help='إصلاح كود')
    fix_parser.add_argument('file', help='ملف الكود')
    fix_parser.add_argument('-e', '--error', help='رسالة الخطأ')
    fix_parser.add_argument('--save', help='حفظ الكود المُصلح')
    
    # optimize
    opt_parser = subparsers.add_parser('optimize', help='تحسين كود')
    opt_parser.add_argument('file', help='ملف الكود')
    opt_parser.add_argument('--type', default='performance',
                           choices=['performance', 'readability', 'size'],
                           help='نوع التحسين')
    opt_parser.add_argument('--save', help='حفظ الكود المحسن')
    
    # classify
    classify_parser = subparsers.add_parser('classify', help='تصنيف كود')
    classify_parser.add_argument('file', help='ملف الكود')
    
    # scan
    scan_parser = subparsers.add_parser('scan', help='كشف ثغرات')
    scan_parser.add_argument('file', help='ملف الكود')
    scan_parser.add_argument('--json', action='store_true', help='إخراج JSON')
    
    # translate
    trans_parser = subparsers.add_parser('translate', help='ترجمة كود')
    trans_parser.add_argument('file', help='ملف الكود')
    trans_parser.add_argument('-t', '--target', required=True,
                             choices=['python', 'javascript', 'java', 'cpp', 'go'],
                             help='اللغة الهدف')
    trans_parser.add_argument('--save', help='حفظ الكود المترجم')
    
    # stats
    stats_parser = subparsers.add_parser('stats', help='عرض الإحصائيات')
    stats_parser.add_argument('--days', type=int, default=30, help='عدد الأيام')
    
    # train
    train_parser = subparsers.add_parser('train', help='تدريب نموذج')
    train_parser.add_argument('model', choices=['classifier', 'bug', 'vuln', 'performance'],
                             help='النموذج للتدريب')
    train_parser.add_argument('--data', required=True, help='ملف بيانات التدريب (JSON)')
    
    # server
    server_parser = subparsers.add_parser('server', help='تشغيل الخادم')
    server_parser.add_argument('--host', default='0.0.0.0', help='عنوان الاستضافة')
    server_parser.add_argument('-p', '--port', type=int, default=5000, help='المنفذ')
    server_parser.add_argument('--debug', action='store_true', help='وضع التصحيح')
    
    return parser


def cmd_generate(args):
    """أمر توليد الكود"""
    if not MODULES_AVAILABLE:
        print("❌ الوحدات غير متاحة")
        return 1
    
    print(f"🤖 توليد كود: {args.description}")
    print(f"   اللغة: {args.language}")
    print(f"   النوع: {args.type}")
    
    engine = CodeGenEngine(use_cache=True)
    
    request = CodeGenerationRequest(
        description=args.description,
        language=args.language,
        code_type=args.type,
        input_signature=args.input,
        output_signature=args.output
    )
    
    result = engine.generate_code(request)
    
    print(f"\n✅ تم التوليد!")
    print(f"📊 درجة الجودة: {result['quality_score']:.1f}/100")
    print(f"⏱️  وقت التنفيذ: {result['execution_time_ms']:.1f}ms")
    
    print("\n" + "=" * 60)
    print("📝 الكود المولد:")
    print("=" * 60)
    print(result['generated_code'])
    
    if args.save:
        with open(args.save, 'w') as f:
            f.write(result['generated_code'])
        print(f"\n💾 تم الحفظ في: {args.save}")
    
    return 0


def cmd_analyze(args):
    """أمر تحليل الكود"""
    if not MODULES_AVAILABLE:
        print("❌ الوحدات غير متاحة")
        return 1
    
    print(f"🔍 تحليل: {args.file}")
    
    code = Path(args.file).read_text()
    analyzer = SmartCodeAnalyzer()
    result = analyzer.analyze(code, args.file)
    
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n📊 النتائج:")
        print(f"   صلاحية الكود: {'✅' if result['is_valid'] else '❌'}")
        print(f"   درجة الجودة: {result['quality_score']:.1f}/100")
        print(f"   اللغة: {result['language']}")
        
        print(f"\n📏 المقاييس:")
        for key, value in result['metrics'].items():
            print(f"   {key}: {value}")
        
        if result['code_smells']:
            print(f"\n⚠️  روائح الكود ({len(result['code_smells'])}):")
            for smell in result['code_smells']:
                print(f"   [{smell['severity']}] {smell['type']}: {smell['message']}")
        
        if result['secrets_detected']:
            print(f"\n🔐 أسرار مكتشفة ({len(result['secrets_detected'])}):")
            for secret in result['secrets_detected']:
                print(f"   {secret['type']} في السطر {secret['line']}")
    
    if args.save:
        with open(args.save, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 تم حفظ التقرير في: {args.save}")
    
    return 0


def cmd_fix(args):
    """أمر إصلاح الكود"""
    if not MODULES_AVAILABLE:
        print("❌ الوحدات غير متاحة")
        return 1
    
    print(f"🔧 إصلاح: {args.file}")
    
    code = Path(args.file).read_text()
    engine = CodeGenEngine(use_cache=False)
    
    result = engine.fix_code(code, args.error or "")
    
    print(f"\n📊 النتائج:")
    print(f"   الحالة: {result['status']}")
    print(f"   صلاحية الإصلاح: {'✅' if result['is_valid'] else '❌'}")
    
    if result['improvements']:
        print(f"\n📈 التحسينات:")
        for imp in result['improvements']:
            print(f"   - {imp}")
    
    print("\n" + "=" * 60)
    print("📝 الكود المُصلح:")
    print("=" * 60)
    print(result['fixed_code'])
    
    if args.save:
        with open(args.save, 'w') as f:
            f.write(result['fixed_code'])
        print(f"\n💾 تم الحفظ في: {args.save}")
    
    return 0


def cmd_optimize(args):
    """أمر تحسين الكود"""
    if not MODULES_AVAILABLE:
        print("❌ الوحدات غير متاحة")
        return 1
    
    print(f"⚡ تحسين: {args.file}")
    print(f"   النوع: {args.type}")
    
    code = Path(args.file).read_text()
    engine = CodeGenEngine(use_cache=False)
    
    result = engine.optimize_code(code, args.type)
    
    print(f"\n📊 النتائج:")
    print(f"   تحسين الأداء: {result['performance_improvement']}")
    print(f"   درجة الجودة الأصلية: {result['original_score']:.1f}")
    print(f"   درجة الجودة المحسنة: {result['optimized_score']:.1f}")
    
    if result['changes']:
        print(f"\n📈 التغييرات:")
        for change in result['changes']:
            print(f"   - {change}")
    
    print("\n" + "=" * 60)
    print("📝 الكود المحسن:")
    print("=" * 60)
    print(result['optimized_code'])
    
    if args.save:
        with open(args.save, 'w') as f:
            f.write(result['optimized_code'])
        print(f"\n💾 تم الحفظ في: {args.save}")
    
    return 0


def cmd_classify(args):
    """أمر تصنيف الكود"""
    if not MODULES_AVAILABLE:
        print("❌ الوحدات غير متاحة")
        return 1
    
    print(f"📦 تصنيف: {args.file}")
    
    code = Path(args.file).read_text()
    engine = CodeMLEngine()
    
    # تدريب سريع إذا لم يكن هناك نموذج
    if engine.classifier is None:
        print("🔄 تدريب نموذج سريع...")
        training_data = [
            ("def web_view(request): return render(request, 'index.html')", "web"),
            ("import numpy as np\ndata = np.mean(dataset)", "data"),
            ("model = RandomForestClassifier()\nmodel.fit(X, y)", "ml"),
            ("hash = sha256(password).hexdigest()", "security"),
        ]
        engine.train_software_classifier(training_data)
    
    result = engine.predict_software_category(code)
    
    print(f"\n📊 النتائج:")
    print(f"   الفئة: {result['category']}")
    print(f"   الثقة: {result['confidence']:.1%}")
    
    if 'top_3' in result:
        print(f"\n🏆 أفضل 3 تخمينات:")
        for cat, conf in result['top_3']:
            print(f"   {cat}: {conf:.1%}")
    
    return 0


def cmd_scan(args):
    """أمر كشف الثغرات"""
    if not MODULES_AVAILABLE:
        print("❌ الوحدات غير متاحة")
        return 1
    
    print(f"🛡️  فحص ثغرات: {args.file}")
    
    code = Path(args.file).read_text()
    engine = CodeMLEngine()
    
    result = engine.detect_vulnerabilities(code)
    
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n📊 النتائج:")
        print(f"   هل يحتوي على ثغرات: {'⚠️  نعم' if result['is_vulnerable'] else '✅ لا'}")
        
        if result['vulnerabilities_found']:
            print(f"\n🐛 الثغرات المكتشفة ({len(result['vulnerabilities_found'])}):")
            for vuln in result['vulnerabilities_found']:
                print(f"\n   [{vuln['severity']}] {vuln['type']}")
                print(f"   الوصف: {vuln['description']}")
                print(f"   الحل: {vuln['fix']}")
        
        print(f"\n💡 التوصية: {result['recommendation']}")
    
    return 0


def cmd_translate(args):
    """أمر ترجمة الكود"""
    if not MODULES_AVAILABLE:
        print("❌ الوحدات غير متاحة")
        return 1
    
    print(f"🌐 ترجمة: {args.file}")
    print(f"   إلى: {args.target}")
    
    code = Path(args.file).read_text()
    engine = CodeGenEngine(use_cache=False)
    
    result = engine.translate_code(code, args.target)
    
    print(f"\n📊 النتائج:")
    print(f"   من: {result['source_language']}")
    print(f"   إلى: {result['target_language']}")
    
    print("\n" + "=" * 60)
    print("📝 الكود المترجم:")
    print("=" * 60)
    print(result['translated_code'])
    
    if args.save:
        with open(args.save, 'w') as f:
            f.write(result['translated_code'])
        print(f"\n💾 تم الحفظ في: {args.save}")
    
    return 0


def cmd_stats(args):
    """أمر الإحصائيات"""
    if not MODULES_AVAILABLE:
        print("❌ الوحدات غير متاحة")
        return 1
    
    print(f"📊 إحصائيات النظام (آخر {args.days} يوم)")
    
    db = LearningDatabase()
    stats = db.get_dashboard_stats()
    
    print(f"\n📈 الإحصائيات:")
    print(f"   إجمالي التوليدات: {stats['total_generations']}")
    print(f"   متوسط الجودة: {stats['average_quality']:.1f}")
    print(f"   التوليدات اليوم: {stats['today_generations']}")
    print(f"   إجمالي الثغرات: {stats['total_vulnerabilities']}")
    
    if stats['top_languages']:
        print(f"\n💻 اللغات الأكثر استخداماً:")
        for lang in stats['top_languages'][:5]:
            print(f"   {lang['language']}: {lang['count']}")
    
    return 0


def cmd_train(args):
    """أمر تدريب النموذج"""
    if not MODULES_AVAILABLE:
        print("❌ الوحدات غير متاحة")
        return 1
    
    print(f"🎓 تدريب نموذج: {args.model}")
    print(f"   البيانات: {args.data}")
    
    # قراءة بيانات التدريب
    with open(args.data, 'r') as f:
        training_data = json.load(f)
    
    engine = CodeMLEngine()
    
    if args.model == 'classifier':
        result = engine.train_software_classifier(training_data)
        print(f"\n✅ تم التدريب!")
        print(f"   الدقة: {result.get('accuracy', 'N/A'):.2%}" if 'accuracy' in result else "")
    else:
        print("❌ نوع النموذج غير مدعوم حالياً")
        return 1
    
    return 0


def cmd_server(args):
    """أمر تشغيل الخادم"""
    print(f"🚀 تشغيل الخادم...")
    print(f"   العنوان: {args.host}")
    print(f"   المنفذ: {args.port}")
    
    try:
        from web.app import app
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except ImportError:
        print("❌ Flask غير مثبت. استخدم: pip install flask")
        return 1
    
    return 0


def main():
    """الدالة الرئيسية"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        'generate': cmd_generate,
        'analyze': cmd_analyze,
        'fix': cmd_fix,
        'optimize': cmd_optimize,
        'classify': cmd_classify,
        'scan': cmd_scan,
        'translate': cmd_translate,
        'stats': cmd_stats,
        'train': cmd_train,
        'server': cmd_server,
    }
    
    if args.command in commands:
        return commands[args.command](args)
    else:
        print(f"❌ أمر غير معروف: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
