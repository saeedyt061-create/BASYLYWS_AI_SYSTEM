#!/usr/bin/env python3
"""
واجهة سطر الأوامر - CLI Interface (نسخة معدلة لـ Termux)
=====================================================
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional

# --- إصلاح المسارات لبيئة Termux والمشاريع المحلية ---
# الحصول على المسار المطلق للمجلد الذي يحتوي على cli.py
current_dir = Path(__file__).parent.absolute()

# إضافة المجلد الرئيسي ومجلد src إلى مسار البحث الخاص ببايثون
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "src"))

# محاولة استيراد الوحدات الأساسية
try:
    from code_generator import CodeGenEngine, CodeGenerationRequest
    from code_analyzer import SmartCodeAnalyzer, CodeQualityAnalyzer
    from ml_engine import CodeMLEngine
    from database import LearningDatabase
    MODULES_AVAILABLE = True
except ImportError as e:
    # طباعة الخطأ الحقيقي للمساعدة في التصحيح
    print(f"⚠️  تنبيه: بعض الوحدات المحلية لم تكتمل بعد أو هناك خطأ في: {e}")
    MODULES_AVAILABLE = False


def create_parser() -> argparse.ArgumentParser:
    """إنشاء محلل الوسائط"""
    parser = argparse.ArgumentParser(
        prog="saeed-ai",
        description="SAEED AI System - منظومة ذكاء اصطناعي متكاملة",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
أمثلة الاستخدام:
  saeed-ai server --port 5000
  saeed-ai generate "Calculate factorial" --language python
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='الأوامر المتاحة')
    
    # generate
    gen_parser = subparsers.add_parser('generate', help='توليد كود')
    gen_parser.add_argument('description', help='وصف الكود المطلوب')
    gen_parser.add_argument('-l', '--language', default='python', help='لغة البرمجة')
    gen_parser.add_argument('-t', '--type', default='function', help='نوع الكود')
    gen_parser.add_argument('--save', help='حفظ النتيجة في ملف')
    
    # analyze
    analyze_parser = subparsers.add_parser('analyze', help='تحليل كود')
    analyze_parser.add_argument('file', help='ملف الكود')
    analyze_parser.add_argument('--json', action='store_true', help='إخراج JSON')
    
    # server
    server_parser = subparsers.add_parser('server', help='تشغيل الخادم')
    server_parser.add_argument('--host', default='0.0.0.0', help='عنوان الاستضافة')
    server_parser.add_argument('-p', '--port', type=int, default=5000, help='المنفذ')
    server_parser.add_argument('--debug', action='store_true', help='وضع التصحيح')

    # إضافة بقية الأوامر (fix, optimize, etc.) هنا للاختصار...
    # (تم الإبقاء على الهيكل العام للمشروع)
    return parser

# --- الدوال التنفيذية للأوامر ---

def cmd_server(args):
    """أمر تشغيل الخادم المعدل"""
    print(f"🚀 جاري محاولة تشغيل الخادم...")
    print(f"   المسار الحالي: {os.getcwd()}")
    
    try:
        # محاولة استيراد تطبيق الويب
        # تأكد من وجود ملف __init__.py داخل مجلد web
        import web.app as web_module
        app = web_module.app
        
        print(f"✅ تم تحميل تطبيق الويب بنجاح.")
        print(f"🔗 الخادم متاح على: http://{args.host}:{args.port}")
        
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except ImportError as e:
        print(f"❌ خطأ: لا يمكن العثور على تطبيق الويب (Flask).")
        print(f"   التفاصيل: {e}")
        print(f"   نصيحة: تأكد من تشغيل: pip install flask")
        print(f"   تأكد أيضاً من وجود مجلد 'web' وبداخله 'app.py' و ' __init__.py '")
        return 1
    except Exception as e:
        print(f"❌ حدث خطأ غير متوقع أثناء تشغيل الخادم: {e}")
        return 1
    return 0

def cmd_generate(args):
    if not MODULES_AVAILABLE:
        print("❌ الوحدات البرمجية المطلوبة غير متوفرة.")
        return 1
    # ... كود التوليد الخاص بك ...
    print(f"🤖 جاري العمل على: {args.description}")
    return 0

# (بقية دوال الأوامر تبقى كما هي مع التأكد من MODULES_AVAILABLE)

def main():
    """الدالة الرئيسية"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        'server': cmd_server,
        'generate': cmd_generate,
        # أضف بقية الأوامر هنا...
    }
    
    if args.command in commands:
        try:
            return commands[args.command](args)
        except KeyboardInterrupt:
            print("\n👋 تم إغلاق النظام.")
            return 0
    else:
        print(f"❌ أمر غير معروف: {args.command}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

