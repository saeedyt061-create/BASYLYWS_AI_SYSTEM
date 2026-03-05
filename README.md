# SAEED AI System 🚀

منظومة ذكاء اصطناعي متكاملة لتوليد وتحليل وتحسين الكود البرمجي مع قدرات أمنية متقدمة.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)

## 📋 المميزات

### 🤖 توليد الكود
- توليد كود تلقائي من الوصف الطبيعي
- دعم Python, JavaScript, Java, C++, Go
- إصلاح الأخطاء تلقائياً
- تحسين الأداء
- ترجمة بين اللغات

### 🧠 تحليل ML
- تصنيف البرمجيات (web, ml, data, security, ...)
- تنبؤ الأخطاء قبل حدوثها
- اكتشاف الثغرات الأمنية
- تنبؤ الأداء

### 🔒 الأمان
- كشف CVEs المعروفة
- فحص SSL/TLS
- اكتشاف الأسرار في الكود
- توليد تقارير أمنية

### 🌐 واجهة ويب
- واجهة مستخدم حديثة
- لوحة تحكم تفاعلية
- رسوم بيانية
- API RESTful

## 🚀 البدء السريع

### التثبيت

```bash
# استنساخ المستودع
git clone https://github.com/BASYLYWS/BASYLYWS_AI_SYSTEM.git

cd BASYLYWS_AI_SYSTEM

# تثبيت المتطلبات
pip install -r requirements.txt
```

### الاستخدام

#### واجهة سطر الأوامر (CLI)

```bash
# توليد كود
python cli.py generate "Calculate factorial" --language python

# تحليل كود
python cli.py analyze myfile.py

# إصلاح كود
python cli.py fix myfile.py --error "SyntaxError"

# تحسين كود
python cli.py optimize myfile.py

# تصنيف كود
python cli.py classify myfile.py

# كشف ثغرات
python cli.py scan myfile.py

# عرض الإحصائيات
python cli.py stats

# تشغيل الخادم
python cli.py server
```

#### واجهة الويب

```bash
# تشغيل الخادم
python web/app.py

# أو
python cli.py server

# افتح المتصفح
open http://localhost:5000
```

### Docker

```bash
# بناء الصورة
docker build -t  BASYLYWS_AI_SYSTEM

# تشغيل
docker run -p 5000:5000 BBASYLYWS_AI_SYSTEM

# أو استخدام Docker Compose
docker-compose up -d
```

## 📁 هيكل المشروع

```
BASYLYWS_AI_SYSTEM/
├── src/                      # الكود المصدري
│   ├── __init__.py
│   ├── code_generator.py     # مولد الكود
│   ├── code_analyzer.py      # محلل الكود
│   ├── ml_engine.py          # محرك ML
│   ├── security_scanner.py   # الماسح الأمني
│   └── database.py           # قاعدة البيانات
├── web/                      # واجهة الويب
│   ├── app.py               # تطبيق Flask
│   ├── templates/           # قوالب HTML
│   └── static/              # ملفات ثابتة
├── tests/                    # الاختبارات
├── data/                     # البيانات
├── models/                   # النماذج المحفوظة
├── cli.py                    # واجهة سطر الأوامر
├── requirements.txt          # المتطلبات
├── Dockerfile               # Docker
├── docker-compose.yml       # Docker Compose
└── README.md                # هذا الملف
```

## 🔧 المتطلبات

### الأساسية
- Python 3.10+
- Flask
- TensorFlow / PyTorch
- Scikit-learn
- SQLite

### الاختيارية
- CUDA (للتسريع بالGPU)
- Redis (للتخزين المؤقت)
- Nginx (للإنتاج)

## 📊 API Endpoints

### توليد الكود
```http
POST /api/generate
Content-Type: application/json

{
  "description": "Calculate factorial",
  "language": "python",
  "type": "function"
}
```

### تحليل الكود
```http
POST /api/analyze
Content-Type: application/json

{
  "code": "def test(): pass"
}
```

### كشف الثغرات
```http
POST /api/detect-vulnerabilities
Content-Type: application/json

{
  "code": "query = f'SELECT * FROM users'"
}
```

### الإحصائيات
```http
GET /api/dashboard/stats
```

## 🧪 الاختبارات

```bash
# تشغيل جميع الاختبارات
python -m pytest tests/

# اختبار وحدة محددة
python -m pytest tests/test_code_analyzer.py

# مع تغطية
python -m pytest --cov=src tests/
```

## 📈 لوحة التحكم

الوصول إلى لوحة التحكم عبر:
```
http://localhost:5000/dashboard
```

تعرض:
- إحصائيات الاستخدام
- اللغات الأكثر استخداماً
- الثغرات المكتشفة
- حالة النماذج

## 🔒 الأمان

> ⚠️ **تحذير**: نظام الفحص الأمني للاستخدام القانوني فقط!

- يجب الحصول على إذن كتابي قبل الفحص
- استخدام ملفات النطاق (Scope) الموقعة
- التوقيع الرقمي للإذن
- زر إيقاف طارئ

## 🤝 المساهمة

نرحب بمساهماتكم!

1. Fork المشروع
2. أنشئ فرعاً (`git checkout -b feature/amazing`)
3. Commit التغييرات (`git commit -m 'Add amazing feature'`)
4. Push للفرع (`git push origin feature/amazing`)
5. افتح Pull Request

## 📝 الترخيص

هذا المشروع مرخص تحت [MIT License](LICENSE).

## 👨‍💻 المطور

**BASYLYWS AI Team**

---

<p align="center">
  Made with ❤️ and ☕
</p>
