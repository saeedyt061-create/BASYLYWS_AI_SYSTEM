#!/usr/bin/env python3
"""
محرك ML للبرمجيات - Code ML Engine
==================================
يحلل: بنية الكود، جودة الكود، الأخطاء المحتملة، فئة البرنامج
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    import tensorflow as tf
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report, confusion_matrix
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow/Scikit-learn غير مثبت")
    TF_AVAILABLE = False

from .code_analyzer import SmartCodeAnalyzer


class CodeMLEngine:
    """
    محرك ML للبرمجيات يحلل:
    - بنية الكود
    - جودة الكود
    - الأخطاء المحتملة
    - فئة البرنامج
    """
    
    # فئات البرمجيات
    SOFTWARE_CATEGORIES = [
        "web", "ml", "data", "security", 
        "automation", "ui", "backend", "api", "script"
    ]
    
    def __init__(self, models_dir: str = "models"):
        """
        تهيئة محرك ML
        
        Args:
            models_dir: مجلد حفظ/تحميل النماذج
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # نماذج ML
        self.classifier: Optional[RandomForestClassifier] = None
        self.bug_predictor: Optional[tf.keras.Model] = None
        self.performance_model: Optional[tf.keras.Model] = None
        self.vulnerability_detector: Optional[IsolationForest] = None
        self.security_classifier: Optional[RandomForestClassifier] = None
        
        # أدوات التحليل
        self.vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2))
        self.code_analyzer = SmartCodeAnalyzer()
        
        # محاول تحميل النماذج المحفوظة
        self._load_saved_models()
        
        print("✅ محرك ML جاهز")
    
    def _load_saved_models(self):
        """تحميل النماذج المحفوظة"""
        model_files = {
            "software_classifier.pkl": "classifier",
            "bug_predictor.pkl": "bug_predictor",
            "vulnerability_detector.pkl": "vulnerability_detector",
            "security_classifier.pkl": "security_classifier"
        }
        
        for filename, attr_name in model_files.items():
            filepath = self.models_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, "rb") as f:
                        setattr(self, attr_name, pickle.load(f))
                    print(f"📦 تم تحميل: {filename}")
                except Exception as e:
                    print(f"⚠️  خطأ في تحميل {filename}: {e}")
    
    # ========== 1. تصنيف البرمجيات ==========
    
    def train_software_classifier(self, code_samples: List[Tuple[str, str]], 
                                   validate: bool = True) -> Dict[str, Any]:
        """
        تدريب نموذج لتصنيف البرامج
        
        Args:
            code_samples: قائمة من (كود, فئة)
            validate: التحقق من النموذج
            
        Returns:
            نتائج التدريب
        """
        if not TF_AVAILABLE:
            return {"error": "ML libraries not available"}
        
        print("\n📦 تدريب نموذج تصنيف البرمجيات...")
        
        features = []
        labels = []
        
        for code, category in code_samples:
            code_features = self.code_analyzer.extract_features(code)
            features.append(list(code_features.values()))
            labels.append(category)
        
        X = np.array(features)
        y = np.array(labels)
        
        # تقسيم البيانات
        if validate and len(X) > 10:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_test, y_test = None, None
        
        # تدريب Random Forest
        self.classifier = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.classifier.fit(X_train, y_train)
        
        # التقييم
        results = {
            "training_samples": len(X_train),
            "features_count": X.shape[1]
        }
        
        if X_test is not None:
            accuracy = self.classifier.score(X_test, y_test)
            predictions = self.classifier.predict(X_test)
            
            results["accuracy"] = accuracy
            results["classification_report"] = classification_report(
                y_test, predictions, output_dict=True
            )
            
            print(f"✅ دقة النموذج: {accuracy:.2%}")
        else:
            accuracy = self.classifier.score(X_train, y_train)
            results["training_accuracy"] = accuracy
            print(f"✅ دقة التدريب: {accuracy:.2%}")
        
        # حفظ النموذج
        self._save_model(self.classifier, "software_classifier.pkl")
        
        return results
    
    def predict_software_category(self, code: str) -> Dict[str, Any]:
        """
        تصنيف شيفرة جديدة تلقائياً
        
        Args:
            code: الكود المراد تصنيفه
            
        Returns:
            نتائج التصنيف مع الثقة
        """
        if self.classifier is None:
            return {"error": "النموذج غير مدرب", "category": "unknown"}
        
        features = np.array([list(self.code_analyzer.extract_features(code).values())])
        
        predicted = self.classifier.predict(features)[0]
        probabilities = self.classifier.predict_proba(features)[0]
        
        # ترتيب الفئات حسب الثقة
        category_scores = {
            cls: float(prob) 
            for cls, prob in zip(self.classifier.classes_, probabilities)
        }
        sorted_categories = sorted(
            category_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return {
            "category": predicted,
            "confidence": float(max(probabilities)),
            "all_scores": dict(sorted_categories[:5]),
            "top_3": sorted_categories[:3]
        }
    
    # ========== 2. تنبؤ الأخطاء ==========
    
    def train_bug_predictor(self, commit_history: List[Dict]) -> Dict[str, Any]:
        """
        تدريب نموذج على سجل commits للتنبؤ بالأخطاء
        
        Args:
            commit_history: سجل الـ commits مع المعلومات
                {
                    "lines": int,
                    "files": int,
                    "complexity": int,
                    "review_time": float,
                    "coverage": float,
                    "experience": int,
                    "has_bug": bool
                }
        """
        if not TF_AVAILABLE:
            return {"error": "TensorFlow not available"}
        
        print("\n🐛 تدريب نموذج تنبؤ الأخطاء...")
        
        features = []
        labels = []
        
        for commit in commit_history:
            feat = {
                "lines_changed": commit.get("lines", 0),
                "files_modified": commit.get("files", 0),
                "complexity_inc": commit.get("complexity", 0),
                "review_hours": commit.get("review_time", 0),
                "test_coverage": commit.get("coverage", 0),
                "author_commits": commit.get("experience", 0),
                "is_friday": 1 if commit.get("day") == "Friday" else 0
            }
            
            features.append(list(feat.values()))
            labels.append(1 if commit.get("has_bug") else 0)
        
        X = np.array(features)
        y = np.array(labels)
        
        if len(X) < 10:
            return {"error": "Insufficient data (need at least 10 samples)"}
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # إنشاء الشبكة العصبية
        self.bug_predictor = self._create_bug_neural_network(len(features[0]))
        
        # تدريب
        history = self.bug_predictor.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # التقييم
        loss, accuracy, precision, recall = self.bug_predictor.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"✅ دقة التنبؤ: {accuracy:.2%}")
        print(f"   Precision: {precision:.2%}")
        print(f"   Recall: {recall:.2%}")
        
        # حفظ
        self.bug_predictor.save(self.models_dir / "bug_predictor.keras")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "training_history": history.history
        }
    
    def _create_bug_neural_network(self, input_dim: int) -> tf.keras.Model:
        """شبكة عصبية لتنبؤ الأخطاء"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_dim=input_dim),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()
            ]
        )
        
        return model
    
    def predict_bug_likelihood(self, commit_stats: Dict) -> Dict[str, Any]:
        """
        تنبؤ باحتمال وجود خطأ
        
        Args:
            commit_stats: إحصائيات الـ commit
        """
        if self.bug_predictor is None:
            return {"error": "النموذج غير مدرب"}
        
        features = np.array([[
            commit_stats.get("lines", 0),
            commit_stats.get("files", 0),
            commit_stats.get("complexity", 0),
            commit_stats.get("review_time", 0),
            commit_stats.get("coverage", 0),
            commit_stats.get("experience", 0),
            1 if commit_stats.get("day") == "Friday" else 0
        ]])
        
        probability = float(self.bug_predictor.predict(features, verbose=0)[0][0])
        
        risk_level = "LOW"
        if probability > 0.7:
            risk_level = "HIGH"
        elif probability > 0.4:
            risk_level = "MEDIUM"
        
        recommendation = "آمن للنشر"
        if probability > 0.7:
            recommendation = "🚨 توقف فوراً - أضف اختبارات إضافية"
        elif probability > 0.5:
            recommendation = "⚠️ أضف اختبارات ومراجعة إضافية"
        elif probability > 0.3:
            recommendation = "ℹ️ مراجعة موصى بها"
        
        return {
            "bug_probability": probability,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "factors": {
                "lines_factor": min(commit_stats.get("lines", 0) / 100, 1.0),
                "complexity_factor": min(commit_stats.get("complexity", 0) / 10, 1.0),
                "coverage_factor": 1 - commit_stats.get("coverage", 0) / 100
            }
        }
    
    # ========== 3. اكتشاف الثغرات ==========
    
    def train_vulnerability_detector(self, 
                                      vulnerable_code_samples: List[Tuple[str, int]]) -> Dict[str, Any]:
        """
        تدريب كاشف الثغرات
        
        Args:
            vulnerable_code_samples: قائمة من (كود, هل يحتوي على ثغرة)
        """
        if not TF_AVAILABLE:
            return {"error": "ML libraries not available"}
        
        print("\n🛡️ تدريب كاشف الثغرات...")
        
        features = []
        labels = []
        
        for code, is_vulnerable in vulnerable_code_samples:
            code_features = self.code_analyzer.extract_features(code)
            features.append(list(code_features.values()))
            labels.append(is_vulnerable)
        
        X = np.array(features)
        y = np.array(labels)
        
        # Isolation Forest للكشف عن الشذوذ
        self.vulnerability_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.vulnerability_detector.fit(X)
        
        # مصنف ثنائي
        self.security_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.security_classifier.fit(X, y)
        
        # حفظ
        self._save_model(self.vulnerability_detector, "vulnerability_detector.pkl")
        self._save_model(self.security_classifier, "security_classifier.pkl")
        
        accuracy = self.security_classifier.score(X, y)
        print(f"✅ كاشف الثغرات جاهز (دقة: {accuracy:.2%})")
        
        return {"accuracy": accuracy}
    
    def detect_vulnerabilities(self, code: str) -> Dict[str, Any]:
        """
        فحص الشيفرة للثغرات
        
        Args:
            code: الكود المراد فحصه
        """
        features = np.array([list(self.code_analyzer.extract_features(code).values())])
        
        # فحص قواعدي
        rule_based_issues = self._rule_based_vulnerability_scan(code)
        
        # فحص ML
        ml_result = None
        if self.vulnerability_detector and self.security_classifier:
            anomaly_score = self.vulnerability_detector.score_samples(features)[0]
            is_vulnerable = self.security_classifier.predict(features)[0]
            confidence = max(self.security_classifier.predict_proba(features)[0])
            
            ml_result = {
                "ml_detected": bool(is_vulnerable),
                "confidence": float(confidence),
                "anomaly_score": float(anomaly_score)
            }
        
        # دمج النتائج
        all_issues = rule_based_issues
        is_vulnerable = len(all_issues) > 0 or (ml_result and ml_result["ml_detected"])
        
        recommendation = "✅ الكود آمن"
        if is_vulnerable:
            if any(i.get("severity") == "CRITICAL" for i in all_issues):
                recommendation = "🚨 توقف فوراً وقُم بالمراجعة الأمنية"
            else:
                recommendation = "⚠️ مراجعة أمنية موصى بها"
        
        return {
            "is_vulnerable": is_vulnerable,
            "vulnerabilities_found": all_issues,
            "ml_analysis": ml_result,
            "recommendation": recommendation,
            "scan_time": datetime.now().isoformat()
        }
    
    def _rule_based_vulnerability_scan(self, code: str) -> List[Dict]:
        """فحص قواعدي للثغرات الشائعة"""
        issues = []
        code_lower = code.lower()
        
        # SQL Injection
        sql_patterns = [
            (r'execute\s*\(\s*f["\']', "SQL Injection via f-string"),
            (r'execute\s*\(\s*["\'].*%s', "SQL Injection via % formatting"),
            (r'execute\s*\(\s*["\'].*\+', "SQL Injection via concatenation"),
            (r'\.format\s*\(.*\).*(?:select|insert|update|delete)', "SQL Injection via format()")
        ]
        
        for pattern, desc in sql_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append({
                    "type": "SQL_INJECTION",
                    "severity": "CRITICAL",
                    "description": desc,
                    "fix": "Use parameterized queries: cursor.execute('SELECT * FROM table WHERE id = %s', (user_input,))"
                })
                break
        
        # XSS
        xss_patterns = [
            (r'innerHTML\s*=\s*.*\+', "XSS via innerHTML"),
            (r'document\.write\s*\(', "XSS via document.write"),
            (r'eval\s*\(', "Dangerous eval() usage")
        ]
        
        for pattern, desc in xss_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append({
                    "type": "XSS",
                    "severity": "CRITICAL",
                    "description": desc,
                    "fix": "Use textContent instead of innerHTML, sanitize all user input"
                })
                break
        
        # Command Injection
        cmd_patterns = [
            (r'os\.system\s*\(.*\+', "Command Injection"),
            (r'subprocess\.call\s*\(\s*["\'].*\+', "Command Injection"),
            (r'subprocess\.run\s*\(\s*["\'].*\+', "Command Injection")
        ]
        
        for pattern, desc in cmd_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append({
                    "type": "COMMAND_INJECTION",
                    "severity": "CRITICAL",
                    "description": desc,
                    "fix": "Use subprocess with list arguments, never concatenate user input"
                })
                break
        
        # Hardcoded secrets
        secret_patterns = [
            (r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9]{16,}["\']', "Hardcoded API key"),
            (r'secret[_-]?key\s*=\s*["\'][a-zA-Z0-9]{8,}["\']', "Hardcoded secret key"),
            (r'token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "Hardcoded token")
        ]
        
        for pattern, desc in secret_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                issues.append({
                    "type": "HARDCODED_SECRET",
                    "severity": "HIGH",
                    "description": desc,
                    "fix": "Use environment variables: os.environ.get('SECRET_KEY')"
                })
                break
        
        # Insecure deserialization
        if re.search(r'pickle\.loads?\s*\(', code):
            issues.append({
                "type": "INSECURE_DESERIALIZATION",
                "severity": "HIGH",
                "description": "Insecure pickle deserialization",
                "fix": "Use json.loads() or validate pickle data before loading"
            })
        
        # Weak crypto
        if re.search(r'md5|sha1', code_lower):
            issues.append({
                "type": "WEAK_CRYPTO",
                "severity": "MEDIUM",
                "description": "Weak hashing algorithm detected",
                "fix": "Use SHA-256 or stronger algorithms"
            })
        
        return issues
    
    # ========== 4. تنبؤ الأداء ==========
    
    def train_performance_predictor(self, performance_data: List[Dict]) -> Dict[str, Any]:
        """
        تدريب نموذج تنبؤ الأداء
        
        Args:
            performance_data: بيانات الأداء
                {
                    "code": str,
                    "execution_time_ms": float
                }
        """
        if not TF_AVAILABLE:
            return {"error": "TensorFlow not available"}
        
        print("\n⚡ تدريب نموذج تنبؤ الأداء...")
        
        features = []
        labels = []
        
        for sample in performance_data:
            code_features = self.code_analyzer.extract_features(sample["code"])
            features.append(list(code_features.values()))
            labels.append(sample["execution_time_ms"])
        
        X = np.array(features)
        y = np.array(labels)
        
        if len(X) < 10:
            return {"error": "Insufficient data"}
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # شبكة للانحدار
        self.performance_model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_dim=len(features[0])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        self.performance_model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.performance_model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=16,
            verbose=0
        )
        
        loss, mae = self.performance_model.evaluate(X_test, y_test, verbose=0)
        
        print(f"✅ نموذج الأداء جاهز (MAE: {mae:.1f}ms)")
        
        self.performance_model.save(self.models_dir / "performance_model.keras")
        
        return {"mae": mae}
    
    def predict_performance(self, code: str) -> Dict[str, Any]:
        """
        تنبؤ بأداء شيفرة جديدة
        
        Args:
            code: الكود المراد تحليله
        """
        if self.performance_model is None:
            return {"error": "النموذج غير مدرب"}
        
        features = np.array([list(self.code_analyzer.extract_features(code).values())])
        
        predicted_time = float(self.performance_model.predict(features, verbose=0)[0][0])
        
        complexity_class = "FAST"
        if predicted_time > 500:
            complexity_class = "SLOW"
        elif predicted_time > 100:
            complexity_class = "MEDIUM"
        
        suggestions = self._suggest_optimizations(predicted_time, code)
        
        return {
            "predicted_time_ms": predicted_time,
            "complexity_class": complexity_class,
            "optimization_suggestions": suggestions
        }
    
    def _suggest_optimizations(self, predicted_time: float, code: str) -> List[str]:
        """إقتراحات تلقائية للتحسين"""
        suggestions = []
        
        # حلقات متداخلة
        if code.count('for') >= 2 or code.count('while') >= 2:
            suggestions.append("🔄 Nested loops detected - Consider vectorization with NumPy")
        
        # list comprehension
        if 'for' in code and '.append(' in code:
            suggestions.append("📋 Use list comprehension instead of append in loop")
        
        # وقت طويل
        if predicted_time > 1000:
            suggestions.append("⚡ High predicted execution time - Consider caching or memoization")
        
        # sleep
        if 'sleep(' in code:
            suggestions.append("⏱️  sleep() detected - Consider async/await for I/O operations")
        
        # recursive
        if re.search(r'def\s+(\w+)\s*\([^)]*\).*\1\s*\(', code, re.DOTALL):
            suggestions.append("🔄 Recursion detected - Consider iterative approach for large inputs")
        
        return suggestions[:5]
    
    # ========== الأدوات المساعدة ==========
    
    def _save_model(self, model, filename: str):
        """حفظ أي نموذج"""
        path = self.models_dir / filename
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"💾 تم حفظ: {filename}")
    
    def generate_report(self, code: str) -> Dict[str, Any]:
        """تقرير ML كامل للشيفرة"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "code_hash": hash(code) % 1000000,
            "analysis": {}
        }
        
        # التصنيف
        if self.classifier:
            report["analysis"]["category"] = self.predict_software_category(code)
        
        # الأخطاء
        if self.bug_predictor:
            report["analysis"]["bug_prediction"] = self.predict_bug_likelihood({
                "lines": 100,
                "files": 2,
                "complexity": 5
            })
        
        # الثغرات
        report["analysis"]["vulnerabilities"] = self.detect_vulnerabilities(code)
        
        # الأداء
        if self.performance_model:
            report["analysis"]["performance"] = self.predict_performance(code)
        
        # روائح الكود
        features = self.code_analyzer.extract_features(code)
        report["code_smells"] = self.code_analyzer.detect_smells(
            code, 
            None, 
            type('Metrics', (), features)()
        )
        
        return report
    
    def get_model_info(self) -> Dict[str, Any]:
        """معلومات عن النماذج المحملة"""
        return {
            "software_classifier": self.classifier is not None,
            "bug_predictor": self.bug_predictor is not None,
            "performance_model": self.performance_model is not None,
            "vulnerability_detector": self.vulnerability_detector is not None,
            "models_directory": str(self.models_dir),
            "available_models": [f.name for f in self.models_dir.glob("*") if f.is_file()]
        }


# اختبار سريع
if __name__ == "__main__":
    engine = CodeMLEngine()
    
    # بيانات تدريب تجريبية
    training_data = [
        ("def web_view(request):\n    return render(request, 'index.html')", "web"),
        ("import numpy as np\ndata = np.mean(dataset, axis=0)", "data"),
        ("model = RandomForestClassifier()\nmodel.fit(X, y)", "ml"),
        ("hash = sha256(password + salt).hexdigest()", "security"),
        ("@app.route('/api')\ndef api():\n    return jsonify(data)", "api"),
        ("class UserForm(forms.Form):\n    name = forms.CharField()", "ui"),
        ("def automate_backup():\n    shutil.copy(src, dst)", "automation"),
        ("class Database:\n    def connect(self):\n        pass", "backend"),
    ]
    
    # تدريب المصنف
    engine.train_software_classifier(training_data)
    
    # اختبار التصنيف
    test_code = """
@app.route('/users')
def get_users():
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])
"""
    
    result = engine.predict_software_category(test_code)
    print(f"\n📦 التصنيف: {result['category']} (ثقة: {result['confidence']:.1%})")
    print(f"🏆 أفضل 3 فئات: {result['top_3']}")
    
    # اختبار كشف الثغرات
    vulnerable_code = """
def login(username, password):
    query = f"SELECT * FROM users WHERE name = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
"""
    
    vuln_result = engine.detect_vulnerabilities(vulnerable_code)
    print(f"\n🛡️  كشف الثغرات:")
    print(f"   هل يحتوي على ثغرات: {vuln_result['is_vulnerable']}")
    for v in vuln_result['vulnerabilities_found']:
        print(f"   ⚠️  {v['type']}: {v['description']}")
