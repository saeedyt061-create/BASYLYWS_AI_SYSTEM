#!/usr/bin/env python3
"""
قاعدة بيانات للتعلم - Learning Database
=======================================
تخزن تاريخ التوليدات والتحليلات والتعلم منها
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class GenerationRecord:
    """سجل توليد كود"""
    id: Optional[int]
    prompt: str
    generated_code: str
    language: str
    quality_score: float
    user_feedback: Optional[str]
    created_at: datetime


class LearningDatabase:
    """
    قاعدة بيانات للتعلم من التوليدات السابقة
    """
    
    def __init__(self, db_path: str = "data/saeed_ai.db"):
        """
        تهيئة قاعدة البيانات
        
        Args:
            db_path: مسار ملف قاعدة البيانات
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        
        self._init_tables()
        print(f"✅ قاعدة البيانات: {db_path}")
    
    def _init_tables(self):
        """إنشاء الجداول"""
        cursor = self.conn.cursor()
        
        # جدول التوليدات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                prompt_hash TEXT UNIQUE,
                generated_code TEXT NOT NULL,
                language TEXT DEFAULT 'python',
                code_type TEXT DEFAULT 'function',
                quality_score REAL,
                execution_time_ms REAL,
                user_rating INTEGER,
                user_feedback TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # جدول التحليلات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_hash TEXT UNIQUE,
                code_snippet TEXT,
                language TEXT,
                metrics TEXT,
                issues TEXT,
                code_smells TEXT,
                quality_score REAL,
                analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # جدول الثغرات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS vulnerabilities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code_hash TEXT,
                vulnerability_type TEXT,
                severity TEXT,
                description TEXT,
                fix_suggestion TEXT,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # جدول المسح الأمني
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS security_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT,
                scope_file TEXT,
                findings_count INTEGER,
                risk_score INTEGER,
                report_path TEXT,
                scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # جدول الإحصائيات
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_type TEXT,
                stat_value REAL,
                stat_date DATE DEFAULT CURRENT_DATE
            )
        ''')
        
        # فهارس
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_gen_prompt ON generations(prompt_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_gen_date ON generations(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analyses_hash ON analyses(code_hash)')
        
        self.conn.commit()
    
    # ========== عمليات التوليد ==========
    
    def save_generation(self, 
                        prompt: str,
                        generated_code: str,
                        language: str = "python",
                        code_type: str = "function",
                        quality_score: float = 0.0,
                        execution_time_ms: float = 0.0,
                        metadata: Dict = None) -> int:
        """
        حفظ توليد جديد
        
        Args:
            prompt: الوصف/الطلب
            generated_code: الكود المولد
            language: لغة البرمجة
            code_type: نوع الكود
            quality_score: درجة الجودة
            execution_time_ms: وقت التنفيذ
            metadata: بيانات إضافية
            
        Returns:
            معرف السجل
        """
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO generations 
            (prompt, prompt_hash, generated_code, language, code_type, 
             quality_score, execution_time_ms, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prompt, prompt_hash, generated_code, language, code_type,
            quality_score, execution_time_ms, json.dumps(metadata or {})
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_generation(self, prompt: str) -> Optional[Dict]:
        """الحصول على توليد بواسطة الوصف"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM generations WHERE prompt_hash = ?
        ''', (prompt_hash,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    
    def get_similar_generations(self, prompt: str, limit: int = 5) -> List[Dict]:
        """الحصول على توليدات مشابهة"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM generations 
            WHERE prompt LIKE ?
            ORDER BY quality_score DESC, created_at DESC
            LIMIT ?
        ''', (f'%{prompt[:50]}%', limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def update_user_feedback(self, generation_id: int, 
                             rating: int, 
                             feedback: str):
        """تحديث تقييم المستخدم"""
        cursor = self.conn.cursor()
        cursor.execute('''
            UPDATE generations 
            SET user_rating = ?, user_feedback = ?
            WHERE id = ?
        ''', (rating, feedback, generation_id))
        self.conn.commit()
    
    # ========== عمليات التحليل ==========
    
    def save_analysis(self,
                      code: str,
                      language: str,
                      metrics: Dict,
                      issues: List,
                      code_smells: List,
                      quality_score: float) -> int:
        """حفظ تحليل كود"""
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO analyses
            (code_hash, code_snippet, language, metrics, issues, code_smells, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            code_hash,
            code[:1000],  # تخزين جزء فقط
            language,
            json.dumps(metrics),
            json.dumps(issues),
            json.dumps(code_smells),
            quality_score
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_analysis(self, code: str) -> Optional[Dict]:
        """الحصول على تحليل سابق"""
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM analyses WHERE code_hash = ?', (code_hash,))
        
        row = cursor.fetchone()
        if row:
            result = dict(row)
            result['metrics'] = json.loads(result['metrics'])
            result['issues'] = json.loads(result['issues'])
            result['code_smells'] = json.loads(result['code_smells'])
            return result
        return None
    
    # ========== عمليات الثغرات ==========
    
    def save_vulnerability(self,
                           code: str,
                           vuln_type: str,
                           severity: str,
                           description: str,
                           fix_suggestion: str):
        """حفظ ثغرة مكتشفة"""
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO vulnerabilities
            (code_hash, vulnerability_type, severity, description, fix_suggestion)
            VALUES (?, ?, ?, ?, ?)
        ''', (code_hash, vuln_type, severity, description, fix_suggestion))
        
        self.conn.commit()
    
    def get_vulnerabilities_by_type(self, vuln_type: str) -> List[Dict]:
        """الحصول على ثغرات حسب النوع"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM vulnerabilities 
            WHERE vulnerability_type = ?
            ORDER BY detected_at DESC
        ''', (vuln_type,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========== عمليات المسح الأمني ==========
    
    def save_security_scan(self,
                           target: str,
                           scope_file: str,
                           findings_count: int,
                           risk_score: int,
                           report_path: str):
        """حفظ نتيجة مسح أمني"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO security_scans
            (target, scope_file, findings_count, risk_score, report_path)
            VALUES (?, ?, ?, ?, ?)
        ''', (target, scope_file, findings_count, risk_score, report_path))
        
        self.conn.commit()
    
    def get_scan_history(self, target: str = None) -> List[Dict]:
        """الحصول على تاريخ المسح"""
        cursor = self.conn.cursor()
        
        if target:
            cursor.execute('''
                SELECT * FROM security_scans 
                WHERE target = ?
                ORDER BY scanned_at DESC
            ''', (target,))
        else:
            cursor.execute('SELECT * FROM security_scans ORDER BY scanned_at DESC')
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========== الإحصائيات ==========
    
    def record_statistic(self, stat_type: str, stat_value: float):
        """تسجيل إحصائية"""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO statistics (stat_type, stat_value)
            VALUES (?, ?)
        ''', (stat_type, stat_value))
        self.conn.commit()
    
    def get_statistics(self, stat_type: str = None, days: int = 30) -> Dict:
        """الحصول على إحصائيات"""
        cursor = self.conn.cursor()
        
        if stat_type:
            cursor.execute('''
                SELECT stat_type, AVG(stat_value) as avg_value, 
                       COUNT(*) as count, MAX(stat_date) as last_date
                FROM statistics
                WHERE stat_type = ? AND stat_date >= date('now', '-{} days')
                GROUP BY stat_type
            '''.format(days), (stat_type,))
        else:
            cursor.execute('''
                SELECT stat_type, AVG(stat_value) as avg_value,
                       COUNT(*) as count, MAX(stat_date) as last_date
                FROM statistics
                WHERE stat_date >= date('now', '-{} days')
                GROUP BY stat_type
            '''.format(days))
        
        return {row['stat_type']: dict(row) for row in cursor.fetchall()}
    
    # ========== التقارير ==========
    
    def get_dashboard_stats(self) -> Dict:
        """إحصائيات لوحة التحكم"""
        cursor = self.conn.cursor()
        
        # إجمالي التوليدات
        cursor.execute('SELECT COUNT(*) FROM generations')
        total_generations = cursor.fetchone()[0]
        
        # متوسط الجودة
        cursor.execute('SELECT AVG(quality_score) FROM generations')
        avg_quality = cursor.fetchone()[0] or 0
        
        # التوليدات اليوم
        cursor.execute('''
            SELECT COUNT(*) FROM generations 
            WHERE date(created_at) = date('now')
        ''')
        today_generations = cursor.fetchone()[0]
        
        # أفضل اللغات
        cursor.execute('''
            SELECT language, COUNT(*) as count 
            FROM generations 
            GROUP BY language 
            ORDER BY count DESC
        ''')
        top_languages = [dict(row) for row in cursor.fetchall()]
        
        # إجمالي الثغرات
        cursor.execute('SELECT COUNT(*) FROM vulnerabilities')
        total_vulns = cursor.fetchone()[0]
        
        # الثغرات حسب الخطورة
        cursor.execute('''
            SELECT severity, COUNT(*) as count 
            FROM vulnerabilities 
            GROUP BY severity
        ''')
        vuln_by_severity = {row['severity']: row['count'] for row in cursor.fetchall()}
        
        return {
            "total_generations": total_generations,
            "average_quality": round(avg_quality, 2),
            "today_generations": today_generations,
            "top_languages": top_languages,
            "total_vulnerabilities": total_vulns,
            "vulnerabilities_by_severity": vuln_by_severity
        }
    
    def get_recent_activity(self, limit: int = 10) -> List[Dict]:
        """النشاط الأخير"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            SELECT 
                'generation' as type,
                prompt as description,
                quality_score,
                created_at
            FROM generations
            UNION ALL
            SELECT 
                'analysis' as type,
                code_snippet as description,
                quality_score,
                analyzed_at as created_at
            FROM analyses
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========== التعلم ==========
    
    def get_training_data(self, min_quality: float = 0.7, limit: int = 1000) -> List[Dict]:
        """الحصول على بيانات للتدريب"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT prompt, generated_code, language, quality_score
            FROM generations
            WHERE quality_score >= ? AND user_rating IS NOT NULL
            ORDER BY user_rating DESC, quality_score DESC
            LIMIT ?
        ''', (min_quality, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_common_issues(self, limit: int = 10) -> List[Dict]:
        """الحصول على المشاكل الشائعة"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                json_extract(value, '$.code') as issue_code,
                json_extract(value, '$.message') as message,
                COUNT(*) as frequency
            FROM analyses, json_each(analyses.issues)
            GROUP BY issue_code
            ORDER BY frequency DESC
            LIMIT ?
        ''', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # ========== الصيانة ==========
    
    def cleanup_old_records(self, days: int = 90):
        """حذف السجلات القديمة"""
        cursor = self.conn.cursor()
        
        # حذف التوليدات القديمة
        cursor.execute('''
            DELETE FROM generations 
            WHERE created_at < date('now', '-{} days')
            AND user_rating IS NULL
        '''.format(days))
        
        generations_deleted = cursor.rowcount
        
        # حذف التحليلات القديمة
        cursor.execute('''
            DELETE FROM analyses 
            WHERE analyzed_at < date('now', '-{} days')
        '''.format(days))
        
        analyses_deleted = cursor.rowcount
        
        self.conn.commit()
        
        print(f"🗑️  تم حذف {generations_deleted} توليد و {analyses_deleted} تحليل")
    
    def vacuum(self):
        """ضغط قاعدة البيانات"""
        self.conn.execute('VACUUM')
        print("✅ تم ضغط قاعدة البيانات")
    
    def close(self):
        """إغلاق الاتصال"""
        self.conn.close()
        print("✅ تم إغلاق قاعدة البيانات")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# اختبار سريع
if __name__ == "__main__":
    with LearningDatabase() as db:
        # حفظ توليد
        gen_id = db.save_generation(
            prompt="Calculate factorial",
            generated_code="def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            language="python",
            quality_score=85.5,
            execution_time_ms=120.5
        )
        print(f"✅ تم حفظ التوليد: {gen_id}")
        
        # حفظ تحليل
        analysis_id = db.save_analysis(
            code="def test(): pass",
            language="python",
            metrics={"lines": 10, "complexity": 2},
            issues=[],
            code_smells=[],
            quality_score=90.0
        )
        print(f"✅ تم حفظ التحليل: {analysis_id}")
        
        # إحصائيات
        stats = db.get_dashboard_stats()
        print("\n📊 إحصائيات لوحة التحكم:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
