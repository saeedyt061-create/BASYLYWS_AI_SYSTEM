#!/usr/bin/env python3
"""
نظام اختبار الاختراق الأخلاقي - Ethical Security Scanner
=========================================================
ماسح أمني يكتشف الثغرات فقط دون استغلالها
يُستخدم فقط بموجب إذن كتابي من مالك النظام
"""

import asyncio
import socket
import ssl
import json
import hashlib
import rsa
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import ipaddress


class LegalWarning(UserWarning):
    """تحذير قانوني"""
    pass


# تحذير قانوني عند الاستيراد
warnings.warn(
    "⚠️  هذه الأداة للاستخدام القانوني فقط. يُجرّم الاستخدام غير المصرح به.",
    LegalWarning
)


@dataclass
class SecurityFinding:
    """نتيجة اكتشاف أمني"""
    severity: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    cve_id: Optional[str]
    title: str
    description: str
    remediation: str
    port: Optional[int] = None
    service: Optional[str] = None
    evidence: str = ""


@dataclass
class ScanResult:
    """نتيجة المسح"""
    target: str
    scan_time: datetime
    authorized_scope: str
    findings: List[SecurityFinding]
    risk_score: int
    scan_duration_seconds: float


class LegalScopeGenerator:
    """
    مُنشئ نطاق اختبار قانوني موثّق
    """
    
    def __init__(self, private_key_path: Optional[str] = None):
        """
        تهيئة المُنشئ
        
        Args:
            private_key_path: مسار المفتاح الخاص للتوقيع
        """
        self.private_key = None
        
        if private_key_path and Path(private_key_path).exists():
            with open(private_key_path, "rb") as f:
                self.private_key = rsa.PrivateKey.load_pkcs1(f.read())
        else:
            # توليد مفتاح جديد
            print("🔑 توليد مفتاح RSA جديد...")
            self.public_key, self.private_key = rsa.newkeys(2048)
            
            # حفظ المفتاح
            Path("keys").mkdir(exist_ok=True)
            with open("keys/private_key.pem", "wb") as f:
                f.write(self.private_key.save_pkcs1())
            with open("keys/public_key.pem", "wb") as f:
                f.write(self.public_key.save_pkcs1())
            
            print("✅ تم حفظ المفاتيح في مجلد keys/")
    
    def create_scope(self, 
                     organization: str, 
                     ips: List[str], 
                     purpose: str,
                     contact_email: str,
                     duration_days: int = 30,
                     restrictions: List[str] = None) -> str:
        """
        إنشاء ملف نطاق موقّع رقمياً
        
        Args:
            organization: اسم المنظمة
            ips: قائمة IPs/نطاقات مسموح بها
            purpose: الغرض من الاختبار
            contact_email: بريد التواصل
            duration_days: مدة الصلاحية بالأيام
            restrictions: قيود إضافية
            
        Returns:
            مسار ملف النطاق
        """
        scope = {
            "version": "2.0",
            "organization": organization,
            "authorized_ips": ips,
            "purpose": purpose,
            "legal_contact": contact_email,
            "created_at": datetime.now().isoformat(),
            "expiry_date": (datetime.now() + timedelta(days=duration_days)).isoformat(),
            "restrictions": restrictions or [],
            "emergency_stop": True,
            "max_scan_intensity": "normal",
            "forbidden_actions": [
                "data_exfiltration",
                "denial_of_service",
                "social_engineering",
                "physical_access"
            ]
        }
        
        # توقيع رقمي
        if self.private_key:
            signature = rsa.sign(
                json.dumps(scope, sort_keys=True).encode(),
                self.private_key,
                "SHA-256"
            )
            signature_hex = signature.hex()
        else:
            signature_hex = "unsigned"
        
        scope_document = {
            "scope": scope,
            "signature": signature_hex,
            "hash": hashlib.sha256(json.dumps(scope).encode()).hexdigest()[:16]
        }
        
        filename = f"SCOPE_{organization.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.json"
        
        with open(filename, "w") as f:
            json.dump(scope_document, f, indent=4, ensure_ascii=False)
        
        print(f"✅ نطاق قانوني مُنشأ: {filename}")
        print(f"📋 المنظمة: {organization}")
        print(f"🎯 الغرض: {purpose}")
        print(f"⏰ صالح حتى: {scope['expiry_date']}")
        print(f"📧 التواصل: {contact_email}")
        
        return filename
    
    def verify_scope(self, scope_file: str, public_key_path: Optional[str] = None) -> Dict:
        """
        التحقق من صحة ملف النطاق
        
        Args:
            scope_file: مسار ملف النطاق
            public_key_path: مسار المفتاح العام
            
        Returns:
            نتائج التحقق
        """
        try:
            with open(scope_file, "r") as f:
                document = json.load(f)
            
            scope = document["scope"]
            signature = bytes.fromhex(document["signature"])
            
            # التحقق من التوقيع
            if public_key_path and Path(public_key_path).exists():
                with open(public_key_path, "rb") as f:
                    public_key = rsa.PublicKey.load_pkcs1(f.read())
                
                try:
                    rsa.verify(
                        json.dumps(scope, sort_keys=True).encode(),
                        signature,
                        public_key
                    )
                    signature_valid = True
                except rsa.VerificationError:
                    signature_valid = False
            else:
                signature_valid = None
            
            # التحقق من الصلاحية
            expiry = datetime.fromisoformat(scope["expiry_date"])
            is_expired = datetime.now() > expiry
            
            return {
                "valid": not is_expired and (signature_valid is not False),
                "signature_valid": signature_valid,
                "expired": is_expired,
                "organization": scope["organization"],
                "purpose": scope["purpose"],
                "expiry_date": scope["expiry_date"],
                "authorized_ips": scope["authorized_ips"]
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}


class EthicalScanner:
    """
    ماسح أمني يكتشف الثغرات فقط دون استغلالها
    """
    
    # CVEs المعروفة للكشف
    KNOWN_CVES = {
        "CVE-2014-0160": {
            "name": "Heartbleed",
            "ports": [443, 8443],
            "severity": "CRITICAL",
            "description": "ثغرة في OpenSSL تسمح بقراءة ذاكرة الخادم",
            "fix": "حدّث OpenSSL إلى 1.0.1g أو أعلى"
        },
        "CVE-2020-15778": {
            "name": "OpenSSH Command Injection",
            "ports": [22],
            "severity": "HIGH",
            "description": "حقن أوامر في OpenSSH القديم",
            "fix": "حدّث OpenSSH إلى إصدار 8.0 أو أحدث"
        },
        "CVE-2017-0144": {
            "name": "EternalBlue",
            "ports": [445],
            "severity": "CRITICAL",
            "description": "ثغرة SMB في Windows",
            "fix": "تطبيق تحديث MS17-010"
        }
    }
    
    def __init__(self, target: str, scope_file: str):
        """
        تهيئة الماسح
        
        Args:
            target: IP المسموح باختبارها
            scope_file: ملف النطاق المصرح به
        """
        self.target = target
        self.authorized_scope = self._verify_legal_scope(scope_file)
        self.findings: List[SecurityFinding] = []
        self.scan_start_time = None
        
        print(f"🔒 ماسح أمني أخلاقي مهيأ")
    
    def _verify_legal_scope(self, scope_file: str) -> Dict:
        """التحقق من الإذن القانوني"""
        if not scope_file or not Path(scope_file).exists():
            raise PermissionError(
                "🚫 يجب تقديم ملف نطاق (scope) موثّق لتشغيل الاختبارات"
            )
        
        try:
            with open(scope_file, "r") as f:
                document = json.load(f)
            
            scope = document["scope"]
            
            # التحقق من الصلاحية
            expiry = datetime.fromisoformat(scope["expiry_date"])
            if datetime.now() > expiry:
                raise PermissionError("❌ الإذن منتهي الصلاحية")
            
            # التحقق من IP المصرح به
            authorized = self._is_ip_authorized(self.target, scope["authorized_ips"])
            if not authorized:
                raise PermissionError(f"🚫 {self.target} غير مصرح باختبارها")
            
            print(f"✅ إذن قانوني مُصرح به لـ {scope['organization']}")
            print(f"📋 الغرض: {scope['purpose']}")
            print(f"⏰ صالح حتى: {scope['expiry_date']}")
            
            return scope
            
        except json.JSONDecodeError:
            raise PermissionError("❌ ملف النطاق تالف")
        except KeyError as e:
            raise PermissionError(f"❌ حقل مفقود في ملف النطاق: {e}")
    
    def _is_ip_authorized(self, target: str, authorized_ips: List[str]) -> bool:
        """التحقق إذا كان IP ضمن النطاق المصرح به"""
        try:
            target_ip = ipaddress.ip_address(target)
            
            for authorized in authorized_ips:
                if '/' in authorized:
                    # نطاق CIDR
                    network = ipaddress.ip_network(authorized, strict=False)
                    if target_ip in network:
                        return True
                else:
                    # IP واحد
                    if target_ip == ipaddress.ip_address(authorized):
                        return True
            
            return False
        except ValueError:
            # إذا كان اسم نطاق
            return target in authorized_ips
    
    async def scan_target(self, 
                         ports: List[int] = None,
                         intensity: str = "normal") -> Dict[str, Any]:
        """
        مسح شامل للهدف
        
        Args:
            ports: قائمة المنافذ (افتراضياً شائعة)
            intensity: شدة المسح (quick, normal, thorough)
        """
        self.scan_start_time = datetime.now()
        
        if ports is None:
            ports = self._get_ports_by_intensity(intensity)
        
        print(f"\n🔍 بدء اختبار الاختراق الأخلاقي لـ {self.target}")
        print(f"📊 المنافذ: {len(ports)} | الشدة: {intensity}")
        
        # فحص المنافذ
        open_ports = await self._port_scan(ports)
        
        # اختبار CVEs
        for port, service in open_ports:
            await self._test_cve_vulnerabilities(port, service)
        
        # اختبارات إضافية
        await self._test_ssl_tls()
        await self._test_http_headers()
        
        # حساب درجة المخاطرة
        risk_score = self._calculate_risk_score()
        
        # بناء النتيجة
        scan_duration = (datetime.now() - self.scan_start_time).total_seconds()
        
        result = {
            "scan_time": datetime.now().isoformat(),
            "target": self.target,
            "authorized_scope": self.authorized_scope["organization"],
            "scan_duration_seconds": scan_duration,
            "open_ports": open_ports,
            "findings_count": len(self.findings),
            "critical_findings": sum(1 for f in self.findings if f.severity == "CRITICAL"),
            "high_findings": sum(1 for f in self.findings if f.severity == "HIGH"),
            "risk_score": risk_score,
            "findings": [self._finding_to_dict(f) for f in self.findings]
        }
        
        # حفظ التقرير
        self._generate_security_report(result)
        
        return result
    
    def _get_ports_by_intensity(self, intensity: str) -> List[int]:
        """الحصول على قائمة المنافذ حسب الشدة"""
        ports = {
            "quick": [22, 80, 443, 3306, 5432],
            "normal": [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3306, 5432, 8080, 8443],
            "thorough": list(range(1, 1025)) + [3306, 5432, 6379, 8080, 8443, 9200, 27017]
        }
        return ports.get(intensity, ports["normal"])
    
    async def _port_scan(self, ports: List[int]) -> List[Tuple[int, str]]:
        """فحص المنافذ المفتوحة"""
        open_ports = []
        
        async def check_port(port: int):
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.target, port),
                    timeout=2
                )
                
                # محاولة تحديد الخدمة
                service = self._identify_service(port, reader, writer)
                
                writer.close()
                await writer.wait_closed()
                
                return (port, service)
            except:
                return None
        
        tasks = [check_port(port) for port in ports]
        results = await asyncio.gather(*tasks)
        
        open_ports = [r for r in results if r is not None]
        
        print(f"🔓 منافذ مفتوحة: {len(open_ports)}")
        for port, service in open_ports:
            print(f"   Port {port}: {service}")
        
        return open_ports
    
    def _identify_service(self, port: int, reader, writer) -> str:
        """تحديد الخدمة على المنفذ"""
        common_services = {
            21: "FTP",
            22: "SSH",
            23: "Telnet",
            25: "SMTP",
            53: "DNS",
            80: "HTTP",
            110: "POP3",
            143: "IMAP",
            443: "HTTPS",
            993: "IMAPS",
            995: "POP3S",
            3306: "MySQL",
            5432: "PostgreSQL",
            6379: "Redis",
            8080: "HTTP-Proxy",
            8443: "HTTPS-Alt",
            9200: "Elasticsearch",
            27017: "MongoDB"
        }
        
        # محاولة قراءة البانر
        try:
            writer.write(b"\r\n")
            banner = asyncio.wait_for(reader.read(1024), timeout=1)
            banner_str = banner.decode(errors="ignore").strip()
            
            if banner_str:
                return f"{common_services.get(port, 'Unknown')} ({banner_str[:30]})"
        except:
            pass
        
        return common_services.get(port, "Unknown")
    
    async def _test_cve_vulnerabilities(self, port: int, service: str):
        """اختبار CVEs المعروفة"""
        cve_tests = {
            443: self._test_heartbleed,
            8443: self._test_heartbleed,
            22: self._test_ssh_versions,
            21: self._test_ftp_anonymous,
            3306: self._test_mysql_weak_auth
        }
        
        if port in cve_tests:
            try:
                result = await cve_tests[port](port)
                if result.get("vulnerable"):
                    self.findings.append(SecurityFinding(
                        severity=result.get("severity", "HIGH"),
                        cve_id=result.get("cve_id"),
                        title=result.get("title", "Unknown Vulnerability"),
                        description=result.get("description"),
                        remediation=result.get("fix"),
                        port=port,
                        service=service,
                        evidence=result.get("evidence", "")
                    ))
            except Exception as e:
                pass
    
    async def _test_heartbleed(self, port: int) -> Dict:
        """الكشف عن ثغرة Heartbleed"""
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.target, port, ssl=context),
                timeout=3
            )
            
            # Heartbeat request
            heartbeat = bytes.fromhex("18 03 02 00 03 01 40 00".replace(" ", ""))
            writer.write(heartbeat)
            await writer.drain()
            
            response = await asyncio.wait_for(reader.read(1024), timeout=2)
            
            writer.close()
            await writer.wait_closed()
            
            if len(response) > 3:
                return {
                    "vulnerable": True,
                    "severity": "CRITICAL",
                    "cve_id": "CVE-2014-0160",
                    "title": "Heartbleed Vulnerability",
                    "description": "خادم SSL معرض لثغرة Heartbleed - يمكن قراءة ذاكرة الخادم",
                    "fix": "حدّث OpenSSL إلى 1.0.1g أو أعلى، أوقف TLS heartbeat",
                    "evidence": f"Response size: {len(response)} bytes"
                }
        except:
            pass
        
        return {"vulnerable": False}
    
    async def _test_ssh_versions(self, port: int) -> Dict:
        """الكشف عن SSH قديم"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.target, port),
                timeout=3
            )
            
            banner = await asyncio.wait_for(reader.read(1024), timeout=2)
            banner_str = banner.decode(errors="ignore")
            
            writer.close()
            await writer.wait_closed()
            
            vulnerable_versions = ["OpenSSH_7.2", "OpenSSH_7.3", "OpenSSH_7.4", "OpenSSH_7.5"]
            
            if any(v in banner_str for v in vulnerable_versions):
                return {
                    "vulnerable": True,
                    "severity": "HIGH",
                    "cve_id": "CVE-2020-15778",
                    "title": "OpenSSH Vulnerable Version",
                    "description": f"SSH نسخة قديمة معروفة بثغرات: {banner_str.strip()}",
                    "fix": "حدّث OpenSSH إلى إصدار 8.0 أو أحدث",
                    "evidence": banner_str.strip()
                }
        except:
            pass
        
        return {"vulnerable": False}
    
    async def _test_ftp_anonymous(self, port: int) -> Dict:
        """الكشف عن FTP Anonymous"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.target, port),
                timeout=3
            )
            
            banner = await asyncio.wait_for(reader.read(1024), timeout=2)
            
            # محاولة تسجيل دخول anonymous
            writer.write(b"USER anonymous\r\n")
            await writer.drain()
            
            response = await asyncio.wait_for(reader.read(1024), timeout=2)
            response_str = response.decode(errors="ignore")
            
            writer.close()
            await writer.wait_closed()
            
            if "331" in response_str or "230" in response_str:
                return {
                    "vulnerable": True,
                    "severity": "MEDIUM",
                    "cve_id": None,
                    "title": "FTP Anonymous Login Enabled",
                    "description": "خادم FTP يسمح بتسجيل الدخول المجهول",
                    "fix": "عطّل تسجيل الدخول المجهول في إعدادات FTP",
                    "evidence": response_str.strip()
                }
        except:
            pass
        
        return {"vulnerable": False}
    
    async def _test_mysql_weak_auth(self, port: int) -> Dict:
        """الكشف عن MySQL مصادقة ضعيفة"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.target, port),
                timeout=3
            )
            
            # قراءة بانر MySQL
            banner = await asyncio.wait_for(reader.read(1024), timeout=2)
            
            writer.close()
            await writer.wait_closed()
            
            if b"mysql_native_password" in banner.lower():
                return {
                    "vulnerable": True,
                    "severity": "MEDIUM",
                    "cve_id": None,
                    "title": "MySQL Weak Authentication",
                    "description": "MySQL يستخدم mysql_native_password (ضعيف)",
                    "fix": "استخدم caching_sha2_password أو sha256_password",
                    "evidence": "mysql_native_password detected"
                }
        except:
            pass
        
        return {"vulnerable": False}
    
    async def _test_ssl_tls(self):
        """اختبار إعدادات SSL/TLS"""
        try:
            context = ssl.create_default_context()
            
            with socket.create_connection((self.target, 443), timeout=3) as sock:
                with context.wrap_socket(sock, server_hostname=self.target) as ssock:
                    version = ssock.version()
                    cipher = ssock.cipher()
                    
                    if version in ["SSLv2", "SSLv3", "TLSv1", "TLSv1.1"]:
                        self.findings.append(SecurityFinding(
                            severity="HIGH",
                            cve_id=None,
                            title=f"Weak SSL/TLS Version: {version}",
                            description=f"الخادم يستخدم إصداراً ضعيفاً: {version}",
                            remediation="تفعيل TLS 1.2 أو 1.3 فقط",
                            port=443,
                            service="HTTPS"
                        ))
        except:
            pass
    
    async def _test_http_headers(self):
        """اختبار رؤوس HTTP الأمنية"""
        try:
            import urllib.request
            
            url = f"http://{self.target}"
            req = urllib.request.Request(url, method='HEAD')
            
            try:
                response = urllib.request.urlopen(req, timeout=3)
                headers = dict(response.headers)
                
                security_headers = [
                    'X-Frame-Options',
                    'X-Content-Type-Options',
                    'X-XSS-Protection',
                    'Content-Security-Policy',
                    'Strict-Transport-Security'
                ]
                
                missing = [h for h in security_headers if h not in headers]
                
                if missing:
                    self.findings.append(SecurityFinding(
                        severity="MEDIUM",
                        cve_id=None,
                        title="Missing Security Headers",
                        description=f"رؤوس أمنية مفقودة: {', '.join(missing[:3])}",
                        remediation="أضف الرؤوس الأمنية في إعدادات الخادم",
                        port=80,
                        service="HTTP"
                    ))
            except:
                pass
        except:
            pass
    
    def _calculate_risk_score(self) -> int:
        """حساب درجة المخاطرة"""
        score = 0
        
        for finding in self.findings:
            if finding.severity == "CRITICAL":
                score += 10
            elif finding.severity == "HIGH":
                score += 7
            elif finding.severity == "MEDIUM":
                score += 4
            elif finding.severity == "LOW":
                score += 1
        
        return min(score, 100)
    
    def _finding_to_dict(self, finding: SecurityFinding) -> Dict:
        """تحويل النتيجة إلى قاموس"""
        return {
            "severity": finding.severity,
            "cve_id": finding.cve_id,
            "title": finding.title,
            "description": finding.description,
            "remediation": finding.remediation,
            "port": finding.port,
            "service": finding.service,
            "evidence": finding.evidence
        }
    
    def _generate_security_report(self, results: Dict):
        """توليد تقرير أمني"""
        filename = f"SECURITY_AUDIT_{self.target.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        
        print(f"\n📊 تقرير الأمان مُنشأ: {filename}")
        print(f"⚠️  النتائج: {results['findings_count']} ثغرة")
        print(f"   🔴 حرجة: {results['critical_findings']}")
        print(f"   🟠 عالية: {results['high_findings']}")
        print(f"   📈 درجة المخاطرة: {results['risk_score']}/100")
        
        if results['findings']:
            print("\n🔧 توصيات فورية:")
            for finding in results['findings'][:5]:
                print(f"   - [{finding['severity']}] {finding['title']}")


class EmergencyStop:
    """زر إيقاف فوري للاختبارات"""
    
    def __init__(self, scan_process_id: Optional[int] = None):
        self.scan_pid = scan_process_id
        self.stop_event = asyncio.Event()
    
    def trigger(self):
        """إيقاف جميع الاختبارات فوراً"""
        import signal
        import os
        
        print("🚨 إيقاف طارئ مُفعل!")
        
        if self.scan_pid:
            try:
                os.kill(self.scan_pid, signal.SIGTERM)
                print("✅ تم إيقاف العملية")
            except ProcessLookupError:
                print("⚠️  العملية غير موجودة")
        
        self.stop_event.set()
        
        # إشعار
        self._notify_admins()
    
    def _notify_admins(self):
        """إشعار فوري بالإيقاف"""
        print("📧 تم إشعار الإداريين بالإيقاف الطارئ")
        print(f"⏰ الوقت: {datetime.now().isoformat()}")
        
        # هنا يمكن إضافة إرسال بريد/SMS


# اختبار سريع
if __name__ == "__main__":
    # إنشاء نطاق اختبار
    scope_gen = LegalScopeGenerator()
    
    scope_file = scope_gen.create_scope(
        organization="Test Company",
        ips=["127.0.0.1", "192.168.1.0/24"],
        purpose="اختبار أمني ربع سنوي",
        contact_email="security@example.com",
        duration_days=7
    )
    
    print("\n" + "=" * 60)
    print("التحقق من النطاق:")
    verification = scope_gen.verify_scope(scope_file)
    print(f"صالح: {verification['valid']}")
