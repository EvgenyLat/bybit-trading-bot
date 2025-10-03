"""
Security Monitoring System
Real-time security monitoring, intrusion detection, and alerting
"""

import logging
import time
import hashlib
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Security event types"""
    API_KEY_EXPOSED = "api_key_exposed"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    CONFIG_CHANGE = "config_change"
    EMERGENCY_STOP = "emergency_stop"
    DATA_BREACH = "data_breach"
    SYSTEM_COMPROMISE = "system_compromise"


@dataclass
class SecurityEvent:
    """Security event container"""
    event_type: SecurityEventType
    severity: str
    description: str
    timestamp: datetime
    source_ip: str
    user_agent: str
    details: Dict[str, Any]
    resolved: bool = False


class SecurityMonitor:
    """Real-time security monitoring"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.events: deque = deque(maxlen=10000)
        self.suspicious_ips: Dict[str, int] = defaultdict(int)
        self.failed_attempts: Dict[str, int] = defaultdict(int)
        self.rate_limits: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.Lock()
        
        # Security thresholds
        self.max_failed_attempts = config.get('max_failed_attempts', 5)
        self.rate_limit_window = config.get('rate_limit_window', 300)  # 5 minutes
        self.max_requests_per_window = config.get('max_requests_per_window', 100)
        
    def log_event(self, event_type: SecurityEventType, severity: str, 
                 description: str, source_ip: str = "127.0.0.1", 
                 user_agent: str = "Unknown", details: Dict[str, Any] = None):
        """Log security event"""
        try:
            event = SecurityEvent(
                event_type=event_type,
                severity=severity,
                description=description,
                timestamp=datetime.now(),
                source_ip=source_ip,
                user_agent=user_agent,
                details=details or {}
            )
            
            with self.lock:
                self.events.append(event)
                
                # Update suspicious IP tracking
                if severity in ['high', 'critical']:
                    self.suspicious_ips[source_ip] += 1
                
                # Check for patterns
                self._analyze_patterns(event)
            
            # Log the event
            log_level = getattr(logging, severity.upper(), logging.INFO)
            logger.log(log_level, f"Security Event: {event_type.value} - {description}")
            
        except Exception as e:
            logger.error(f"Error logging security event: {e}")
    
    def _analyze_patterns(self, event: SecurityEvent):
        """Analyze security patterns"""
        try:
            # Check for brute force attempts
            if event.event_type == SecurityEventType.UNAUTHORIZED_ACCESS:
                self.failed_attempts[event.source_ip] += 1
                
                if self.failed_attempts[event.source_ip] >= self.max_failed_attempts:
                    self.log_event(
                        SecurityEventType.SUSPICIOUS_ACTIVITY,
                        'high',
                        f"Brute force attack detected from {event.source_ip}",
                        event.source_ip,
                        event.user_agent
                    )
            
            # Check rate limiting
            current_time = time.time()
            self.rate_limits[event.source_ip].append(current_time)
            
            # Remove old requests
            while (self.rate_limits[event.source_ip] and 
                   current_time - self.rate_limits[event.source_ip][0] > self.rate_limit_window):
                self.rate_limits[event.source_ip].popleft()
            
            # Check if rate limit exceeded
            if len(self.rate_limits[event.source_ip]) > self.max_requests_per_window:
                self.log_event(
                    SecurityEventType.RATE_LIMIT_EXCEEDED,
                    'medium',
                    f"Rate limit exceeded from {event.source_ip}",
                    event.source_ip,
                    event.user_agent
                )
                
        except Exception as e:
            logger.error(f"Error analyzing security patterns: {e}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate security report"""
        try:
            with self.lock:
                recent_events = [
                    event for event in self.events
                    if event.timestamp > datetime.now() - timedelta(hours=24)
                ]
                
                critical_events = [
                    event for event in recent_events
                    if event.severity == 'critical'
                ]
                
                high_events = [
                    event for event in recent_events
                    if event.severity == 'high'
                ]
                
                return {
                    'total_events_24h': len(recent_events),
                    'critical_events': len(critical_events),
                    'high_events': len(high_events),
                    'suspicious_ips': len(self.suspicious_ips),
                    'failed_attempts': sum(self.failed_attempts.values()),
                    'top_suspicious_ips': dict(sorted(
                        self.suspicious_ips.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]),
                    'recent_critical_events': [
                        {
                            'type': event.event_type.value,
                            'description': event.description,
                            'timestamp': event.timestamp.isoformat(),
                            'source_ip': event.source_ip
                        }
                        for event in critical_events[-5:]
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error generating security report: {e}")
            return {}
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        try:
            with self.lock:
                return (self.failed_attempts.get(ip, 0) >= self.max_failed_attempts or
                        self.suspicious_ips.get(ip, 0) >= 3)
                        
        except Exception as e:
            logger.error(f"Error checking IP block status: {e}")
            return False
    
    def block_ip(self, ip: str, reason: str):
        """Block IP address"""
        try:
            self.log_event(
                SecurityEventType.UNAUTHORIZED_ACCESS,
                'high',
                f"IP {ip} blocked: {reason}",
                ip
            )
            
            with self.lock:
                self.failed_attempts[ip] = self.max_failed_attempts
                self.suspicious_ips[ip] = 10  # High suspicion level
                
        except Exception as e:
            logger.error(f"Error blocking IP {ip}: {e}")
    
    def unblock_ip(self, ip: str):
        """Unblock IP address"""
        try:
            with self.lock:
                self.failed_attempts.pop(ip, None)
                self.suspicious_ips.pop(ip, None)
                
            self.log_event(
                SecurityEventType.UNAUTHORIZED_ACCESS,
                'low',
                f"IP {ip} unblocked",
                ip
            )
            
        except Exception as e:
            logger.error(f"Error unblocking IP {ip}: {e}")


class IntrusionDetectionSystem:
    """Intrusion detection system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.security_monitor = SecurityMonitor(config)
        self.anomaly_threshold = config.get('anomaly_threshold', 0.8)
        self.baseline_metrics = {}
        self.current_metrics = {}
        
    def analyze_behavior(self, behavior_data: Dict[str, Any]) -> bool:
        """Analyze behavior for anomalies"""
        try:
            # Update current metrics
            self.current_metrics.update(behavior_data)
            
            # Check for anomalies
            anomalies = []
            
            # Check API call frequency
            if 'api_calls_per_minute' in behavior_data:
                if behavior_data['api_calls_per_minute'] > 100:  # Unusually high
                    anomalies.append('High API call frequency')
            
            # Check trading frequency
            if 'trades_per_hour' in behavior_data:
                if behavior_data['trades_per_hour'] > 50:  # Unusually high
                    anomalies.append('High trading frequency')
            
            # Check position sizes
            if 'avg_position_size' in behavior_data:
                if behavior_data['avg_position_size'] > 0.1:  # 10% of portfolio
                    anomalies.append('Large position sizes')
            
            # Check error rates
            if 'error_rate' in behavior_data:
                if behavior_data['error_rate'] > 0.1:  # 10% error rate
                    anomalies.append('High error rate')
            
            # If anomalies detected, log security event
            if anomalies:
                self.security_monitor.log_event(
                    SecurityEventType.SUSPICIOUS_ACTIVITY,
                    'medium',
                    f"Behavioral anomalies detected: {', '.join(anomalies)}",
                    details={'anomalies': anomalies, 'metrics': behavior_data}
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error analyzing behavior: {e}")
            return False
    
    def detect_config_tampering(self, config_hash: str) -> bool:
        """Detect configuration tampering"""
        try:
            # Compare with stored hash
            stored_hash = self.config.get('config_hash')
            
            if stored_hash and config_hash != stored_hash:
                self.security_monitor.log_event(
                    SecurityEventType.CONFIG_CHANGE,
                    'high',
                    'Configuration tampering detected',
                    details={'stored_hash': stored_hash, 'current_hash': config_hash}
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting config tampering: {e}")
            return False
    
    def detect_data_breach(self, data_access_log: List[Dict]) -> bool:
        """Detect potential data breach"""
        try:
            # Check for unusual data access patterns
            unusual_access = []
            
            for access in data_access_log:
                # Check for access outside normal hours
                hour = access.get('timestamp', datetime.now()).hour
                if hour < 6 or hour > 22:  # Outside 6 AM - 10 PM
                    unusual_access.append('Access outside normal hours')
                
                # Check for large data downloads
                if access.get('data_size', 0) > 1000000:  # 1MB
                    unusual_access.append('Large data download')
                
                # Check for access from new IP
                if access.get('ip') not in self.baseline_metrics.get('known_ips', []):
                    unusual_access.append('Access from unknown IP')
            
            if unusual_access:
                self.security_monitor.log_event(
                    SecurityEventType.DATA_BREACH,
                    'high',
                    f'Potential data breach detected: {", ".join(unusual_access)}',
                    details={'unusual_access': unusual_access}
                )
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting data breach: {e}")
            return False


class SecurityAlertSystem:
    """Security alert system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alert_channels = config.get('alert_channels', [])
        self.alert_thresholds = config.get('alert_thresholds', {})
        
    def send_alert(self, event: SecurityEvent):
        """Send security alert"""
        try:
            alert_message = self._format_alert(event)
            
            # Send to configured channels
            for channel in self.alert_channels:
                if channel['type'] == 'telegram':
                    self._send_telegram_alert(alert_message, channel)
                elif channel['type'] == 'email':
                    self._send_email_alert(alert_message, channel)
                elif channel['type'] == 'webhook':
                    self._send_webhook_alert(alert_message, channel)
            
        except Exception as e:
            logger.error(f"Error sending security alert: {e}")
    
    def _format_alert(self, event: SecurityEvent) -> str:
        """Format alert message"""
        return f"""
🚨 SECURITY ALERT 🚨

Type: {event.event_type.value}
Severity: {event.severity.upper()}
Description: {event.description}
Timestamp: {event.timestamp.isoformat()}
Source IP: {event.source_ip}
User Agent: {event.user_agent}

Details: {json.dumps(event.details, indent=2)}
        """
    
    def _send_telegram_alert(self, message: str, channel: Dict):
        """Send Telegram alert"""
        try:
            import requests
            
            url = f"https://api.telegram.org/bot{channel['token']}/sendMessage"
            data = {
                'chat_id': channel['chat_id'],
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending Telegram alert: {e}")
    
    def _send_email_alert(self, message: str, channel: Dict):
        """Send email alert"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            
            msg = MIMEText(message)
            msg['Subject'] = 'Security Alert - Trading Bot'
            msg['From'] = channel['from_email']
            msg['To'] = channel['to_email']
            
            server = smtplib.SMTP(channel['smtp_server'], channel['smtp_port'])
            server.starttls()
            server.login(channel['username'], channel['password'])
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _send_webhook_alert(self, message: str, channel: Dict):
        """Send webhook alert"""
        try:
            import requests
            
            data = {
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'source': 'trading_bot_security'
            }
            
            response = requests.post(
                channel['url'], 
                json=data, 
                headers=channel.get('headers', {}),
                timeout=10
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")

