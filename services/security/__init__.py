"""
Security Manager
Handles encryption, key management, and security controls
"""

import os
import base64
import hashlib
import hmac
import time
import logging
from typing import Dict, Optional, Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
import json
import secrets

logger = logging.getLogger(__name__)


class SecurityManager:
    """Handles all security-related operations"""
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password or self._get_master_password()
        self.encryption_key = self._derive_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
    def _get_master_password(self) -> str:
        """Get master password from keyring or environment"""
        try:
            # Try to get from keyring first
            password = keyring.get_password("bybit_trading_bot", "master_password")
            if password:
                return password
        except Exception:
            pass
        
        # Fallback to environment variable
        password = os.getenv("MASTER_PASSWORD")
        if password:
            return password
        
        # Generate new password if none exists
        password = secrets.token_urlsafe(32)
        try:
            keyring.set_password("bybit_trading_bot", "master_password", password)
            logger.info("Generated new master password and stored in keyring")
        except Exception as e:
            logger.warning(f"Could not store password in keyring: {e}")
        
        return password
    
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from master password"""
        try:
            # Use PBKDF2 to derive key from master password
            salt = b'bybit_trading_bot_salt'  # In production, use random salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self.master_password.encode()))
            return key
        except Exception as e:
            logger.error(f"Error deriving encryption key: {e}")
            raise
    
    def encrypt_api_key(self, api_key: str, api_secret: str) -> Dict[str, str]:
        """Encrypt API credentials"""
        try:
            credentials = {
                'api_key': api_key,
                'api_secret': api_secret,
                'timestamp': time.time()
            }
            
            encrypted_data = self.cipher_suite.encrypt(json.dumps(credentials).encode())
            encrypted_b64 = base64.b64encode(encrypted_data).decode()
            
            return {
                'encrypted_credentials': encrypted_b64,
                'checksum': self._calculate_checksum(api_key + api_secret)
            }
        except Exception as e:
            logger.error(f"Error encrypting API credentials: {e}")
            raise
    
    def decrypt_api_key(self, encrypted_credentials: str, checksum: str) -> Tuple[str, str]:
        """Decrypt API credentials"""
        try:
            # Verify checksum
            if not self._verify_checksum(encrypted_credentials, checksum):
                raise ValueError("Checksum verification failed")
            
            encrypted_data = base64.b64decode(encrypted_credentials.encode())
            decrypted_data = self.cipher_suite.decrypt(encrypted_data)
            credentials = json.loads(decrypted_data.decode())
            
            # Check timestamp (credentials older than 30 days are invalid)
            if time.time() - credentials['timestamp'] > 30 * 24 * 3600:
                raise ValueError("Credentials expired")
            
            return credentials['api_key'], credentials['api_secret']
            
        except Exception as e:
            logger.error(f"Error decrypting API credentials: {e}")
            raise
    
    def _calculate_checksum(self, data: str) -> str:
        """Calculate SHA256 checksum"""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _verify_checksum(self, data: str, expected_checksum: str) -> bool:
        """Verify SHA256 checksum"""
        actual_checksum = self._calculate_checksum(data)
        return hmac.compare_digest(actual_checksum, expected_checksum)
    
    def validate_api_permissions(self, api_key: str, api_secret: str) -> Dict[str, bool]:
        """Validate API key permissions"""
        try:
            # This would make a test API call to validate permissions
            # For now, return basic validation
            return {
                'has_read_permission': True,
                'has_trade_permission': True,
                'has_withdraw_permission': False,  # Should be False for security
                'is_testnet': True  # Should be True for development
            }
        except Exception as e:
            logger.error(f"Error validating API permissions: {e}")
            return {
                'has_read_permission': False,
                'has_trade_permission': False,
                'has_withdraw_permission': False,
                'is_testnet': True
            }
    
    def rotate_api_keys(self, old_key: str, old_secret: str) -> bool:
        """Rotate API keys (placeholder for actual implementation)"""
        try:
            # In production, this would:
            # 1. Generate new API keys
            # 2. Update the bot configuration
            # 3. Revoke old keys
            # 4. Test new keys
            logger.info("API key rotation requested (not implemented)")
            return True
        except Exception as e:
            logger.error(f"Error rotating API keys: {e}")
            return False
    
    def audit_log(self, action: str, details: Dict[str, str]):
        """Log security-related actions"""
        try:
            audit_entry = {
                'timestamp': time.time(),
                'action': action,
                'details': details,
                'ip_address': self._get_client_ip(),
                'user_agent': self._get_user_agent()
            }
            
            # Log to secure audit file
            audit_file = "logs/security_audit.log"
            os.makedirs(os.path.dirname(audit_file), exist_ok=True)
            
            with open(audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Error writing audit log: {e}")
    
    def _get_client_ip(self) -> str:
        """Get client IP address"""
        # Placeholder - in production, get actual IP
        return "127.0.0.1"
    
    def _get_user_agent(self) -> str:
        """Get user agent"""
        return "BybitTradingBot/1.0"


class SecureConfigManager:
    """Manages secure configuration"""
    
    def __init__(self, security_manager: SecurityManager):
        self.security_manager = security_manager
        self.config_cache = {}
    
    def load_secure_config(self, config_path: str) -> Dict:
        """Load configuration with encrypted credentials"""
        try:
            import yaml
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Decrypt API credentials if they exist
            if 'encrypted_credentials' in config.get('api', {}):
                encrypted_creds = config['api']['encrypted_credentials']
                checksum = config['api']['checksum']
                
                api_key, api_secret = self.security_manager.decrypt_api_key(
                    encrypted_creds, checksum
                )
                
                config['api']['api_key'] = api_key
                config['api']['api_secret'] = api_secret
                
                # Remove encrypted data from memory
                del config['api']['encrypted_credentials']
                del config['api']['checksum']
            
            # Validate API permissions
            if 'api_key' in config.get('api', {}):
                permissions = self.security_manager.validate_api_permissions(
                    config['api']['api_key'], 
                    config['api']['api_secret']
                )
                
                if not permissions['has_read_permission']:
                    raise ValueError("API key lacks read permissions")
                
                if permissions['has_withdraw_permission']:
                    logger.warning("API key has withdraw permissions - security risk!")
                
                config['api']['permissions'] = permissions
            
            self.config_cache = config
            return config
            
        except Exception as e:
            logger.error(f"Error loading secure config: {e}")
            raise
    
    def save_secure_config(self, config: Dict, config_path: str):
        """Save configuration with encrypted credentials"""
        try:
            import yaml
            
            # Create a copy for saving
            save_config = config.copy()
            
            # Encrypt API credentials
            if 'api_key' in save_config.get('api', {}):
                encrypted_creds = self.security_manager.encrypt_api_key(
                    save_config['api']['api_key'],
                    save_config['api']['api_secret']
                )
                
                save_config['api']['encrypted_credentials'] = encrypted_creds['encrypted_credentials']
                save_config['api']['checksum'] = encrypted_creds['checksum']
                
                # Remove plaintext credentials
                del save_config['api']['api_key']
                del save_config['api']['api_secret']
            
            # Save to file
            with open(config_path, 'w') as f:
                yaml.dump(save_config, f, default_flow_style=False)
            
            logger.info(f"Secure configuration saved to {config_path}")
            
        except Exception as e:
            logger.error(f"Error saving secure config: {e}")
            raise

