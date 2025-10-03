"""
Machine Learning Module for Trading Bot
Implements LSTM, XGBoost, and other ML models for signal prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import joblib
import os

# Machine Learning imports
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Attention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM/GRU models will be disabled.")

logger = logging.getLogger(__name__)


class MLSignalPredictor:
    """Machine Learning signal prediction system"""
    
    def __init__(self, config: Dict):
        """
        Initialize ML Signal Predictor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.ml_config = config.get('ml', {})
        
        # Model parameters
        self.model_type = self.ml_config.get('model_type', 'lstm')
        self.features = self.ml_config.get('features', ['price', 'volume', 'rsi', 'macd'])
        self.lookback_window = self.ml_config.get('lookback_window', 60)
        self.prediction_horizon = self.ml_config.get('prediction_horizon', 5)
        self.confidence_threshold = self.ml_config.get('confidence_threshold', 0.6)
        
        # Model storage
        self.model = None
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.is_trained = False
        
        # Performance tracking
        self.model_performance = {}
        self.prediction_history = []
        
        # Model directory
        self.model_dir = "models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"ML Signal Predictor initialized with {self.model_type} model")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for ML model
        
        Args:
            df: DataFrame with OHLCV and indicator data
            
        Returns:
            DataFrame with prepared features
        """
        try:
            logger.info("Preparing features for ML model")
            
            features_df = df.copy()
            
            # Price-based features
            if 'price' in self.features:
                features_df['price_change'] = features_df['close'].pct_change()
                features_df['price_change_2'] = features_df['close'].pct_change(2)
                features_df['price_change_5'] = features_df['close'].pct_change(5)
                features_df['high_low_ratio'] = features_df['high'] / features_df['low']
                features_df['close_open_ratio'] = features_df['close'] / features_df['open']
            
            # Volume features
            if 'volume' in self.features:
                features_df['volume_change'] = features_df['volume'].pct_change()
                features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
                features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
                features_df['price_volume'] = features_df['close'] * features_df['volume']
            
            # Technical indicators (if available)
            if 'rsi' in self.features and 'rsi' in df.columns:
                features_df['rsi_change'] = df['rsi'].diff()
                features_df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
                features_df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            
            if 'macd' in self.features and 'macd' in df.columns:
                features_df['macd_signal'] = df['macd'] - df.get('macd_signal', 0)
                features_df['macd_histogram'] = df.get('macd_histogram', 0)
            
            if 'bb' in self.features and 'bb_upper' in df.columns:
                features_df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
                features_df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            if 'atr' in self.features and 'atr' in df.columns:
                features_df['atr_ratio'] = df['atr'] / df['close']
                features_df['volatility'] = df['atr'].rolling(14).mean()
            
            # Additional features
            features_df['hour'] = features_df.index.hour
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
            
            # Remove NaN values
            features_df = features_df.dropna()
            
            logger.info(f"Prepared {len(features_df.columns)} features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def create_sequences(self, data: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series models
        
        Args:
            data: Prepared feature data
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            logger.info("Creating sequences for time series model")
            
            # Select feature columns
            feature_cols = [col for col in data.columns if col not in ['target', 'timestamp']]
            X_data = data[feature_cols].values
            
            # Create sequences
            X, y = [], []
            
            for i in range(self.lookback_window, len(data)):
                X.append(X_data[i-self.lookback_window:i])
                y.append(data[target_col].iloc[i])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Created {len(X)} sequences with shape {X.shape}")
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sequences: {e}")
            raise
    
    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for prediction
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with target variables
        """
        try:
            logger.info("Creating target variables")
            
            data = df.copy()
            
            # Price direction prediction (classification)
            data['price_change_future'] = data['close'].shift(-self.prediction_horizon) / data['close'] - 1
            
            # Binary classification: 1 for up, 0 for down
            data['target'] = (data['price_change_future'] > 0).astype(int)
            
            # Multi-class classification: strong up, up, down, strong down
            data['target_multiclass'] = pd.cut(
                data['price_change_future'],
                bins=[-np.inf, -0.02, 0, 0.02, np.inf],
                labels=[0, 1, 2, 3]  # strong down, down, up, strong up
            ).astype(int)
            
            # Regression target
            data['target_regression'] = data['price_change_future']
            
            logger.info("Target variables created")
            return data
            
        except Exception as e:
            logger.error(f"Error creating targets: {e}")
            raise
    
    def train_lstm_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Train LSTM model
        
        Args:
            X: Input sequences
            y: Target values
            
        Returns:
            Trained LSTM model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for LSTM training")
        
        try:
            logger.info("Training LSTM model")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(
                X_train.reshape(-1, X_train.shape[-1])
            ).reshape(X_train.shape)
            X_val_scaled = self.feature_scaler.transform(
                X_val.reshape(-1, X_val.shape[-1])
            ).reshape(X_val.shape)
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Callbacks
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=5)
            ]
            
            # Train model
            history = model.fit(
                X_train_scaled, y_train,
                validation_data=(X_val_scaled, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            val_loss, val_acc = model.evaluate(X_val_scaled, y_val, verbose=0)
            logger.info(f"LSTM model trained. Validation accuracy: {val_acc:.4f}")
            
            self.model_performance = {
                'model_type': 'lstm',
                'validation_accuracy': val_acc,
                'validation_loss': val_loss,
                'training_history': history.history
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise
    
    def train_xgboost_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Train XGBoost model
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Trained XGBoost model
        """
        try:
            logger.info("Training XGBoost model")
            
            # Reshape for XGBoost (flatten sequences)
            X_flat = X.reshape(X.shape[0], -1)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train XGBoost model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            logger.info(f"XGBoost model trained. Validation accuracy: {accuracy:.4f}")
            
            self.model_performance = {
                'model_type': 'xgboost',
                'validation_accuracy': accuracy,
                'feature_importance': model.feature_importances_.tolist()
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Error training XGBoost model: {e}")
            raise
    
    def train_lightgbm_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """
        Train LightGBM model
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Trained LightGBM model
        """
        try:
            logger.info("Training LightGBM model")
            
            # Reshape for LightGBM (flatten sequences)
            X_flat = X.reshape(X.shape[0], -1)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X_flat, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train LightGBM model
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Evaluate model
            y_pred = model.predict(X_val, num_iteration=model.best_iteration)
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_val, y_pred_binary)
            
            logger.info(f"LightGBM model trained. Validation accuracy: {accuracy:.4f}")
            
            self.model_performance = {
                'model_type': 'lightgbm',
                'validation_accuracy': accuracy,
                'best_iteration': model.best_iteration
            }
            
            return model
            
        except Exception as e:
            logger.error(f"Error training LightGBM model: {e}")
            raise
    
    def train_model(self, df: pd.DataFrame) -> bool:
        """
        Train the ML model
        
        Args:
            df: DataFrame with OHLCV and indicator data
            
        Returns:
            True if training successful
        """
        try:
            logger.info(f"Training {self.model_type} model")
            
            # Prepare features and targets
            features_df = self.prepare_features(df)
            features_df = self.create_targets(features_df)
            
            # Remove rows with NaN targets
            features_df = features_df.dropna(subset=['target'])
            
            if len(features_df) < self.lookback_window + 100:
                logger.error("Insufficient data for training")
                return False
            
            # Create sequences
            X, y = self.create_sequences(features_df, 'target')
            
            if len(X) == 0:
                logger.error("No sequences created")
                return False
            
            # Train model based on type
            if self.model_type == 'lstm':
                self.model = self.train_lstm_model(X, y)
            elif self.model_type == 'xgboost':
                self.model = self.train_xgboost_model(X, y)
            elif self.model_type == 'lightgbm':
                self.model = self.train_lightgbm_model(X, y)
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            logger.info(f"{self.model_type} model trained successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict_signal(self, df: pd.DataFrame) -> Dict:
        """
        Predict trading signal
        
        Args:
            df: DataFrame with recent market data
            
        Returns:
            Dictionary with prediction results
        """
        try:
            if not self.is_trained or self.model is None:
                logger.warning("Model not trained, returning neutral signal")
                return {
                    'signal': 0,
                    'confidence': 0.0,
                    'prediction': 'neutral',
                    'timestamp': datetime.now()
                }
            
            # Prepare features
            features_df = self.prepare_features(df)
            
            if len(features_df) < self.lookback_window:
                logger.warning("Insufficient data for prediction")
                return {
                    'signal': 0,
                    'confidence': 0.0,
                    'prediction': 'insufficient_data',
                    'timestamp': datetime.now()
                }
            
            # Get latest sequence
            feature_cols = [col for col in features_df.columns if col not in ['target', 'timestamp']]
            X_data = features_df[feature_cols].values
            
            # Create sequence for prediction
            X_pred = X_data[-self.lookback_window:].reshape(1, self.lookback_window, -1)
            
            # Make prediction
            if self.model_type == 'lstm':
                X_pred_scaled = self.feature_scaler.transform(
                    X_pred.reshape(-1, X_pred.shape[-1])
                ).reshape(X_pred.shape)
                prediction_proba = self.model.predict(X_pred_scaled)[0][0]
            elif self.model_type == 'xgboost':
                X_pred_flat = X_pred.reshape(1, -1)
                prediction_proba = self.model.predict_proba(X_pred_flat)[0][1]
            elif self.model_type == 'lightgbm':
                X_pred_flat = X_pred.reshape(1, -1)
                prediction_proba = self.model.predict(X_pred_flat, num_iteration=self.model.best_iteration)[0]
            else:
                prediction_proba = 0.5
            
            # Convert to signal
            confidence = abs(prediction_proba - 0.5) * 2  # Scale to 0-1
            
            if prediction_proba > 0.5:
                signal = 1 if confidence >= self.confidence_threshold else 0
                prediction = 'buy' if signal == 1 else 'weak_buy'
            else:
                signal = -1 if confidence >= self.confidence_threshold else 0
                prediction = 'sell' if signal == -1 else 'weak_sell'
            
            result = {
                'signal': signal,
                'confidence': confidence,
                'prediction': prediction,
                'probability': prediction_proba,
                'timestamp': datetime.now()
            }
            
            # Store prediction history
            self.prediction_history.append(result)
            
            logger.info(f"ML prediction: {prediction} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {
                'signal': 0,
                'confidence': 0.0,
                'prediction': 'error',
                'timestamp': datetime.now()
            }
    
    def save_model(self):
        """Save trained model and scalers"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            if self.model_type == 'lstm':
                model_path = os.path.join(self.model_dir, f"{self.model_type}_model_{timestamp}.h5")
                self.model.save(model_path)
            else:
                model_path = os.path.join(self.model_dir, f"{self.model_type}_model_{timestamp}.pkl")
                joblib.dump(self.model, model_path)
            
            # Save scalers
            scaler_path = os.path.join(self.model_dir, f"feature_scaler_{timestamp}.pkl")
            joblib.dump(self.feature_scaler, scaler_path)
            
            # Save performance metrics
            performance_path = os.path.join(self.model_dir, f"performance_{timestamp}.json")
            import json
            with open(performance_path, 'w') as f:
                json.dump(self.model_performance, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, model_path: str):
        """Load trained model and scalers"""
        try:
            # Load model
            if self.model_type == 'lstm' and TENSORFLOW_AVAILABLE:
                self.model = tf.keras.models.load_model(model_path)
            else:
                self.model = joblib.load(model_path)
            
            # Load scalers (assume they exist in same directory)
            model_dir = os.path.dirname(model_path)
            scaler_files = [f for f in os.listdir(model_dir) if f.startswith('feature_scaler_')]
            if scaler_files:
                scaler_path = os.path.join(model_dir, scaler_files[-1])  # Latest scaler
                self.feature_scaler = joblib.load(scaler_path)
            
            self.is_trained = True
            logger.info(f"Model loaded from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_model_performance(self) -> Dict:
        """Get model performance metrics"""
        return self.model_performance.copy()
    
    def get_prediction_history(self, limit: int = 100) -> List[Dict]:
        """Get recent prediction history"""
        return self.prediction_history[-limit:] if self.prediction_history else []

