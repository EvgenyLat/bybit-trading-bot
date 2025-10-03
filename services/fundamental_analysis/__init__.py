"""
Fundamental Analysis Service
News sentiment, on-chain metrics, and macro analysis
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import yaml
import json
from transformers import pipeline
import asyncio
import aiohttp

logger = logging.getLogger(__name__)


class NewsSentimentAnalyzer:
    """Analyze news sentiment for trading decisions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.news_api_key = config.get('news_api_key')
        self.sentiment_pipeline = pipeline("sentiment-analysis", 
                                          model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        
    async def fetch_crypto_news(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Fetch recent crypto news"""
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': f'{symbol} cryptocurrency',
                'from': (datetime.now() - timedelta(hours=hours)).isoformat(),
                'sortBy': 'publishedAt',
                'apiKey': self.news_api_key,
                'language': 'en',
                'pageSize': 50
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('articles', [])
                    else:
                        logger.error(f"News API error: {response.status}")
                        return []
                        
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text"""
        try:
            result = self.sentiment_pipeline(text)
            
            # Convert to numerical score
            sentiment_map = {'LABEL_0': -1, 'LABEL_1': 0, 'LABEL_2': 1}  # Negative, Neutral, Positive
            score = sentiment_map.get(result[0]['label'], 0) * result[0]['score']
            
            return {
                'sentiment': result[0]['label'],
                'confidence': result[0]['score'],
                'score': score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5, 'score': 0}
    
    async def get_sentiment_score(self, symbol: str) -> Dict:
        """Get overall sentiment score for symbol"""
        try:
            news_articles = await self.fetch_crypto_news(symbol)
            
            if not news_articles:
                return {'score': 0, 'confidence': 0, 'article_count': 0}
            
            sentiments = []
            for article in news_articles:
                title_sentiment = self.analyze_sentiment(article.get('title', ''))
                description_sentiment = self.analyze_sentiment(article.get('description', ''))
                
                # Weight title more heavily
                combined_score = title_sentiment['score'] * 0.7 + description_sentiment['score'] * 0.3
                sentiments.append(combined_score)
            
            # Calculate weighted average
            avg_sentiment = np.mean(sentiments)
            confidence = 1 - np.std(sentiments)  # Lower std = higher confidence
            
            return {
                'score': avg_sentiment,
                'confidence': confidence,
                'article_count': len(news_articles),
                'sentiments': sentiments
            }
            
        except Exception as e:
            logger.error(f"Error getting sentiment score: {e}")
            return {'score': 0, 'confidence': 0, 'article_count': 0}


class OnChainAnalyzer:
    """Analyze on-chain metrics for crypto assets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.api_keys = config.get('onchain_apis', {})
        
    async def fetch_btc_metrics(self) -> Dict:
        """Fetch Bitcoin on-chain metrics"""
        try:
            metrics = {}
            
            # Glassnode API for Bitcoin metrics
            if 'glassnode' in self.api_keys:
                glassnode_key = self.api_keys['glassnode']
                
                # Active addresses
                url = "https://api.glassnode.com/v1/metrics/addresses/active_count"
                params = {
                    'a': 'BTC',
                    'api_key': glassnode_key,
                    'f': 'JSON',
                    'i': '1d',
                    's': int((datetime.now() - timedelta(days=30)).timestamp())
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data:
                                metrics['active_addresses'] = data[-1]['v']
                                metrics['active_addresses_ma7'] = np.mean([d['v'] for d in data[-7:]])
            
            # CoinMetrics API
            if 'coinmetrics' in self.api_keys:
                cm_key = self.api_keys['coinmetrics']
                
                # Network value to transactions ratio
                url = f"https://api.coinmetrics.io/v4/timeseries/asset-metrics"
                params = {
                    'assets': 'btc',
                    'metrics': 'NVT',
                    'api_key': cm_key,
                    'start_time': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data and 'data' in data:
                                nvt_values = [float(d['NVT']) for d in data['data'] if d['NVT']]
                                if nvt_values:
                                    metrics['nvt'] = nvt_values[-1]
                                    metrics['nvt_ma7'] = np.mean(nvt_values[-7:])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching BTC metrics: {e}")
            return {}
    
    async def fetch_eth_metrics(self) -> Dict:
        """Fetch Ethereum on-chain metrics"""
        try:
            metrics = {}
            
            # Ethereum-specific metrics
            if 'etherscan' in self.api_keys:
                etherscan_key = self.api_keys['etherscan']
                
                # Gas price analysis
                url = "https://api.etherscan.io/api"
                params = {
                    'module': 'gastracker',
                    'action': 'gasoracle',
                    'apikey': etherscan_key
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('status') == '1':
                                result = data['result']
                                metrics['gas_price_fast'] = int(result['FastGasPrice'])
                                metrics['gas_price_standard'] = int(result['StandardGasPrice'])
                                metrics['gas_price_safe'] = int(result['SafeGasPrice'])
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error fetching ETH metrics: {e}")
            return {}
    
    def calculate_onchain_signals(self, metrics: Dict) -> Dict:
        """Calculate trading signals from on-chain metrics"""
        try:
            signals = {}
            
            # Active addresses signal
            if 'active_addresses' in metrics and 'active_addresses_ma7' in metrics:
                active_ratio = metrics['active_addresses'] / metrics['active_addresses_ma7']
                if active_ratio > 1.1:
                    signals['active_addresses'] = 'bullish'
                elif active_ratio < 0.9:
                    signals['active_addresses'] = 'bearish'
                else:
                    signals['active_addresses'] = 'neutral'
            
            # NVT signal
            if 'nvt' in metrics and 'nvt_ma7' in metrics:
                nvt_ratio = metrics['nvt'] / metrics['nvt_ma7']
                if nvt_ratio < 0.9:  # Low NVT = undervalued
                    signals['nvt'] = 'bullish'
                elif nvt_ratio > 1.1:  # High NVT = overvalued
                    signals['nvt'] = 'bearish'
                else:
                    signals['nvt'] = 'neutral'
            
            # Gas price signal for ETH
            if 'gas_price_standard' in metrics:
                if metrics['gas_price_standard'] > 50:  # High gas = high activity
                    signals['gas_price'] = 'bullish'
                elif metrics['gas_price_standard'] < 20:  # Low gas = low activity
                    signals['gas_price'] = 'bearish'
                else:
                    signals['gas_price'] = 'neutral'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating on-chain signals: {e}")
            return {}


class MacroAnalyzer:
    """Analyze macroeconomic indicators"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.fred_api_key = config.get('fred_api_key')
        
    async def fetch_dxy(self) -> Optional[float]:
        """Fetch Dollar Index (DXY)"""
        try:
            if not self.fred_api_key:
                return None
                
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'DTWEXBGS',
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'limit': 1,
                'sort_order': 'desc'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('observations'):
                            return float(data['observations'][0]['value'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching DXY: {e}")
            return None
    
    async def fetch_vix(self) -> Optional[float]:
        """Fetch VIX (Volatility Index)"""
        try:
            if not self.fred_api_key:
                return None
                
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                'series_id': 'VIXCLS',
                'api_key': self.fred_api_key,
                'file_type': 'json',
                'limit': 1,
                'sort_order': 'desc'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('observations'):
                            return float(data['observations'][0]['value'])
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching VIX: {e}")
            return None
    
    def calculate_macro_signals(self, dxy: Optional[float], vix: Optional[float]) -> Dict:
        """Calculate macro trading signals"""
        try:
            signals = {}
            
            # DXY signal
            if dxy is not None:
                if dxy > 105:  # Strong dollar
                    signals['dxy'] = 'bearish'  # Bearish for crypto
                elif dxy < 95:  # Weak dollar
                    signals['dxy'] = 'bullish'  # Bullish for crypto
                else:
                    signals['dxy'] = 'neutral'
            
            # VIX signal
            if vix is not None:
                if vix > 30:  # High volatility/fear
                    signals['vix'] = 'bearish'  # Risk-off
                elif vix < 15:  # Low volatility/complacency
                    signals['vix'] = 'bullish'  # Risk-on
                else:
                    signals['vix'] = 'neutral'
            
            return signals
            
        except Exception as e:
            logger.error(f"Error calculating macro signals: {e}")
            return {}


class FundamentalAnalysisService:
    """Main fundamental analysis service"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.news_analyzer = NewsSentimentAnalyzer(config)
        self.onchain_analyzer = OnChainAnalyzer(config)
        self.macro_analyzer = MacroAnalyzer(config)
        
    async def get_comprehensive_analysis(self, symbol: str) -> Dict:
        """Get comprehensive fundamental analysis"""
        try:
            logger.info(f"Performing fundamental analysis for {symbol}")
            
            # Get all analysis components
            sentiment_task = self.news_analyzer.get_sentiment_score(symbol)
            
            if symbol.upper().startswith('BTC'):
                onchain_task = self.onchain_analyzer.fetch_btc_metrics()
            elif symbol.upper().startswith('ETH'):
                onchain_task = self.onchain_analyzer.fetch_eth_metrics()
            else:
                onchain_task = asyncio.create_task(asyncio.sleep(0))  # No-op
            
            dxy_task = self.macro_analyzer.fetch_dxy()
            vix_task = self.macro_analyzer.fetch_vix()
            
            # Wait for all tasks
            sentiment_result = await sentiment_task
            onchain_metrics = await onchain_task
            dxy = await dxy_task
            vix = await vix_task
            
            # Calculate signals
            onchain_signals = self.onchain_analyzer.calculate_onchain_signals(onchain_metrics)
            macro_signals = self.macro_analyzer.calculate_macro_signals(dxy, vix)
            
            # Combine all analysis
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'sentiment': sentiment_result,
                'onchain_metrics': onchain_metrics,
                'onchain_signals': onchain_signals,
                'macro_signals': macro_signals,
                'overall_score': self._calculate_overall_score(sentiment_result, onchain_signals, macro_signals)
            }
            
            logger.info(f"Fundamental analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_score(self, sentiment: Dict, onchain_signals: Dict, macro_signals: Dict) -> float:
        """Calculate overall fundamental score"""
        try:
            score = 0.0
            weight_sum = 0.0
            
            # Sentiment weight: 40%
            if 'score' in sentiment:
                score += sentiment['score'] * 0.4
                weight_sum += 0.4
            
            # On-chain signals weight: 35%
            onchain_score = 0.0
            for signal in onchain_signals.values():
                if signal == 'bullish':
                    onchain_score += 1
                elif signal == 'bearish':
                    onchain_score -= 1
                # neutral = 0
            
            if onchain_signals:
                onchain_score = onchain_score / len(onchain_signals)
                score += onchain_score * 0.35
                weight_sum += 0.35
            
            # Macro signals weight: 25%
            macro_score = 0.0
            for signal in macro_signals.values():
                if signal == 'bullish':
                    macro_score += 1
                elif signal == 'bearish':
                    macro_score -= 1
                # neutral = 0
            
            if macro_signals:
                macro_score = macro_score / len(macro_signals)
                score += macro_score * 0.25
                weight_sum += 0.25
            
            # Normalize by actual weight sum
            if weight_sum > 0:
                score = score / weight_sum
            
            return max(-1, min(1, score))  # Clamp between -1 and 1
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0

