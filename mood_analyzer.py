import streamlit as st
import pandas as pd
import numpy as np
# from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class MoodAnalyzer:
    """AI-powered mood analysis and pattern detection"""
    
    def __init__(self):
        """Initialize the mood analyzer with required models"""
        self.sentiment_analyzer = None
        self.nltk_analyzer = None
        self.vectorizer = None
        self.scaler = StandardScaler()
        
        # Initialize components
        self._initialize_sentiment_analyzer()
        self._initialize_nltk()
    
    @st.cache_resource
    def _initialize_sentiment_analyzer(_self):
        """Initialize HuggingFace sentiment analysis pipeline"""
        try:
            # HuggingFace transformers not available, using NLTK only
            # _self.sentiment_analyzer = pipeline(
            #     "sentiment-analysis", 
            #     model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            #     return_all_scores=True
            # )
            return False  # Return False to indicate HF model not available
        except Exception as e:
            st.warning(f"Could not load HuggingFace model: {e}")
            return False
    
    def _initialize_nltk(self):
        """Initialize NLTK VADER sentiment analyzer"""
        try:
            nltk.download('vader_lexicon', quiet=True)
            self.nltk_analyzer = SentimentIntensityAnalyzer()
            return True
        except Exception as e:
            st.warning(f"Could not initialize NLTK: {e}")
            return False
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for analysis"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^\w\s\.\!\?\:\)\(\-\;\,]', '', text)
        
        return text.strip()
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using multiple approaches"""
        processed_text = self.preprocess_text(text)
        
        result = {
            'compound': 0.0,
            'pos': 0.0,
            'neu': 0.0,
            'neg': 0.0,
            'emotion': 'neutral',
            'confidence': 0.0
        }
        
        if not processed_text:
            return result
        
        # NLTK VADER Analysis (always available)
        if self.nltk_analyzer:
            vader_scores = self.nltk_analyzer.polarity_scores(processed_text)
            result.update({
                'compound': vader_scores['compound'],
                'pos': vader_scores['pos'],
                'neu': vader_scores['neu'],
                'neg': vader_scores['neg']
            })
        
        # HuggingFace Analysis (if available)
        if self.sentiment_analyzer:
            try:
                hf_results = self.sentiment_analyzer(processed_text)[0]
                
                # Map HuggingFace labels to emotions
                label_to_emotion = {
                    'LABEL_0': 'negative',
                    'LABEL_1': 'neutral', 
                    'LABEL_2': 'positive',
                    'NEGATIVE': 'negative',
                    'POSITIVE': 'positive'
                }
                
                # Find the highest confidence prediction
                best_prediction = max(hf_results, key=lambda x: x['score'])
                result['emotion'] = label_to_emotion.get(best_prediction['label'], 'neutral')
                result['confidence'] = best_prediction['score']
                
                # Adjust compound score based on HF results
                if result['emotion'] == 'positive':
                    result['compound'] = max(result['compound'], 0.1)
                elif result['emotion'] == 'negative':
                    result['compound'] = min(result['compound'], -0.1)
                    
            except Exception as e:
                st.warning(f"HuggingFace analysis failed: {e}")
        
        # Determine final emotion based on compound score
        if result['compound'] >= 0.05:
            result['emotion'] = 'positive'
        elif result['compound'] <= -0.05:
            result['emotion'] = 'negative'
        else:
            result['emotion'] = 'neutral'
        
        return result
    
    def extract_features_for_clustering(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for clustering analysis"""
        features = []
        
        # Numerical features
        numerical_cols = ['energy_level', 'calm_level', 'mood_rating', 'sentiment_score']
        for col in numerical_cols:
            if col in data.columns:
                features.append(data[col].values.reshape(-1, 1))
        
        # Sentiment components
        sentiment_cols = ['pos', 'neu', 'neg']
        for col in sentiment_cols:
            if col in data.columns:
                features.append(data[col].values.reshape(-1, 1))
        
        # Time-based features
        if 'timestamp' in data.columns:
            data_copy = data.copy()
            data_copy['hour'] = data_copy['timestamp'].dt.hour
            data_copy['day_of_week'] = data_copy['timestamp'].dt.dayofweek
            features.extend([
                data_copy['hour'].values.reshape(-1, 1),
                data_copy['day_of_week'].values.reshape(-1, 1)
            ])
        
        # Boolean features
        boolean_cols = ['work_stress', 'social_interaction', 'exercise']
        for col in boolean_cols:
            if col in data.columns:
                features.append(data[col].astype(int).values.reshape(-1, 1))
        
        if not features:
            return np.array([]).reshape(0, 0)
        
        # Combine all features
        combined_features = np.hstack(features)
        
        # Standardize features
        return self.scaler.fit_transform(combined_features)
    
    def perform_clustering(self, data: pd.DataFrame, method: str = "KMeans", n_clusters: int = 3) -> np.ndarray:
        """Perform clustering on mood data"""
        features = self.extract_features_for_clustering(data)
        
        if features.size == 0:
            return np.zeros(len(data))
        
        try:
            if method == "KMeans":
                # Determine optimal number of clusters
                n_samples = len(data)
                n_clusters = min(n_clusters, max(2, n_samples // 3))
                
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = clusterer.fit_predict(features)
                
            elif method == "DBSCAN":
                # Use DBSCAN for density-based clustering
                clusterer = DBSCAN(eps=0.5, min_samples=3)
                labels = clusterer.fit_predict(features)
                
                # Handle noise points (label -1)
                if -1 in labels:
                    labels = np.where(labels == -1, 0, labels + 1)
            
            return labels
            
        except Exception as e:
            st.warning(f"Clustering failed: {e}")
            return np.zeros(len(data))
    
    def generate_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate insights from mood data patterns"""
        insights = []
        
        if data.empty:
            return insights
        
        try:
            # Sentiment trends
            recent_sentiment = data.tail(7)['sentiment_score'].mean()
            overall_sentiment = data['sentiment_score'].mean()
            
            if recent_sentiment > overall_sentiment + 0.1:
                insights.append("📈 Your mood has been trending upward recently!")
            elif recent_sentiment < overall_sentiment - 0.1:
                insights.append("📉 Your recent mood has been lower than usual. Consider what might be contributing to this.")
            
            # Energy vs Calm correlation
            if 'energy_level' in data.columns and 'calm_level' in data.columns:
                correlation = data['energy_level'].corr(data['calm_level'])
                if correlation > 0.5:
                    insights.append("⚖️ You tend to feel more energetic when you're calm - great balance!")
                elif correlation < -0.5:
                    insights.append("⚡ Your energy and calm levels are inversely related. High energy might come with some restlessness.")
            
            # Activity impacts
            if 'exercise' in data.columns and 'mood_rating' in data.columns:
                exercise_mood = data[data['exercise'] == True]['mood_rating'].mean()
                no_exercise_mood = data[data['exercise'] == False]['mood_rating'].mean()
                
                if exercise_mood > no_exercise_mood + 5:
                    insights.append("🏃‍♂️ Exercise seems to boost your mood significantly!")
            
            if 'work_stress' in data.columns and 'sentiment_score' in data.columns:
                stress_sentiment = data[data['work_stress'] == True]['sentiment_score'].mean()
                no_stress_sentiment = data[data['work_stress'] == False]['sentiment_score'].mean()
                
                if stress_sentiment < no_stress_sentiment - 0.2:
                    insights.append("💼 Work stress appears to significantly impact your emotional well-being.")
            
            # Sleep quality impact
            if 'sleep_quality' in data.columns and 'energy_level' in data.columns:
                sleep_data = data[data['sleep_quality'].isin(['Good', 'Excellent'])]
                if not sleep_data.empty:
                    good_sleep_energy = sleep_data['energy_level'].mean()
                    poor_sleep_data = data[data['sleep_quality'].isin(['Poor', 'Fair'])]
                    if not poor_sleep_data.empty:
                        poor_sleep_energy = poor_sleep_data['energy_level'].mean()
                        if good_sleep_energy > poor_sleep_energy + 10:
                            insights.append("😴 Quality sleep has a strong positive impact on your energy levels!")
            
            # Weekly patterns
            if 'timestamp' in data.columns:
                data_copy = data.copy()
                data_copy['day_of_week'] = data_copy['timestamp'].dt.dayofweek
                day_sentiment = data_copy.groupby('day_of_week')['sentiment_score'].mean()
                
                best_day = day_sentiment.idxmax()
                worst_day = day_sentiment.idxmin()
                
                day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                if day_sentiment.max() - day_sentiment.min() > 0.3:
                    insights.append(f"📅 {day_names[best_day]}s tend to be your best days, while {day_names[worst_day]}s are typically more challenging.")
            
            # Consistency insights
            sentiment_std = data['sentiment_score'].std()
            if sentiment_std < 0.3:
                insights.append("🎯 Your mood has been quite consistent lately - you seem to be in a stable emotional state.")
            elif sentiment_std > 0.7:
                insights.append("🎢 Your mood has been quite variable. Consider what factors might be causing these fluctuations.")
            
        except Exception as e:
            st.warning(f"Error generating insights: {e}")
        
        if not insights:
            insights.append("📊 Keep logging your mood to discover personalized insights about your emotional patterns!")
        
        return insights
    
    def calculate_mood_stability(self, data: pd.DataFrame) -> float:
        """Calculate mood stability score (0-100, higher is more stable)"""
        if data.empty or 'sentiment_score' not in data.columns:
            return 50.0
        
        # Calculate coefficient of variation (lower = more stable)
        mean_sentiment = data['sentiment_score'].mean()
        std_sentiment = data['sentiment_score'].std()
        
        if mean_sentiment == 0:
            return 50.0
        
        cv = abs(std_sentiment / mean_sentiment)
        
        # Convert to 0-100 scale (inverted so higher = more stable)
        stability = max(0, min(100, 100 - (cv * 50)))
        
        return stability
