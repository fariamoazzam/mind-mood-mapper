import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import re

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    
    # Initialize mood analyzer state
    if 'mood_analyzer_loaded' not in st.session_state:
        st.session_state.mood_analyzer_loaded = False
    
    # Initialize form state
    if 'form_cleared' not in st.session_state:
        st.session_state.form_cleared = False
    
    # Initialize data cache
    if 'data_cache' not in st.session_state:
        st.session_state.data_cache = None
    
    # Initialize cache timestamp
    if 'cache_timestamp' not in st.session_state:
        st.session_state.cache_timestamp = datetime.now()

def format_date(date_obj: datetime) -> str:
    """Format datetime object for display"""
    if pd.isna(date_obj):
        return "Unknown"
    
    if isinstance(date_obj, str):
        try:
            date_obj = pd.to_datetime(date_obj)
        except:
            return date_obj
    
    now = datetime.now()
    diff = now - date_obj
    
    if diff.days == 0:
        return "Today"
    elif diff.days == 1:
        return "Yesterday"
    elif diff.days < 7:
        return f"{diff.days} days ago"
    else:
        return date_obj.strftime("%Y-%m-%d")

def validate_mood_entry(entry_data: Dict[str, Any]) -> Dict[str, str]:
    """Validate mood entry data and return errors"""
    errors = {}
    
    # Required fields
    if not entry_data.get('journal_text', '').strip():
        errors['journal_text'] = "Journal text is required"
    
    # Validate numerical ranges
    for field, min_val, max_val in [
        ('energy_level', 0, 100),
        ('calm_level', 0, 100),
        ('mood_rating', 0, 100)
    ]:
        value = entry_data.get(field)
        if value is None or not isinstance(value, (int, float)):
            errors[field] = f"{field} must be a number"
        elif not min_val <= value <= max_val:
            errors[field] = f"{field} must be between {min_val} and {max_val}"
    
    # Validate timestamp
    timestamp = entry_data.get('timestamp')
    if not timestamp:
        errors['timestamp'] = "Timestamp is required"
    elif isinstance(timestamp, str):
        try:
            pd.to_datetime(timestamp)
        except:
            errors['timestamp'] = "Invalid timestamp format"
    
    return errors

def clean_text_input(text: str) -> str:
    """Clean and sanitize text input"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove potentially harmful characters but keep basic punctuation
    text = re.sub(r'[<>{}[\]|\\`]', '', text)
    
    # Limit length
    if len(text) > 1000:
        text = text[:1000] + "..."
    
    return text

def calculate_mood_trends(data: pd.DataFrame, days: int = 7) -> Dict[str, float]:
    """Calculate mood trends over specified period"""
    if data.empty or len(data) < 2:
        return {}
    
    # Get recent data
    recent_data = data.tail(days) if days > 0 else data
    
    trends = {}
    
    # Calculate trends for numerical columns
    for column in ['mood_rating', 'energy_level', 'calm_level', 'sentiment_score']:
        if column in recent_data.columns:
            values = recent_data[column].dropna()
            if len(values) >= 2:
                # Simple linear trend
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                trends[column] = slope
    
    return trends

def get_mood_insights(data: pd.DataFrame) -> List[str]:
    """Generate quick mood insights from data"""
    insights = []
    
    if data.empty:
        return ["Start logging your mood to see insights!"]
    
    # Recent trend
    trends = calculate_mood_trends(data, days=7)
    
    if 'mood_rating' in trends:
        if trends['mood_rating'] > 1:
            insights.append("📈 Your mood has been trending upward this week!")
        elif trends['mood_rating'] < -1:
            insights.append("📉 Your mood has been declining this week. Consider what might help.")
    
    # Consistency check
    if 'mood_rating' in data.columns and len(data) >= 5:
        recent_std = data.tail(7)['mood_rating'].std()
        if recent_std < 10:
            insights.append("🎯 Your mood has been quite stable recently.")
        elif recent_std > 25:
            insights.append("🎢 Your mood has been quite variable lately.")
    
    # Activity correlation
    if 'exercise' in data.columns and 'mood_rating' in data.columns:
        exercise_data = data[data['exercise'] == True]
        no_exercise_data = data[data['exercise'] == False]
        
        if len(exercise_data) > 0 and len(no_exercise_data) > 0:
            exercise_avg = exercise_data['mood_rating'].mean()
            no_exercise_avg = no_exercise_data['mood_rating'].mean()
            
            if exercise_avg > no_exercise_avg + 5:
                insights.append("🏃‍♂️ Exercise seems to boost your mood!")
    
    if not insights:
        insights.append("📊 Keep logging to discover patterns in your mood!")
    
    return insights[:3]  # Limit to 3 insights

def export_mood_summary(data: pd.DataFrame) -> str:
    """Generate a text summary of mood data for export"""
    if data.empty:
        return "No mood data available."
    
    summary = []
    summary.append("🧠 MIND MOOD MAPPER - DATA SUMMARY")
    summary.append("=" * 40)
    summary.append("")
    
    # Basic stats
    summary.append(f"📊 OVERVIEW")
    summary.append(f"Total entries: {len(data)}")
    summary.append(f"Date range: {data['timestamp'].min().strftime('%Y-%m-%d')} to {data['timestamp'].max().strftime('%Y-%m-%d')}")
    summary.append("")
    
    # Averages
    if 'mood_rating' in data.columns:
        summary.append(f"📈 AVERAGES")
        summary.append(f"Average mood: {data['mood_rating'].mean():.1f}/100")
        
        if 'energy_level' in data.columns:
            summary.append(f"Average energy: {data['energy_level'].mean():.1f}/100")
        
        if 'calm_level' in data.columns:
            summary.append(f"Average calm: {data['calm_level'].mean():.1f}/100")
        
        if 'sentiment_score' in data.columns:
            summary.append(f"Average sentiment: {data['sentiment_score'].mean():.2f}")
        
        summary.append("")
    
    # Recent trends
    trends = calculate_mood_trends(data, days=7)
    if trends:
        summary.append(f"📉📈 RECENT TRENDS (7 days)")
        for metric, trend in trends.items():
            direction = "↗️" if trend > 0.5 else "↘️" if trend < -0.5 else "➡️"
            summary.append(f"{metric}: {direction} {trend:+.1f}")
        summary.append("")
    
    # Insights
    insights = get_mood_insights(data)
    summary.append(f"💡 INSIGHTS")
    for insight in insights:
        summary.append(f"• {insight}")
    summary.append("")
    
    summary.append("Generated by Mind Mood Mapper")
    summary.append(f"Export date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    return "\n".join(summary)

def get_color_for_sentiment(sentiment: float) -> str:
    """Get color code based on sentiment score"""
    if sentiment >= 0.1:
        return "#2ecc71"  # Green for positive
    elif sentiment <= -0.1:
        return "#e74c3c"  # Red for negative
    else:
        return "#f39c12"  # Orange for neutral

def format_sentiment_label(sentiment: float) -> str:
    """Format sentiment score as readable label"""
    if sentiment >= 0.5:
        return "Very Positive"
    elif sentiment >= 0.1:
        return "Positive"
    elif sentiment <= -0.5:
        return "Very Negative"
    elif sentiment <= -0.1:
        return "Negative"
    else:
        return "Neutral"

def calculate_wellness_score(data: pd.DataFrame, days: int = 30) -> Dict[str, Any]:
    """Calculate overall wellness score based on multiple factors"""
    if data.empty:
        return {"score": 50, "components": {}, "recommendation": "Start logging to track wellness"}
    
    recent_data = data.tail(days) if days > 0 else data
    
    components = {}
    
    # Mood stability (0-25 points)
    if 'mood_rating' in recent_data.columns:
        mood_avg = recent_data['mood_rating'].mean()
        mood_std = recent_data['mood_rating'].std()
        stability_score = max(0, 25 - (mood_std * 0.5))  # Lower std = higher stability
        components['mood_stability'] = min(25, stability_score)
    
    # Average mood level (0-25 points)
    if 'mood_rating' in recent_data.columns:
        mood_score = (recent_data['mood_rating'].mean() / 100) * 25
        components['mood_level'] = mood_score
    
    # Energy-calm balance (0-25 points)
    if 'energy_level' in recent_data.columns and 'calm_level' in recent_data.columns:
        energy_avg = recent_data['energy_level'].mean()
        calm_avg = recent_data['calm_level'].mean()
        balance_score = 25 - (abs(energy_avg - calm_avg) * 0.25)  # Penalty for imbalance
        components['energy_calm_balance'] = max(0, balance_score)
    
    # Sentiment trend (0-25 points)
    if 'sentiment_score' in recent_data.columns:
        sentiment_avg = recent_data['sentiment_score'].mean()
        # Convert sentiment (-1 to 1) to 0-25 scale
        sentiment_score = ((sentiment_avg + 1) / 2) * 25
        components['sentiment'] = sentiment_score
    
    # Calculate total score
    total_score = sum(components.values())
    
    # Generate recommendation
    if total_score >= 80:
        recommendation = "Excellent wellness! Keep up the great work! 🌟"
    elif total_score >= 60:
        recommendation = "Good wellness overall. Focus on areas for improvement. 👍"
    elif total_score >= 40:
        recommendation = "Moderate wellness. Consider lifestyle changes. 🔄"
    else:
        recommendation = "Wellness needs attention. Consider professional support. 💙"
    
    return {
        "score": round(total_score, 1),
        "components": components,
        "recommendation": recommendation
    }

# Error handling decorators
def handle_data_errors(func):
    """Decorator to handle common data processing errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Data processing error in {func.__name__}: {str(e)}")
            return None
    return wrapper

def safe_numeric_conversion(value, default=0.0):
    """Safely convert value to numeric, return default if fails"""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default
