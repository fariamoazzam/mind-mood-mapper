import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from mood_analyzer import MoodAnalyzer
from data_manager import DataManager
from visualizer import Visualizer
from utils import initialize_session_state, format_date

# Configure page
st.set_page_config(
    page_title="Mind Mood Mapper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    
    # Initialize session state and components
    initialize_session_state()
    
    if 'mood_analyzer' not in st.session_state:
        with st.spinner("Initializing AI mood analyzer..."):
            st.session_state.mood_analyzer = MoodAnalyzer()
    
    if 'data_manager' not in st.session_state:
        st.session_state.data_manager = DataManager()
    
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()
    
    # Header
    st.title("🧠 Mind Mood Mapper")
    st.markdown("*Transform your inner states into visible patterns*")
    
    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio(
            "Choose a view:",
            ["📝 Log Mood", "📊 Cognitive Landscape", "📈 Timeline Analysis", "📋 Data Export"],
            index=0
        )
        
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        # Load and display quick stats
        data = st.session_state.data_manager.load_data()
        if not data.empty:
            total_entries = len(data)
            days_tracked = (data['timestamp'].max() - data['timestamp'].min()).days + 1
            
            # Check if sentiment_score column exists
            if 'sentiment_score' in data.columns:
                avg_sentiment = data['sentiment_score'].mean()
                st.metric("Avg Sentiment", f"{avg_sentiment:.2f}")
            
            st.metric("Total Entries", total_entries)
            st.metric("Days Tracked", days_tracked)
        else:
            st.info("No data yet. Start logging your mood!")
    
    # Main content based on selected page
    if page == "📝 Log Mood":
        show_mood_logging()
    elif page == "📊 Cognitive Landscape":
        show_cognitive_landscape()
    elif page == "📈 Timeline Analysis":
        show_timeline_analysis()
    elif page == "📋 Data Export":
        show_data_export()

def show_mood_logging():
    """Display mood logging interface"""
    st.header("Log Your Current Mood")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Journal Entry")
        journal_text = st.text_area(
            "How are you feeling right now? Describe your thoughts, emotions, or experiences:",
            height=150,
            placeholder="I'm feeling energetic today after a good night's sleep. Work was challenging but rewarding..."
        )
        
        st.subheader("Mood Dimensions")
        energy_level = st.slider("Energy Level", 0, 100, 50, help="Low = Tired/Lethargic, High = Energetic/Alert")
        calm_level = st.slider("Calm Level", 0, 100, 50, help="Low = Anxious/Stressed, High = Peaceful/Relaxed")
        mood_rating = st.slider("Overall Mood", 0, 100, 50, help="Low = Negative, High = Positive")
        
        # Optional tags
        st.subheader("Context Tags (Optional)")
        col_tags1, col_tags2 = st.columns(2)
        
        with col_tags1:
            work_stress = st.checkbox("Work Stress")
            social_interaction = st.checkbox("Social Interaction")
            exercise = st.checkbox("Exercise")
            
        with col_tags2:
            sleep_quality = st.selectbox("Sleep Quality", ["Not specified", "Poor", "Fair", "Good", "Excellent"])
            weather = st.selectbox("Weather", ["Not specified", "Sunny", "Cloudy", "Rainy", "Stormy"])
    
    with col2:
        st.subheader("Preview")
        
        if journal_text:
            # Get real-time sentiment analysis
            with st.spinner("Analyzing sentiment..."):
                sentiment_result = st.session_state.mood_analyzer.analyze_sentiment(journal_text)
            
            st.metric("Sentiment Score", f"{sentiment_result['compound']:.2f}")
            st.metric("Emotion", sentiment_result['emotion'])
            
            # Show sentiment breakdown
            st.write("**Sentiment Breakdown:**")
            st.write(f"Positive: {sentiment_result['pos']:.2f}")
            st.write(f"Neutral: {sentiment_result['neu']:.2f}")
            st.write(f"Negative: {sentiment_result['neg']:.2f}")
        else:
            st.info("Enter journal text to see sentiment analysis")
    
    # Submit button
    if st.button("💾 Save Mood Entry", type="primary", use_container_width=True):
        if journal_text.strip():
            # Prepare entry data
            entry_data = {
                'timestamp': datetime.now(),
                'journal_text': journal_text,
                'energy_level': energy_level,
                'calm_level': calm_level,
                'mood_rating': mood_rating,
                'work_stress': work_stress,
                'social_interaction': social_interaction,
                'exercise': exercise,
                'sleep_quality': sleep_quality,
                'weather': weather
            }
            
            # Analyze sentiment and add to entry
            sentiment_result = st.session_state.mood_analyzer.analyze_sentiment(journal_text)
            entry_data.update(sentiment_result)
            
            # Save entry
            success = st.session_state.data_manager.save_entry(entry_data)
            
            if success:
                st.success("✅ Mood entry saved successfully!")
                st.balloons()
                # Clear form
                st.rerun()
            else:
                st.error("❌ Failed to save mood entry. Please try again.")
        else:
            st.warning("⚠️ Please enter some journal text before saving.")

def show_cognitive_landscape():
    """Display cognitive landscape visualization"""
    st.header("🗺️ Cognitive Landscape Map")
    st.markdown("*Explore the hidden patterns in your emotional states*")
    
    # Load data
    data = st.session_state.data_manager.load_data()
    
    if data.empty:
        st.info("No mood entries found. Start by logging your mood!")
        return
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days_back = st.selectbox("Time Period", [7, 14, 30, 90, 365], index=2)
    
    with col2:
        min_entries = st.slider("Min entries for clustering", 3, 20, 5)
    
    with col3:
        cluster_method = st.selectbox("Clustering Method", ["KMeans", "DBSCAN"])
    
    # Filter data by time period
    cutoff_date = datetime.now() - timedelta(days=days_back)
    filtered_data = data[data['timestamp'] >= cutoff_date]
    
    if len(filtered_data) < min_entries:
        st.warning(f"Need at least {min_entries} entries for clustering. You have {len(filtered_data)} entries in the selected time period.")
        return
    
    # Generate visualizations
    st.subheader("Emotional Dimension Map")
    
    # Create the main landscape plot
    landscape_fig = st.session_state.visualizer.create_landscape_map(
        filtered_data, cluster_method=cluster_method
    )
    st.plotly_chart(landscape_fig, use_container_width=True)
    
    # Additional visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sentiment Distribution")
        sentiment_fig = st.session_state.visualizer.create_sentiment_distribution(filtered_data)
        st.plotly_chart(sentiment_fig, use_container_width=True)
    
    with col2:
        st.subheader("Mood Patterns by Tags")
        patterns_fig = st.session_state.visualizer.create_pattern_analysis(filtered_data)
        st.plotly_chart(patterns_fig, use_container_width=True)
    
    # Insights section
    st.subheader("🔍 Pattern Insights")
    insights = st.session_state.mood_analyzer.generate_insights(filtered_data)
    
    for insight in insights:
        st.info(insight)

def show_timeline_analysis():
    """Display timeline analysis and trends"""
    st.header("📈 Timeline Analysis")
    st.markdown("*Track your mood evolution over time*")
    
    # Load data
    data = st.session_state.data_manager.load_data()
    
    if data.empty:
        st.info("No mood entries found. Start by logging your mood!")
        return
    
    # Time period selector
    col1, col2 = st.columns(2)
    
    with col1:
        time_grouping = st.selectbox("Group by", ["Day", "Week", "Month"])
    
    with col2:
        smoothing = st.checkbox("Apply smoothing", value=True)
    
    # Main timeline chart
    timeline_fig = st.session_state.visualizer.create_timeline_chart(
        data, grouping=time_grouping.lower(), smoothing=smoothing
    )
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Detailed metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Correlation Matrix")
        correlation_fig = st.session_state.visualizer.create_correlation_matrix(data)
        st.plotly_chart(correlation_fig, use_container_width=True)
    
    with col2:
        st.subheader("Weekly Patterns")
        weekly_fig = st.session_state.visualizer.create_weekly_pattern(data)
        st.plotly_chart(weekly_fig, use_container_width=True)
    
    # Statistics summary
    st.subheader("📊 Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    recent_data = data.tail(7)  # Last 7 entries
    
    with col1:
        avg_sentiment = recent_data['sentiment_score'].mean()
        st.metric("Avg Sentiment (7 days)", f"{avg_sentiment:.2f}")
    
    with col2:
        avg_energy = recent_data['energy_level'].mean()
        st.metric("Avg Energy (7 days)", f"{avg_energy:.1f}")
    
    with col3:
        avg_calm = recent_data['calm_level'].mean()
        st.metric("Avg Calm (7 days)", f"{avg_calm:.1f}")
    
    with col4:
        streak = st.session_state.data_manager.get_logging_streak(data)
        st.metric("Logging Streak", f"{streak} days")

def show_data_export():
    """Display data export options"""
    st.header("📋 Data Export & Management")
    st.markdown("*Export your data or manage your mood history*")
    
    # Load data
    data = st.session_state.data_manager.load_data()
    
    if data.empty:
        st.info("No mood entries found. Start by logging your mood!")
        return
    
    # Data overview
    st.subheader("Data Overview")
    st.write(f"Total entries: {len(data)}")
    st.write(f"Date range: {data['timestamp'].min().strftime('%Y-%m-%d')} to {data['timestamp'].max().strftime('%Y-%m-%d')}")
    
    # Preview data
    st.subheader("Data Preview")
    preview_data = data.copy()
    preview_data['timestamp'] = preview_data['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(preview_data.head(10), use_container_width=True)
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Export Data")
        
        # CSV export
        csv_data = data.to_csv(index=False)
        st.download_button(
            label="📄 Download as CSV",
            data=csv_data,
            file_name=f"mood_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # JSON export
        json_data = data.to_json(orient='records', date_format='iso')
        st.download_button(
            label="📋 Download as JSON",
            data=json_data,
            file_name=f"mood_data_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )
    
    with col2:
        st.subheader("Data Management")
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            if st.button("⚠️ Confirm Delete", type="secondary"):
                st.session_state.data_manager.clear_all_data()
                st.success("All data cleared!")
                st.rerun()
    
    # Privacy notice
    st.subheader("🔒 Privacy & Security")
    st.info("""
    **Your data stays private:**
    - All mood data is stored locally on this device
    - No data is sent to external servers
    - You have full control over your information
    - Export your data anytime for backup or migration
    """)

if __name__ == "__main__":
    main()
