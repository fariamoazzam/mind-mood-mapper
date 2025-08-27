import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Any
from mood_analyzer import MoodAnalyzer

class Visualizer:
    """Create interactive visualizations for mood data"""
    
    def __init__(self):
        """Initialize visualizer with color schemes and layouts"""
        self.mood_analyzer = MoodAnalyzer()
        
        # Color schemes
        self.color_palette = {
            'positive': '#2ecc71',
            'neutral': '#f39c12', 
            'negative': '#e74c3c',
            'energy': '#3498db',
            'calm': '#9b59b6',
            'background': '#ecf0f1',
            'text': '#2c3e50'
        }
        
        # Emotion colors for clustering
        self.cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    def create_landscape_map(self, data: pd.DataFrame, cluster_method: str = "KMeans") -> go.Figure:
        """Create the main cognitive landscape visualization"""
        if data.empty:
            return self._create_empty_plot("No data available for landscape mapping")
        
        # Perform clustering
        cluster_labels = self.mood_analyzer.perform_clustering(data, method=cluster_method)
        data_copy = data.copy()
        data_copy['cluster'] = cluster_labels
        
        # Create the main scatter plot
        fig = go.Figure()
        
        # Add scatter plot for each cluster
        unique_clusters = sorted(data_copy['cluster'].unique())
        
        for i, cluster in enumerate(unique_clusters):
            cluster_data = data_copy[data_copy['cluster'] == cluster]
            
            # Use energy and calm as primary dimensions
            x_col = 'energy_level' if 'energy_level' in cluster_data.columns else 'sentiment_score'
            y_col = 'calm_level' if 'calm_level' in cluster_data.columns else 'mood_rating'
            
            # Create hover text
            hover_text = []
            for _, row in cluster_data.iterrows():
                text = f"""
                Date: {row['timestamp'].strftime('%Y-%m-%d %H:%M')}
                Energy: {row.get('energy_level', 'N/A')}
                Calm: {row.get('calm_level', 'N/A')}
                Mood: {row.get('mood_rating', 'N/A')}
                Sentiment: {row.get('sentiment_score', 0):.2f}
                Emotion: {row.get('emotion', 'N/A')}
                """.strip()
                hover_text.append(text)
            
            # Add scatter trace
            fig.add_trace(go.Scatter(
                x=cluster_data[x_col],
                y=cluster_data[y_col],
                mode='markers',
                marker=dict(
                    size=cluster_data.get('mood_rating', [50] * len(cluster_data)) / 5 + 8,
                    color=self.cluster_colors[i % len(self.cluster_colors)],
                    opacity=0.7,
                    line=dict(width=2, color='white')
                ),
                name=f'Pattern {cluster + 1}',
                hovertext=hover_text,
                hoverinfo='text'
            ))
        
        # Update layout
        fig.update_layout(
            title="🗺️ Cognitive Landscape Map",
            xaxis_title=f"Energy Level" if x_col == 'energy_level' else "Sentiment Score",
            yaxis_title=f"Calm Level" if y_col == 'calm_level' else "Mood Rating",
            showlegend=True,
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_sentiment_distribution(self, data: pd.DataFrame) -> go.Figure:
        """Create sentiment distribution visualization"""
        if data.empty:
            return self._create_empty_plot("No data available for sentiment distribution")
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Sentiment Score Distribution', 'Emotion Categories'),
            specs=[[{"type": "xy"}],
                   [{"type": "domain"}]],
            vertical_spacing=0.15
        )
        
        # Sentiment score histogram
        if 'sentiment_score' in data.columns:
            fig.add_trace(
                go.Histogram(
                    x=data['sentiment_score'],
                    nbinsx=20,
                    marker_color=self.color_palette['energy'],
                    opacity=0.7,
                    name='Sentiment Distribution'
                ),
                row=1, col=1
            )
        
        # Emotion category pie chart
        if 'emotion' in data.columns:
            emotion_counts = data['emotion'].value_counts()
            
            colors = [self.color_palette.get(emotion, '#95a5a6') for emotion in emotion_counts.index]
            
            fig.add_trace(
                go.Pie(
                    labels=emotion_counts.index,
                    values=emotion_counts.values,
                    marker_colors=colors,
                    name='Emotions'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_pattern_analysis(self, data: pd.DataFrame) -> go.Figure:
        """Create pattern analysis based on context tags"""
        if data.empty:
            return self._create_empty_plot("No data available for pattern analysis")
        
        # Analyze mood by different factors
        factors = []
        
        # Work stress impact
        if 'work_stress' in data.columns and 'mood_rating' in data.columns:
            stress_mood = data[data['work_stress'] == True]['mood_rating'].mean()
            no_stress_mood = data[data['work_stress'] == False]['mood_rating'].mean()
            factors.append({
                'factor': 'Work Stress',
                'with_factor': stress_mood,
                'without_factor': no_stress_mood,
                'impact': stress_mood - no_stress_mood
            })
        
        # Exercise impact
        if 'exercise' in data.columns and 'mood_rating' in data.columns:
            exercise_mood = data[data['exercise'] == True]['mood_rating'].mean()
            no_exercise_mood = data[data['exercise'] == False]['mood_rating'].mean()
            factors.append({
                'factor': 'Exercise',
                'with_factor': exercise_mood,
                'without_factor': no_exercise_mood,
                'impact': exercise_mood - no_exercise_mood
            })
        
        # Social interaction impact
        if 'social_interaction' in data.columns and 'mood_rating' in data.columns:
            social_mood = data[data['social_interaction'] == True]['mood_rating'].mean()
            no_social_mood = data[data['social_interaction'] == False]['mood_rating'].mean()
            factors.append({
                'factor': 'Social Interaction',
                'with_factor': social_mood,
                'without_factor': no_social_mood,
                'impact': social_mood - no_social_mood
            })
        
        if not factors:
            return self._create_empty_plot("Not enough tagged data for pattern analysis")
        
        # Create bar chart
        fig = go.Figure()
        
        factor_names = [f['factor'] for f in factors]
        impacts = [f['impact'] for f in factors]
        
        # Color bars based on impact (positive/negative)
        bar_colors = [self.color_palette['positive'] if impact > 0 else self.color_palette['negative'] 
                     for impact in impacts]
        
        fig.add_trace(go.Bar(
            x=factor_names,
            y=impacts,
            marker_color=bar_colors,
            text=[f"{impact:+.1f}" for impact in impacts],
            textposition='outside'
        ))
        
        fig.update_layout(
            title="📊 Mood Impact by Life Factors",
            yaxis_title="Mood Impact (Rating Points)",
            xaxis_title="Life Factors",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def create_timeline_chart(self, data: pd.DataFrame, grouping: str = "day", smoothing: bool = True) -> go.Figure:
        """Create timeline visualization of mood trends"""
        if data.empty:
            return self._create_empty_plot("No data available for timeline")
        
        data_copy = data.copy()
        
        # Group data by time period
        if grouping == "day":
            data_copy['period'] = data_copy['timestamp'].dt.date
        elif grouping == "week":
            data_copy['period'] = data_copy['timestamp'].dt.to_period('W').dt.start_time
        elif grouping == "month":
            data_copy['period'] = data_copy['timestamp'].dt.to_period('M').dt.start_time
        
        # Aggregate by period
        agg_data = data_copy.groupby('period').agg({
            'mood_rating': 'mean',
            'energy_level': 'mean',
            'calm_level': 'mean',
            'sentiment_score': 'mean'
        }).reset_index()
        
        # Create timeline plot
        fig = go.Figure()
        
        # Add mood rating line
        if 'mood_rating' in agg_data.columns:
            y_values = agg_data['mood_rating']
            if smoothing and len(y_values) > 3:
                y_values = self._apply_smoothing(y_values)
            
            fig.add_trace(go.Scatter(
                x=agg_data['period'],
                y=y_values,
                mode='lines+markers',
                name='Mood Rating',
                line=dict(color=self.color_palette['positive'], width=3),
                marker=dict(size=8)
            ))
        
        # Add energy level line
        if 'energy_level' in agg_data.columns:
            y_values = agg_data['energy_level']
            if smoothing and len(y_values) > 3:
                y_values = self._apply_smoothing(y_values)
                
            fig.add_trace(go.Scatter(
                x=agg_data['period'],
                y=y_values,
                mode='lines+markers',
                name='Energy Level',
                line=dict(color=self.color_palette['energy'], width=2),
                marker=dict(size=6)
            ))
        
        # Add calm level line
        if 'calm_level' in agg_data.columns:
            y_values = agg_data['calm_level']
            if smoothing and len(y_values) > 3:
                y_values = self._apply_smoothing(y_values)
                
            fig.add_trace(go.Scatter(
                x=agg_data['period'],
                y=y_values,
                mode='lines+markers',
                name='Calm Level',
                line=dict(color=self.color_palette['calm'], width=2),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title=f"📈 Mood Timeline ({grouping.title()} View)",
            xaxis_title="Time",
            yaxis_title="Level (0-100)",
            height=500,
            showlegend=True,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Add grid
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        
        return fig
    
    def create_correlation_matrix(self, data: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap"""
        if data.empty:
            return self._create_empty_plot("No data available for correlation analysis")
        
        # Select numerical columns for correlation
        numerical_cols = ['mood_rating', 'energy_level', 'calm_level', 'sentiment_score']
        available_cols = [col for col in numerical_cols if col in data.columns]
        
        if len(available_cols) < 2:
            return self._create_empty_plot("Not enough numerical data for correlation analysis")
        
        # Calculate correlation matrix
        corr_matrix = data[available_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 12},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="🔗 Mood Dimensions Correlation",
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def create_weekly_pattern(self, data: pd.DataFrame) -> go.Figure:
        """Create weekly pattern analysis"""
        if data.empty:
            return self._create_empty_plot("No data available for weekly patterns")
        
        data_copy = data.copy()
        data_copy['day_of_week'] = data_copy['timestamp'].dt.day_name()
        data_copy['hour'] = data_copy['timestamp'].dt.hour
        
        # Calculate average mood by day of week
        daily_mood = data_copy.groupby('day_of_week')['mood_rating'].mean().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ])
        
        # Create polar bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Barpolar(
            r=daily_mood.values,
            theta=daily_mood.index,
            marker_color=daily_mood.values,
            marker_colorscale='Viridis',
            marker_line_color="black",
            marker_line_width=2,
            opacity=0.8
        ))
        
        fig.update_layout(
            title="📅 Weekly Mood Pattern",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100]),
                angularaxis=dict(direction='clockwise')
            ),
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        return fig
    
    def _apply_smoothing(self, values: pd.Series, window: int = 3) -> pd.Series:
        """Apply smoothing to time series data"""
        return values.rolling(window=window, min_periods=1, center=True).mean()
    
    def _create_empty_plot(self, message: str) -> go.Figure:
        """Create empty plot with message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=400
        )
        
        return fig
