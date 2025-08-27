import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import os
from typing import Dict, Any, Optional
import json

class DataManager:
    """Handle data persistence and management for mood entries"""
    
    def __init__(self, data_file: str = "mood_data.csv"):
        """Initialize data manager with file path"""
        self.data_file = data_file
        self.backup_file = f"mood_data_backup_{datetime.now().strftime('%Y%m%d')}.csv"
        
        # Ensure data file exists
        self._ensure_data_file_exists()
    
    def _ensure_data_file_exists(self):
        """Create data file with headers if it doesn't exist"""
        if not os.path.exists(self.data_file):
            # Create empty DataFrame with expected columns
            empty_df = pd.DataFrame(columns=[
                'timestamp', 'journal_text', 'energy_level', 'calm_level', 
                'mood_rating', 'work_stress', 'social_interaction', 'exercise',
                'sleep_quality', 'weather', 'sentiment_score', 'pos', 'neu', 
                'neg', 'emotion', 'confidence'
            ])
            empty_df.to_csv(self.data_file, index=False)
    
    def save_entry(self, entry_data: Dict[str, Any]) -> bool:
        """Save a new mood entry to the data file"""
        try:
            # Convert entry to DataFrame row
            entry_df = pd.DataFrame([entry_data])
            
            # Load existing data
            existing_data = self.load_data()
            
            # Append new entry
            if existing_data.empty:
                combined_data = entry_df
            else:
                combined_data = pd.concat([existing_data, entry_df], ignore_index=True)
            
            # Save to file
            combined_data.to_csv(self.data_file, index=False)
            
            # Create backup periodically
            if len(combined_data) % 10 == 0:  # Backup every 10 entries
                self._create_backup()
            
            return True
            
        except Exception as e:
            st.error(f"Error saving entry: {e}")
            return False
    
    def load_data(self) -> pd.DataFrame:
        """Load mood data from file"""
        try:
            if not os.path.exists(self.data_file):
                return pd.DataFrame()
            
            df = pd.read_csv(self.data_file)
            
            if df.empty:
                return df
            
            # Convert timestamp column to datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure boolean columns are properly typed
            boolean_cols = ['work_stress', 'social_interaction', 'exercise']
            for col in boolean_cols:
                if col in df.columns:
                    df[col] = df[col].astype(bool)
            
            # Ensure numerical columns are properly typed
            numerical_cols = ['energy_level', 'calm_level', 'mood_rating', 
                            'sentiment_score', 'pos', 'neu', 'neg', 'confidence']
            for col in numerical_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def get_data_in_range(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get mood data within a specific date range"""
        data = self.load_data()
        
        if data.empty or 'timestamp' not in data.columns:
            return pd.DataFrame()
        
        mask = (data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)
        return data.loc[mask]
    
    def get_recent_data(self, days: int = 7) -> pd.DataFrame:
        """Get mood data from the last N days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.get_data_in_range(start_date, end_date)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the mood data"""
        data = self.load_data()
        
        if data.empty:
            return {}
        
        stats = {
            'total_entries': len(data),
            'date_range': {
                'start': data['timestamp'].min(),
                'end': data['timestamp'].max()
            },
            'average_mood': data['mood_rating'].mean() if 'mood_rating' in data.columns else None,
            'average_energy': data['energy_level'].mean() if 'energy_level' in data.columns else None,
            'average_calm': data['calm_level'].mean() if 'calm_level' in data.columns else None,
            'average_sentiment': data['sentiment_score'].mean() if 'sentiment_score' in data.columns else None,
        }
        
        # Add frequency statistics
        if 'timestamp' in data.columns:
            dates = data['timestamp'].dt.date
            unique_dates = dates.nunique()
            total_days = (data['timestamp'].max() - data['timestamp'].min()).days + 1
            
            stats['logging_frequency'] = len(data) / total_days if total_days > 0 else 0
            stats['days_logged'] = unique_dates
            stats['total_days'] = total_days
        
        return stats
    
    def get_logging_streak(self, data: Optional[pd.DataFrame] = None) -> int:
        """Calculate the current logging streak in days"""
        if data is None:
            data = self.load_data()
        
        if data.empty or 'timestamp' not in data.columns:
            return 0
        
        # Get unique dates
        dates = data['timestamp'].dt.date.unique()
        dates = sorted(dates, reverse=True)  # Most recent first
        
        if not dates:
            return 0
        
        # Check if logged today
        today = datetime.now().date()
        if dates[0] != today and dates[0] != today - timedelta(days=1):
            return 0
        
        # Count consecutive days
        streak = 0
        current_date = dates[0]
        expected_date = today
        
        for date in dates:
            if date == expected_date or date == expected_date - timedelta(days=1):
                streak += 1
                expected_date = date - timedelta(days=1)
            else:
                break
        
        return streak
    
    def _create_backup(self):
        """Create a backup of the current data file"""
        try:
            if os.path.exists(self.data_file):
                data = self.load_data()
                data.to_csv(self.backup_file, index=False)
        except Exception as e:
            st.warning(f"Could not create backup: {e}")
    
    def clear_all_data(self):
        """Clear all mood data (with backup)"""
        try:
            # Create backup before clearing
            self._create_backup()
            
            # Clear the file
            empty_df = pd.DataFrame(columns=[
                'timestamp', 'journal_text', 'energy_level', 'calm_level', 
                'mood_rating', 'work_stress', 'social_interaction', 'exercise',
                'sleep_quality', 'weather', 'sentiment_score', 'pos', 'neu', 
                'neg', 'emotion', 'confidence'
            ])
            empty_df.to_csv(self.data_file, index=False)
            
            return True
            
        except Exception as e:
            st.error(f"Error clearing data: {e}")
            return False
    
    def export_data(self, format: str = 'csv') -> Optional[str]:
        """Export data in specified format"""
        data = self.load_data()
        
        if data.empty:
            return None
        
        try:
            if format.lower() == 'csv':
                return data.to_csv(index=False)
            elif format.lower() == 'json':
                return data.to_json(orient='records', date_format='iso', indent=2)
            else:
                st.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            st.error(f"Error exporting data: {e}")
            return None
    
    def import_data(self, file_content: str, format: str = 'csv') -> bool:
        """Import data from file content"""
        try:
            if format.lower() == 'csv':
                # Parse CSV content
                from io import StringIO
                df = pd.read_csv(StringIO(file_content))
            elif format.lower() == 'json':
                # Parse JSON content
                data_list = json.loads(file_content)
                df = pd.DataFrame(data_list)
            else:
                st.error(f"Unsupported import format: {format}")
                return False
            
            # Validate required columns
            required_cols = ['timestamp', 'journal_text', 'mood_rating']
            if not all(col in df.columns for col in required_cols):
                st.error("Import file missing required columns")
                return False
            
            # Load existing data and append
            existing_data = self.load_data()
            
            if existing_data.empty:
                combined_data = df
            else:
                combined_data = pd.concat([existing_data, df], ignore_index=True)
            
            # Remove duplicates based on timestamp and journal_text
            combined_data = combined_data.drop_duplicates(
                subset=['timestamp', 'journal_text'], 
                keep='first'
            )
            
            # Save combined data
            combined_data.to_csv(self.data_file, index=False)
            
            return True
            
        except Exception as e:
            st.error(f"Error importing data: {e}")
            return False
