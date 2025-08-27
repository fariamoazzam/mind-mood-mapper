# Overview

Mind Mood Mapper is a Streamlit-based web application that transforms personal mood and mental state data into interactive visualizations. The application enables users to log daily mood entries through journal text and quantitative metrics, then uses AI-powered sentiment analysis to create visual patterns and insights about emotional trends over time. The core concept is to make inner mental states visible through data visualization, helping users understand their mood patterns and triggers.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses Streamlit as the primary web framework, providing a multi-page interface with four main views: mood logging, cognitive landscape visualization, timeline analysis, and data export. The interface employs a sidebar navigation system and leverages Streamlit's session state management for component persistence and caching.

## Core Components
The system follows a modular architecture with four main components:

- **MoodAnalyzer**: Handles AI-powered sentiment analysis using both HuggingFace transformers (RoBERTa model) and NLTK's VADER sentiment analyzer. Includes clustering capabilities using KMeans and DBSCAN algorithms for pattern detection.

- **DataManager**: Manages data persistence using CSV file storage with automatic backup functionality. Handles mood entry validation, data loading, and export operations.

- **Visualizer**: Creates interactive visualizations using Plotly, including scatter plots for cognitive landscapes, timeline charts, and clustering visualizations. Implements color-coded emotion mapping and pattern recognition displays.

- **Utils**: Provides utility functions for session state initialization, date formatting, and data validation.

## Data Storage
The application uses a file-based storage approach with CSV format for simplicity and portability. Mood entries include timestamp, journal text, quantitative ratings (energy, calm, mood, work stress, social interaction, exercise, sleep quality, weather), and computed sentiment metrics (sentiment scores, emotion classification, confidence levels).

## AI Integration
Sentiment analysis is performed using a hybrid approach combining HuggingFace's pre-trained RoBERTa model for robust sentiment classification and NLTK's VADER for quick lexicon-based analysis. The system includes fallback mechanisms and error handling for model loading failures.

## Visualization Strategy
Interactive visualizations are built with Plotly to create "cognitive landscapes" where mood data points are plotted in multi-dimensional space. Clustering algorithms group similar mood states, and color coding represents different emotional categories and intensity levels.

# External Dependencies

## AI/ML Libraries
- **HuggingFace Transformers**: Pre-trained RoBERTa model for advanced sentiment analysis
- **NLTK**: VADER sentiment analyzer and text processing utilities
- **scikit-learn**: Clustering algorithms (KMeans, DBSCAN) and data preprocessing tools

## Visualization and Data Processing
- **Plotly**: Interactive plotting library for creating cognitive landscape visualizations
- **Streamlit**: Web application framework for the user interface
- **Pandas**: Data manipulation and CSV file handling
- **NumPy**: Numerical computations and array operations

## Text Processing
- **TfidfVectorizer**: Text feature extraction for clustering analysis
- **Regular expressions**: Text cleaning and validation utilities

The application is designed to run locally without requiring external databases or cloud services, making it privacy-focused and self-contained.