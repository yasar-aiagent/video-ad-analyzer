# Video Ad Analyzer

## Overview
An AI-powered video ad analysis tool that evaluates Meta video ads across 200+ metrics including visual elements, production quality, branding, content analysis, and performance metrics.

## Project Structure
- `app.py` - Main Streamlit application with video analysis functionality
- `.streamlit/config.toml` - Streamlit server configuration
- `pyproject.toml` - Python dependencies

## Features

### Video Input
- Upload video files (MP4, MOV, AVI, MKV formats)
- Video URL input for direct analysis from web URLs

### AI-Powered Analysis
Using OpenAI GPT-4o Vision API to analyze video frames across 200+ metrics:
- **Actor & Human Elements** (30 metrics) - Actor presence, demographics, expressions, movement
- **Color & Visual Style** (25 metrics) - Color schemes, contrast, consistency, mood
- **Video Production** (35 metrics) - Quality, lighting, camera work, effects, transitions
- **Text & Typography** (35 metrics) - Overlays, captions, fonts, messaging
- **Branding & Logo** (16 metrics) - Logo placement, brand consistency
- **Call-to-Action** (18 metrics) - CTA presence, visibility, effectiveness
- **Audio Elements** (16 metrics) - Music, voiceover, sound effects (estimated from visuals)
- **Content & Messaging** (37 metrics) - Product visibility, storytelling, emotional appeal
- **Engagement Elements** (15 metrics) - Scroll-stopping power, pacing, shareability
- **Platform Optimization** (10 metrics) - Mobile optimization, format compatibility

### Meta Performance Metrics
- Manual entry for ad performance data
- CSV upload from Meta Ads exports
- Metrics: Impressions, Clicks, Spend, CTR, CPM, CPC, ROAS, Leads, Purchases

### Dashboard & Export
- Interactive charts with Plotly (radar charts, bar charts)
- 13 category tabs for organized metric viewing
- Export to CSV and JSON formats
- Visual scoring system (0-100 scale)

## Dependencies
- streamlit - Web application framework
- openai - AI analysis via GPT-5 Vision
- opencv-python-headless - Video frame extraction
- pandas - Data manipulation and CSV handling
- plotly - Interactive visualizations
- pillow - Image processing
- requests - URL video downloading

## System Dependencies
- libGL, mesa - Required for OpenCV video processing

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Recent Changes
- December 11, 2025: Initial implementation of video ad analyzer
  - Added video URL input option
  - Added Meta performance metrics input (CSV and manual entry)
  - Integrated performance metrics into dashboard
  - Added Performance tab to results view
  - Fixed OpenCV system dependencies

## User Preferences
- None recorded yet

## Architecture Decisions
- Using OpenAI GPT-5 for multimodal video frame analysis
- Extracting 5 key frames by default for balanced accuracy/speed
- Results organized by 13 categories with visual charts
- Export functionality for CSV and JSON formats
- Session state for persisting analysis results within session
