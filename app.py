from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

import streamlit as st

# Set page config FIRST
st.set_page_config(
    page_title="Video Ad Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Get API key - Try Streamlit secrets first, then environment
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Import other libraries
import cv2
import base64
import json
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from openai import OpenAI
from PIL import Image
import io
import requests
from urllib.parse import urlparse
import yt_dlp

METRIC_CATEGORIES = {
    "Actor & Human Elements": [
        "Has Human Actor", "Male Actor Present", "Female Actor Present",
        "Animated Character", "Celebrity/Influencer", "Actor Age Group",
        "Actor Body Type", "Actor Diversity", "Number of Actors",
        "Actor Speaking", "Actor Demonstrating Product", "Facial Expression",
        "Direct Eye Contact", "Hand Gestures", "Actor Movement",
        "Clothing Style", "Authenticity Level", "Energy Level",
        "Relatability Score", "Actor Screen Time", "Actor Positioning",
        "Close Up Shots", "Full Body Shots", "Group Shots",
        "Product Interaction", "Emotion Range", "Wardrobe Changes",
        "Props Used", "Setting Changes"
    ],
    "Color & Visual Style": [
        "Color Contrast Level", "Dark Color Scheme", "Light Color Scheme",
        "Vibrant Colors", "Pastel Colors", "Monochrome Style",
        "Warm Colors", "Cool Colors", "Color Consistency",
        "Background Color Intensity", "Brand Colors Used", "Seasonal Colors",
        "Cultural Color Significance", "Gradient Usage", "B&W Filter",
        "Neon Colors", "Earth Tones", "Metallic Colors",
        "Color Saturation", "Color Temperature", "Dominant Color",
        "Secondary Color", "Accent Color", "Color Harmony", "Color Mood"
    ],
    "Video Production": [
        "Aspect Ratio", "Video Quality", "Lighting Quality", "Lighting Type",
        "Lighting Direction", "Background Type", "Background Blur",
        "Background Complexity", "Focus Quality", "Depth of Field",
        "Camera Movement", "Camera Angle", "Shot Composition",
        "Transition Style", "Transition Frequency", "Speed Changes",
        "Slow Motion", "Fast Motion", "Timelapse", "Color Filter",
        "Color Grading Style", "Special Effects", "VFX Type",
        "Screen Recording", "UGC Style", "B-Roll Footage",
        "Stock Footage", "Split Screen", "Before/After", "Picture-in-Picture",
        "Green Screen", "Motion Graphics", "3D Elements", "Lens Flare",
        "Film Grain", "Vignette Effect"
    ],
    "Text & Typography": [
        "Text Overlay Amount", "Subtitles/Captions", "Auto Captions Style",
        "Text Size", "Font Style", "Font Weight", "Text Animation",
        "Text Animation Type", "Headline Present", "Subheadline Present",
        "Caption Length", "Emojis Used", "Numbers/Stats Shown",
        "Percentage Shown", "Bullet Points", "Question Asked",
        "Urgency Language", "Scarcity Language", "Benefits Highlighted",
        "Features Listed", "Problem Stated", "Solution Presented",
        "Social Proof", "Testimonial Quote", "Price/Discount Shown",
        "Original Price Shown", "Savings Highlighted", "Free Shipping",
        "Guarantee Mentioned", "Text Contrast", "Text Background Overlay",
        "Text Shadow", "Text Outline", "Text Position", "Text Alignment"
    ],
    "Branding & Logo": [
        "Logo Placement", "Logo Position", "Logo Style", "Logo Size",
        "Logo Duration", "Logo Animation", "Brand Name Text",
        "Brand Tagline", "Brand Colors Consistency", "Brand Font Consistency",
        "Brand Consistency Score", "Brand Watermark", "Brand Sound",
        "Website URL Shown", "Social Handles Shown", "QR Code Present"
    ],
    "Call-to-Action": [
        "CTA Present", "CTA Placement", "CTA Timing", "CTA Button Visible",
        "CTA Button Color", "CTA Button Animation", "Arrow/Pointer Used",
        "Swipe Up Indicator", "Tap Animation", "CTA Visibility",
        "CTA Repetition", "CTA Clarity", "CTA Text Type", "CTA Urgency",
        "CTA Size", "CTA Contrast", "Multiple CTAs", "Verbal CTA"
    ],
    "Audio Elements": [
        "Background Music Likely", "Music Tempo", "Music Genre",
        "Music Energy Level", "Music Mood", "Trending Audio",
        "Voiceover Likely", "Voiceover Gender", "Voiceover Tone",
        "Voiceover Speed", "Natural Sound", "Sound Effects",
        "ASMR Elements", "Silence Used", "Audio Ducking", "Beat Sync"
    ],
    "Content & Messaging": [
        "Product Shown", "Product Close Up", "Product In Use",
        "Product Packaging", "Product Variety", "Unboxing Content",
        "Animated Graphics", "Infographics", "Testimonial Present",
        "Customer Review", "Star Rating", "Influencer Endorsement",
        "Expert Endorsement", "Tutorial Style", "How-To Content",
        "Demonstration", "Comparison Content", "Storytelling Approach",
        "Emotional Appeal Type", "Humor Used", "Fear/Urgency Appeal",
        "FOMO Appeal", "Educational Content", "Entertainment Value",
        "Inspirational Content", "Lifestyle Content", "Behind The Scenes",
        "Hook Strength", "Hook Type", "Opening Style", "Closing Style",
        "Pattern Interrupt", "Curiosity Gap", "Transformation Shown",
        "Results Shown", "Social Proof Type"
    ],
    "Engagement Elements": [
        "Scroll Stopping Power", "First 3 Seconds Impact", "Visual Variety",
        "Pacing Score", "Scene Changes Per 5 Sec", "Movement Intensity",
        "Eye Catching Elements", "Pattern Breaks", "Surprise Elements",
        "Interactive Elements", "Loop Potential", "Shareability Score",
        "Comment Bait", "Save Worthiness", "Rewatchability"
    ],
    "Platform Optimization": [
        "Mobile Optimized", "Vertical Format Score", "Sound Off Friendly",
        "Caption Dependency", "Thumbnail Appeal", "Feed Stopping Power",
        "Stories Format Fit", "Reels Format Fit", "Ad Format Guess",
        "Platform Native Feel"
    ],
    "Technical Metrics": [
        "Video Duration", "Frame Rate", "Resolution", "Total Frames"
    ]
}

def get_video_frames(video_path, num_frames=5):
    """Extract key frames from video for analysis"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Could not open video file")
        return frames, {}
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_indices = [int(i * total_frames / (num_frames + 1)) for i in range(1, num_frames + 1)]
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
    
    cap.release()
    
    video_info = {
        "duration": duration,
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height
    }
    
    return frames, video_info

def frame_to_base64(frame):
    """Convert numpy frame to base64 string"""
    img = Image.fromarray(frame)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()

def is_social_media_url(url):
    """Check if URL is from a social media platform that requires yt-dlp"""
    social_domains = ['facebook.com', 'fb.com', 'instagram.com', 'tiktok.com', 
                      'youtube.com', 'youtu.be', 'twitter.com', 'x.com', 'vimeo.com']
    parsed = urlparse(url)
    return any(domain in parsed.netloc.lower() for domain in social_domains)

def download_video_from_url(url):
    """Download video from URL and save to temp file"""
    try:
        parsed = urlparse(url)
        if not parsed.scheme in ['http', 'https']:
            return None, "Invalid URL scheme. Please use http or https."
        
        if is_social_media_url(url):
            return download_with_ytdlp(url)
        
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '')
        if 'video' not in content_type and not url.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            return None, "URL does not appear to be a video file."
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            return tmp_file.name, None
            
    except requests.exceptions.Timeout:
        return None, "Request timed out. Please try a different URL."
    except requests.exceptions.RequestException as e:
        return None, f"Failed to download video: {str(e)}"

def download_with_ytdlp(url):
    """Download video from social media using yt-dlp"""
    try:
        tmp_dir = tempfile.mkdtemp()
        output_template = os.path.join(tmp_dir, 'video.%(ext)s')
        
        ydl_opts = {
            'outtmpl': output_template,
            'format': 'bestvideo[vcodec^=avc1]+bestaudio/bestvideo[vcodec^=h264]+bestaudio/best[vcodec^=avc1]/best[vcodec^=h264]/bestvideo[height<=720]+bestaudio/best',
            'quiet': True,
            'no_warnings': True,
            'merge_output_format': 'mp4',
            'postprocessors': [{
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }],
            'postprocessor_args': ['-c:v', 'libx264', '-preset', 'fast', '-crf', '23'],
            'socket_timeout': 60,
            'retries': 3,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            
            for f in os.listdir(tmp_dir):
                file_path = os.path.join(tmp_dir, f)
                if os.path.isfile(file_path) and os.path.getsize(file_path) > 1000:
                    final_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    os.rename(file_path, final_path)
                    return final_path, None
            
            return None, "Failed to download video file."
                
    except yt_dlp.utils.DownloadError as e:
        error_msg = str(e)
        if "Private video" in error_msg or "login" in error_msg.lower():
            return None, "This video is private or requires login. Please use a public video."
        elif "not available" in error_msg.lower():
            return None, "This video is not available. It may have been removed or is geo-restricted."
        return None, f"Could not download video: {error_msg}"
    except Exception as e:
        return None, f"Error downloading video: {str(e)}"

def parse_meta_csv(uploaded_csv):
    """Parse Meta ad performance CSV file"""
    try:
        df = pd.read_csv(uploaded_csv)
        required_cols = []
        optional_cols = ['Ad ID', 'Ad Name', 'Campaign Name', 'AdSet Name', 'Impressions', 
                        'Outbound Clicks', 'Landing Page Views', 'Spend ($)', 'Cost Per Click',
                        'Cost Per 1000 Impressions', 'Click Through Rate (%)', 'Total Leads',
                        'Total Purchases', 'Total Registrations', 'Cost Per Lead', 'Return on Ad Spend']
        
        performance_data = {}
        for col in df.columns:
            if df[col].iloc[0] is not None:
                performance_data[col] = df[col].iloc[0]
        
        return performance_data, None
    except Exception as e:
        return None, f"Error parsing CSV: {str(e)}"

def calculate_derived_metrics(perf_data):
    """Calculate derived performance metrics"""
    derived = {}
    
    impressions = perf_data.get('Impressions', 0)
    clicks = perf_data.get('Outbound Clicks', perf_data.get('Clicks', 0))
    spend = perf_data.get('Spend ($)', perf_data.get('Spend', 0))
    purchases = perf_data.get('Total Purchases', 0)
    
    if impressions and clicks:
        derived['Click Through Rate (%)'] = round((clicks / impressions) * 100, 2)
    if clicks and spend:
        derived['Cost Per Click'] = round(spend / clicks, 2)
    if impressions and spend:
        derived['Cost Per 1000 Impressions'] = round((spend / impressions) * 1000, 2)
    if purchases and spend:
        derived['Cost Per Purchase'] = round(spend / purchases, 2)
    
    return derived

def analyze_frames_with_openai(frames, video_info):
    """Analyze video frames using OpenAI Vision API"""
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Please add OPENAI_API_KEY to your secrets.")
        return None
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    base64_frames = [frame_to_base64(frame) for frame in frames]
    
    analysis_prompt = """Analyze these video ad frames comprehensively across 200+ metrics for Meta video ads.
For each metric, provide a score (0-100) where applicable, or a descriptive value.

IMPORTANT: Return your response as a valid JSON object with ALL these categories and metrics:

{
    "actor_human_elements": {
        "has_human_actor": 0-100,
        "male_actor_present": 0-100,
        "female_actor_present": 0-100,
        "animated_character": 0-100,
        "celebrity_influencer": 0-100,
        "actor_age_group": "string (child/teen/young_adult/adult/senior/mixed)",
        "actor_body_type": "string (slim/average/athletic/plus_size/varied)",
        "actor_diversity": "string (low/medium/high)",
        "number_of_actors": number,
        "actor_style": "string",
        "actor_speaking": 0-100,
        "actor_demonstrating_product": 0-100,
        "facial_expression": "string",
        "direct_eye_contact": 0-100,
        "hand_gestures_used": 0-100,
        "actor_movement": "string (static/minimal/moderate/dynamic)",
        "clothing_style": "string",
        "authenticity_level": "string (low/medium/high)",
        "energy_level": "string (low/medium/high)",
        "relatability_score": 0-100,
        "actor_screen_time_percent": 0-100,
        "actor_positioning": "string (center/left/right/varied)",
        "actor_close_up_shots": 0-100,
        "actor_full_body_shots": 0-100,
        "actor_group_shots": 0-100,
        "actor_interaction_with_product": 0-100,
        "actor_emotion_range": "string (single/varied/dynamic)",
        "actor_wardrobe_changes": number,
        "actor_props_used": 0-100,
        "actor_setting_changes": number
    },
    "color_visual_style": {
        "color_contrast_level": "string (low/medium/high)",
        "dark_color_scheme": 0-100,
        "light_color_scheme": 0-100,
        "vibrant_colors": 0-100,
        "pastel_colors": 0-100,
        "monochrome_style": 0-100,
        "warm_colors": 0-100,
        "cool_colors": 0-100,
        "color_consistency": "string (low/medium/high)",
        "background_color_intensity": "string (low/medium/high)",
        "brand_colors_used": 0-100,
        "seasonal_colors": 0-100,
        "cultural_color_significance": 0-100,
        "gradient_usage": 0-100,
        "black_white_filter": 0-100,
        "neon_colors": 0-100,
        "earth_tones": 0-100,
        "metallic_colors": 0-100,
        "color_saturation_level": "string (desaturated/normal/saturated/oversaturated)",
        "color_temperature": "string (cold/neutral/warm/very_warm)",
        "dominant_color": "string",
        "secondary_color": "string",
        "accent_color": "string",
        "color_harmony_type": "string (complementary/analogous/triadic/monochromatic)",
        "color_mood": "string (energetic/calm/professional/playful/luxurious)"
    },
    "video_production": {
        "aspect_ratio": "string (16:9/9:16/1:1/4:5)",
        "video_quality": "string (low/medium/high/professional)",
        "lighting_quality": "string (poor/average/good/excellent)",
        "lighting_type": "string (natural/studio/mixed/dramatic/soft)",
        "lighting_direction": "string (front/side/back/top/mixed)",
        "background_type": "string",
        "background_blur": 0-100,
        "background_complexity": "string (simple/moderate/complex)",
        "focus_quality": "string (poor/average/good/excellent)",
        "depth_of_field": "string (shallow/medium/deep)",
        "camera_movement": "string (static/pan/zoom/tracking/handheld)",
        "camera_angle": "string (eye_level/low/high/dutch/varied)",
        "shot_composition": "string (rule_of_thirds/centered/symmetrical/dynamic)",
        "transition_style": "string",
        "transition_frequency": "string (none/few/moderate/frequent)",
        "speed_changes": 0-100,
        "slow_motion_used": 0-100,
        "fast_motion_used": 0-100,
        "timelapse_used": 0-100,
        "color_filter_applied": 0-100,
        "color_grading_style": "string (natural/cinematic/vintage/modern/dramatic)",
        "special_effects": 0-100,
        "vfx_type": "string (none/subtle/moderate/heavy)",
        "screen_recording": 0-100,
        "ugc_style": 0-100,
        "b_roll_footage": 0-100,
        "stock_footage_type": "string (none/minimal/moderate/heavy)",
        "split_screen": 0-100,
        "before_after_comparison": 0-100,
        "picture_in_picture": 0-100,
        "green_screen_used": 0-100,
        "motion_graphics": 0-100,
        "3d_elements": 0-100,
        "lens_flare": 0-100,
        "film_grain": 0-100,
        "vignette_effect": 0-100
    },
    "text_typography": {
        "text_overlay_amount": "string (none/minimal/moderate/heavy)",
        "subtitles_captions": 0-100,
        "auto_captions_style": 0-100,
        "text_size": "string (small/medium/large/extra_large)",
        "font_style": "string",
        "font_weight": "string (light/regular/bold/extra_bold)",
        "text_animation": 0-100,
        "text_animation_type": "string (none/fade/slide/pop/kinetic)",
        "headline_present": 0-100,
        "subheadline_present": 0-100,
        "caption_length": "string (short/medium/long)",
        "emojis_used": 0-100,
        "numbers_stats_shown": 0-100,
        "percentage_shown": 0-100,
        "bullet_points": 0-100,
        "question_asked": 0-100,
        "urgency_language": 0-100,
        "scarcity_language": 0-100,
        "benefits_highlighted": 0-100,
        "features_listed": 0-100,
        "problem_stated": 0-100,
        "solution_presented": 0-100,
        "social_proof": 0-100,
        "testimonial_quote": 0-100,
        "price_discount_shown": 0-100,
        "original_price_shown": 0-100,
        "savings_highlighted": 0-100,
        "free_shipping_mentioned": 0-100,
        "guarantee_mentioned": 0-100,
        "text_contrast_ratio": "string (low/medium/high)",
        "text_background_overlay": 0-100,
        "text_shadow_used": 0-100,
        "text_outline_used": 0-100,
        "text_position": "string (top/center/bottom/varied)",
        "text_alignment": "string (left/center/right/justified)"
    },
    "branding_logo": {
        "logo_placement": "string (none/corner/center/end_card/throughout)",
        "logo_position": "string (top_left/top_right/bottom_left/bottom_right/center)",
        "logo_style": "string (full_color/monochrome/watermark/animated)",
        "logo_size": "string (none/small/medium/large)",
        "logo_duration": "string (none/brief/moderate/throughout)",
        "logo_animation": 0-100,
        "brand_name_text_shown": 0-100,
        "brand_tagline_shown": 0-100,
        "brand_colors_consistency": 0-100,
        "brand_font_consistency": 0-100,
        "brand_consistency_score": 0-100,
        "brand_watermark": 0-100,
        "brand_sound_likely": 0-100,
        "website_url_shown": 0-100,
        "social_handles_shown": 0-100,
        "qr_code_present": 0-100
    },
    "call_to_action": {
        "cta_present": 0-100,
        "cta_placement": "string (none/beginning/middle/end/throughout)",
        "cta_timing": "string (early/middle/late/multiple)",
        "cta_button_visible": 0-100,
        "cta_button_color": "string",
        "cta_button_animation": 0-100,
        "arrow_pointer_used": 0-100,
        "swipe_up_indicator": 0-100,
        "tap_animation": 0-100,
        "cta_visibility": 0-100,
        "cta_repetition": "string (none/once/twice/multiple)",
        "cta_clarity": "string (unclear/average/clear/very_clear)",
        "cta_text_type": "string (shop_now/learn_more/sign_up/download/get_started/other)",
        "cta_urgency": 0-100,
        "cta_size": "string (small/medium/large)",
        "cta_contrast": "string (low/medium/high)",
        "multiple_ctas": 0-100,
        "verbal_cta_likely": 0-100
    },
    "audio_elements": {
        "background_music_likely": 0-100,
        "music_tempo_estimate": "string (slow/medium/fast/varied)",
        "music_genre_estimate": "string (pop/electronic/hip_hop/acoustic/cinematic/corporate/other)",
        "music_energy_level": "string (low/medium/high/building)",
        "music_mood": "string (upbeat/calm/dramatic/inspirational/playful)",
        "trending_audio_likely": 0-100,
        "voiceover_likely": 0-100,
        "voiceover_gender_estimate": "string (male/female/both/unclear)",
        "voiceover_tone_estimate": "string (professional/casual/excited/calm/authoritative)",
        "voiceover_speed_estimate": "string (slow/normal/fast)",
        "natural_sound_likely": 0-100,
        "sound_effects_likely": 0-100,
        "asmr_elements": 0-100,
        "silence_used": 0-100,
        "audio_ducking_likely": 0-100,
        "music_beat_sync_likely": 0-100
    },
    "content_messaging": {
        "product_shown": 0-100,
        "product_close_up": 0-100,
        "product_in_use": 0-100,
        "product_packaging_shown": 0-100,
        "product_variety_shown": number,
        "unboxing_content": 0-100,
        "animated_graphics": 0-100,
        "infographics_used": 0-100,
        "testimonial_present": 0-100,
        "customer_review_shown": 0-100,
        "star_rating_shown": 0-100,
        "influencer_endorsement": 0-100,
        "expert_endorsement": 0-100,
        "tutorial_style": 0-100,
        "how_to_content": 0-100,
        "demonstration_content": 0-100,
        "comparison_content": 0-100,
        "storytelling_approach": "string (narrative/problem_solution/testimonial/lifestyle/educational)",
        "emotional_appeal_type": "string (happiness/fear/trust/excitement/nostalgia/aspiration)",
        "humor_used": 0-100,
        "fear_urgency_appeal": 0-100,
        "fomo_appeal": 0-100,
        "educational_content": 0-100,
        "entertainment_value": 0-100,
        "inspirational_content": 0-100,
        "lifestyle_content": 0-100,
        "behind_the_scenes": 0-100,
        "hook_strength": "string (weak/average/strong/very_strong)",
        "hook_type": "string (question/statement/visual/action/statistic)",
        "opening_style": "string (problem/benefit/action/question/story)",
        "closing_style": "string (cta/summary/cliffhanger/loop/fade)",
        "pattern_interrupt": 0-100,
        "curiosity_gap": 0-100,
        "transformation_shown": 0-100,
        "results_shown": 0-100,
        "social_proof_type": "string (none/reviews/testimonials/numbers/awards/media)"
    },
    "engagement_elements": {
        "scroll_stopping_power": 0-100,
        "first_3_seconds_impact": 0-100,
        "visual_variety": 0-100,
        "pacing_score": 0-100,
        "scene_changes_per_5_sec": number,
        "movement_intensity": "string (low/medium/high)",
        "eye_catching_elements": 0-100,
        "pattern_breaks": number,
        "surprise_elements": 0-100,
        "interactive_elements_suggested": 0-100,
        "loop_potential": 0-100,
        "shareability_score": 0-100,
        "comment_bait": 0-100,
        "save_worthiness": 0-100,
        "rewatchability": 0-100
    },
    "platform_optimization": {
        "mobile_optimized": 0-100,
        "vertical_format_score": 0-100,
        "sound_off_friendly": 0-100,
        "caption_dependency": 0-100,
        "thumbnail_appeal": 0-100,
        "feed_stopping_power": 0-100,
        "stories_format_fit": 0-100,
        "reels_format_fit": 0-100,
        "ad_format_guess": "string (feed/stories/reels/in_stream)",
        "platform_native_feel": 0-100
    },
    "overall_assessment": {
        "estimated_target_audience": "string",
        "target_age_range": "string (18-24/25-34/35-44/45-54/55+/broad)",
        "target_gender": "string (male/female/all)",
        "ad_objective_guess": "string (awareness/consideration/conversion)",
        "funnel_stage": "string (top/middle/bottom)",
        "industry_category": "string",
        "creative_quality_score": 0-100,
        "engagement_potential": 0-100,
        "conversion_potential": 0-100,
        "professionalism_score": 0-100,
        "uniqueness_score": 0-100,
        "trend_alignment": 0-100,
        "brand_safety_score": 0-100,
        "compliance_score": 0-100,
        "key_strengths": ["list of strings"],
        "areas_for_improvement": ["list of strings"],
        "similar_ad_style": "string",
        "recommended_optimizations": ["list of strings"],
        "estimated_performance_tier": "string (low/average/good/excellent)"
    }
}

Analyze all frames collectively. Provide accurate scores based on visual evidence. Be thorough and precise."""

    content = [{"type": "text", "text": analysis_prompt}]
    
    for i, b64_frame in enumerate(base64_frames):
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_frame}"}
        })
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_tokens=4096
        )
        
        response_content = response.choices[0].message.content
        if not response_content:
            st.error("No response received from AI. Please try again.")
            return None
        
        result = json.loads(response_content)
        
        result["technical_metrics"] = {
            "video_duration_seconds": round(video_info["duration"], 2),
            "frame_rate": round(video_info["fps"], 2),
            "resolution": f"{video_info['width']}x{video_info['height']}",
            "total_frames": video_info["total_frames"]
        }
        
        return result
        
    except json.JSONDecodeError as e:
        st.error(f"Error parsing AI response. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error analyzing video: {str(e)}")
        return None

def display_metric_card(title, value, is_score=False):
    """Display a metric in a styled card"""
    if is_score and isinstance(value, (int, float)):
        color = "#28a745" if value >= 70 else "#ffc107" if value >= 40 else "#dc3545"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%); 
                    padding: 15px; border-radius: 10px; margin: 5px 0;
                    border-left: 4px solid {color};">
            <p style="color: #888; margin: 0; font-size: 12px;">{title}</p>
            <p style="color: {color}; margin: 0; font-size: 24px; font-weight: bold;">{value}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%); 
                    padding: 15px; border-radius: 10px; margin: 5px 0;
                    border-left: 4px solid #6c5ce7;">
            <p style="color: #888; margin: 0; font-size: 12px;">{title}</p>
            <p style="color: #fff; margin: 0; font-size: 18px; font-weight: bold;">{value}</p>
        </div>
        """, unsafe_allow_html=True)

def flatten_results(results, prefix=""):
    """Flatten nested dictionary for CSV export"""
    items = {}
    for key, value in results.items():
        new_key = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_results(value, new_key))
        elif isinstance(value, list):
            items[new_key] = ", ".join(str(v) for v in value)
        else:
            items[new_key] = value
    return items

def create_radar_chart(results):
    """Create a radar chart for key metrics"""
    categories = []
    values = []
    
    key_metrics = [
        ("Creative Quality", results.get("overall_assessment", {}).get("creative_quality_score", 0)),
        ("Engagement", results.get("overall_assessment", {}).get("engagement_potential", 0)),
        ("Conversion", results.get("overall_assessment", {}).get("conversion_potential", 0)),
        ("Professionalism", results.get("overall_assessment", {}).get("professionalism_score", 0)),
        ("Uniqueness", results.get("overall_assessment", {}).get("uniqueness_score", 0)),
        ("Scroll Stop", results.get("engagement_elements", {}).get("scroll_stopping_power", 0)),
        ("First 3 Sec", results.get("engagement_elements", {}).get("first_3_seconds_impact", 0)),
        ("Mobile Ready", results.get("platform_optimization", {}).get("mobile_optimized", 0)),
        ("CTA Strength", results.get("call_to_action", {}).get("cta_present", 0)),
        ("Brand", results.get("branding_logo", {}).get("brand_consistency_score", 0)),
    ]
    
    for cat, val in key_metrics:
        if isinstance(val, (int, float)):
            categories.append(cat)
            values.append(val)
    
    if categories:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(108, 92, 231, 0.3)',
            line=dict(color='#6c5ce7', width=2),
            name='Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(color='#888'),
                    gridcolor='#333'
                ),
                angularaxis=dict(
                    tickfont=dict(color='#fff', size=10),
                    gridcolor='#333'
                ),
                bgcolor='#1e1e2e'
            ),
            showlegend=False,
            paper_bgcolor='#1e1e2e',
            font=dict(color='#fff'),
            margin=dict(l=80, r=80, t=40, b=40)
        )
        return fig
    return None

def create_bar_chart(data, title):
    """Create a horizontal bar chart for category scores"""
    df = pd.DataFrame(data)
    fig = px.bar(
        df, 
        x='Score', 
        y='Metric',
        orientation='h',
        color='Score',
        color_continuous_scale=['#dc3545', '#ffc107', '#28a745'],
        range_color=[0, 100]
    )
    fig.update_layout(
        title=title,
        paper_bgcolor='#1e1e2e',
        plot_bgcolor='#1e1e2e',
        font=dict(color='#fff'),
        xaxis=dict(gridcolor='#333', range=[0, 100]),
        yaxis=dict(gridcolor='#333'),
        coloraxis_showscale=False,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def main():
    st.markdown("""
    <style>
    .stApp {
        background-color: #0e0e1a;
    }
    .main-header {
        background: linear-gradient(135deg, #6c5ce7 0%, #a29bfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-section {
        background: #1e1e2e;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">Video Ad Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered analysis of 200+ video ad metrics</p>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Settings")
        num_frames = st.slider("Frames to analyze", 3, 10, 5, help="More frames = more accurate but slower analysis")
        
        st.markdown("---")
        st.markdown("### Analysis Categories")
        for category in METRIC_CATEGORIES.keys():
            st.markdown(f"- {category}")
    
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "video_frames" not in st.session_state:
        st.session_state.video_frames = None
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = {}
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "video_ready" not in st.session_state:
        st.session_state.video_ready = False
    
    st.markdown("### Step 1: Upload Video")
    
    # Check for video_id parameter ONLY if not already processed
    if 'initial_load_done' not in st.session_state:
        query_params = st.query_params
        video_id = query_params.get("video_id", None)
        
        if video_id:
            st.session_state.default_method = "Video URL"
            st.session_state.prefilled_url = f"https://www.facebook.com/reel/{video_id}"
            st.session_state.auto_load = True
        else:
            st.session_state.default_method = "Upload File"
            st.session_state.prefilled_url = ""
            st.session_state.auto_load = False
        
        st.session_state.initial_load_done = True
    
    # Get the default method (will persist across reruns)
    default_index = 1 if st.session_state.get('default_method', "Upload File") == "Video URL" else 0
    
    input_method = st.radio(
        "Choose input method:", 
        ["Upload File", "Video URL"], 
        index=default_index,
        horizontal=True
    )
    
    if input_method == "Upload File":
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your video ad",
                type=["mp4", "mov", "avi", "mkv"],
                help="Upload a video file to analyze"
            )
        with col2:
            st.markdown("### Quick Stats")
            if uploaded_file:
                st.success(f"File: {uploaded_file.name}")
                st.info(f"Size: {uploaded_file.size / (1024*1024):.2f} MB")
        
        if uploaded_file:
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                try:
                    os.unlink(st.session_state.video_path)
                except:
                    pass
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                st.session_state.video_path = tmp_file.name
            st.session_state.video_ready = True
            
            st.markdown("### Video Preview")
            st.markdown("""
                <style>
                    .stVideo {
                        max-width: 350px;
                        margin: 0 auto;
                    }
                    .stVideo video {
                        max-height: 600px;
                        width: auto !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            st.video(uploaded_file)
    else:
        # Get prefilled URL from session state
        default_url = st.session_state.get('prefilled_url', '')
        
        video_url = st.text_input(
            "Enter video URL:", 
            value=default_url,
            placeholder="https://example.com/video.mp4"
        )
        
        # Auto-load if URL was prefilled and not yet loaded
        if st.session_state.get('auto_load', False) and video_url:
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                try:
                    os.unlink(st.session_state.video_path)
                except:
                    pass
            
            with st.spinner("Downloading video..."):
                tmp_path, error = download_video_from_url(video_url)
            if error:
                st.error(error)
                st.session_state.video_ready = False
            elif tmp_path:
                st.session_state.video_path = tmp_path
                st.session_state.video_ready = True
                st.success("Video downloaded successfully!")
            
            # Disable auto-load after first attempt
            st.session_state.auto_load = False
        
        # Manual load button
        if st.button("Load Video", key="load_url"):
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                try:
                    os.unlink(st.session_state.video_path)
                except:
                    pass
            
            with st.spinner("Downloading video..."):
                tmp_path, error = download_video_from_url(video_url)
            if error:
                st.error(error)
                st.session_state.video_ready = False
            elif tmp_path:
                st.session_state.video_path = tmp_path
                st.session_state.video_ready = True
                st.success("Video downloaded successfully!")
        
        if st.session_state.video_ready and st.session_state.video_path:
            st.markdown("### Video Preview")
            st.markdown("""
                <style>
                    .stVideo {
                        max-width: 350px;
                        margin: 0 auto;
                        border-radius: 10px;
                    }
                    .stVideo video {
                        max-height: 600px;
                        width: auto !important;
                    }
                        .st-emotion-cache-1vo6xi6{
                        text-align: center;
                        }
                        .st-emotion-cache-1q82h82{
                            font-size: 16px;
                        text-align: left;
                        }
                </style>
            """, unsafe_allow_html=True)
            st.video(st.session_state.video_path)



    st.markdown("---")
    st.markdown("### Step 2: Meta Performance Metrics (Optional)")
    st.markdown("Add your Meta Ads performance data to combine with visual analysis")
    
    # Check for metrics in URL parameters
    if 'metrics_loaded_from_url' not in st.session_state:
        query_params = st.query_params
        
        # Check if we have ad metrics in URL
        if query_params.get("ad_name"):
            try:
                perf_data = {
                    "Ad ID": query_params.get("video_id", ""),
                    "Ad Name": query_params.get("ad_name", ""),
                    "Campaign Name": query_params.get("campaign", ""),
                    "Adset Name": query_params.get("adset", ""),
                    "Status": query_params.get("status", ""),
                    "Title": query_params.get("title", ""),
                    "Body": query_params.get("body", ""),
                    "Link": query_params.get("link", ""),
                    "CTA": query_params.get("cta", ""),
                    "Impressions": int(query_params.get("impressions", 0)),
                    "Outbound Clicks": int(query_params.get("clicks", 0)),
                    "Landing Page Views": int(query_params.get("landing_page_views", 0)),
                    "Total Leads": int(query_params.get("leads", 0)),
                    "Total Purchases": int(query_params.get("purchases", 0)),
                    "Total Registrations": int(query_params.get("registrations", 0)),
                    "Spend (₹)": float(query_params.get("spend", 0)),
                    "Target CPA (₹)": float(query_params.get("target_cpa", 0)),
                    "Actual CPA (₹)": float(query_params.get("cpa", 0)),
                    "Return on Ad Spend": float(query_params.get("roas", 0)),
                    "CTR (%)": float(query_params.get("ctr", 0)),
                    "CPC (₹)": float(query_params.get("cpc", 0))
                }
                
                # Calculate additional derived metrics
                derived = calculate_derived_metrics(perf_data)
                perf_data.update(derived)
                
                st.session_state.performance_metrics = perf_data
                st.session_state.metrics_loaded_from_url = True
                st.success("Performance metrics loaded from URL!")
            except Exception as e:
                st.warning(f"Could not load metrics from URL: {str(e)}")
                st.session_state.metrics_loaded_from_url = False
        else:
            st.session_state.metrics_loaded_from_url = False
    
    perf_input_method = st.radio("Choose input method:", ["Manual Entry", "Upload CSV"], horizontal=True, key="perf_method")
    
    if perf_input_method == "Manual Entry":
        with st.expander("Enter Performance Metrics", expanded=False):
            pcol1, pcol2, pcol3 = st.columns(3)
            with pcol1:
                ad_id = st.text_input("Ad ID", value=st.session_state.performance_metrics.get("Ad ID", "") if st.session_state.performance_metrics else "", key="ad_id")
                ad_name = st.text_input("Ad Name", value=st.session_state.performance_metrics.get("Ad Name", "") if st.session_state.performance_metrics else "", key="ad_name")
                campaign_name = st.text_input("Campaign Name", value=st.session_state.performance_metrics.get("Campaign Name", "") if st.session_state.performance_metrics else "", key="campaign_name")
                adset_name = st.text_input("Adset Name", value=st.session_state.performance_metrics.get("Adset Name", "") if st.session_state.performance_metrics else "", key="adset_name")
                status = st.text_input("Status", value=st.session_state.performance_metrics.get("Status", "") if st.session_state.performance_metrics else "", key="status")
            with pcol2:
                impressions = st.number_input("Impressions", min_value=0, value=st.session_state.performance_metrics.get("Impressions", 0) if st.session_state.performance_metrics else 0, key="impressions")
                clicks = st.number_input("Outbound Clicks", min_value=0, value=st.session_state.performance_metrics.get("Outbound Clicks", 0) if st.session_state.performance_metrics else 0, key="clicks")
                landing_views = st.number_input("Landing Page Views", min_value=0, value=st.session_state.performance_metrics.get("Landing Page Views", 0) if st.session_state.performance_metrics else 0, key="landing_views")
                leads = st.number_input("Total Leads", min_value=0, value=st.session_state.performance_metrics.get("Total Leads", 0) if st.session_state.performance_metrics else 0, key="leads")
                purchases = st.number_input("Total Purchases", min_value=0, value=st.session_state.performance_metrics.get("Total Purchases", 0) if st.session_state.performance_metrics else 0, key="purchases")
            with pcol3:
                registrations = st.number_input("Total Registrations", min_value=0, value=st.session_state.performance_metrics.get("Total Registrations", 0) if st.session_state.performance_metrics else 0, key="registrations")
                spend = st.number_input("Spend (₹)", min_value=0.0, value=st.session_state.performance_metrics.get("Spend (₹)", 0.0) if st.session_state.performance_metrics else 0.0, format="%.2f", key="spend")
                target_cpa = st.number_input("Target CPA (₹)", min_value=0.0, value=st.session_state.performance_metrics.get("Target CPA (₹)", 0.0) if st.session_state.performance_metrics else 0.0, format="%.2f", key="target_cpa")
                actual_cpa = st.number_input("Actual CPA (₹)", min_value=0.0, value=st.session_state.performance_metrics.get("Actual CPA (₹)", 0.0) if st.session_state.performance_metrics else 0.0, format="%.2f", key="actual_cpa")
                roas = st.number_input("Return on Ad Spend", min_value=0.0, value=st.session_state.performance_metrics.get("Return on Ad Spend", 0.0) if st.session_state.performance_metrics else 0.0, format="%.2f", key="roas")
            
            # Text fields
            st.markdown("#### Ad Creative Content")
            title = st.text_input("Title", value=st.session_state.performance_metrics.get("Title", "") if st.session_state.performance_metrics else "", key="title")
            body = st.text_area("Body", value=st.session_state.performance_metrics.get("Body", "") if st.session_state.performance_metrics else "", height=150, key="body")
            link = st.text_input("Link URL", value=st.session_state.performance_metrics.get("Link", "") if st.session_state.performance_metrics else "", key="link")
            cta = st.text_input("CTA Type", value=st.session_state.performance_metrics.get("CTA", "") if st.session_state.performance_metrics else "", key="cta")
            
            if st.button("Save Performance Metrics"):
                perf_data = {
                    "Ad ID": ad_id,
                    "Ad Name": ad_name,
                    "Campaign Name": campaign_name,
                    "Adset Name": adset_name,
                    "Status": status,
                    "Title": title,
                    "Body": body,
                    "Link": link,
                    "CTA": cta,
                    "Impressions": impressions,
                    "Outbound Clicks": clicks,
                    "Landing Page Views": landing_views,
                    "Spend (₹)": spend,
                    "Total Leads": leads,
                    "Total Purchases": purchases,
                    "Total Registrations": registrations,
                    "Target CPA (₹)": target_cpa,
                    "Actual CPA (₹)": actual_cpa,
                    "Return on Ad Spend": roas
                }
                derived = calculate_derived_metrics(perf_data)
                perf_data.update(derived)
                st.session_state.performance_metrics = perf_data
                st.success("Performance metrics saved!")
    else:
        uploaded_csv = st.file_uploader("Upload Meta Ads CSV", type=["csv"], key="csv_upload")
        if uploaded_csv:
            perf_data, error = parse_meta_csv(uploaded_csv)
            if error:
                st.error(error)
            else:
                derived = calculate_derived_metrics(perf_data)
                perf_data.update(derived)
                st.session_state.performance_metrics = perf_data
                st.success(f"Loaded {len(perf_data)} metrics from CSV!")
                with st.expander("Preview loaded metrics"):
                    st.json(perf_data)
    
    if st.session_state.performance_metrics:
        st.info(f"Performance metrics loaded: {len(st.session_state.performance_metrics)} fields")
        with st.expander("View All Saved Metrics", expanded=False):
            metrics = st.session_state.performance_metrics
            
            st.markdown("#### Campaign Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Ad ID", metrics.get("Ad ID", "N/A"))
                st.metric("Ad Name", metrics.get("Ad Name", "N/A"))
            with col2:
                st.metric("Campaign", metrics.get("Campaign Name", "N/A"))
                st.metric("Adset", metrics.get("Adset Name", "N/A"))
            with col3:
                st.metric("Status", metrics.get("Status", "N/A"))
            
            st.markdown("#### Performance Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Impressions", f"{metrics.get('Impressions', 0):,}")
                st.metric("Clicks", f"{metrics.get('Outbound Clicks', 0):,}")
            with col2:
                st.metric("Landing Page Views", f"{metrics.get('Landing Page Views', 0):,}")
                st.metric("Leads", f"{metrics.get('Total Leads', 0):,}")
            with col3:
                st.metric("Purchases", f"{metrics.get('Total Purchases', 0):,}")
                st.metric("Registrations", f"{metrics.get('Total Registrations', 0):,}")
            with col4:
                st.metric("Spend", f"₹{metrics.get('Spend (₹)', 0):,.2f}")
                st.metric("CPA", f"₹{metrics.get('Actual CPA (₹)', 0):,.2f}")
            
            st.markdown("#### Target vs Actual")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Target CPA", f"₹{metrics.get('Target CPA (₹)', 0):,.2f}")
                st.metric("Actual CPA", f"₹{metrics.get('Actual CPA (₹)', 0):,.2f}")
            with col2:
                st.metric("Actual ROAS", f"{metrics.get('Return on Ad Spend', 0):.2f}")
            
            st.markdown("#### Calculated Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("CTR", f"{metrics.get('CTR (%)', 0):.2f}%")
            with col2:
                st.metric("CPC", f"₹{metrics.get('CPC (₹)', 0):,.2f}")
            with col3:
                st.metric("ROAS", f"{metrics.get('Return on Ad Spend', 0):.2f}")
            
            st.markdown("#### Creative Content")
            if metrics.get("Title"):
                st.markdown(f"**Title:** {metrics.get('Title')}")
            if metrics.get("Body"):
                st.markdown(f"**Body:** {metrics.get('Body')}")
            if metrics.get("Link"):
                st.markdown(f"**Link:** {metrics.get('Link')}")
            if metrics.get("CTA"):
                st.markdown(f"**CTA:** {metrics.get('CTA')}")


    st.markdown("---")
    st.markdown("### Step 3: Analyze Video")
    
    if st.session_state.video_ready and st.session_state.video_path:
        if st.button("Analyze Video", type="primary", use_container_width=True):
            with st.spinner("Extracting frames from video..."):
                frames, video_info = get_video_frames(st.session_state.video_path, num_frames)
            
            if frames:
                st.success(f"Extracted {len(frames)} frames for analysis")
                
                cols = st.columns(min(len(frames), 5))
                for i, (col, frame) in enumerate(zip(cols, frames[:5])):
                    with col:
                        st.image(frame, caption=f"Frame {i+1}", use_container_width=True)
                
                with st.spinner("Analyzing video with AI... This may take a minute..."):
                    results = analyze_frames_with_openai(frames, video_info)
                
                if results:
                    if st.session_state.performance_metrics:
                        results["performance_metrics"] = st.session_state.performance_metrics
                    st.session_state.analysis_results = results
                    st.session_state.video_frames = frames
                    st.success("Analysis complete!")
                    
                    if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                        try:
                            os.unlink(st.session_state.video_path)
                        except:
                            pass
                    st.session_state.video_path = None
                    st.session_state.video_ready = False
                    st.rerun()
    else:
        st.info("Please upload a video or provide a URL to analyze.")
    
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        st.markdown("---")
        st.markdown("## Analysis Results")
        
        overview = results.get("overall_assessment", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            display_metric_card("Creative Quality", overview.get("creative_quality_score", "N/A"), True)
        with col2:
            display_metric_card("Engagement Potential", overview.get("engagement_potential", "N/A"), True)
        with col3:
            display_metric_card("Professionalism", overview.get("professionalism_score", "N/A"), True)
        with col4:
            display_metric_card("Uniqueness", overview.get("uniqueness_score", "N/A"), True)
        
        col1, col2 = st.columns(2)
        with col1:
            display_metric_card("Target Audience", overview.get("estimated_target_audience", "N/A"))
        with col2:
            display_metric_card("Ad Objective", overview.get("ad_objective_guess", "N/A"))
        
        st.markdown("### Performance Overview")
        radar_fig = create_radar_chart(results)
        if radar_fig:
            st.plotly_chart(radar_fig, use_container_width=True)
        
        tabs = st.tabs([
            "Performance",
            "Actor & Human",
            "Colors & Style",
            "Production",
            "Text & Typography",
            "Branding",
            "CTA",
            "Audio",
            "Content",
            "Engagement",
            "Platform",
            "Technical",
            "Summary"
        ])
        
        with tabs[0]:
            st.markdown("### Meta Performance Metrics")
            perf_data = results.get("performance_metrics", {})
            
            if perf_data:
                # Campaign Information
                st.markdown("#### Campaign Information")
                ccol1, ccol2, ccol3, ccol4 = st.columns(4)
                with ccol1:
                    display_metric_card("Ad Name", perf_data.get('Ad Name', 'N/A'))
                with ccol2:
                    display_metric_card("Campaign", perf_data.get('Campaign Name', 'N/A'))
                with ccol3:
                    display_metric_card("Adset", perf_data.get('Adset Name', 'N/A'))
                with ccol4:
                    display_metric_card("Status", perf_data.get('Status', 'N/A'))
                
                st.markdown("---")
                
                # Performance Metrics - Row 1
                st.markdown("#### Performance Metrics")
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                with pcol1:
                    display_metric_card("Impressions", f"{perf_data.get('Impressions', 0):,}")
                with pcol2:
                    display_metric_card("Clicks", f"{perf_data.get('Outbound Clicks', 0):,}")
                with pcol3:
                    display_metric_card("Landing Page Views", f"{perf_data.get('Landing Page Views', 0):,}")
                with pcol4:
                    display_metric_card("Leads", f"{perf_data.get('Total Leads', 0):,}")
                
                # Performance Metrics - Row 2
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                with pcol1:
                    display_metric_card("Purchases", f"{perf_data.get('Total Purchases', 0):,}")
                with pcol2:
                    display_metric_card("Registrations", f"{perf_data.get('Total Registrations', 0):,}")
                with pcol3:
                    display_metric_card("Spend", f"₹{perf_data.get('Spend (₹)', 0):,.2f}")
                with pcol4:
                    display_metric_card("CPA", f"₹{perf_data.get('Actual CPA (₹)', 0):,.2f}")
                
                st.markdown("---")
                
                # Calculated Metrics
                st.markdown("#### Calculated Metrics")
                pcol1, pcol2, pcol3, pcol4 = st.columns(4)
                with pcol1:
                    ctr = perf_data.get('CTR (%)', 0)
                    display_metric_card("CTR", f"{ctr:.2f}%", is_score=False)
                with pcol2:
                    display_metric_card("CPC", f"₹{perf_data.get('CPC (₹)', 0):,.2f}")
                with pcol3:
                    display_metric_card("CPM", f"₹{perf_data.get('Cost Per 1000 Impressions', 0):,.2f}")
                with pcol4:
                    display_metric_card("ROAS", f"{perf_data.get('Return on Ad Spend', 0):.2f}x")
                
                st.markdown("---")
                
                # Target vs Actual
                st.markdown("#### Target vs Actual")
                tcol1, tcol2, tcol3, tcol4 = st.columns(4)
                with tcol1:
                    display_metric_card("Target CPA", f"₹{perf_data.get('Target CPA (₹)', 0):,.2f}")
                with tcol2:
                    display_metric_card("Actual CPA", f"₹{perf_data.get('Actual CPA (₹)', 0):,.2f}")
                with tcol3:
                    display_metric_card("Target ROAS", f"{perf_data.get('Target ROAS', 0):.2f}x")
                with tcol4:
                    display_metric_card("Actual ROAS", f"{perf_data.get('Return on Ad Spend', 0):.2f}x")
                
                st.markdown("---")
                
                # Creative Content
                st.markdown("#### Creative Content")
                if perf_data.get("Title"):
                    st.markdown(f"**Title:** {perf_data.get('Title')}")
                if perf_data.get("Body"):
                    with st.expander("View Ad Body Text", expanded=False):
                        st.markdown(perf_data.get('Body'))
                if perf_data.get("Link"):
                    st.markdown(f"**Link:** [{perf_data.get('Link')}]({perf_data.get('Link')})")
                if perf_data.get("CTA"):
                    st.markdown(f"**CTA:** {perf_data.get('CTA')}")
                
                st.markdown("---")
                
                # All Performance Data Table
                st.markdown("#### All Performance Data")
                perf_df = pd.DataFrame([perf_data]).T
                perf_df.columns = ["Value"]
                st.dataframe(perf_df, use_container_width=True)
            else:
                st.info("No performance metrics were provided. Add Meta Ads data in Step 2 to see performance analysis.")

        with tabs[1]:
            st.markdown("### Actor & Human Elements")
            actor_data = results.get("actor_human_elements", {})
            
            score_metrics = []
            for key, value in actor_data.items():
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    score_metrics.append({"Metric": key.replace("_", " ").title(), "Score": value})
            
            if score_metrics:
                fig = create_bar_chart(score_metrics, "Actor Metrics Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            cols = st.columns(3)
            i = 0
            for key, value in actor_data.items():
                if not isinstance(value, (int, float)) or value > 100:
                    with cols[i % 3]:
                        display_metric_card(key.replace("_", " ").title(), value)
                    i += 1
        
        with tabs[2]:
            st.markdown("### Color & Visual Style")
            color_data = results.get("color_visual_style", {})
            
            score_metrics = []
            for key, value in color_data.items():
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    score_metrics.append({"Metric": key.replace("_", " ").title(), "Score": value})
            
            if score_metrics:
                fig = create_bar_chart(score_metrics, "Color & Style Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            cols = st.columns(3)
            i = 0
            for key, value in color_data.items():
                if not isinstance(value, (int, float)) or value > 100:
                    with cols[i % 3]:
                        display_metric_card(key.replace("_", " ").title(), value)
                    i += 1
        
        with tabs[3]:
            st.markdown("### Video Production")
            prod_data = results.get("video_production", {})
            
            score_metrics = []
            for key, value in prod_data.items():
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    score_metrics.append({"Metric": key.replace("_", " ").title(), "Score": value})
            
            if score_metrics:
                fig = create_bar_chart(score_metrics, "Production Quality Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            cols = st.columns(3)
            i = 0
            for key, value in prod_data.items():
                if not isinstance(value, (int, float)) or value > 100:
                    with cols[i % 3]:
                        display_metric_card(key.replace("_", " ").title(), value)
                    i += 1
        
        with tabs[4]:
            st.markdown("### Text & Typography")
            text_data = results.get("text_typography", {})
            
            score_metrics = []
            for key, value in text_data.items():
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    score_metrics.append({"Metric": key.replace("_", " ").title(), "Score": value})
            
            if score_metrics:
                fig = create_bar_chart(score_metrics, "Text & Typography Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            cols = st.columns(3)
            i = 0
            for key, value in text_data.items():
                if not isinstance(value, (int, float)) or value > 100:
                    with cols[i % 3]:
                        display_metric_card(key.replace("_", " ").title(), value)
                    i += 1
        
        with tabs[5]:
            st.markdown("### Branding & Logo")
            brand_data = results.get("branding_logo", {})
            
            cols = st.columns(3)
            i = 0
            for key, value in brand_data.items():
                with cols[i % 3]:
                    is_score = isinstance(value, (int, float)) and 0 <= value <= 100
                    display_metric_card(key.replace("_", " ").title(), value, is_score)
                i += 1
        
        with tabs[6]:
            st.markdown("### Call-to-Action Analysis")
            cta_data = results.get("call_to_action", {})
            
            score_metrics = []
            for key, value in cta_data.items():
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    score_metrics.append({"Metric": key.replace("_", " ").title(), "Score": value})
            
            if score_metrics:
                fig = create_bar_chart(score_metrics, "CTA Effectiveness Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            cols = st.columns(3)
            i = 0
            for key, value in cta_data.items():
                if not isinstance(value, (int, float)) or value > 100:
                    with cols[i % 3]:
                        display_metric_card(key.replace("_", " ").title(), value)
                    i += 1
        
        with tabs[7]:
            st.markdown("### Audio Elements")
            audio_data = results.get("audio_elements", {})
            
            score_metrics = []
            for key, value in audio_data.items():
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    score_metrics.append({"Metric": key.replace("_", " ").title(), "Score": value})
            
            if score_metrics:
                fig = create_bar_chart(score_metrics, "Audio Analysis Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            cols = st.columns(3)
            i = 0
            for key, value in audio_data.items():
                if not isinstance(value, (int, float)) or value > 100:
                    with cols[i % 3]:
                        display_metric_card(key.replace("_", " ").title(), value)
                    i += 1
        
        with tabs[8]:
            st.markdown("### Content & Messaging")
            content_data = results.get("content_messaging", {})
            
            score_metrics = []
            for key, value in content_data.items():
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    score_metrics.append({"Metric": key.replace("_", " ").title(), "Score": value})
            
            if score_metrics:
                fig = create_bar_chart(score_metrics, "Content Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            cols = st.columns(3)
            i = 0
            for key, value in content_data.items():
                if not isinstance(value, (int, float)) or value > 100:
                    with cols[i % 3]:
                        display_metric_card(key.replace("_", " ").title(), value)
                    i += 1
        
        with tabs[9]:
            st.markdown("### Engagement Elements")
            engagement_data = results.get("engagement_elements", {})
            
            score_metrics = []
            for key, value in engagement_data.items():
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    score_metrics.append({"Metric": key.replace("_", " ").title(), "Score": value})
            
            if score_metrics:
                fig = create_bar_chart(score_metrics, "Engagement Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            cols = st.columns(3)
            i = 0
            for key, value in engagement_data.items():
                if not isinstance(value, (int, float)) or value > 100:
                    with cols[i % 3]:
                        display_metric_card(key.replace("_", " ").title(), value)
                    i += 1
        
        with tabs[10]:
            st.markdown("### Platform Optimization")
            platform_data = results.get("platform_optimization", {})
            
            score_metrics = []
            for key, value in platform_data.items():
                if isinstance(value, (int, float)) and 0 <= value <= 100:
                    score_metrics.append({"Metric": key.replace("_", " ").title(), "Score": value})
            
            if score_metrics:
                fig = create_bar_chart(score_metrics, "Platform Optimization Scores")
                st.plotly_chart(fig, use_container_width=True)
            
            cols = st.columns(3)
            i = 0
            for key, value in platform_data.items():
                if not isinstance(value, (int, float)) or value > 100:
                    with cols[i % 3]:
                        display_metric_card(key.replace("_", " ").title(), value)
                    i += 1
        
        with tabs[11]:
            st.markdown("### Technical Metrics")
            tech_data = results.get("technical_metrics", {})
            
            cols = st.columns(4)
            for i, (key, value) in enumerate(tech_data.items()):
                with cols[i % 4]:
                    display_metric_card(key.replace("_", " ").title(), value)
        
        with tabs[12]:
            st.markdown("### Key Strengths")
            strengths = overview.get("key_strengths", [])
            if strengths:
                for strength in strengths:
                    st.markdown(f"- {strength}")
            else:
                st.info("No specific strengths identified")
            
            st.markdown("### Areas for Improvement")
            improvements = overview.get("areas_for_improvement", [])
            if improvements:
                for improvement in improvements:
                    st.markdown(f"- {improvement}")
            else:
                st.info("No specific improvements identified")
            
            st.markdown("### Recommended Optimizations")
            optimizations = overview.get("recommended_optimizations", [])
            if optimizations:
                for opt in optimizations:
                    st.markdown(f"- {opt}")
            else:
                st.info("No specific optimizations recommended")
            
            st.markdown("### Similar Ad Style")
            st.info(overview.get("similar_ad_style", "Not determined"))
            
            col1, col2 = st.columns(2)
            with col1:
                display_metric_card("Performance Tier", overview.get("estimated_performance_tier", "N/A"))
            with col2:
                display_metric_card("Trend Alignment", overview.get("trend_alignment", "N/A"), True)
        
        st.markdown("---")
        st.markdown("### Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            flat_results = flatten_results(results)
            flat_results["analysis_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            df = pd.DataFrame([flat_results])
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="Download as CSV",
                data=csv,
                file_name=f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            json_str = json.dumps(results, indent=2)
            st.download_button(
                label="Download as JSON",
                data=json_str,
                file_name=f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

if __name__ == "__main__":
    main()
