from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import base64
import json
import tempfile
import os
from openai import OpenAI
from PIL import Image
import io
import requests
from urllib.parse import urlparse
import yt_dlp
import pandas as pd

app = Flask(__name__)
CORS(app)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

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

def parse_meta_csv(csv_file):
    """Parse Meta ad performance CSV file"""
    try:
        df = pd.read_csv(csv_file)
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
        return None
    except Exception as e:
        return None

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    try:
        video_file = request.files.get('video')
        video_url = request.form.get('url')
        num_frames = int(request.form.get('num_frames', 5))
        
        # Handle video file or URL
        if video_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                video_file.save(tmp.name)
                video_path = tmp.name
        elif video_url:
            video_path, error = download_video_from_url(video_url)
            if error:
                return jsonify({'error': error}), 400
        else:
            return jsonify({'error': 'No video provided'}), 400
        
        # Extract frames
        frames, video_info = get_video_frames(video_path, num_frames)
        
        if not frames:
            return jsonify({'error': 'Failed to extract frames'}), 500
        
        # Convert frames to base64 for preview
        frames_base64 = [frame_to_base64(frame) for frame in frames[:5]]
        
        # Analyze with OpenAI
        results = analyze_frames_with_openai(frames, video_info)
        
        # Cleanup
        if os.path.exists(video_path):
            os.unlink(video_path)
        
        if not results:
            return jsonify({'error': 'Analysis failed'}), 500
        
        return jsonify({
            'results': results,
            'frames': frames_base64,
            'video_info': video_info
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze-performance', methods=['POST'])
def analyze_performance():
    try:
        csv_file = request.files.get('csv')
        manual_data = request.get_json() if request.is_json else None
        
        if csv_file:
            perf_data, error = parse_meta_csv(csv_file)
            if error:
                return jsonify({'error': error}), 400
        elif manual_data:
            perf_data = manual_data
        else:
            return jsonify({'error': 'No performance data provided'}), 400
        
        derived = calculate_derived_metrics(perf_data)
        perf_data.update(derived)
        
        return jsonify(perf_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)