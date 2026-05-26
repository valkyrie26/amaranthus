"""
References:
- Farnebäck, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion. (Dense Optical Flow).
- Russell, J. A. (1980). A Circumplex Model of Affect. (Valence-Arousal Matrix Mapping).
- Pech-Pacheco, J. L. (2000). Diatom autofocusing in brightfield microscopy: a comparative study. (Laplacian Variance).
"""
import cv2
import numpy as np

def extract_environmental_signals(current_frame, previous_frame=None):
    """
    Analyzes landscapes and non-facial cinematography to calculate psychological metrics.
    
    Combines:
    1. Kinetic Energy (Dense Optical Flow via Farnebäck) -> Maps to Arousal.
    2. Color Harmony (HSV Space Luminance & Saturation) -> Maps to Valence.
    3. Structural Complexity (Laplacian Variance Spatial Edges) -> Modulates Mood Profile.
    """
    # Initialize baseline structural matrices
    valence_score = 0.0
    arousal_score = 0.0
    
    # Ensure images are in standard 8-bit sizing models
    gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # -------------------------------------------------------------------------
    # LAYER 1: SIGNAL KINETICS (Optical Flow Engine -> Arousal)
    # -------------------------------------------------------------------------
    if previous_frame is not None:
        gray_prev = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate full-grid pixel displacement vectors (Dense Optical Flow)
        # Parameters configured for stable macro cinematic pacing tracking
        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray_curr, flow=None, 
            pyr_scale=0.5, levels=3, winsize=15, 
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        # Translate Cartesian coordinates (X, Y displacement) to Polar Magnitude maps
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_motion = float(np.mean(magnitude))
        motion_variance = float(np.var(magnitude))
        
        # Cinematographic Pacing Rules:
        # High uniform velocity = Excitation. High chaotic/unstable velocity = Terror/Anger.
        if mean_motion > 1.5:
            if motion_variance > 2.0:
                # Chaotic camera movement (Shaky handheld tracking, frantic cuts)
                arousal_score += 0.7
                valence_score -= 0.3  # Skews towards anxious/fearful aesthetics
            else:
                # Uniform linear velocity (Smooth drone paths, tracking crane dollys)
                arousal_score += 0.5
                valence_score += 0.2  # Skews towards epic/vibrant aesthetics
        else:
            # Micro tracking layouts (Static lock-offs, tripod compositions)
            arousal_score -= 0.4
    
    # -------------------------------------------------------------------------
    # LAYER 2: CHROMATIC HARMONY (HSV Color Profiling -> Valence)
    # -------------------------------------------------------------------------
    hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
    avg_sat = float(np.mean(hsv[:, :, 1])) / 255.0
    avg_val = float(np.mean(hsv[:, :, 2])) / 255.0
    
    # Cinematic Palette Classification:
    # High-Key (Bright/Saturated) = Positive. Low-Key (Dark/Desaturated) = Melancholic.
    if avg_val > 0.55:
        if avg_sat > 0.35:
            valence_score += 0.4   # Vibrant / Commercial / Joyful tones
        else:
            valence_score += 0.1   # Soft Airy High-key / Minimalist tones
    elif avg_val < 0.35:
        valence_score -= 0.5       # Dark Low-Key / Shadows / Heavy Narrative tones
        if avg_sat < 0.25:
            arousal_score -= 0.2   # Depressed / Somber / Desaturated profile tracking
    
    # -------------------------------------------------------------------------
    # LAYER 3: SPATIAL TEXTURE (Laplacian Edge Variance -> Complexity Modifier)
    # -------------------------------------------------------------------------
    # Computes the second spatial derivative to measure texture sharpness.
    # Smooth landscapes (beaches, skies) vs Jagged environments (cities, thorns)
    spatial_variance = float(cv2.Laplacian(gray_curr, cv2.CV_64F).var())
    
    if spatial_variance > 300.0 and valence_score < 0.0:
        # High spatial frequency paired with dark tones generates tension/suspense
        arousal_score += 0.2
    
    # Final clamping pass to secure data inside safe emotional matrix vectors
    final_v = np.clip(valence_score, -1.0, 1.0)
    final_a = np.clip(arousal_score, -1.0, 1.0)
    
    # -------------------------------------------------------------------------
    # LAYER 4: GENRE PROFILE INTERPRETATION MATRIX
    # -------------------------------------------------------------------------
    # Map the coordinates directly to our pre-configured Cinematic Look Profiles
    if final_v < -0.2:
        assigned_mood = "thriller_grit" if final_a > 0.1 else "melancholic"
    elif final_v > 0.2:
        assigned_mood = "vibrant_pop" if final_a > 0.0 else "epic_scenic"
    else:
        assigned_mood = "suspense" if final_a > 0.2 else "neutral"
        
    return {
        "valence": final_v,
        "arousal": final_a,
        "mood_profile": assigned_mood,
        "complexity": spatial_variance
    }