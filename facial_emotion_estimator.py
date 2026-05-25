"""
References:
- DeepFace Library, Serengil et al. (Facial Attribute Analysis).
- Circumplex Model of Affect, James A. Russell (Valence/Arousal Mapping).
"""
from deepface import DeepFace
import cv2
import numpy as np

# Mapping standard psychological indices directly to a 2D coordinate vector grid space.
# Index Layout formatting: (Valence [-1.0 to 1.0], Arousal [-1.0 to 1.0])
VA_BASE = {
    "neutral":  (0.0, 0.0),
    "calm":     (0.1, -0.2),
    "happy":    (0.8, 0.4),
    "sad":      (-0.8, -0.4),
    "angry":    (-0.6, 0.8),
    "fear":     (-0.5, 0.7),
    "disgust":  (-0.7, 0.3),
    "surprise": (0.2, 0.9)
}

def estimate_emotions_ai(frame):
    """
    Extracts high-fidelity psychological intent values from tracking data arrays.
    
    Transforms arbitrary classification values into contextual coordinate positions.
    Includes fallback heuristics that suppress tracking jitter and prevent look-flashing errors.
    """
    try:
        # Enforce headless execution optimization using standard openCV classification engines
        results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='opencv')
        if results:
            # Map structural raw scale distribution properties across floating intervals
            raw = {k: v / 100.0 for k, v in results[0]['emotion'].items()}
            
            # Anchor Correction Layer: Dampen tracking jitter within high-variance clusters (e.g. Sad vs Fear)
            if raw.get('sad', 0) > 0.2:
                arousal_hint = raw.get('surprise', 0) + raw.get('fear', 0)
                if arousal_hint > 0.05:
                    raw['fear'] = raw.get('fear', 0) + (raw['sad'] * 0.5)
                    raw['sad'] *= 0.5

            # Synthesize final coordinate vectors through the mathematical Circumplex Matrix grid
            v_final, a_final = 0.0, 0.0
            for emo, score in raw.items():
                if emo in VA_BASE:
                    v_base, a_base = VA_BASE[emo]
                    v_final += v_base * score
                    a_final += a_base * score

            # Calculate the structural vector distance to derive emotional thematic intensity
            magnitude = np.sqrt(v_final**2 + a_final**2)

            return {
                "valence": np.clip(v_final, -1.0, 1.0),
                "arousal": np.clip(a_final, -1.0, 1.0),
                "intensity": 1.0 + magnitude, 
                "raw": raw
            }
    except Exception:
        # Graceful diagnostic failover preservation layout tracking
        pass
    return {"valence": 0.0, "arousal": 0.0, "intensity": 1.0, "raw": {"neutral": 1.0}}