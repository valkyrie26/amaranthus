"""
References:
- ASC Technology Committee, DI Subcommittee (ASC CDL Transfer Functions Spec v1.2).
- Rec. 709 (ITU-R BT.709-6) Luma Parameter Constants.
- Linear Interpolation (LERP) Principles for Temporal Smoothing in Video Post-Processing.
- Subtractive Color Synthesis in Film Stock Emulation (Kodak 2383/Fujifilm 3514).
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import cv2

def lerp_tuple(t1, t2, alpha):
    """
    Linearly interpolates between two numeric tuples based on an alpha factor (0.0 to 1.0).
    Used for smooth temporal transitions between cinematic look states.
    """
    return tuple(val1 + (val2 - val1) * alpha for val1, val2 in zip(t1, t2))

def apply_cdl_to_frame(frame, exact_cdl):
    """
    Applies the standardized ASC CDL primary color grading transform matrix using a
    strictly compliant Linear-Light processing pipeline.
    
    Professional Colorist Standards applied:
    1. Inverse Gamma mapping from display/storage space to Linear-Light space.
    2. ASC SOP Node Math: Out = clamp(In * slope + offset)^power
    3. Rec. 709 Luminance-Preserving Saturation Node calculation.
    4. Subtractive Color Density Emulation: High saturation programmatically darkens 
       luminescence curves to mimic physical celluloid emulsion behavior.
    5. Re-application of Display Gamma mapping to avoid digital clipping or washouts.
    """
    # Step 1: Normalize and map from Non-Linear Display/Gamma Space (sRGB/Rec.709) to Linear-Light Space.
    # Color calculations executed directly on gamma-encoded values yield unpredictable, harsh digital artifacts.
    img = (frame.astype(np.float32) / 255.0) ** 2.2
    
    s = exact_cdl['slope']
    o = exact_cdl['offset']
    p = exact_cdl['power']
    sat = exact_cdl['sat']

    # Step 2: Emulate Subtractive Color Density (The Film Look)
    # In physical film, high saturation naturally darkens color fields. In unconstrained digital color spaces,
    # high saturation blooms unnaturally. We modulate the Power (gamma curve) dynamically against the saturation coefficient.
    if sat > 1.0:
        density_backoff = 1.0 + ((sat - 1.0) * 0.12)
        p = tuple(val * density_backoff for val in p)

    # Step 3: Apply the standardized ASC CDL SOP (Slope, Offset, Power) equation block.
    # Note: OpenCV operates in BGR layout (Index 0 = Blue, Index 1 = Green, Index 2 = Red).
    # Parameter channel configurations are strictly locked to their corresponding structural mappings.
    img[:,:,2] = np.power(np.maximum((img[:,:,2] * s[0]) + o[0], 0.0), p[0]) # Red Channel
    img[:,:,1] = np.power(np.maximum((img[:,:,1] * s[1]) + o[1], 0.0), p[1]) # Green Channel
    img[:,:,0] = np.power(np.maximum((img[:,:,0] * s[2]) + o[2], 0.0), p[2]) # Blue Channel
    
    # Step 4: Isolate Luma using ITU-R BT.709-6 Rec. 709 industry standard coefficients.
    luma = 0.2126 * img[:,:,2] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,0]
    luma = np.expand_dims(luma, axis=-1)
    
    # Step 5: Execute ASC SatNode scaling logic relative to isolated luminance boundaries.
    img = luma + sat * (img - luma)
    
    # Step 6: Hard clamp values to 0.0-1.0 boundary safety envelopes, apply Display Gamma encoding (1/2.2),
    # and re-cast the localized array parameters into standard 8-bit integer space.
    img = np.clip(img, 0.0, 1.0) ** (1.0 / 2.2)
    return (img * 255.0).astype(np.uint8)

def generate_cdl_xml(segments):
    """
    Constructs a fully compliant, standardized ASC Color Decision List (.cdl / XML schema) 
    intended for direct metadata ingestion into professional NLE systems (DaVinci Resolve, Premiere, Avid).
    """
    root = ET.Element("ColorDecisionList", xmlns="urn:ASC:CDL:v1.01")
    for i, seg in enumerate(segments):
        decision = ET.SubElement(root, "ColorDecision")
        cc = ET.SubElement(decision, "ColorCorrection", id=f"Seg_{i}_{seg['emotion']}")
        sop = ET.SubElement(cc, "SOPNode")
        s, o, p = seg['cdl']['slope'], seg['cdl']['offset'], seg['cdl']['power']
        ET.SubElement(sop, "Slope").text = f"{round(s[0],3)} {round(s[1],3)} {round(s[2],3)}"
        ET.SubElement(sop, "Offset").text = f"{round(o[0],3)} {round(o[1],3)} {round(o[2],3)}"
        ET.SubElement(sop, "Power").text = f"{round(p[0],3)} {round(p[1],3)} {round(p[2],3)}"
        sat = ET.SubElement(cc, "SatNode")
        ET.SubElement(sat, "Saturation").text = str(round(seg['cdl']['sat'], 3))
    return minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

# Standardized Cinematic Base Palette Templates initialized to neutral vectors.
# Modified inside the main application engine using contextual Valence/Arousal transforms.
RAVDESS_BASE = {
    "neutral":  {"slope": (1.0, 1.0, 1.0), "offset": (0.0, 0.0, 0.0), "power": (1.0, 1.0, 1.0), "sat": 1.0},
    "calm":     {"slope": (0.95, 0.98, 1.0), "offset": (0.0, 0.01, 0.02), "power": (0.95, 0.95, 0.92), "sat": 0.85},
    "happy":    {"slope": (1.18, 1.08, 0.88), "offset": (0.01, 0.0, -0.02), "power": (0.95, 0.98, 1.02), "sat": 1.45},
    "sad":      {"slope": (0.82, 0.88, 1.05), "offset": (-0.04, -0.02, 0.05), "power": (1.15, 1.10, 0.95), "sat": 0.45},
    "angry":    {"slope": (1.42, 0.78, 0.72), "offset": (0.06, -0.03, -0.04), "power": (1.25, 0.95, 0.90), "sat": 1.25},
    "fearful":  {"slope": (0.95, 1.12, 0.85), "offset": (-0.05, -0.01, 0.01), "power": (1.45, 1.35, 1.50), "sat": 0.55},
    "disgust":  {"slope": (0.88, 1.22, 0.78), "offset": (-0.02, 0.05, -0.03), "power": (1.05, 0.90, 1.15), "sat": 0.90},
    "surprise": {"slope": (1.25, 1.25, 1.10), "offset": (0.0, 0.0, 0.0), "power": (0.85, 0.85, 0.90), "sat": 1.50}
}

# Standardized Environmental Mood Look Profiles for Tab 2
# Designed around classic cinematography genre palettes
SCENE_MOOD_BASE = {
    "vibrant_pop":   {"slope": (1.15, 1.08, 0.92), "offset": (0.01, 0.0, -0.01), "power": (0.95, 0.98, 1.02), "sat": 1.40},
    "melancholic":   {"slope": (0.85, 0.90, 1.05), "offset": (-0.03, -0.01, 0.04), "power": (1.12, 1.08, 0.98), "sat": 0.50},
    "thriller_grit": {"slope": (1.20, 1.10, 0.80), "offset": (0.04, -0.02, -0.05), "power": (1.20, 0.95, 0.85), "sat": 0.85},
    "epic_scenic":   {"slope": (1.05, 1.05, 1.10), "offset": (0.0, 0.0, 0.01), "power": (0.90, 0.92, 0.95), "sat": 1.20},
    "suspense":      {"slope": (0.90, 1.05, 0.90), "offset": (-0.02, 0.03, -0.02), "power": (1.30, 1.15, 1.25), "sat": 0.65},
    "neutral":       {"slope": (1.0, 1.0, 1.0), "offset": (0.0, 0.0, 0.0), "power": (1.0, 1.0, 1.0), "sat": 1.0}
}