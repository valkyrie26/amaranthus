"""
References:
- ASC Technology Committee, DI Subcommittee (ASC CDL Transfer Functions Spec v1.2).
- Rec. 709 (ITU-R BT.709-6) Luma Parameter Constants.
- Linear Interpolation (LERP) Principles for Temporal Smoothing in Video Post-Processing.
"""
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import cv2

def lerp_tuple(t1, t2, alpha):
    """Linearly interpolates between two numeric tuples based on an alpha factor (0.0 to 1.0)."""
    return tuple(val1 + (val2 - val1) * alpha for val1, val2 in zip(t1, t2))

def apply_cdl_to_frame(frame, exact_cdl):
    """
    Applies the standardized ASC CDL primary color grading transform matrix.
    Formula: out = clamp(in * slope + offset)^power
    Followed by Rec. 709 Luminance-preserving Saturation matrix.
    """
    img = frame.astype(np.float32) / 255.0
    
    s = exact_cdl['slope']
    o = exact_cdl['offset']
    p = exact_cdl['power']
    sat = exact_cdl['sat']

    # OpenCV frames are in BGR order (0=B, 1=G, 2=R)
    img[:,:,2] = np.power(np.maximum((img[:,:,2] * s[0]) + o[0], 0.0), p[0]) # Red
    img[:,:,1] = np.power(np.maximum((img[:,:,1] * s[1]) + o[1], 0.0), p[1]) # Green
    img[:,:,0] = np.power(np.maximum((img[:,:,0] * s[2]) + o[2], 0.0), p[2]) # Blue
    
    # Rec. 709 standard luma conversion values 
    luma = 0.2126 * img[:,:,2] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,0]
    luma = np.expand_dims(luma, axis=-1)
    
    # Apply standard SatNode transformation matrix
    img = luma + sat * (img - luma)
    
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

def generate_cdl_xml(segments):
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