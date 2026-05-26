"""
AuraGrade Unified Architecture (Production Baseline)

References:
- Multimodal Affective Computing Protocols (Combining Facial & Environmental Cues).
- Farneback, G. (2003). Two-Frame Motion Estimation Based on Polynomial Expansion.
- Russell, J. A. (1980). A Circumplex Model of Affect. (Valence-Arousal Metric Mapping).
- American Society of Cinematographers Color Decision List (ASC CDL) v1.0 Metadata Exchange.
"""

import os
import streamlit as st
import cv2
import tempfile
import pandas as pd
import numpy as np
from utils_cdl import generate_cdl_xml, RAVDESS_BASE, apply_cdl_to_frame

# --- GLOBAL SYSTEM CONFIGURATIONS ---
# Force legacy Keras loading flags to secure historical environment tracking layers
os.environ["TF_USE_LEGACY_KERAS"] = "1"

st.set_page_config(page_title="AuraGrade Unified Suite", layout="wide")
st.title("🎬 AuraGrade: Unified Motion & Emotion Grading Engine")

# --- CORE SESSION STATE RECOVERY HUB ---
# Initialize and link standard multi-channel grading preset parameters
if 'templates' not in st.session_state:
    st.session_state['templates'] = RAVDESS_BASE.copy()
if 'unified_segments' not in st.session_state:
    st.session_state['unified_segments'] = []

# --- SIDEBAR: TEMPLATE CONFIGURATION LAB ---
with st.sidebar:
    st.header("Template Lab")
    edit_target = st.selectbox("Select Target Master Template", list(st.session_state['templates'].keys()))
    t = st.session_state['templates'][edit_target]
    
    st.markdown("---")
    st.subheader("ASC CDL Parameters (10 Channels Explicit)")
    
    # 1. Slope Vectors (RGB Multipliers / Highlight Control Gains)
    st.markdown("**Slope (Highlights / Gain)**")
    s_r = st.slider("Slope R", 0.0, 2.5, float(t['slope'][0]), key="s_r")
    s_g = st.slider("Slope G", 0.0, 2.5, float(t['slope'][1]), key="s_g")
    s_b = st.slider("Slope B", 0.0, 2.5, float(t['slope'][2]), key="s_b")
    
    # 2. Offset Vectors (RGB Additive Constants / Black Point Lift Adjustments)
    st.markdown("**Offset (Shadows / Lift)**")
    o_r = st.slider("Offset R", -0.5, 0.5, float(t['offset'][0]), key="o_r")
    o_g = st.slider("Offset G", -0.5, 0.5, float(t['offset'][1]), key="o_g")
    o_b = st.slider("Offset B", -0.5, 0.5, float(t['offset'][2]), key="o_b")
    
    # 3. Power Vectors (RGB Exponential Non-linear Curves / Midtone Gamma Controls)
    st.markdown("**Power (Midtones / Gamma)**")
    p_r = st.slider("Power R", 0.5, 2.5, float(t['power'][0]), key="p_r")
    p_g = st.slider("Power G", 0.5, 2.5, float(t['power'][1]), key="p_g")
    p_b = st.slider("Power B", 0.5, 2.5, float(t['power'][2]), key="p_b")
    
    # 4. Saturation Component (Luminance Weights L1=0.2126, L2=0.7152, L3=0.0722)
    st.markdown("**Saturation**")
    sat = st.slider("Global Saturation", 0.0, 3.0, float(t['sat']), key="sat_slider")

    # Dynamic update callback mapping parameters back into library registry
    st.session_state['templates'][edit_target] = {
        "slope": (s_r, s_g, s_b), "offset": (o_r, o_g, o_b), "power": (p_r, p_g, p_b), "sat": sat
    }

# --- UNIFIED WORKSPACE FRONTEND ---
uploaded_video = st.file_uploader("Upload Any Video Asset (Faces, Action, or Landscapes)", type=["mp4", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    c_left, c_right = st.columns([2, 1])

    with c_left:
        st.video(tfile.name)
        if st.button("Step 1: Execute Unified Kinetic & Affective Analysis", type="primary"):
            cap = cv2.VideoCapture(tfile.name)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = 24.0 if fps <= 0 else fps
            
            raw_timeline = []
            prev_gray = None
            
            # Spatial analysis sampling looping over a structured 20-frame stride window
            for fno in range(0, total, 20):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                ret, frame = cap.read()
                if not ret: 
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                    # Calculate dense optical flow displacement maps via polynomial expansion
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    
                    # Compute vector magnitude (Velocity) and vector variance (Directional Chaos)
                    kinetic_energy = float(np.mean(flow_magnitude))
                    motion_chaos = float(np.var(flow_angle)) if kinetic_energy > 0.1 else 0.0
                    
                    # --- 8-EMOTION KINETIC LOOK SIGNALS MATRIX ---
                    # Intersects velocity profiles (Arousal) and directional chaos (Valence proxy)
                    if kinetic_energy > 6.0:
                        assigned_recommendation = "fearful"
                        confidence_weight = min(kinetic_energy / 12.0, 1.0)
                        
                    elif kinetic_energy > 3.5:
                        if motion_chaos > 2.5:
                            assigned_recommendation = "angry"
                        else:
                            assigned_recommendation = "surprise"
                        confidence_weight = 0.85
                        
                    elif kinetic_energy > 1.5:
                        if motion_chaos > 2.5:
                            assigned_recommendation = "disgust"
                        else:
                            assigned_recommendation = "happy"
                        confidence_weight = 0.75
                        
                    elif kinetic_energy > 0.5:
                        assigned_recommendation = "neutral"
                        confidence_weight = 0.60
                        
                    else:
                        # Baseline micro-velocity validation: use brightness boundaries to resolve ambiguity
                        avg_brightness = float(np.mean(gray)) / 255.0
                        if avg_brightness < 0.35:
                            assigned_recommendation = "sad"
                        else:
                            assigned_recommendation = "calm"
                        confidence_weight = 0.80
                        
                    raw_timeline.append({"f": int(fno), "emo": assigned_recommendation, "w": confidence_weight})
                else:
                    raw_timeline.append({"f": int(fno), "emo": "neutral", "w": 0.5})
                    
                prev_gray = gray.copy()
            cap.release()
            
            # Process sequential data iterations into continuous chronological block segments
            f_segs = []
            if raw_timeline:
                curr = {"s_f": raw_timeline[0]['f'], "emo": raw_timeline[0]['emo'], "w_sum": raw_timeline[0]['w'], "cnt": 1}
                
                def format_time(frame_num, frame_rate):
                    seconds = int(frame_num / frame_rate)
                    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

                for i in range(1, len(raw_timeline)):
                    if raw_timeline[i]['emo'] != curr['emo']:
                        end_f = raw_timeline[i]['f']
                        f_segs.append({
                            "start": format_time(curr['s_f'], fps), "end": format_time(end_f, fps),
                            "s_f": int(curr['s_f']), "e_f": int(end_f), 
                            "emotion": str(curr['emo']), "weight": float(curr['w_sum']/curr['cnt'])
                        })
                        curr = {"s_f": end_f, "emo": raw_timeline[i]['emo'], "w_sum": raw_timeline[i]['w'], "cnt": 1}
                    else:
                        curr['w_sum'] += raw_timeline[i]['w']
                        curr['cnt'] += 1
                        
                f_segs.append({
                    "start": format_time(curr['s_f'], fps), "end": format_time(total, fps),
                    "s_f": int(curr['s_f']), "e_f": int(total), 
                    "emotion": str(curr['emo']), "weight": float(curr['w_sum']/curr['cnt'])
                })
                
                st.session_state['unified_segments'] = f_segs
                st.session_state['unified_fps'] = fps

    # Render management UI tracking columns if valid segments exist
    if st.session_state['unified_segments']:
        with c_right:
            st.subheader("📋 Master Timeline Overrides")
            st.write("Modify automated aesthetic recommendations below:")
            
            transition_seconds = st.slider("🎬 Blend Window (Seconds)", 0.0, 3.0, 1.0, 0.1)
            transition_frames = int(transition_seconds * st.session_state['unified_fps'])

            for idx, s in enumerate(st.session_state['unified_segments']):
                with st.expander(f"Track block {idx}: {s['emotion'].upper()} ({round(s['weight']*100)}%)"):
                    choice = st.selectbox("Assign Look Template", list(st.session_state['templates'].keys()), 
                                         index=list(st.session_state['templates'].keys()).index(s['emotion']) if s['emotion'] in st.session_state['templates'] else 0,
                                         key=f"uni_swap_{idx}")
                    s['active_template'] = choice
                    base = st.session_state['templates'][choice]
                    
                    # Interpolation step scales lookup weights to avoid harsh digital jumps
                    s['cdl'] = {
                        "slope": tuple(1.0 + (v - 1.0) * s['weight'] for v in base['slope']),
                        "offset": tuple(v * s['weight'] for v in base['offset']),
                        "power": tuple(1.0 + (v - 1.0) * s['weight'] for v in base['power']),
                        "sat": 1.0 + (base['sat'] - 1.0) * s['weight']
                    }
                    
            st.divider()
            if st.toggle("View Exportable CDL XML"):
                st.code(generate_cdl_xml(st.session_state['unified_segments']), language="xml")
                
            st.download_button("📥 Export Grading Data (.cdl)", generate_cdl_xml(st.session_state['unified_segments']), "AuraGrade_Master.cdl")
            
            if st.button("Step 2: Render Master Preview", type="primary"):
                with st.spinner("Rendering pipeline configurations directly to frame pixels..."):
                    out_path = "final_unified_render.mp4"
                    cap = cv2.VideoCapture(tfile.name)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), st.session_state['unified_fps'], (int(cap.get(3)), int(cap.get(4))))
                    
                    # Construct structural coordinate arrays across full clip layout
                    frame_parameters = []
                    for f_idx in range(total_frames):
                        seg = next((sg for sg in st.session_state['unified_segments'] if sg['s_f'] <= f_idx <= sg['e_f']), None)
                        if seg: 
                            frame_parameters.append({"slope": seg['cdl']['slope'], "offset": seg['cdl']['offset'], "power": seg['cdl']['power'], "sat": seg['cdl']['sat']})
                        else: 
                            frame_parameters.append({"slope": (1.0, 1.0, 1.0), "offset": (0.0, 0.0, 0.0), "power": (1.0, 1.0, 1.0), "sat": 1.0})
                    
                    # Parameter Smoothing Filtering Layer (Prevents Look Stuttering)
                    smoothed_parameters = []
                    for f_idx in range(total_frames):
                        start_window = max(0, f_idx - transition_frames // 2)
                        end_window = min(total_frames - 1, f_idx + transition_frames // 2)
                        
                        slopes = [frame_parameters[i]['slope'] for i in range(start_window, end_window + 1)]
                        offsets = [frame_parameters[i]['offset'] for i in range(start_window, end_window + 1)]
                        powers = [frame_parameters[i]['power'] for i in range(start_window, end_window + 1)]
                        sats = [frame_parameters[i]['sat'] for i in range(start_window, end_window + 1)]
                        
                        smoothed_parameters.append({
                            "slope": tuple(np.mean([s[ch] for s in slopes]) for ch in range(3)),
                            "offset": tuple(np.mean([o[ch] for o in offsets]) for ch in range(3)),
                            "power": tuple(np.mean([p[ch] for p in powers]) for ch in range(3)),
                            "sat": float(np.mean(sats))
                        })

                    f_idx = 0
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: 
                            break
                        vw.write(apply_cdl_to_frame(frame, smoothed_parameters[f_idx]))
                        f_idx += 1
                    cap.release(); vw.release()
                    st.session_state['master_render_path'] = out_path

        with c_left:
            if 'master_render_path' in st.session_state:
                st.video(st.session_state['master_render_path'])
                
            st.subheader("Film Strip Comparison")
            for s in st.session_state['unified_segments']:
                mid_f = (s['s_f'] + s['e_f']) // 2
                cap = cv2.VideoCapture(tfile.name)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    st.write(f"**{s['start']} - {s['end']}** | Active Style Target: **{s['active_template'].upper()}**")
                    c1, c2 = st.columns(2)
                    c1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    graded = apply_cdl_to_frame(frame, {"slope": s['cdl']['slope'], "offset": s['cdl']['offset'], "power": s['cdl']['power'], "sat": s['cdl']['sat']})
                    c2.image(cv2.cvtColor(graded, cv2.COLOR_BGR2RGB), use_container_width=True)
