"""
References:
- Streamlit Layout Documentation: st.tabs widget configuration.
- Real-Time Frame Buffering and Multi-Pass Parametric Interpolation Protocols.
- Advanced Psychological Look Profiles: Mapping Emotional Coordinates to Grading Aesthetics.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import cv2, tempfile, pandas as pd, numpy as np
from facial_emotion_estimator import estimate_emotions_ai
from scene_sentiment_engine import extract_environmental_signals
from utils_cdl import generate_cdl_xml, RAVDESS_BASE, SCENE_MOOD_BASE, apply_cdl_to_frame, lerp_tuple

st.set_page_config(page_title="AuraGrade Prototype 17.5", layout="wide")
st.title("🎬 AuraGrade Prototype 17.5: Custom ASC CDL Suite")

# Initialize and lock default data caches inside the Session Engine
if 'templates' not in st.session_state:
    st.session_state['templates'] = RAVDESS_BASE.copy()
if 'scene_templates' not in st.session_state:
    st.session_state['scene_templates'] = SCENE_MOOD_BASE.copy()
if 'segments' not in st.session_state:
    st.session_state['segments'] = []
if 'scene_segments' not in st.session_state:
    st.session_state['scene_segments'] = []

# --- SIDEBAR: TEMPLATE LAB & SAVING (Shared across both workflows) ---
with st.sidebar:
    st.header("Template Lab")
    
    # Allow colorist to toggle manual parameter overrides across libraries
    mode_target = st.radio("Library Target", ["Character Profiles (Tab 1)", "Environmental Profiles (Tab 2)"])
    active_lib = st.session_state['templates'] if mode_target == "Character Profiles (Tab 1)" else st.session_state['scene_templates']
    
    edit_target = st.selectbox("Select Template to Edit/View", list(active_lib.keys()))
    t = active_lib[edit_target]
    
    st.markdown("---")
    st.subheader("ASC CDL Parameters (10 Channels Explicit)")
    
    st.markdown("**Slope (Highlights / Gain)**")
    s_r = st.slider("Slope R", 0.0, 2.5, float(t['slope'][0]), key="s_r")
    s_g = st.slider("Slope G", 0.0, 2.5, float(t['slope'][1]), key="s_g")
    s_b = st.slider("Slope B", 0.0, 2.5, float(t['slope'][2]), key="s_b")
    
    st.markdown("**Offset (Shadows / Lift)**")
    o_r = st.slider("Offset R", -0.5, 0.5, float(t['offset'][0]), key="o_r")
    o_g = st.slider("Offset G", -0.5, 0.5, float(t['offset'][1]), key="o_g")
    o_b = st.slider("Offset B", -0.5, 0.5, float(t['offset'][2]), key="o_b")
    
    st.markdown("**Power (Midtones / Gamma)**")
    p_r = st.slider("Power R", 0.5, 2.5, float(t['power'][0]), key="p_r")
    p_g = st.slider("Power G", 0.5, 2.5, float(t['power'][1]), key="p_g")
    p_b = st.slider("Power B", 0.5, 2.5, float(t['power'][2]), key="p_b")
    
    st.markdown("**Saturation**")
    sat = st.slider("Global Saturation", 0.0, 3.0, float(t['sat']), key="sat_slider")

    # Save changes dynamically back to specific selected profile indexes
    active_lib[edit_target] = {
        "slope": (s_r, s_g, s_b), "offset": (o_r, o_g, o_b), "power": (p_r, p_g, p_b), "sat": sat
    }

    st.markdown("---")
    new_t_name = st.text_input("New Template Name", placeholder="e.g., Crimson_Shadow")
    if st.button("💾 Save as New Template"):
        if new_t_name:
            if new_t_name not in active_lib:
                active_lib[new_t_name] = active_lib[edit_target].copy()
                st.success(f"Saved '{new_t_name}' to target library!")
                st.rerun()
            else:
                st.error("Template name already exists!")
        else:
            st.error("Please enter a name for the new template.")

# --- MAIN WORKSPACE WORKFLOW TABS ---
tab_people, tab_scenes = st.tabs(["👤 Character-Centric Analysis (People)", "🏞️ Environmental Analysis (Scenes/Landscapes)"])

# ==============================================================================
# TAB 1: CHARACTER-CENTRIC ANALYSIS (Unchanged Original Architecture)
# ==============================================================================
with tab_people:
    uploaded_video = st.file_uploader("Upload Video Asset containing Faces", type=["mp4", "mov"], key="video_uploader_people")

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        c_left, c_right = st.columns([2, 1])

        with c_left:
            st.video(tfile.name)
            if st.button("Step 1: AI Analysis & Auto-Assign", type="primary", key="btn_analysis_people"):
                cap = cv2.VideoCapture(tfile.name)
                fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                raw_data = []
                for fno in range(0, total, 20):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                    ret, frame = cap.read()
                    if not ret: break
                    res = estimate_emotions_ai(frame)
                    top = max(res['raw'], key=res['raw'].get)
                    raw_data.append({"f": fno, "emo": top, "w": res['raw'][top]})
                cap.release()

                segs = []
                if raw_data:
                    curr = {"s_f": 0, "emo": raw_data[0]['emo'], "w_sum": raw_data[0]['w'], "cnt": 1}
                    for i in range(1, len(raw_data)):
                        if raw_data[i]['emo'] != curr['emo']:
                            end_f = raw_data[i]['f']
                            segs.append({
                                "start": str(pd.to_timedelta(curr['s_f']/fps, unit='s'))[7:15],
                                "end": str(pd.to_timedelta(end_f/fps, unit='s'))[7:15],
                                "s_f": curr['s_f'], "e_f": end_f, "emotion": curr['emo'], "weight": curr['w_sum']/curr['cnt']
                            })
                            curr = {"s_f": end_f, "emo": raw_data[i]['emo'], "w_sum": raw_data[i]['w'], "cnt": 1}
                        else:
                            curr['w_sum'] += raw_data[i]['w']; curr['cnt'] += 1
                    segs.append({"start": str(pd.to_timedelta(curr['s_f']/fps, unit='s'))[7:15], "end": str(pd.to_timedelta(total/fps, unit='s'))[7:15],
                                 "s_f": curr['s_f'], "e_f": total, "emotion": curr['emo'], "weight": curr['w_sum']/curr['cnt']})
                    st.session_state['segments'] = segs
                    st.session_state['fps'] = fps

        if st.session_state['segments']:
            with c_right:
                st.subheader("📋 Override Menu")
                transition_seconds = st.slider("🎬 Transition Smoothness (Seconds)", 0.0, 3.0, 1.0, 0.1, key="smoothness_slider_people")
                transition_frames = int(transition_seconds * st.session_state.get('fps', 24.0))

                for idx, s in enumerate(st.session_state['segments']):
                    with st.expander(f"Seg {idx}: {s['emotion'].upper()} ({round(s['weight']*100)}%)"):
                        choice = st.selectbox("Swap Template", list(st.session_state['templates'].keys()), 
                                             index=list(st.session_state['templates'].keys()).index(s['emotion']) if s['emotion'] in st.session_state['templates'] else 0,
                                             key=f"swap_{idx}")
                        s['active_template'] = choice
                        base = st.session_state['templates'][choice]
                        s['cdl'] = {
                            "slope": tuple(1.0 + (v - 1.0) * s['weight'] for v in base['slope']),
                            "offset": tuple(v * s['weight'] for v in base['offset']),
                            "power": tuple(1.0 + (v - 1.0) * s['weight'] for v in base['power']),
                            "sat": 1.0 + (base['sat'] - 1.0) * s['weight']
                        }

                st.divider()
                st.download_button("📥 Export Metadata", generate_cdl_xml(st.session_state['segments']), "AuraGrade.cdl", key="download_xml_people")
                
                if st.button("Step 2: Render Master Preview", type="primary", key="btn_render_people"):
                    with st.spinner("Baking smoothed grades into video..."):
                        out_path = "final_render.mp4"
                        cap = cv2.VideoCapture(tfile.name)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), st.session_state['fps'], (int(cap.get(3)), int(cap.get(4))))
                        
                        frame_parameters = []
                        for f_idx in range(total_frames):
                            seg = next((sg for sg in st.session_state['segments'] if sg['s_f'] <= f_idx <= sg['e_f']), None)
                            if seg: frame_parameters.append({"slope": seg['cdl']['slope'], "offset": seg['cdl']['offset'], "power": seg['cdl']['power'], "sat": seg['cdl']['sat']})
                            else: frame_parameters.append({"slope": (1.0, 1.0, 1.0), "offset": (0.0, 0.0, 0.0), "power": (1.0, 1.0, 1.0), "sat": 1.0})
                        
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
                            if not ret: break
                            vw.write(apply_cdl_to_frame(frame, smoothed_parameters[f_idx]))
                            f_idx += 1
                        cap.release(); vw.release()
                        st.session_state['render_path'] = out_path

            with c_left:
                if 'render_path' in st.session_state: st.video(st.session_state['render_path'])
                st.subheader("Film Strip Comparison")
                for s in st.session_state['segments']:
                    mid_f = (s['s_f'] + s['e_f']) // 2
                    cap = cv2.VideoCapture(tfile.name); cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f); ret, frame = cap.read(); cap.release()
                    if ret:
                        st.write(f"**{s['start']}** | Applied Template: **{s['active_template']}**")
                        c1, c2 = st.columns(2)
                        c1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        graded = apply_cdl_to_frame(frame, {"slope": s['cdl']['slope'], "offset": s['cdl']['offset'], "power": s['cdl']['power'], "sat": s['cdl']['sat']})
                        c2.image(cv2.cvtColor(graded, cv2.COLOR_BGR2RGB), use_container_width=True)

# ==============================================================================
# TAB 2: ENVIRONMENTAL ANALYSIS (Fully Realized Feature Pipeline)
# ==============================================================================
with tab_scenes:
    st.subheader("🏞️ Scene Sentiment Extraction Engine")
    st.write("Extracts structural and emotional lookup parameters from landscapes and backgrounds using motion-pacing arrays and color balance tracking.")
    
    uploaded_scene_video = st.file_uploader("Upload Scenic Video Asset", type=["mp4", "mov"], key="video_uploader_scenes")
    
    if uploaded_scene_video:
        tfile_scene = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_scene.write(uploaded_scene_video.read())
        sc_left, sc_right = st.columns([2, 1])
        
        with sc_left:
            st.video(tfile_scene.name)
            if st.button("Step 1: Extract Environmental Palette Elements", type="primary", key="btn_analysis_scenes"):
                cap = cv2.VideoCapture(tfile_scene.name)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                raw_scene_data = []
                prev_frame = None
                
                # Analyze background dynamics using a structured 20-frame spatial sample stride
                for fno in range(0, total, 20):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Call our new landscape processing library
                    res = extract_environmental_signals(frame, prev_frame)
                    raw_scene_data.append({"f": fno, "mood": res['mood_profile'], "v": res['valence'], "a": res['arousal']})
                    prev_frame = frame.copy()
                cap.release()
                
                # Convert processed frame data points into continuous visual timeline bands
                segs = []
                if raw_scene_data:
                    curr = {"s_f": 0, "mood": raw_scene_data[0]['mood'], "v_sum": raw_scene_data[0]['v'], "a_sum": raw_scene_data[0]['a'], "cnt": 1}
                    for i in range(1, len(raw_scene_data)):
                        if raw_scene_data[i]['mood'] != curr['mood']:
                            end_f = raw_scene_data[i]['f']
                            segs.append({
                                "start": str(pd.to_timedelta(curr['s_f']/fps, unit='s'))[7:15],
                                "end": str(pd.to_timedelta(end_f/fps, unit='s'))[7:15],
                                "s_f": curr['s_f'], "e_f": end_f,
                                "emotion": curr['mood'], # Maps directly back to XML structuring utilities
                                "valence": curr['v_sum']/curr['cnt'], "arousal": curr['a_sum']/curr['cnt']
                            })
                            curr = {"s_f": end_f, "mood": raw_scene_data[i]['mood'], "v_sum": raw_scene_data[i]['v'], "a_sum": raw_scene_data[i]['a'], "cnt": 1}
                        else:
                            curr['v_sum'] += raw_scene_data[i]['v']; curr['a_sum'] += raw_scene_data[i]['a']; curr['cnt'] += 1
                    
                    segs.append({
                        "start": str(pd.to_timedelta(curr['s_f']/fps, unit='s'))[7:15], "end": str(pd.to_timedelta(total/fps, unit='s'))[7:15],
                        "s_f": curr['s_f'], "e_f": total, "emotion": curr['mood'],
                        "valence": curr['v_sum']/curr['cnt'], "arousal": curr['a_sum']/curr['cnt']
                    })
                    st.session_state['scene_segments'] = segs
                    st.session_state['scene_fps'] = fps

        if st.session_state['scene_segments']:
            with sc_right:
                st.subheader("📋 Scene Override Graph")
                st.write("Tune environmental profiles to match narrative pacing structures.")
                
                sc_smooth_seconds = st.slider("🎬 Blend Window Pacing (Seconds)", 0.0, 3.0, 1.5, 0.1, key="smoothness_slider_scenes")
                sc_smooth_frames = int(sc_smooth_seconds * st.session_state.get('scene_fps', 24.0))

                for idx, s in enumerate(st.session_state['scene_segments']):
                    with st.expander(f"Scene Track {idx}: {s['emotion'].upper().replace('_', ' ')}"):
                        choice = st.selectbox("Assign Look Template", list(st.session_state['scene_templates'].keys()), 
                                             index=list(st.session_state['scene_templates'].keys()).index(s['emotion']) if s['emotion'] in st.session_state['scene_templates'] else 0,
                                             key=f"scene_swap_{idx}")
                        s['active_template'] = choice
                        base = st.session_state['scene_templates'][choice]
                        
                        # Apply look factors scaled against the intensity of the scene parameters
                        intensity_factor = max(0.5, (abs(s['valence']) + abs(s['arousal'])) / 2.0)
                        s['cdl'] = {
                            "slope": tuple(1.0 + (v - 1.0) * intensity_factor for v in base['slope']),
                            "offset": tuple(v * intensity_factor for v in base['offset']),
                            "power": tuple(1.0 + (v - 1.0) * intensity_factor for v in base['power']),
                            "sat": 1.0 + (base['sat'] - 1.0) * intensity_factor
                        }

                st.divider()
                st.download_button("📥 Export Scene CDL Metadata", generate_cdl_xml(st.session_state['scene_segments']), "AuraGrade_Scene.cdl", key="download_xml_scenes")
                
                if st.button("Step 2: Render Master Grading Preview", type="primary", key="btn_render_scenes"):
                    with st.spinner("Executing structural frame transformations..."):
                        out_scene_path = "final_scene_render.mp4"
                        cap = cv2.VideoCapture(tfile_scene.name)
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        vw = cv2.VideoWriter(out_scene_path, cv2.VideoWriter_fourcc(*'mp4v'), st.session_state['scene_fps'], (int(cap.get(3)), int(cap.get(4))))
                        
                        frame_parameters = []
                        for f_idx in range(total_frames):
                            seg = next((sg for sg in st.session_state['scene_segments'] if sg['s_f'] <= f_idx <= sg['e_f']), None)
                            if seg: frame_parameters.append({"slope": seg['cdl']['slope'], "offset": seg['cdl']['offset'], "power": seg['cdl']['power'], "sat": seg['cdl']['sat']})
                            else: frame_parameters.append({"slope": (1.0, 1.0, 1.0), "offset": (0.0, 0.0, 0.0), "power": (1.0, 1.0, 1.0), "sat": 1.0})
                        
                        # Cross-channel parameter interpolation maps
                        smoothed_parameters = []
                        for f_idx in range(total_frames):
                            start_window = max(0, f_idx - sc_smooth_frames // 2)
                            end_window = min(total_frames - 1, f_idx + sc_smooth_frames // 2)
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
                            if not ret: break
                            vw.write(apply_cdl_to_frame(frame, smoothed_parameters[f_idx]))
                            f_idx += 1
                        cap.release(); vw.release()
                        st.session_state['scene_render_path'] = out_scene_path

            with sc_left:
                if 'scene_render_path' in st.session_state: 
                    st.video(st.session_state['scene_render_path'])
                    
                st.subheader("Cinematographic Landscape Monitor")
                for s in st.session_state['scene_segments']:
                    mid_f = (s['s_f'] + s['e_f']) // 2
                    cap = cv2.VideoCapture(tfile_scene.name); cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f); ret, frame = cap.read(); cap.release()
                    if ret:
                        st.write(f"🎞️ **{s['start']} - {s['end']}** | Look: **{s['active_template'].upper().replace('_', ' ')}** (V: {round(s['valence'],2)}, A: {round(s['arousal'],2)})")
                        c1, c2 = st.columns(2)
                        c1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        graded = apply_cdl_to_frame(frame, {"slope": s['cdl']['slope'], "offset": s['cdl']['offset'], "power": s['cdl']['power'], "sat": s['cdl']['sat']})
                        c2.image(cv2.cvtColor(graded, cv2.COLOR_BGR2RGB), use_container_width=True)