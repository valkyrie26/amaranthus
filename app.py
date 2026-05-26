"""
References:
- Streamlit Layout Documentation: st.tabs widget configuration.
- Real-Time Frame Buffering and Multi-Pass Parametric Interpolation Protocols.
- OpenCV Farneback Dense Optical Flow Transfer Functions.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import cv2, tempfile, pandas as pd, numpy as np
from facial_emotion_estimator import estimate_emotions_ai
from utils_cdl import generate_cdl_xml, RAVDESS_BASE, apply_cdl_to_frame, lerp_tuple

st.set_page_config(page_title="AuraGrade Prototype 17.5", layout="wide")
st.title("🎬 AuraGrade Prototype 25.5: Custom ASC CDL Suite")

# Initialize Library
if 'templates' not in st.session_state:
    st.session_state['templates'] = RAVDESS_BASE.copy()
if 'segments' not in st.session_state:
    st.session_state['segments'] = []
if 'scene_segments' not in st.session_state:
    st.session_state['scene_segments'] = []

# --- SIDEBAR: TEMPLATE LAB & SAVING (Shared across both workflows) ---
with st.sidebar:
    st.header("Template Lab")
    
    edit_target = st.selectbox("Select Template to Edit/View", list(st.session_state['templates'].keys()))
    t = st.session_state['templates'][edit_target]
    
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

    st.session_state['templates'][edit_target] = {
        "slope": (s_r, s_g, s_b), "offset": (o_r, o_g, o_b), "power": (p_r, p_g, p_b), "sat": sat
    }

    st.markdown("---")
    new_t_name = st.text_input("New Template Name", placeholder="e.g., Noir_Cold")
    if st.button("💾 Save as New Template"):
        if new_t_name:
            if new_t_name not in st.session_state['templates']:
                st.session_state['templates'][new_t_name] = st.session_state['templates'][edit_target].copy()
                st.success(f"Saved '{new_t_name}' to library!")
                st.rerun()
            else:
                st.error("Template name already exists!")
        else:
            st.error("Please enter a name for the new template.")

# --- MAIN WORKSPACE WORKFLOW TABS ---
tab_people, tab_scenes = st.tabs(["👤 Character-Centric Analysis (People)", "🏞️ Environmental Analysis (Scenes/Landscapes)"])

# ==============================================================================
# TAB 1: CHARACTER-CENTRIC ANALYSIS (Completely Original Code)
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
                st.write("Swap AI recommendations with your custom templates.")
                
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
                if st.toggle("View Exportable CDL", key="toggle_xml_people"):
                    st.code(generate_cdl_xml(st.session_state['segments']), language="xml")
                
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
                            if seg:
                                frame_parameters.append({
                                    "slope": seg['cdl']['slope'], "offset": seg['cdl']['offset'], "power": seg['cdl']['power'], "sat": seg['cdl']['sat']
                                })
                            else:
                                frame_parameters.append({
                                    "slope": (1.0, 1.0, 1.0), "offset": (0.0, 0.0, 0.0), "power": (1.0, 1.0, 1.0), "sat": 1.0
                                })
                        
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
                            exact_cdl = smoothed_parameters[f_idx]
                            vw.write(apply_cdl_to_frame(frame, exact_cdl))
                            f_idx += 1
                            
                        cap.release(); vw.release()
                        st.session_state['render_path'] = out_path

            with c_left:
                if 'render_path' in st.session_state:
                    st.video(st.session_state['render_path'])
                
                st.subheader("Film Strip Comparison")
                for s in st.session_state['segments']:
                    mid_f = (s['s_f'] + s['e_f']) // 2
                    cap = cv2.VideoCapture(tfile.name)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_f)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        st.write(f"**{s['start']}** | Applied Template: **{s['active_template']}**")
                        c1, c2 = st.columns(2)
                        c1.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                        
                        dummy_cdl = {
                            "slope": s['cdl']['slope'], "offset": s['cdl']['offset'], "power": s['cdl']['power'], "sat": s['cdl']['sat']
                        }
                        graded = apply_cdl_to_frame(frame, dummy_cdl)
                        c2.image(cv2.cvtColor(graded, cv2.COLOR_BGR2RGB), use_container_width=True)

# ==============================================================================
# TAB 2: ENVIRONMENTAL ANALYSIS (FEATURE: Pure Optical Flow Pacing)
# ==============================================================================
with tab_scenes:
    st.subheader("🏞️ Scene Pacing Engine (Optical Flow)")
    st.write("Tracks camera movement speed to determine scene urgency. Fast movement defaults to 'surprise' (high excitement), slow movement defaults to 'calm'.")
    
    uploaded_scene_video = st.file_uploader("Upload Scenic Video Asset", type=["mp4", "mov"], key="video_uploader_scenes")
    
    if uploaded_scene_video:
        tfile_scene = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile_scene.write(uploaded_scene_video.read())
        sc_left, sc_right = st.columns([2, 1])
        
        with sc_left:
            st.video(tfile_scene.name)
            if st.button("Step 1: Analyze Motion Pacing", type="primary", key="btn_analysis_scenes"):
                cap = cv2.VideoCapture(tfile_scene.name)
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Dynamic fallback for invalid or missing frame rate readings
                if fps <= 0:
                    fps = 24.0
                
                raw_scene_data = []
                prev_gray = None
                
                # Analyze movement pacing across video frames using 20-frame intervals
                for fno in range(0, total, 20):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
                    ret, frame = cap.read()
                    if not ret: break
                    
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    
                    if prev_gray is not None:
                        # Farneback Dense Optical Flow computation
                        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        motion_speed = float(np.mean(magnitude))
                        
                        # Operational Rule: High pacing threshold assignments mapping to existing template labels
                        assigned_mood = "surprise" if motion_speed > 1.2 else "calm"
                        raw_scene_data.append({"f": int(fno), "emo": assigned_mood, "w": float(min(motion_speed / 3.0, 1.0))})
                    else:
                        # Baseline starter frame fallbacks
                        raw_scene_data.append({"f": int(fno), "emo": "calm", "w": 0.5})
                        
                    prev_gray = gray.copy()
                cap.release()
                
                # Build time-block tracks out of our movement classifications
                sc_segs = []
                if raw_scene_data:
                    curr = {"s_f": 0, "emo": raw_scene_data[0]['emo'], "w_sum": raw_scene_data[0]['w'], "cnt": 1}
                    
                    # Safe localized helper for frame-to-timestamp calculation
                    def format_time(frame_num, frame_rate):
                        total_seconds = int(frame_num / frame_rate)
                        hours = total_seconds // 3600
                        minutes = (total_seconds % 3600) // 60
                        seconds = total_seconds % 60
                        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                    for i in range(1, len(raw_scene_data)):
                        if raw_scene_data[i]['emo'] != curr['emo']:
                            end_f = raw_scene_data[i]['f']
                            sc_segs.append({
                                "start": format_time(curr['s_f'], fps),
                                "end": format_time(end_f, fps),
                                "s_f": int(curr['s_f']), "e_f": int(end_f), "emotion": curr['emo'], "weight": float(curr['w_sum']/curr['cnt'])
                            })
                            curr = {"s_f": end_f, "emo": raw_scene_data[i]['emo'], "w_sum": raw_scene_data[i]['w'], "cnt": 1}
                        else:
                            curr['w_sum'] += raw_scene_data[i]['w']; curr['cnt'] += 1
                            
                    sc_segs.append({
                        "start": format_time(curr['s_f'], fps), 
                        "end": format_time(total, fps),
                        "s_f": int(curr['s_f']), "e_f": int(total), "emotion": curr['emo'], "weight": float(curr['w_sum']/curr['cnt'])
                    })
                    
                    st.session_state['scene_segments'] = sc_segs
                    st.session_state['scene_fps'] = fps

        if st.session_state['scene_segments']:
            with sc_right:
                st.subheader("📋 Pacing Overrides")
                
                for idx, s in enumerate(st.session_state['scene_segments']):
                    with st.expander(f"Track {idx}: {s['emotion'].upper()} (Motion Score: {round(s['weight']*100)}%)"):
                        choice = st.selectbox("Swap Template Profile", list(st.session_state['templates'].keys()), 
                                             index=list(st.session_state['templates'].keys()).index(s['emotion']) if s['emotion'] in st.session_state['templates'] else 0,
                                             key=f"scene_swap_{idx}")
                        s['active_template'] = choice
                        base = st.session_state['templates'][choice]
                        s['cdl'] = {
                            "slope": tuple(1.0 + (v - 1.0) * s['weight'] for v in base['slope']),
                            "offset": tuple(v * s['weight'] for v in base['offset']),
                            "power": tuple(1.0 + (v - 1.0) * s['weight'] for v in base['power']),
                            "sat": 1.0 + (base['sat'] - 1.0) * s['weight']
                        }
                        
                st.divider()
                st.download_button("📥 Export Scene Metadata", generate_cdl_xml(st.session_state['scene_segments']), "AuraGrade_Scene.cdl", key="download_xml_scenes")