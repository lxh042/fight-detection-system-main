import cv2
import datetime
import os
import tempfile
import time

import numpy as np
import streamlit as st
from PIL import Image

import mindspore as ms
from mindspore import Tensor, context, nn
from mindspore.train.serialization import load

from utills.frame_to_keypoits import extract_keypoints
from components.alert_box import render_alert_box
from components.incident_cards import render_incident_cards


from utills.session_utill import initialize_session_state
initialize_session_state()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# 配置 MindSpore 运行环境
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

# Load models
mindir_path = os.path.join(MODEL_DIR, 'version3.mindir')
graph = load(mindir_path)
model = nn.GraphCell(graph)

# Streamlit UI setup
st.set_page_config(page_title="Conflict Alert System", layout="wide")

coll1, coll2 = st.columns([3, 1])
frame_buffer = []
prob_buffer = []  # 新增：用于概率平滑

with coll2:
    alert_box_placeholder = st.empty()  # use this later
    st.markdown("### 🚨 Live Incident Cards")
    incidentsPlaceHolder = st.empty()

with coll1:
    camera_feed = st.empty()

    # 新增：输入源与推理参数设置
    st.markdown("### Input Source & Inference Settings")
    src = st.selectbox("输入源", ["摄像头", "本地视频"], index=0, key="source_mode")
    if src == "摄像头":
        cam_index = st.number_input("摄像头索引", min_value=0, max_value=10, value=0, step=1, key="camera_index")
        # 清理已存在的临时上传
        if st.session_state.get("tmp_video_path") and os.path.exists(st.session_state["tmp_video_path"]):
            try:
                os.remove(st.session_state["tmp_video_path"])
            except Exception:
                pass
            st.session_state["tmp_video_path"] = None
    else:
        # 路径输入
        st.text_input("本地视频路径", value=st.session_state.get("video_path_input", ""), key="video_path_input")
        # 文件上传
        uploaded = st.file_uploader("上传视频文件", type=["mp4", "avi", "mov", "mkv"], key="video_uploader")
        if uploaded is not None:
            # 创建/覆盖临时文件
            suffix = os.path.splitext(uploaded.name)[-1] or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.flush(); tmp.close()
            # 清理旧的临时文件
            old_tmp = st.session_state.get("tmp_video_path")
            if old_tmp and os.path.exists(old_tmp):
                try:
                    os.remove(old_tmp)
                except Exception:
                    pass
            st.session_state["tmp_video_path"] = tmp.name
            st.success(f"已保存上传视频到临时文件：{tmp.name}")

    # 新增：阈值与平滑窗口大小
    st.slider("分类阈值", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="cls_threshold")
    st.slider("平滑窗口大小（预测滑动平均）", min_value=1, max_value=20, value=5, step=1, key="smooth_k")

    # 控制按钮
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶️ Start"):
            st.session_state.start_camera = True
            prob_buffer.clear()
            frame_buffer.clear()
    with col_stop:
        if st.button("⛔ Stop"):
            st.session_state.start_camera = False

    # Analytical portion is displayed here
    st.markdown("### Analytics Overview")
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        # st.metric("Total Incidents", st.session_state.incident_count, "+15%")
        incident_placeholder = st.empty()
        incident_placeholder.metric("Total Incidents", st.session_state.incident_count, "+15%")
    with col_b:
        st.metric("Detection Accuracy", "98%", "⬆ 2.3%")
    with col_c:
        st.metric("Response Time", "45s", "⬇ 5s faster")
    with col_d:
        # st.metric("Current Status", "No Fight")
        status_placeholder = st.empty()
        status_placeholder.markdown(
            "<h4> <span style='color: green;'>🟢 No Fight</span></h4>",
             unsafe_allow_html=True)
        
    if st.session_state.start_camera:
        # 根据输入源选择 VideoCapture
        if st.session_state.source_mode == "摄像头":
            cap = cv2.VideoCapture(int(st.session_state.get("camera_index", 0)))
        else:
            video_path = st.session_state.get("tmp_video_path") or st.session_state.get("video_path_input")
            if not video_path or not os.path.exists(video_path):
                st.warning("未找到有效的视频文件，请提供有效路径或上传文件。")
                st.session_state.start_camera = False
                # 终止并清理
                if st.session_state.get("tmp_video_path") and os.path.exists(st.session_state["tmp_video_path"]):
                    try: os.remove(st.session_state["tmp_video_path"])
                    except Exception: pass
                    st.session_state["tmp_video_path"] = None
            else:
                cap = cv2.VideoCapture(video_path)

        cooldown = 5  # seconds
        while st.session_state.start_camera and 'cap' in locals() and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # 本地视频播完或相机读取失败
                break

            # Extract Keypoints for each frame
            keypoints, result = extract_keypoints(frame)
            annotated_frame = result.plot()

            if keypoints.shape[0] == 153:
                frame_buffer.append(keypoints)
                if len(frame_buffer) > 41:
                    frame_buffer.pop(0)

                if len(frame_buffer) == 41:
                    input_data = np.expand_dims(frame_buffer, axis=0).astype(np.float32)
                    ms_out = model(Tensor(input_data, ms.float32))
                    prob = float(ms_out.asnumpy().reshape(-1)[0])

                    # 平滑
                    prob_buffer.append(prob)
                    if len(prob_buffer) > int(st.session_state.get("smooth_k", 5)):
                        prob_buffer.pop(0)
                    smoothed = float(np.mean(prob_buffer))

                    threshold = float(st.session_state.get("cls_threshold", 0.5))
                    is_fight = smoothed > threshold

                    label = "🔴 Fighting!" if is_fight else "🟢 No Fight"
                    color = (0, 0, 255) if is_fight else (0, 255, 0)
                    cv2.putText(annotated_frame, f"{label} (p={smoothed:.2f}, thr={threshold:.2f})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    status_placeholder.markdown(f"### **{label}**")

                    current_time = time.time()
                    if is_fight and (current_time - st.session_state.last_incident_time > cooldown):
                        st.session_state.incident_count += 1
                        st.session_state.color = 'red'
                        # Append new incident info
                        incident_info = {
                            "location": "Numl Ghazali",  # You can update dynamically
                            "cam_id": "#3459",
                            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "status": "Unattended"
                        }
                        st.session_state.incidents.append(incident_info)
                        
                        # Render Incident Cards
                        incidentsPlaceHolder.markdown(render_incident_cards(st.session_state.incidents), unsafe_allow_html=True)
                        
                        st.session_state.last_incident_time = current_time
                        incident_placeholder.metric("Total Incidents", st.session_state.incident_count, "+15%")
                        
                    if (not is_fight) and (current_time - st.session_state.last_incident_time > cooldown):
                        st.session_state.color = 'green'
                    # For Changing the Status
                    status_placeholder.markdown(
                        f"<h4><span style='color: {color};'>{label}</span></h4>",
                        unsafe_allow_html=True)

            # Displaying the alert box
            alert_box_placeholder.markdown(render_alert_box(st.session_state.color), unsafe_allow_html=True)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)
            img_resized = img_pil.resize((600, 300))
            camera_feed.image(img_resized)
            time.sleep(0.01)

        # 退出循环清理资源
        if 'cap' in locals():
            cap.release()
        camera_feed.empty()
        # 自动清理临时文件
        if st.session_state.get("tmp_video_path") and os.path.exists(st.session_state["tmp_video_path"]):
            try:
                os.remove(st.session_state["tmp_video_path"])
            except Exception:
                pass
            st.session_state["tmp_video_path"] = None




