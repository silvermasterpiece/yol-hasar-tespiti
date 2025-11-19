import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="AI Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# --- DİL SÖZLÜĞÜ ---
TEXTS = {
    "tr": {
        "title": "🛣️ AI Destekli Yol Hasar Analizi",
        "subtitle": "Yüksek Performanslı İşleme ve Raporlama Modu",
        "sidebar_header": "⚙️ Analiz Ayarları",
        "conf_label": "Güven Eşiği (Confidence)",
        "info_msg": "ℹ️ Video işlendikten sonra grafik ve sonuçlar gösterilir.",
        "dev": "Geliştirici: Anıl GÜMÜŞ",
        "upload_label": "Analiz edilecek videoyu yükleyin (MP4, AVI, MOV)",
        "orig_video": "🎥 Orijinal Video",
        "start_btn": "🚀 ANALİZİ BAŞLAT",
        "results_header": "🔍 Sonuç ve Rapor",
        "success_msg": "Analiz {:.1f} saniyede tamamlandı!",
        "video_err": "Video tarayıcıda oynatılamadı. Lütfen indirip izleyin.",
        "download_btn": "📥 İŞLENMİŞ VİDEOYU İNDİR",
        "metric_header": "📊 Toplam Hasar Özeti",
        "clean_msg": "✅ Temiz Yol: Hiçbir hasar tespit edilmedi.",
        "chart_header": "📈 Hasar Yoğunluk Grafiği",
        "chart_caption": "Bu grafik, videonun hangi saniyesinde ne kadar yoğun hasar tespit edildiğini gösterir.",
        "processing": "Analiz ediliyor... %{}",
        "model_err": "Model yüklenemedi! Hata: {}",
        "class_names": {0: "Timsah Sirti", 1: "Boyuna Catlak", 2: "Cukur/Obruk", 3: "Enine Catlak"},
        "wait_msg": "Video işleniyor, lütfen bekleyin...",
        "file_err": "Video dosyası oluşturulamadı! (Sunucu Codec Hatası)"
    },
    "en": {
        "title": "🛣️ AI Road Damage Detection",
        "subtitle": "High Performance Processing & Reporting Mode",
        "sidebar_header": "⚙️ Analysis Settings",
        "conf_label": "Confidence Threshold",
        "info_msg": "ℹ️ Results and charts will be shown after processing.",
        "dev": "Developer: Anıl GÜMÜŞ",
        "upload_label": "Upload a video for analysis (MP4, AVI, MOV)",
        "orig_video": "🎥 Original Video",
        "start_btn": "🚀 START ANALYSIS",
        "results_header": "🔍 Results & Report",
        "success_msg": "Analysis completed in {:.1f} seconds!",
        "video_err": "Video could not be played in browser. Please download.",
        "download_btn": "📥 DOWNLOAD PROCESSED VIDEO",
        "metric_header": "📊 Total Damage Summary",
        "clean_msg": "✅ Clean Road: No damage detected.",
        "chart_header": "📈 Damage Density Chart",
        "chart_caption": "This chart shows the density of detected damages over time (seconds).",
        "processing": "Processing... %{}",
        "model_err": "Model failed to load! Error: {}",
        "class_names": {0: "Alligator Crack", 1: "Longitudinal Crack", 2: "Pothole", 3: "Transverse Crack"},
        "wait_msg": "Processing video, please wait...",
        "file_err": "Video file could not be created! (Server Codec Error)"
    }
}

# --- CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 { color: #00ffcc; text-align: center; font-family: 'Helvetica'; }
    .stButton>button { width: 100%; background-color: #00ffcc; color: black; font-weight: bold; border: none; height: 50px; border-radius: 10px; }
    .stButton>button:hover { background-color: #00ccaa; color: white; }
    .stProgress > div > div > div > div { background-color: #00ffcc; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
</style>
""", unsafe_allow_html=True)

# --- KENAR ÇUBUĞU ---
with st.sidebar:
    lang_option = st.radio("🌐 Language / Dil", ["Türkçe", "English"])
    lang_code = "tr" if lang_option == "Türkçe" else "en"
    t = TEXTS[lang_code]

    st.write("---")
    st.header(t["sidebar_header"]) 
    model_path = 'best.pt' 
    conf_threshold = st.slider(t["conf_label"], 0.10, 1.0, 0.25, 0.05)
    st.info(t["info_msg"])
    st.write("---")
    st.write(t["dev"])

# --- ANA BAŞLIK ---
st.title(t["title"])
st.markdown(f"<h5 style='text-align: center; color: gray;'>{t['subtitle']}</h5>", unsafe_allow_html=True)

# --- RENK PALETİ ---
COLORS = {0: (255, 140, 0), 1: (0, 255, 255), 2: (255, 0, 80), 3: (50, 255, 50)}

# --- İŞLEME FONKSİYONU ---
def process_entire_video(input_path, output_path, model, conf_thresh, lang_texts):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Cloud için en güvenli codec: mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stats = {} 
    timeline_data = {} 
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    class_names = lang_texts["class_names"]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        current_second = int(frame_count / fps)
        
        results = model(frame, conf=conf_thresh, verbose=False)
        detections_in_frame = len(results[0].boxes)
        timeline_data[current_second] = timeline_data.get(current_second, 0) + detections_in_frame
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                color_bgr = COLORS.get(cls_id, (255, 255, 255))[::-1] 
                name = class_names.get(cls_id, "Unknown")
                stats[name] = stats.get(name, 0) + 1
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        out.write(frame)
        
        if frame_count % 5 == 0:
            prog = frame_count / total_frames
            progress_bar.progress(prog)
            status_text.text(lang_texts["processing"].format(int(prog*100)))

    cap.release()
    out.release()
    progress_bar.progress(100)
    status_text.empty()
    
    return stats, timeline_data

# --- ANA AKIŞ ---
uploaded_file = st.file_uploader(t["upload_label"], type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    
    # Çıktı yolu (Daha güvenli bir path)
    output_path = os.path.join(os.getcwd(), "sonuc.mp4")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(t["orig_video"])
        st.video(tfile.name)

    if st.sidebar.button(t["start_btn"]):
        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(t["model_err"].format(e))
            st.stop()
            
        with col2:
            st.subheader(t["results_header"])
            start_time = time.time()
            
            with st.spinner(t["wait_msg"]):
                final_stats, timeline_data = process_entire_video(tfile.name, output_path, model, conf_threshold, t)
            
            duration = time.time() - start_time
            st.success(t["success_msg"].format(duration))
            
            # --- GÜVENLİ DOSYA KONTROLÜ (BURASI YENİ) ---
            if os.path.exists(output_path):
                try:
                    with open(output_path, 'rb') as video_file:
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                except:
                    st.warning(t["video_err"])

                with open(output_path, 'rb') as f:
                    st.download_button(t["download_btn"], f, file_name='analiz_sonucu.mp4')
            else:
                st.error(t["file_err"])
            # ----------------------------------------------

            st.write("---")
            st.markdown(f"### {t['metric_header']}")
            stat_cols = st.columns(4)
            idx = 0
            for damage_name, count in final_stats.items():
                with stat_cols[idx % 4]:
                    st.metric(label=damage_name, value=f"{count}")
                idx += 1
            
            if not final_stats:
                st.info(t["clean_msg"])
            
            if timeline_data:
                st.write("---")
                st.markdown(f"### {t['chart_header']}")
                chart_data = pd.DataFrame(list(timeline_data.items()), columns=['Seconds', 'Damage Count']).set_index('Seconds')
                st.area_chart(chart_data, color="#00ffcc")
                st.caption(t["chart_caption"])