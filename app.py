import streamlit as st
import cv2
import tempfile
import os
import time
import subprocess
from ultralytics import YOLO
import numpy as np
import pandas as pd
import imageio_ffmpeg

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="AI Yol Hasar Analizi",
    page_icon="🛣️",
    layout="wide"
)

# --- MODERN CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 { color: #00ffcc; text-align: center; font-family: 'Helvetica'; }
    .stButton>button { width: 100%; background-color: #00ffcc; color: black; font-weight: bold; border: none; height: 50px; border-radius: 10px; }
    .stButton>button:hover { background-color: #00ccaa; color: white; }
    .stProgress > div > div > div > div { background-color: #00ffcc; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00ffcc; }
    .empty-state {
        border: 2px dashed #333;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

# --- SABİT AYARLAR (Türkçe) ---
CLASS_NAMES = {
    0: "Timsah Sirti", 
    1: "Boyuna Catlak", 
    2: "Cukur/Obruk", 
    3: "Enine Catlak"
}

COLORS = {
    0: (255, 140, 0),   # Turuncu
    1: (0, 255, 255),   # Cyan
    2: (255, 0, 80),    # Kırmızı
    3: (50, 255, 50)    # Yeşil
}

# --- YARDIMCI FONKSİYON: FFmpeg Dönüştürücü ---
def convert_video_to_h264(input_path, output_path):
    """Videoyu tarayıcı uyumlu H.264 formatına çevirir"""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, '-y', 
        '-i', input_path, 
        '-vcodec', 'libx264', 
        '-pix_fmt', 'yuv420p', 
        output_path
    ]
    if os.name == 'nt':
        subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        subprocess.run(command, check=True)

# --- İŞLEME FONKSİYONU ---
def process_entire_video(input_path, output_path, model, conf_thresh):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Geçici ham dosya
    temp_output = output_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stats = {} 
    timeline_data = {} 
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        current_second = int(frame_count / fps)
        
        # Model Tahmini
        results = model(frame, conf=conf_thresh, verbose=False)
        
        # Grafik verisi (o saniyedeki yoğunluk)
        detections_in_frame = len(results[0].boxes)
        timeline_data[current_second] = timeline_data.get(current_second, 0) + detections_in_frame
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0]) # Skor
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                color_bgr = COLORS.get(cls_id, (255, 255, 255))[::-1] # RGB -> BGR
                name = CLASS_NAMES.get(cls_id, "Bilinmeyen")
                stats[name] = stats.get(name, 0) + 1
                
                # Kutu Çiz
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                
                # --- GÜNCELLENEN KISIM: İsim + Skor ---
                label_text = f"{name} %{int(conf * 100)}"
                cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        out.write(frame)
        
        # İlerleme çubuğunu güncelle (Her 5 karede bir)
        if frame_count % 5 == 0:
            prog = frame_count / total_frames
            progress_bar.progress(prog)
            status_text.text(f"Analiz ediliyor... %{int(prog*100)}")

    cap.release()
    out.release()
    progress_bar.progress(100)
    
    status_text.text("Video web formatına çevriliyor (FFmpeg)...")
    try:
        convert_video_to_h264(temp_output, output_path)
        if os.path.exists(temp_output):
            os.remove(temp_output)
    except Exception as e:
        st.error(f"Video dönüştürme hatası: {e}")
        return stats, timeline_data, False

    status_text.empty()
    return stats, timeline_data, True

# --- ARAYÜZ BAŞLANGICI ---

# Sidebar
with st.sidebar:
    st.header("⚙️ Analiz Ayarları")
    
    uploaded_file = st.file_uploader("Video Yükle (MP4, AVI)", type=['mp4', 'avi', 'mov'])
    
    st.write("---")
    model_path = 'best.pt' 
    conf_threshold = st.slider("Güven Eşiği (Hassasiyet)", 0.10, 1.0, 0.25, 0.05)
    
    st.info("ℹ️ Analiz tamamlandıktan sonra sonuçlar ekrana gelir.")
    st.write("---")
    st.write("Geliştirici: Anıl GÜMÜŞ")

# Ana Başlık
st.title("🛣️ AI Destekli Yol Hasar Analizi")
st.markdown("<h5 style='text-align: center; color: gray;'>Yüksek Performanslı İşleme Modu</h5>", unsafe_allow_html=True)

# Kılavuz
with st.expander("ℹ️ Nasıl Kullanılır?"):
    st.markdown("""
    1. Sol menüden **Video Yükle** alanını kullanın.
    2. Ayarları isteğe bağlı değiştirin.
    3. Sağ tarafta belirecek **Analizi Başlat** butonuna basın.
    4. İşlem bitince raporu ve videoyu indirebilirsiniz.
    """)

# Ana Akış Düzeni
col1, col2 = st.columns([1, 1])

# Sol Kolon: Video Gösterimi
with col1:
    st.subheader("🎥 Orijinal Video")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.video(tfile.name)
    else:
        st.markdown("""
        <div class="empty-state">
            <h1>🎥</h1>
            <p>👈 Lütfen sol menüden bir video yükleyin.</p>
        </div>
        """, unsafe_allow_html=True)

# Sağ Kolon: İşlem ve Sonuçlar
with col2:
    st.subheader("🔍 Sonuç ve Rapor")
    
    if uploaded_file is None:
        st.markdown("""
        <div class="empty-state">
            <h1>📊</h1>
            <h3>Analiz Bekleniyor</h3>
        </div>
        """, unsafe_allow_html=True)
    
    elif uploaded_file is not None:
        start_analyze = st.button("🚀 ANALİZİ BAŞLAT", use_container_width=True)
        
        if not start_analyze:
             st.info("Video yüklendi. Analizi başlatmak için yukarıdaki butona tıklayın.")

        if start_analyze:
            try:
                model = YOLO(model_path)
            except Exception as e:
                st.error(f"Model yüklenemedi! Hata: {e}")
                st.stop()
                
            output_path = os.path.join(os.getcwd(), "sonuc.mp4")
            start_time = time.time()
            
            with st.spinner("Yapay zeka videoyu inceliyor..."):
                final_stats, timeline_data, success = process_entire_video(tfile.name, output_path, model, conf_threshold)
            
            duration = time.time() - start_time
            
            if success:
                st.success(f"Analiz {duration:.1f} saniyede tamamlandı!")
                
                if os.path.exists(output_path):
                    try:
                        with open(output_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes, format="video/mp4")
                    except:
                        st.warning("Video tarayıcıda oynatılamadı.")
                    
                    # İndirme Butonları
                    btn_col1, btn_col2 = st.columns(2)
                    with btn_col1:
                        with open(output_path, 'rb') as f:
                            st.download_button("📹 VİDEOYU İNDİR", f, file_name='analiz_sonucu.mp4', use_container_width=True)
                    
                    with btn_col2:
                        if timeline_data:
                            df_report = pd.DataFrame(list(timeline_data.items()), columns=['Saniye', 'Hasar_Sayisi'])
                            csv = df_report.to_csv(index=False).encode('utf-8')
                            st.download_button("📄 RAPORU İNDİR (CSV)", csv, file_name='hasar_raporu.csv', mime='text/csv', use_container_width=True)

            else:
                st.error("Video işlendi fakat kaydedilemedi.")

            st.write("---")
            # İstatistikler
            st.markdown("### 📊 Toplam Hasar Özeti")
            stat_cols = st.columns(4)
            idx = 0
            for damage_name, count in final_stats.items():
                with stat_cols[idx % 4]:
                    st.metric(label=damage_name, value=f"{count}")
                idx += 1
            
            if not final_stats:
                st.info("✅ Temiz Yol: Hiçbir hasar tespit edilmedi.")
            
            # Grafik
            if timeline_data:
                st.write("---")
                st.markdown("### 📈 Hasar Yoğunluk Grafiği")
                chart_data = pd.DataFrame(list(timeline_data.items()), columns=['Saniye', 'Hasar Sayısı']).set_index('Saniye')
                st.area_chart(chart_data, color="#00ffcc")
                st.caption("Grafik, videonun hangi saniyesinde ne kadar hasar tespit edildiğini gösterir.")