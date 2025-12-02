import streamlit as st
import cv2
import tempfile
import os
import time
import subprocess
from ultralytics import YOLO
import pandas as pd
import imageio_ffmpeg

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="Silver Road - Hasar Analiz Platformu",
    page_icon="🛣️",
    layout="wide"
)
# --- YENİ KURUMSAL CSS TEMASI ---
st.markdown("""
<style>
    /* Ana Arka Plan */
    .main {
        background-color: #f8f9fa; 
    }
    /* Başlıklar */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 600;
    }
    /* Sidebar (Yan Menü) */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    /* Butonlar */
    .stButton>button {
        background-color: #e74c3c; /* Kiremit Kırmızısı */
        color: white;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: none;
        height: 50px;
        font-weight: 600;
    }
    .stButton>button:hover {
        background-color: #c0392b;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.2);
        color: white;
    }
    /* Metrik Kutuları */
    div[data-testid="stMetricValue"] {
        color: #e74c3c;
        font-weight: bold;
    }
    /* Progress Bar Rengi */
    .stProgress > div > div > div > div {
        background-color: #e74c3c;
    }
    /* Boş Durum Kutusu (Empty State) */
    .empty-state {
        border: 2px dashed #bdc3c7;
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        color: #7f8c8d;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- MODEL ÖNBELLEKLEME (HIZLANDIRMA) ---
@st.cache_resource
def load_model(path):
    return YOLO(path)

# --- SABİT AYARLAR ---
CLASS_NAMES = {
    0: "Catlak", 
    1: "Cukur", 
    2: "Kasis"
}

COLORS = {
    0: (0, 255, 255),   # Çatlak (Sarı/Cyan)
    1: (255, 0, 0),     # Çukur (Kırmızı)
    2: (255, 165, 0)    # Kasis (Turuncu)
}

# --- YARDIMCI FONKSİYONLAR ---
def convert_video_to_h264(input_path, output_path):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg_exe, '-y', '-i', input_path, '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', output_path
    ]
    if os.name == 'nt':
        subprocess.run(command, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        subprocess.run(command, check=True)

def process_entire_video(input_path, output_path, model, conf_thresh, hood_mask_ratio):
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    temp_output = output_path.replace(".mp4", "_raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stats = {} 
    timeline_data = {} 
    ignore_threshold = int(height * (1 - hood_mask_ratio))

    progress_bar = st.progress(0)
    status_text = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        current_second = int(frame_count / fps)
        results = model(frame, conf=conf_thresh, verbose=False)
        detections_in_frame = 0
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                if (y1 + y2) / 2 > ignore_threshold: continue 

                detections_in_frame += 1
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = CLASS_NAMES.get(cls_id, "Bilinmeyen")
                color_bgr = COLORS.get(cls_id, (255, 255, 255))[::-1]
                
                stats[name] = stats.get(name, 0) + 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
                
                label_text = f"{name} {int(conf * 100)}%"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color_bgr, -1)
                cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if hood_mask_ratio > 0:
            cv2.line(frame, (0, ignore_threshold), (width, ignore_threshold), (0, 0, 255), 1)

        timeline_data[current_second] = timeline_data.get(current_second, 0) + detections_in_frame
        out.write(frame)
        
        if frame_count % 5 == 0:
            prog = frame_count / total_frames
            progress_bar.progress(prog)
            status_text.text(f"İşleniyor... %{int(prog*100)}")

    cap.release()
    out.release()
    progress_bar.progress(100)
    
    status_text.text("Format dönüştürülüyor...")
    try:
        convert_video_to_h264(temp_output, output_path)
        if os.path.exists(temp_output): os.remove(temp_output)
    except Exception as e:
        return stats, timeline_data, False

    status_text.empty()
    return stats, timeline_data, True

# --- ARAYÜZ (SIDEBAR) ---
with st.sidebar:
    # --- LOGO ENTEGRASYONU (DÜZELTİLDİ) ---
    logo_path = "silveroad.png"  # Logo dosyasının adı
    if os.path.exists(logo_path):
        # DÜZELTME BURADA: use_container_width=True kullanıldı
        st.image(logo_path, use_container_width=True)
    else:
        # Logo yoksa metin göster
        st.title("🛣️ Silver Road")
        st.warning(f"'{logo_path}' dosyası bulunamadı.")

    st.markdown("---")
    st.header("⚙️ Analiz Parametreleri")
    
    uploaded_file = st.file_uploader("Video Dosyası Seçin", type=['mp4', 'avi', 'mov'])
    
    st.markdown("### 🎛️ Model Ayarları")
    model_path = 'best.pt'
    conf_threshold = st.slider("Güven Eşiği", 0.10, 1.0, 0.40, 0.05)
    hood_mask = st.slider("🚘 Kaput Maskesi (%)", 0, 50, 15, 5)
    hood_ratio = hood_mask / 100.0
    
    st.markdown("---")
    st.caption("SilveRoad - By Anıl GÜMÜŞ")

# --- ANA EKRAN ---
st.title("Yol Kusur Tespit ve Analiz Sistemi")
st.markdown("Yapay zeka destekli otonom yol denetim arayüzü.")

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown("### 🎥 Kaynak Görüntü")
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        st.video(tfile.name)
    else:
        st.markdown('<div class="empty-state"><h3>Video Bekleniyor</h3><p>Analiz için sol menüden dosya yükleyiniz.</p></div>', unsafe_allow_html=True)

with col2:
    st.markdown("### 📊 Analiz Raporu")
    
    if uploaded_file is None:
        st.markdown('<div class="empty-state"><h3>Sonuçlar</h3><p>İşlem başlatıldığında veriler burada görünecektir.</p></div>', unsafe_allow_html=True)
    
    elif uploaded_file is not None:
        if st.button("ANALİZİ BAŞLAT", use_container_width=True):
            if not os.path.exists(model_path):
                st.error("Model dosyası (best.pt) bulunamadı.")
                st.stop()
                
            try:
                model = load_model(model_path)
            except Exception as e:
                st.error(f"Model Hatası: {e}")
                st.stop()
                
            output_path = os.path.join(os.getcwd(), "sonuc.mp4")
            
            with st.spinner("AI Görüntü İşleme Motoru Çalışıyor..."):
                final_stats, timeline_data, success = process_entire_video(tfile.name, output_path, model, conf_threshold, hood_ratio)
            
            if success:
                st.success("İşlem Başarıyla Tamamlandı.")
                
                # Video Oynatıcı
                if os.path.exists(output_path):
                    with open(output_path, 'rb') as vf:
                        vbytes = vf.read()
                        st.video(vbytes)
                    
                    # Buton Grubu
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button("📥 İşlenmiş Videoyu İndir", vbytes, file_name='detected_road.mp4', mime="video/mp4", use_container_width=True)
                    with c2:
                        if timeline_data:
                            df = pd.DataFrame(list(timeline_data.items()), columns=['Saniye', 'Hasar'])
                            csv = df.to_csv(index=False).encode('utf-8')
                            st.download_button("📊 CSV Raporu İndir", csv, file_name='data.csv', mime='text/csv', use_container_width=True)
                
                # İstatistikler
                st.markdown("#### Tespit Özeti")
                if final_stats:
                    scols = st.columns(len(final_stats))
                    for i, (k, v) in enumerate(final_stats.items()):
                        scols[i].metric(label=k, value=v)
                else:
                    st.info("Kusur tespit edilmedi.")

                # Grafik
                if timeline_data:
                    st.markdown("#### Hasar/Zaman Grafiği")
                    chart_data = pd.DataFrame(list(timeline_data.items()), columns=['Sn', 'Adet']).set_index('Sn')
                    st.line_chart(chart_data)
            else:
                st.error("Video işleme sırasında bir hata oluştu.")