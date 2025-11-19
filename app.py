import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd # Grafik verisi için 

# --- SAYFA AYARLARI ---
st.set_page_config(
    page_title="AI Yol Hasar Tespiti",
    page_icon="🛣️",
    layout="wide"
)

# --- CSS İLE MODERN GÖRÜNÜM ---
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

# --- BAŞLIK VE AÇIKLAMA ---
st.title("🛣️ AI Destekli Yol Hasar Analizi")
st.markdown("<h5 style='text-align: center; color: gray;'>Yüksek Performanslı İşleme ve Raporlama Modu</h5>", unsafe_allow_html=True)

# --- KENAR ÇUBUĞU (Ayarlar) ---
with st.sidebar:
    st.header("⚙️ Analiz Ayarları")
    model_path = 'best.pt' 
    
    conf_threshold = st.slider("Güven Eşiği (Confidence)", 0.10, 1.0, 0.25, 0.05)
    
    st.info("ℹ️ Video işlendikten sonra grafik ve sonuçlar gösterilir.")
    st.write("---")
    st.write("Geliştirici: Anıl GÜMÜŞ")

# --- RENK PALETİ ---
COLORS = {
    0: (255, 140, 0),   # Timsah (RGB)
    1: (0, 255, 255),   # Boyuna (RGB)
    2: (255, 0, 80),    # Cukur (RGB)
    3: (50, 255, 50)    # Enine (RGB)
}
NAMES = {0: "Timsah Sirti", 1: "Boyuna Catlak", 2: "Cukur/Obruk", 3: "Enine Catlak"}

# --- İŞLEME FONKSİYONU ---
def process_entire_video(input_path, output_path, model, conf_thresh):
    """Videoyu işler, kaydeder ve saniye bazlı istatistik toplar"""
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    except:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    stats = {} # Toplam sayılar
    timeline_data = {} # Saniye başına hasar sayısı {0.sn: 2, 1.sn: 0, ...}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        current_second = int(frame_count / fps) # Hangi saniyedeyiz?
        
        # --- MODEL TAHMİNİ ---
        results = model(frame, conf=conf_thresh, verbose=False)
        
        # --- GRAFİK VERİSİ TOPLAMA ---
        # Bu karede kaç hasar var?
        detections_in_frame = len(results[0].boxes)
        
        # O saniyedeki toplam hasar skoruna ekle
        # (Not: Bu yöntem o saniyedeki tüm karelerdeki toplam tespit sayısını biriktirir, 
        # yoğunluğu göstermek için idealdir)
        timeline_data[current_second] = timeline_data.get(current_second, 0) + detections_in_frame
        
        # --- ÇİZİM VE İSTATİSTİK ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        overlay = Image.new('RGBA', pil_img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                color_rgb = COLORS.get(cls_id, (255, 255, 255))
                name = NAMES.get(cls_id, "Bilinmeyen")
                stats[name] = stats.get(name, 0) + 1
                
                # Yarı saydam kutu
                draw.rectangle([x1, y1, x2, y2], fill=color_rgb + (50,), outline=color_rgb, width=3)

        pil_img.paste(overlay, (0,0), overlay)
        
        final_frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        out.write(final_frame)
        
        if frame_count % 5 == 0:
            prog = frame_count / total_frames
            progress_bar.progress(prog)
            status_text.text(f"Analiz ediliyor... %{int(prog*100)}")

    cap.release()
    out.release()
    progress_bar.progress(100)
    status_text.empty()
    
    return stats, timeline_data

# --- ANA AKIŞ ---
uploaded_file = st.file_uploader("Analiz edilecek videoyu yükleyin (MP4, AVI, MOV)", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    output_path = os.path.join(tempfile.gettempdir(), "islenmis_video_sonuc.mp4")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎥 Orijinal Video")
        st.video(tfile.name)

    if st.sidebar.button("🚀 ANALİZİ BAŞLAT"):
        try:
            model = YOLO(model_path)
        except Exception as e:
            st.error(f"Model yüklenemedi! Hata: {e}")
            st.stop()
            
        with col2:
            st.subheader("🔍 Sonuç ve Rapor")
            start_time = time.time()
            
            # Fonksiyondan artık 2 değer dönüyor: İstatistikler ve Grafik Verisi
            final_stats, timeline_data = process_entire_video(tfile.name, output_path, model, conf_threshold)
            
            duration = time.time() - start_time
            st.success(f"Analiz {duration:.1f} saniyede tamamlandı!")
            
            try:
                with open(output_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            except:
                st.warning("Video tarayıcıda oynatılamadı.")

            with open(output_path, 'rb') as f:
                st.download_button('📥 İŞLENMİŞ VİDEOYU İNDİR', f, file_name='analiz_sonucu.mp4')

            st.write("---")
            
            # --- 1. METRİK KARTLARI ---
            st.markdown("### 📊 Toplam Hasar Özeti")
            stat_cols = st.columns(4)
            idx = 0
            for damage_name, count in final_stats.items():
                with stat_cols[idx % 4]:
                    st.metric(label=damage_name, value=f"{count}")
                idx += 1
            
            if not final_stats:
                st.info("✅ Temiz Yol: Hiçbir hasar tespit edilmedi.")
            
            # --- 2. ZAMAN ÇİZELGESİ GRAFİĞİ (YENİ) ---
            if timeline_data:
                st.write("---")
                st.markdown("### 📈 Hasar Yoğunluk Grafiği")
                
                # Veriyi Pandas DataFrame'e çevir
                chart_data = pd.DataFrame(
                    list(timeline_data.items()),
                    columns=['Saniye', 'Hasar Yoğunluğu']
                ).set_index('Saniye')
                
                # Alan grafiği çiz
                st.area_chart(chart_data, color="#00ffcc")
                st.caption("Bu grafik, videonun hangi saniyesinde ne kadar yoğun hasar tespit edildiğini gösterir. Yüksek tepeler, yolun o kısmının çok bozuk olduğunu işaret eder.")