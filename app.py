import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


# 1. Menyiapkan data historis sederhana
# Fitur: [Iklan (Juta), Diskon (%)]
X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
# Target: Keuntungan (Juta)
y_train = np.array([50, 80, 110, 90, 150])

# 2. Melatih model (Mesin Replika)
model = LinearRegression().fit(X_train, y_train)

# 3. Menetapkan Skenario Dasar (Baseline)
# Kondisi saat ini: Iklan 10 Juta, Diskon 10%
baseline_input = np.array([[10, 10]])
baseline_pred = model.predict(baseline_input)[0]

print(f"Baseline: Rp {baseline_pred:.2f} Juta")


def run_simulation(new_iklan, new_diskon):
    # Input baru dari user (Intervensi)
    intervention_input = np.array([[new_iklan, new_diskon]])

    # Prediksi hasil intervensi
    prediction = model.predict(intervention_input)[0]

    # Menghitung Delta (Selisih)
    delta_y = prediction - baseline_pred

    return prediction, delta_y

st.title("🚀 Simulator Kebijakan Ekonomi") 
st.header("Analisis Skenario What-If") 
st.write("Aplikasi ini mensimulasikan dampak perubahan variabel terhadap keuntungan.") 

# --- CSS KHUSUS SLIDER (Letakkan di sini agar terbaca) ---
st.markdown("""
    <style>
    /* Menargetkan track slider agar menjadi gradasi warna */
    div[data-testid="stSlider"] input[type=range] {
        background: linear-gradient(to right, #ff4b4b, #ffda4b, #4bff7d) !important;
        height: 10px !important;
        border-radius: 5px !important;
        appearance: none !important;
    }
    /* Menghilangkan tick bar bawaan agar tampilan bersih */
    div[data-testid="stSlider"] > div[data-testid="stTickBar"] {
        display: none !important;
    }
    /* Mengubah bentuk thumb (tombol geser) agar lebih elegan */
    div[data-testid="stSlider"] input[type=range]::-webkit-slider-thumb {
        background: #f0f2f6 !important;
        border: 2px solid #888 !important;
        width: 18px !important;
        height: 18px !important;
        border-radius: 50% !important;
        cursor: default !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: Variabel Kontrol ---
st.sidebar.header("Tuas Kebijakan (Intervensi)")
# Widget Slider: (Label, Nilai Min, Nilai Max, Nilai Default) 
iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta)", 0, 50, 10)
diskon_slider = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 10)

# --- ENGINE: Jalankan Simulasi ---
hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

# --- UI SLIDER GRADASI ---
st.write("### Skala Keuntungan Relatif")
# Kita gunakan key yang unik untuk slider indikator ini
st.slider("indikator", 0, 200, int(hasil_pred), disabled=True, label_visibility="collapsed")
col_label1, col_label2, col_label3 = st.columns(3)
col_label1.write("0%")
col_label2.write("50%")
col_label3.write("100%")

# --- UI: Tampilkan Hasil ---
col1, col2 = st.columns(2)
col1.metric("Prediksi Keuntungan", f"Rp {hasil_pred:.2f} Jt", f"{delta:.2f} Jt")
col2.write(f"Skenario ini menghasilkan perubahan sebesar {delta:.2f} Juta dibandingkan kondisi "
            f"baseline.")

st.subheader("📊 Kontribusi Variabel")

kontribusi_iklan = model.coef_[0] * iklan_slider
kontribusi_diskon = model.coef_[1] * diskon_slider

col1, col2 = st.columns(2)

col1.metric(
    "Kontribusi Iklan",
    f"{kontribusi_iklan:.2f}"
)

col2.metric(
    "Kontribusi Diskon",
    f"{kontribusi_diskon:.2f}"
)
# Visualisasi Perbandingan
data_plot = pd.DataFrame({
    'Skenario': ['Baseline', 'Intervensi'],
    'Keuntungan': [baseline_pred, hasil_pred]
})
st.bar_chart(data=data_plot, x='Skenario', y='Keuntungan')
