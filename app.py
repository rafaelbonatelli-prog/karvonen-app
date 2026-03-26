import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# 1. CONFIGURAÇÃO DA PÁGINA E DESIGN (O seu CSS limpo)
st.set_page_config(page_title="Karvonen Efficiency App", layout="centered")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
      :root { color-scheme: light dark; }
      body {
        font-family: 'Inter', Helvetica, Arial, sans-serif;
      }
      .stMetric {
        background-color: light-dark(#F8F8F7, #191919);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid light-dark(#E2E3E4, #3E3E3E);
      }
    </style>
""", unsafe_allow_html=True)

# 2. INTERFACE E CABEÇALHO
st.title("🏃 Running Efficiency Analyzer")
st.write("Análise de esforço Karvonen versus Ritmo Mecânico.")

# 3. BARRA LATERAL (ZONAS DO ATLETA)
st.sidebar.header("⚙️ Suas Zonas Biológicas")
fc_repouso = st.sidebar.number_input("FC de Repouso (bpm)", value=45)
fc_limiar = st.sidebar.number_input("FC de Limiar (bpm)", value=157)
pace_input = st.sidebar.text_input("Pace de Limiar (MM:SS)", value="4:40")

# Converter o Pace para matemática
try:
    m, s = pace_input.split(':')
    PACE_LIMIAR_MIN_KM = int(m) + int(s)/60.0
except:
    st.sidebar.error("Formato de Pace inválido. Use MM:SS (ex: 4:40)")
    PACE_LIMIAR_MIN_KM = 4 + 40/60.0

VELOCIDADE_LIMIAR_MS = 1000 / (PACE_LIMIAR_MIN_KM * 60)
FC_RESERVA_RANGE = fc_limiar - fc_repouso
JANELA_SUAVIZACAO_SEG = 45
CUTOFF_SPEED_MS = 3.2

# 4. ÁREA DE UPLOAD
arquivo_tcx = st.file_uploader("Arraste seu treino (.tcx)", type=['tcx'])

if arquivo_tcx is not None:
    with st.spinner("Processando satélites e batimentos..."):
        try:
            # 5. O MOTOR MATEMÁTICO (O nosso script original)
            NS = {
                'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2',
                'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'
            }
            
            data = []
            tree = ET.parse(arquivo_tcx)
            root = tree.getroot()
            
            for trackpoint in root.findall('.//ns:Trackpoint', NS):
                point_data = {}
                time_elem = trackpoint.find('ns:Time', NS)
                if time_elem is not None: point_data['time'] = time_elem.text
                hr_elem = trackpoint.find('.//ns:HeartRateBpm/ns:Value', NS)
                if hr_elem is not None: point_data['heart_rate'] = float(hr_elem.text)
                speed_elem = trackpoint.find('.//ns3:TPX/ns3:Speed', NS)
                if speed_elem is not None: point_data['speed'] = float(speed_elem.text)
                data.append(point_data)
            
            df = pd.DataFrame(data)
            df['time'] = pd.to_datetime(df['time'])
            df.sort_values('time', inplace=True)
            
            df['dt_segundos'] = df['time'].diff().dt.total_seconds().fillna(0)
            df['speed_filled'] = df['speed'].fillna(0)
            df['dist_delta'] = df['speed_filled'] * df['dt_segundos']
            df['distance'] = df['dist_delta'].cumsum()
            df['elapsed_time'] = df['dt_segundos'].cumsum()
            
            df.dropna(subset=['heart_rate', 'speed'], inplace=True)
            df = df[df['speed'] > 0.5] 
            
            df['velocidade_suave'] = df['speed'].rolling(window=JANELA_SUAVIZACAO_SEG, min_periods=15).mean()
            df['fc_suave'] = df['heart_rate'].rolling(window=JANELA_SUAVIZACAO_SEG, min_periods=15).mean()
            
            df['pct_fc_reserva'] = (df['fc_suave'] - fc_repouso) / FC_RESERVA_RANGE
            df['pct_velocidade_threshold'] = df['velocidade_suave'] / VELOCIDADE_LIMIAR_MS
            df['idx_karvonen'] = df['pct_velocidade_threshold'] - df['pct_fc_reserva']
            
            # 6. EXIBIÇÃO DOS RESULTADOS NA TELA
            distancia_total = df['distance'].max() / 1000
            tempo_total_min = df['elapsed_time'].max() / 60
            fc_media = df['heart_rate'].mean()
            saldo_global = df['idx_karvonen'].mean() * 100
            
            # Paineis superiores (Métricas rápidas)
            st.markdown("### Resumo do Motor")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Distância", f"{distancia_total:.2f} km")
            col2.metric("Tempo", f"{tempo_total_min:.0f} min")
            col3.metric("FC Média", f"{fc_media:.0f} bpm")
            col4.metric("Saldo Karvonen", f"{saldo_global:+.1f}%", delta=f"{saldo_global:+.1f}%", delta_color="normal")
            
            # Gráfico Principal
            st.markdown("### Radiografia do Esforço")
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            x_axis = df['distance'] / 1000
            
            ax1.set_ylabel('% Limiar', fontweight='bold')
            ax1.plot(x_axis, df['pct_velocidade_threshold']*100, color='blue', label='% Velocidade', linewidth=2)
            ax1.plot(x_axis, df['pct_fc_reserva']*100, color='red', label='% FC Reserva', linewidth=2)
            ax1.axhline(100, color='gray', linestyle='--', alpha=0.5)
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            y_axis = df['idx_karvonen']
            ax2.plot(x_axis, y_axis, color='black', linewidth=1.5, alpha=0.6)
            ax2.fill_between(x_axis, y_axis, 0, where=(y_axis >= 0), color='green', alpha=0.3, interpolate=True)
            ax2.fill_between(x_axis, y_axis, 0, where=(y_axis < 0), color='red', alpha=0.3, interpolate=True)
            ax2.axhline(0, color='black', linestyle='--')
            ax2.set_ylabel('Eficiência Real', fontweight='bold')
            ax2.set_xlabel('Distância (km)')
            ax2.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")