import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import io

st.set_page_config(page_title="Karvonen Efficiency App", layout="centered")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
      :root { color-scheme: light dark; }
      body { font-family: 'Inter', Helvetica, Arial, sans-serif; }
      .stMetric { background-color: light-dark(#F8F8F7, #191919); padding: 15px; border-radius: 10px; border: 1px solid light-dark(#E2E3E4, #3E3E3E); }
    </style>
""", unsafe_allow_html=True)

st.title("🏃 Running Efficiency Analyzer")
st.write("Análise de esforço Karvonen versus Ritmo Mecânico.")

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Suas Zonas Biológicas")
fc_repouso = st.sidebar.number_input("FC de Repouso (bpm)", value=45)
fc_limiar = st.sidebar.number_input("FC de Limiar (bpm)", value=157)
pace_input = st.sidebar.text_input("Pace de Limiar (MM:SS)", value="4:40")

st.sidebar.header("📊 Configurações do Gráfico")
JANELA_SUAVIZACAO_SEG = st.sidebar.selectbox(
    "Suavização da Média Móvel (segundos)", 
    options=[30, 45, 60, 90, 120], 
    index=2,
    help="Janelas maiores ignoram picos curtos do GPS. Janelas menores mostram dados crus."
)

try:
    m, s = pace_input.split(':')
    PACE_LIMIAR_MIN_KM = int(m) + int(s)/60.0
except:
    st.sidebar.error("Formato de Pace inválido. Use MM:SS (ex: 4:40)")
    PACE_LIMIAR_MIN_KM = 4 + 40/60.0

VELOCIDADE_LIMIAR_MS = 1000 / (PACE_LIMIAR_MIN_KM * 60)
FC_RESERVA_RANGE = fc_limiar - fc_repouso
CUTOFF_SPEED_MS = 3.2 
STEADY_STATE_TRIM_S = 45 

# --- UPLOAD DE MÚLTIPLOS ARQUIVOS ---
arquivos_tcx = st.file_uploader("Selecione um ou mais treinos (.tcx)", type=['tcx'], accept_multiple_files=True)

if arquivos_tcx:
    with st.spinner("Processando o histórico de treinos..."):
        historico = []
        NS = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2', 'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'}
        
        for arquivo in arquivos_tcx:
            try:
                arquivo.seek(0)
                data = []
                tree = ET.parse(arquivo)
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
                
                if df.empty:
                    continue
                
                df['velocidade_suave'] = df['speed'].rolling(window=JANELA_SUAVIZACAO_SEG, min_periods=15).mean()
                df['fc_suave'] = df['heart_rate'].rolling(window=JANELA_SUAVIZACAO_SEG, min_periods=15).mean()
                
                df['pct_fc_reserva'] = (df['fc_suave'] - fc_repouso) / FC_RESERVA_RANGE
                df['pct_velocidade_threshold'] = df['velocidade_suave'] / VELOCIDADE_LIMIAR_MS
                df['idx_karvonen'] = df['pct_velocidade_threshold'] - df['pct_fc_reserva']
                
                # --- AUTO-LAP E TIROS ---
                df['is_quality_raw'] = df['velocidade_suave'] > CUTOFF_SPEED_MS
                df['block_change'] = df['is_quality_raw'].ne(df['is_quality_raw'].shift()).cumsum()
                df['block_duration'] = df.groupby('block_change')['elapsed_time'].transform(lambda x: x - x.min())
                
                quality_blocks = df[df['is_quality_raw']].groupby('block_change')
                splits = []
                block_num = 1
                for name, group in quality_blocks:
                    dist = group['dist_delta'].sum()
                    if dist > 200: 
                        steady_group = group[group['block_duration'] > STEADY_STATE_TRIM_S]
                        if not steady_group.empty:
                            avg_eff = steady_group['idx_karvonen'].mean()
                        else:
                            avg_eff = group['idx_karvonen'].mean()
                        splits.append({'km': block_num, 'avg_eff': avg_eff, 'x_pos': group['distance'].iloc[-1]})
                        block_num += 1
                
                # Resumo
                data_treino = df['time'].min()
                distancia_total = df['distance'].max() / 1000
                tempo_total_min = df['elapsed_time'].max() / 60
                pace_decimal = tempo_total_min / distancia_total
                pace_str = f"{int(pace_decimal)}:{int((pace_decimal%1)*60):02d}"
                fc_media = df['heart_rate'].mean()
                saldo_global = df['idx_karvonen'].mean() * 100
                
                historico.append({
                    'Data Original': data_treino,
                    'Data': data_treino.strftime('%d/%m/%Y'),
                    'Distância (km)': round(distancia_total, 2),
