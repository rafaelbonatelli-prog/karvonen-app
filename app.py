import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

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
    options=[30, 60, 90, 120], 
    index=1, # O padrão será 60 segundos
    help="Janelas maiores ignoram picos curtos do GPS (melhor para longos). Janelas menores mostram dados mais crus (melhor para tiros curtos)."
)

try:
    m, s = pace_input.split(':')
    PACE_LIMIAR_MIN_KM = int(m) + int(s)/60.0
except:
    st.sidebar.error("Formato de Pace inválido. Use MM:SS (ex: 4:40)")
    PACE_LIMIAR_MIN_KM = 4 + 40/60.0

VELOCIDADE_LIMIAR_MS = 1000 / (PACE_LIMIAR_MIN_KM * 60)
FC_RESERVA_RANGE = fc_limiar - fc_repouso

# --- UPLOAD DE MÚLTIPLOS ARQUIVOS ---
arquivos_tcx = st.file_uploader("Selecione um ou mais treinos (.tcx)", type=['tcx'], accept_multiple_files=True)

if arquivos_tcx:
    with st.spinner("Processando o histórico de treinos..."):
        historico = []
        NS = {'ns': 'http://www.garmin.com/xmlschemas/TrainingCenterDatabase/v2', 'ns3': 'http://www.garmin.com/xmlschemas/ActivityExtension/v2'}
        
        for arquivo in arquivos_tcx:
            try:
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
                
                df['velocidade_suave'] = df['speed'].rolling(window=JANELA_SUAVIZACAO_SEG, min_periods=15).mean()
                df['fc_suave'] = df['heart_rate'].rolling(window=JANELA_SUAVIZACAO_SEG, min_periods=15).mean()
                
                df['pct_fc_reserva'] = (df['fc_suave'] - fc_repouso) / FC_RESERVA_RANGE
                df['pct_velocidade_threshold'] = df['velocidade_suave'] / VELOCIDADE_LIMIAR_MS
                df['idx_karvonen'] = df['pct_velocidade_threshold'] - df['pct_fc_reserva']
                
                # Resumo para a tabela
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
                    'Pace': pace_str,
                    'FC Média': round(fc_media, 0),
                    'Eficiência': f"{saldo_global:+.1f}%",
                    'df': df,
                    'saldo_num': saldo_global
                })
            except Exception as e:
                st.error(f"Erro ao processar {arquivo.name}: {e}")

        if historico:
            # Ordenar do treino mais antigo para o mais recente
            historico = sorted(historico, key=lambda x: x['Data Original'])
            
            # 1. TABELA DE COMPARAÇÃO
            st.markdown("### 📅 Histórico de Treinos")
            df_historico = pd.DataFrame(historico).drop(columns=['Data Original', 'df', 'saldo_num'])
            st.dataframe(df_historico, use_container_width=True, hide_index=True)
            
            # 2. SELETOR DE TREINO PARA O GRÁFICO
            st.markdown("### 🔬 Análise Detalhada")
            opcoes_treino = [f"{h['Data']} - {h['Distância (km)']}km ({h['Eficiência']})" for h in historico]
            
            # O selectbox escolhe por padrão o último treino da lista (o mais recente)
            treino_selecionado = st.selectbox("Escolha um treino para gerar o gráfico Karvonen:", opcoes_treino, index=len(opcoes_treino)-1)
            
            idx_selecionado = opcoes_treino.index(treino_selecionado)
            treino_dados = historico[idx_selecionado]
            df_plot = treino_dados['df']
            
            # 3. MÉTRICAS RÁPIDAS
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Distância", f"{treino_dados['Distância (km)']} km")
            col2.metric("Tempo", f"{int(df_plot['elapsed_time'].max() / 60)} min")
            col3.metric("FC Média", f"{treino_dados['FC Média']} bpm")
            col4.metric("Karvonen", treino_dados['Eficiência'], delta=treino_dados['Eficiência'], delta_color="normal")
            
            # 4. GRÁFICO PRINCIPAL
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
            x_axis = df_plot['distance'] / 1000
            
            ax1.set_ylabel('% Limiar', fontweight='bold')
            ax1.plot(x_axis, df_plot['pct_velocidade_threshold']*100, color='blue', label='% Velocidade', linewidth=2)
            ax1.plot(x_axis, df_plot['pct_fc_reserva']*100, color='red', label='% FC Reserva', linewidth=2)
            ax1.axhline(100, color='gray', linestyle='--', alpha=0.5)
            ax1.legend(loc='upper right')
            ax1.grid(True, linestyle='--', alpha=0.3)
            
            y_axis = df_plot['idx_karvonen']
            ax2.plot(x_axis, y_axis, color='black', linewidth=1.5, alpha=0.6)
            ax2.fill_between(x_axis, y_axis, 0, where=(y_axis >= 0), color='green', alpha=0.3, interpolate=True)
            ax2.fill_between(x_axis, y_axis, 0, where=(y_axis < 0), color='red', alpha=0.3, interpolate=True)
            ax2.axhline(0, color='black', linestyle='--')
            ax2.set_ylabel('Eficiência Real', fontweight='bold')
            ax2.set_xlabel('Distância (km)')
            ax2.grid(True, linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
