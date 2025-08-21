import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Judul dashboard
st.title('Dashboard Analisis Kualitas Udara')
st.markdown('Analisis polutan utama dan tren kualitas udara berdasarkan dataset main_data.csv')
st.markdown('---')

# Load data dari main_data.csv
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('PRSA_Data_Dongsi_20130301-20170228.csv')
        st.success("✅ Data berhasil dimuat dari PRSA_Data_Dongsi_20130301-20170228.csv")
        
        # Convert kolom timestamp jika ada
        timestamp_cols = ['timestamp', 'date', 'time', 'datetime', 'tanggal']
        for col in timestamp_cols:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col])
                break
        
        return data
    except FileNotFoundError:
        st.error("❌ File main_data.csv tidak ditemukan. Pastikan file berada di direktori yang sama.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

# Memuat data
data = load_data()

if data.empty:
    st.warning("Tidak ada data yang dapat dimuat. Dashboard tidak dapat dilanjutkan.")
    st.stop()

# Sidebar untuk informasi dataset
st.sidebar.header('Informasi Dataset')
st.sidebar.write(f"Jumlah baris: {data.shape[0]}")
st.sidebar.write(f"Jumlah kolom: {data.shape[1]}")
st.sidebar.write(f"Periode data: {data.iloc[0,0]} hingga {data.iloc[-1,0]}" if 'timestamp' in data.columns else "Tidak ada kolom timestamp")

# Identifikasi kolom
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
datetime_columns = data.select_dtypes(include=[np.datetime64]).columns.tolist()
timestamp_col = datetime_columns[0] if datetime_columns else None

# Sidebar untuk filter
st.sidebar.header('Filter Data')
selected_pollutant = st.sidebar.selectbox('Pilih Polutan', numeric_columns)

# Tampilkan preview data
st.header('Preview Data')
st.write("5 baris pertama dari main_data.csv:")
st.dataframe(data.head())

# Statistik deskriptif
st.header('📊 Statistik Deskriptif Polutan')

# Statistik untuk polutan terpilih
st.subheader(f'Statistik untuk {selected_pollutant}')
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rata-rata", f"{data[selected_pollutant].mean():.2f}")
with col2:
    st.metric("Median", f"{data[selected_pollutant].median():.2f}")
with col3:
    st.metric("Std Dev", f"{data[selected_pollutant].std():.2f}")
with col4:
    st.metric("Range", f"{data[selected_pollutant].min():.2f} - {data[selected_pollutant].max():.2f}")

# Tabel statistik lengkap
st.subheader('Statistik Lengkap Semua Polutan')
stats_all = data[numeric_columns].describe().round(3)
st.dataframe(stats_all)

# Analisis polutan utama
st.header('🥇 Polutan Utama')
pollutant_means = data[numeric_columns].mean().sort_values(ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.subheader('Rata-rata Konsentrasi')
    for i, (pollutant, value) in enumerate(pollutant_means.items()):
        st.write(f"{i+1}. {pollutant}: {value:.2f}")

with col2:
    fig_bar = px.bar(x=pollutant_means.index, y=pollutant_means.values,
                    labels={'x': 'Polutan', 'y': 'Rata-rata Konsentrasi'},
                    title='Rata-rata Konsentrasi Polutan')
    st.plotly_chart(fig_bar)

# Analisis tren waktu jika ada timestamp
if timestamp_col:
    st.header('📈 Tren Kualitas Udara dari Waktu ke Waktu')
    
    # Pilihan agregasi
    aggregation = st.selectbox('Pilih Agregasi Waktu', ['Harian', 'Mingguan', 'Bulanan', 'Tahunan'])
    
    freq_map = {'Harian': 'D', 'Mingguan': 'W', 'Bulanan': 'M', 'Tahunan': 'Y'}
    freq = freq_map[aggregation]
    
    # Agregasi data
    aggregated_data = data.groupby(pd.Grouper(key=timestamp_col, freq=freq))[selected_pollutant].mean().reset_index()
    aggregated_data = aggregated_data.dropna()  # Hapus nilai NaN
    
    if not aggregated_data.empty:
        # Hitung perubahan tren
        first_value = aggregated_data[selected_pollutant].iloc[0]
        last_value = aggregated_data[selected_pollutant].iloc[-1]
        trend_change = ((last_value - first_value) / first_value) * 100
        
        st.metric(
            label=f"Perubahan Tren {aggregation}",
            value=f"{last_value:.2f}",
            delta=f"{trend_change:.1f}%",
            delta_color="inverse" if trend_change > 0 else "normal"
        )
        
        # Plot tren
        fig_trend = px.line(aggregated_data, x=timestamp_col, y=selected_pollutant,
                           title=f'Tren {selected_pollutant} ({aggregation})',
                           labels={timestamp_col: 'Waktu', selected_pollutant: 'Konsentrasi'})
        st.plotly_chart(fig_trend)
        
        # Tampilkan data agregat
        with st.expander("Lihat Data Agregat"):
            st.dataframe(aggregated_data)
    else:
        st.warning("Tidak ada data yang cukup untuk analisis tren.")
else:
    st.warning("⚠️ Tidak ditemukan kolom timestamp untuk analisis tren waktu.")

# Analisis korelasi antar polutan
st.header('🔗 Korelasi Antar Polutan')
if len(numeric_columns) > 1:
    correlation_matrix = data[numeric_columns].corr()
    
    fig_corr = px.imshow(correlation_matrix,
                        title='Matriks Korelasi Antar Polutan',
                        color_continuous_scale='RdBu_r',
                        aspect="auto")
    st.plotly_chart(fig_corr)
    
    # Tampilkan korelasi terkuat
    st.subheader('Korelasi Terkuat')
    correlations = []
    for i in range(len(numeric_columns)):
        for j in range(i+1, len(numeric_columns)):
            corr_value = correlation_matrix.iloc[i, j]
            correlations.append((numeric_columns[i], numeric_columns[j], corr_value))
    
    # Urutkan berdasarkan absolute value
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for i, (col1, col2, corr) in enumerate(correlations[:3]):
        st.write(f"{i+1}. {col1} vs {col2}: {corr:.3f}")
else:
    st.info("Minimal 2 polutan diperlukan untuk analisis korelasi.")

# Download data hasil analisis
st.header('📥 Ekspor Data')
if st.button('Download Statistik Deskriptif sebagai CSV'):
    stats_all.to_csv('statistik_polutan.csv')
    st.success('File statistik_polutan.csv berhasil diunduh!')

# Footer
st.markdown('---')
st.caption('Dashboard Analisis Kualitas Udara | Data sumber: main_data.csv | Dibuat dengan Streamlit')
