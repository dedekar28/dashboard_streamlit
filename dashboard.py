import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os
import io

# Judul dashboard
st.title('Dashboard Analisis Kualitas Udara')
st.markdown('Analisis data kualitas udara dari dataset PRSA atau file upload')
st.markdown('---')

# Fungsi untuk membuat sample data PRSA
@st.cache_data(show_spinner=False)
def make_sample_data():
    """Generate sample data PRSA untuk demonstrasi"""
    dates = pd.date_range('2013-03-01', '2017-02-28', freq='H')
    n = len(dates)
    
    # Buat data sample meniru format PRSA
    sample_data = pd.DataFrame({
        'No': range(1, n+1),
        'year': dates.year,
        'month': dates.month,
        'day': dates.day,
        'hour': dates.hour,
        'PM2.5': 35 + 10*np.sin(2*np.pi*np.linspace(0, 4, n)) + np.random.normal(0, 8, n),
        'PM10': 60 + 15*np.cos(2*np.pi*np.linspace(0, 3, n)) + np.random.normal(0, 12, n),
        'SO2': 12 + 3*np.sin(2*np.pi*np.linspace(0, 2, n)) + np.random.normal(0, 2, n),
        'NO2': 25 + 6*np.cos(2*np.pi*np.linspace(0, 1.5, n)) + np.random.normal(0, 4, n),
        'CO': 0.9 + 0.2*np.sin(2*np.pi*np.linspace(0, 5, n)) + np.random.normal(0, 0.1, n),
        'O3': 40 + 8*np.sin(2*np.pi*np.linspace(0, 1, n)) + np.random.normal(0, 5, n),
    })
    
    # Tambahkan missing values seperti dataset asli
    for col in ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']:
        mask = np.random.random(n) < 0.1  # 10% missing
        sample_data.loc[mask, col] = np.nan
    
    # Buat kolom datetime
    sample_data['datetime'] = pd.to_datetime(
        sample_data[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1) + ' ' + 
        sample_data['hour'].astype(str) + ':00:00'
    )
    
    return sample_data

# Fungsi load data dengan opsi multiple
@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None):
    """Load data dari berbagai sumber"""
    if uploaded_file is not None:
        # Load dari uploaded file
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(uploaded_file)
            else:
                st.error("Format file tidak didukung. Gunakan CSV atau Excel.")
                return pd.DataFrame(), None
                
            st.success(f"✅ File {uploaded_file.name} berhasil dimuat")
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return pd.DataFrame(), None
            
    else:
        # Coba load dari file PRSA
        filename = "PRSA_Data_Dongsi_20130301-20170228.csv"
        if os.path.exists(filename):
            try:
                data = pd.read_csv(filename)
                st.success(f"✅ Data berhasil dimuat dari {filename}")
            except Exception as e:
                st.error(f"❌ Error loading {filename}: {str(e)}")
                return pd.DataFrame(), None
        else:
            # Gunakan sample data
            data = make_sample_data()
            st.info("ℹ️ Menggunakan data sample untuk demonstrasi")
    
    # Preprocessing data
    data = data.replace(-999, np.nan)  # Handle missing values format PRSA
    
    # Coba buat kolom datetime
    datetime_col = None
    if all(col in data.columns for col in ['year', 'month', 'day', 'hour']):
        try:
            data['datetime'] = pd.to_datetime(
                data[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1) + ' ' + 
                data['hour'].astype(str) + ':00:00'
            )
            datetime_col = 'datetime'
        except:
            pass
    
    # Coba kolom datetime lainnya
    if datetime_col is None:
        datetime_cols = ['date', 'time', 'timestamp', 'datetime']
        for col in datetime_cols:
            if col in data.columns:
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                    datetime_col = col
                    break
                except:
                    continue
    
    return data, datetime_col

# Sidebar untuk pemilihan data
st.sidebar.header('Sumber Data')

data_option = st.sidebar.radio(
    "Pilih sumber data:",
    ["File PRSA (otomatis)", "Upload File", "Gunakan Sample Data"]
)

uploaded_file = None
if data_option == "Upload File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload file CSV atau Excel",
        type=['csv', 'xlsx', 'xls'],
        help="Upload file data kualitas udara Anda"
    )
elif data_option == "Gunakan Sample Data":
    st.sidebar.info("Akan menggunakan data sample")

# Memuat data
data, datetime_col = load_data(uploaded_file)

if data.empty:
    st.warning("""
    ❌ Tidak ada data yang dapat dimuat. 
    
    **Pilihan:**
    1. Upload file data Anda
    2. Gunakan sample data
    3. Pastikan file PRSA_Data_Dongsi_20130301-20170228.csv berada di folder yang sama
    """)
    
    # Tampilkan instruksi
    with st.expander("📋 Cara Menggunakan Dashboard"):
        st.write("""
        1. **Upload File**: Klik 'Upload File' dan pilih file CSV/Excel Anda
        2. **Sample Data**: Pilih 'Gunakan Sample Data' untuk demo
        3. **File PRSA**: Letakkan file PRSA_Data_Dongsi_20130301-20170228.csv di folder ini
        
        **Format Data yang Didukung:**
        - Kolom polutan: PM2.5, PM10, SO2, NO2, CO, O3
        - Kolom datetime: timestamp, datetime, atau kolom year/month/day/hour terpisah
        """)
    
    st.stop()

# Identifikasi kolom polutan
prsa_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
available_pollutants = [col for col in prsa_pollutants if col in data.columns]

# Jika tidak ada polutan standar, cari kolom numerik lainnya
if not available_pollutants:
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    available_pollutants = numeric_columns[:6]  # Ambil maksimal 6 kolom numerik

# Sidebar untuk filter
st.sidebar.header('Konfigurasi Analisis')

# Dropdown untuk polutan
if available_pollutants:
    selected_pollutant = st.sidebar.selectbox(
        'Pilih Polutan', 
        available_pollutants,
        help="Pilih polutan untuk dianalisis"
    )
else:
    selected_pollutant = None
    st.sidebar.error("❌ Tidak ditemukan kolom polutan")

# Informasi dataset
st.sidebar.header('Informasi Dataset')
st.sidebar.write(f"Jumlah baris: {data.shape[0]:,}")
st.sidebar.write(f"Jumlah kolom: {data.shape[1]}")
if datetime_col:
    date_min = data[datetime_col].min()
    date_max = data[datetime_col].max()
    if pd.notna(date_min) and pd.notna(date_max):
        st.sidebar.write(f"Periode: {date_min.strftime('%Y-%m-%d')} hingga {date_max.strftime('%Y-%m-%d')}")

# Tampilkan preview data
st.header('📋 Preview Data')
st.write(f"5 baris pertama dari dataset:")
st.dataframe(data.head())

# Tampilkan informasi dataset
with st.expander("📊 Informasi Dataset"):
    st.write("**Kolom Polutan:**", available_pollutants)
    st.write("**Kolom Datetime:**", datetime_col if datetime_col else "Tidak ditemukan")
    st.write("**Kolom Lainnya:**", [col for col in data.columns if col not in available_pollutants + [datetime_col]])
    
    # Info missing values
    st.subheader("Missing Values")
    missing_info = data[available_pollutants].isnull().sum()
    if missing_info.sum() > 0:
        for col, count in missing_info.items():
            if count > 0:
                st.write(f"- {col}: {count} missing values ({count/len(data)*100:.1f}%)")
    else:
        st.write("Tidak ada missing values")

# Statistik deskriptif
st.header('📊 Statistik Deskriptif Polutan')

if selected_pollutant:
    # Hitung statistik
    stats = data[selected_pollutant].describe()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rata-rata", f"{stats['mean']:.2f}")
        st.metric("Median", f"{stats['50%']:.2f}")
    with col2:
        st.metric("Std Dev", f"{stats['std']:.2f}")
        st.metric("Range", f"{stats['min']:.2f} - {stats['max']:.2f}")
    with col3:
        st.metric("Percentile 25%", f"{stats['25%']:.2f}")
        st.metric("Percentile 75%", f"{stats['75%']:.2f}")
    with col4:
        count = stats['count']
        missing = len(data) - count
        st.metric("Jumlah Data", f"{int(count):,}")
        st.metric("Data Hilang", f"{missing:,} ({missing/len(data)*100:.1f}%)")

# Statistik lengkap
st.subheader('Statistik Lengkap Semua Polutan')
if available_pollutants:
    stats_all = data[available_pollutants].describe().round(3)
    st.dataframe(stats_all)

# Analisis tren waktu
if datetime_col and selected_pollutant:
    st.header('📈 Analisis Tren Waktu')
    
    # Pilihan agregasi
    aggregation = st.selectbox('Tingkat Agregasi', ['Harian', 'Mingguan', 'Bulanan', 'Tahunan'])
    freq_map = {'Harian': 'D', 'Mingguan': 'W', 'Bulanan': 'M', 'Tahunan': 'Y'}
    
    try:
        # Agregasi data
        trend_data = data[[datetime_col, selected_pollutant]].dropna()
        trend_data = trend_data.set_index(datetime_col)
        aggregated_data = trend_data.resample(freq_map[aggregation]).mean().reset_index()
        
        if not aggregated_data.empty:
            fig = px.line(aggregated_data, x=datetime_col, y=selected_pollutant,
                         title=f'Tren {selected_pollutant} ({aggregation})',
                         labels={datetime_col: 'Waktu', selected_pollutant: 'Konsentrasi'})
            st.plotly_chart(fig)
        else:
            st.warning("Tidak ada data yang cukup untuk analisis tren")
    except Exception as e:
        st.error(f"Error dalam analisis tren: {str(e)}")

# Analisis polutan utama
st.header('🥇 Polutan Utama')
if available_pollutants:
    pollutant_means = data[available_pollutants].mean().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Ranking Polutan')
        for i, (pollutant, value) in enumerate(pollutant_means.items()):
            st.write(f"{i+1}. **{pollutant}**: {value:.2f}")
    
    with col2:
        fig_bar = px.bar(x=pollutant_means.index, y=pollutant_means.values,
                        title='Rata-rata Konsentrasi Polutan',
                        labels={'x': 'Polutan', 'y': 'Konsentrasi Rata-rata'})
        st.plotly_chart(fig_bar)

# Footer
st.markdown('---')
st.caption('Dashboard Analisis Kualitas Udara | Dibuat dengan Streamlit')

# Tampilkan warning jika menggunakan sample data
if data_option == "Gunakan Sample Data":
    st.warning("⚠️ Sedang menggunakan data sample. Upload file Anda sendiri untuk analisis data aktual.")
