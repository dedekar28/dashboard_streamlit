import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import os

# Judul dashboard
st.title('Dashboard Analisis Kualitas Udara - PRSA Dongsi')
st.markdown('Analisis data kualitas udara dari stasiun Dongsi (2013-2017)')
st.markdown('---')

# Load data dari PRSA_Data_Dongsi_20130301-20170228.csv
@st.cache_data(show_spinner=False)
def load_data():
    """Load data dari PRSA_Data_Dongsi_20130301-20170228.csv"""
    filename = "PRSA_Data_Dongsi_20130301-20170228.csv"
    
    try:
        if os.path.exists(filename):
            data = pd.read_csv(filename)
            st.success(f"✅ Data berhasil dimuat dari {filename}")
            
            # Preprocessing data PRSA
            # Handle missing values (biasanya ditandai dengan NaN atau -999)
            data = data.replace(-999, np.nan)
            
            # Gabungkan kolom tahun, bulan, hari, jam menjadi datetime
            if all(col in data.columns for col in ['year', 'month', 'day', 'hour']):
                data['datetime'] = pd.to_datetime(
                    data[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1) + ' ' + data['hour'].astype(str) + ':00:00'
                )
                datetime_col = 'datetime'
            else:
                # Coba kolom datetime lainnya
                datetime_cols = ['date', 'time', 'timestamp']
                for col in datetime_cols:
                    if col in data.columns:
                        data[col] = pd.to_datetime(data[col], errors='coerce')
                        datetime_col = col
                        break
                else:
                    datetime_col = None
            
            return data, datetime_col
            
        else:
            st.error(f"❌ File {filename} tidak ditemukan.")
            st.info("Pastikan file PRSA_Data_Dongsi_20130301-20170228.csv berada di direktori yang sama")
            return pd.DataFrame(), None
            
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame(), None

# Memuat data
data, datetime_col = load_data()

if data.empty:
    st.warning("Tidak ada data yang dapat dimuat. Dashboard tidak dapat dilanjutkan.")
    st.stop()

# Identifikasi kolom polutan pada dataset PRSA
# Kolom polutan umum pada dataset PRSA
prsa_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
available_pollutants = [col for col in prsa_pollutants if col in data.columns]

# Kolom meteorologi (jika ada)
meteo_columns = ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']

# Kolom lainnya
other_columns = [col for col in data.columns if col not in available_pollutants + meteo_columns + ['year', 'month', 'day', 'hour', 'No', 'station']]

# Sidebar untuk filter
st.sidebar.header('Konfigurasi Analisis')

# Informasi dataset
st.sidebar.header('Informasi Dataset')
st.sidebar.write(f"Jumlah baris: {data.shape[0]:,}")
st.sidebar.write(f"Jumlah kolom: {data.shape[1]}")
st.sidebar.write(f"Stasiun: Dongsi")
if datetime_col:
    date_min = data[datetime_col].min()
    date_max = data[datetime_col].max()
    st.sidebar.write(f"Periode: {date_min.strftime('%Y-%m-%d')} hingga {date_max.strftime('%Y-%m-%d')}")

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

# Tampilkan preview data
st.header('📋 Preview Data')
st.write("5 baris pertama dari dataset PRSA Dongsi:")
st.dataframe(data.head())

# Tampilkan informasi kolom
with st.expander("Lihat Informasi Kolom Dataset"):
    st.write("**Kolom Polutan:**", available_pollutants)
    st.write("**Kolom Meteorologi:**", [col for col in meteo_columns if col in data.columns])
    st.write("**Kolom Lainnya:**", other_columns)
    st.write("**Kolom Datetime:**", datetime_col if datetime_col else "Tidak ditemukan")

# Statistik deskriptif
st.header('📊 Statistik Deskriptif Polutan')

if selected_pollutant:
    # Hitung statistik untuk polutan terpilih
    pollutant_data = data[selected_pollutant].dropna()
    
    if len(pollutant_data) > 0:
        stats = {
            'Rata-rata': pollutant_data.mean(),
            'Median': pollutant_data.median(),
            'Std Dev': pollutant_data.std(),
            'Minimum': pollutant_data.min(),
            'Maksimum': pollutant_data.max(),
            'Percentile 25%': pollutant_data.quantile(0.25),
            'Percentile 75%': pollutant_data.quantile(0.75),
            'Jumlah Data': len(pollutant_data),
            'Data Hilang': data[selected_pollutant].isnull().sum(),
            'Persentase Hilang': (data[selected_pollutant].isnull().sum() / len(data)) * 100
        }
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rata-rata", f"{stats['Rata-rata']:.2f}")
            st.metric("Median", f"{stats['Median']:.2f}")
        with col2:
            st.metric("Std Dev", f"{stats['Std Dev']:.2f}")
            st.metric("Range", f"{stats['Minimum']:.2f} - {stats['Maksimum']:.2f}")
        with col3:
            st.metric("Percentile 25%", f"{stats['Percentile 25%']:.2f}")
            st.metric("Percentile 75%", f"{stats['Percentile 75%']:.2f}")
        with col4:
            st.metric("Jumlah Data", f"{stats['Jumlah Data']:,}")
            st.metric("Data Hilang", f"{stats['Data Hilang']:,} ({stats['Persentase Hilang']:.1f}%)")
    else:
        st.warning(f"Tidak ada data yang valid untuk polutan {selected_pollutant}")

# Statistik lengkap semua polutan
st.subheader('Statistik Lengkap Semua Polutan')
if available_pollutants:
    stats_all = data[available_pollutants].describe().round(3)
    st.dataframe(stats_all)
else:
    st.warning("Tidak ada polutan yang dapat dianalisis")

# Analisis tren waktu jika ada datetime
if datetime_col and selected_pollutant:
    st.header('📈 Analisis Tren Waktu')
    
    # Filter data untuk polutan terpilih
    trend_data = data[[datetime_col, selected_pollutant]].dropna()
    
    if len(trend_data) > 0:
        # Pilihan agregasi
        aggregation = st.selectbox('Tingkat Agregasi', ['Harian', 'Mingguan', 'Bulanan', 'Tahunan'])
        freq_map = {'Harian': 'D', 'Mingguan': 'W', 'Bulanan': 'M', 'Tahunan': 'Y'}
        
        try:
            # Set index untuk grouping
            trend_data = trend_data.set_index(datetime_col)
            
            # Agregasi data
            aggregated_data = trend_data.resample(freq_map[aggregation]).mean().reset_index()
            aggregated_data = aggregated_data.dropna()
            
            if not aggregated_data.empty:
                # Hitung perubahan tren
                first_value = aggregated_data[selected_pollutant].iloc[0]
                last_value = aggregated_data[selected_pollutant].iloc[-1]
                trend_change = ((last_value - first_value) / first_value) * 100 if first_value != 0 else 0
                
                # Tampilkan metric tren
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        f"Konsentrasi Terakhir ({aggregation})",
                        f"{last_value:.2f} μg/m³"
                    )
                with col2:
                    st.metric(
                        f"Perubahan Tren",
                        f"{trend_change:.1f}%",
                        delta_color="inverse" if trend_change > 0 else "normal"
                    )
                
                # Plot tren
                fig = px.line(aggregated_data, x=datetime_col, y=selected_pollutant,
                             title=f'Tren {selected_pollutant} di Stasiun Dongsi ({aggregation})',
                             labels={datetime_col: 'Waktu', selected_pollutant: 'Konsentrasi (μg/m³)'})
                st.plotly_chart(fig)
                
            else:
                st.warning("Tidak ada data yang cukup untuk analisis tren.")
        except Exception as e:
            st.error(f"Error dalam analisis tren: {str(e)}")
    else:
        st.warning("Tidak ada data yang valid untuk analisis tren.")

# Analisis polutan utama
st.header('🥇 Polutan Utama')
if available_pollutants:
    pollutant_means = data[available_pollutants].mean().sort_values(ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Ranking Polutan berdasarkan Konsentrasi Rata-rata')
        for i, (pollutant, value) in enumerate(pollutant_means.items()):
            st.write(f"{i+1}. **{pollutant}**: {value:.2f} μg/m³")
    
    with col2:
        fig_bar = px.bar(x=pollutant_means.index, y=pollutant_means.values,
                        title='Rata-rata Konsentrasi Polutan di Stasiun Dongsi',
                        labels={'x': 'Polutan', 'y': 'Konsentrasi Rata-rata (μg/m³)'})
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar)

# Analisis distribusi
st.header('📊 Distribusi Polutan')
if selected_pollutant:
    fig_hist = px.histogram(data, x=selected_pollutant,
                           title=f'Distribusi {selected_pollutant} di Stasiun Dongsi',
                           labels={selected_pollutant: 'Konsentrasi (μg/m³)', 'count': 'Frekuensi'})
    st.plotly_chart(fig_hist)

# Analisis musiman jika ada bulan
if 'month' in data.columns and selected_pollutant:
    st.header('🌤️ Analisis Musiman')
    
    seasonal_data = data.groupby('month')[selected_pollutant].mean().reset_index()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    seasonal_data['month_name'] = seasonal_data['month'].apply(lambda x: month_names[x-1] if 1 <= x <= 12 else f'Month {x}')
    
    fig_seasonal = px.bar(seasonal_data, x='month_name', y=selected_pollutant,
                         title=f'Konsentrasi Rata-rata {selected_pollutant} per Bulan',
                         labels={'month_name': 'Bulan', selected_pollutant: 'Konsentrasi Rata-rata (μg/m³)'})
    st.plotly_chart(fig_seasonal)

# Analisis korelasi antar polutan
if len(available_pollutants) > 1:
    st.header('🔗 Korelasi Antar Polutan')
    
    correlation_matrix = data[available_pollutants].corr()
    
    fig_corr = px.imshow(correlation_matrix,
                        title='Matriks Korelasi Antar Polutan di Stasiun Dongsi',
                        color_continuous_scale='RdBu_r',
                        aspect="auto",
                        zmin=-1, zmax=1)
    st.plotly_chart(fig_corr)

# Footer
st.markdown('---')
st.caption('Dashboard Analisis Kualitas Udara | Data: PRSA_Data_Dongsi_20130301-20170228.csv | Dibuat dengan Streamlit')
