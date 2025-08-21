import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta
import io

# Judul dashboard
st.title('Dashboard Analisis Kualitas Udara')
st.markdown('Analisis polutan utama dan tren kualitas udara berdasarkan dataset main_data.csv')
st.markdown('---')

# Load data dari main_data.csv
@st.cache_data(show_spinner=False)
def load_data():
    """Load data dari main_data.csv dengan parsing datetime otomatis."""
    try:
        data = pd.read_csv('main_data.csv')
        st.success("✅ Data berhasil dimuat dari main_data.csv")
        
        # Daftar kolom yang mungkin berisi tanggal/waktu
        date_columns = ['timestamp', 'datetime', 'date', 'time', 'tanggal', 'waktu']
        
        # Coba parsing kolom tanggal secara otomatis
        for col in data.columns:
            if col.lower() in [c.lower() for c in date_columns]:
                data[col] = pd.to_datetime(data[col], errors='coerce')
            elif data[col].dtype == 'object':
                # Coba parsing jika kolom object bisa jadi datetime
                try:
                    data[col] = pd.to_datetime(data[col], errors='coerce')
                except:
                    pass
        
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

# Identifikasi kolom dengan benar
all_columns = data.columns.tolist()

# Deteksi kolom datetime
datetime_columns = []
for col in all_columns:
    if pd.api.types.is_datetime64_any_dtype(data[col]):
        datetime_columns.append(col)

# Deteksi kolom polutan (numerik)
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

# Daftar nama polutan umum untuk deteksi tambahan
common_pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'PM25', 'NOx', 'CO2']

# Coba konversi kolom yang mungkin polutan tetapi belum numerik
for col in all_columns:
    if col in common_pollutants and col not in numeric_columns:
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            if col not in numeric_columns:
                numeric_columns.append(col)
        except:
            pass

# Sidebar untuk filter
st.sidebar.header('Konfigurasi Analisis')

# Dropdown untuk kolom tanggal/waktu - HANYA kolom datetime
if datetime_columns:
    timestamp_col = st.sidebar.selectbox(
        'Kolom tanggal/waktu', 
        datetime_columns,
        help="Pilih kolom yang berisi informasi waktu untuk analisis tren"
    )
else:
    timestamp_col = None
    st.sidebar.warning("⚠️ Tidak ditemukan kolom tanggal/waktu")

# Dropdown untuk polutan - HANYA kolom numerik
if numeric_columns:
    selected_pollutant = st.sidebar.selectbox(
        'Pilih Polutan', 
        numeric_columns,
        help="Pilih polutan untuk dianalisis"
    )
else:
    selected_pollutant = None
    st.sidebar.error("❌ Tidak ditemukan kolom polutan (numerik)")

# Informasi dataset
st.sidebar.header('Informasi Dataset')
st.sidebar.write(f"Jumlah baris: {data.shape[0]}")
st.sidebar.write(f"Jumlah kolom: {data.shape[1]}")
if timestamp_col:
    date_min = data[timestamp_col].min()
    date_max = data[timestamp_col].max()
    st.sidebar.write(f"Periode: {date_min.strftime('%Y-%m-%d') if pd.notna(date_min) else 'N/A'} hingga {date_max.strftime('%Y-%m-%d') if pd.notna(date_max) else 'N/A'}")

# Tampilkan preview data
st.header('📋 Preview Data')
st.write("5 baris pertama dari main_data.csv:")
st.dataframe(data.head())

# Tampilkan informasi kolom
with st.expander("Lihat Informasi Kolom"):
    st.write("**Kolom Datetime:**", datetime_columns)
    st.write("**Kolom Numerik (Polutan):**", numeric_columns)
    st.write("**Kolom Lainnya:**", [col for col in all_columns if col not in datetime_columns + numeric_columns])

if not numeric_columns:
    st.error("Tidak ada kolom numerik yang dapat dianalisis. Pastikan dataset berisi data polutan.")
    st.stop()

# Statistik deskriptif
st.header('📊 Statistik Deskriptif')

if selected_pollutant:
    st.subheader(f'Statistik untuk {selected_pollutant}')
    
    # Hitung statistik
    stats = {
        'Rata-rata': data[selected_pollutant].mean(),
        'Median': data[selected_pollutant].median(),
        'Std Dev': data[selected_pollutant].std(),
        'Minimum': data[selected_pollutant].min(),
        'Maksimum': data[selected_pollutant].max(),
        'Jumlah Data': data[selected_pollutant].count(),
        'Data Hilang': data[selected_pollutant].isnull().sum()
    }
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rata-rata", f"{stats['Rata-rata']:.2f}")
    with col2:
        st.metric("Median", f"{stats['Median']:.2f}")
    with col3:
        st.metric("Std Dev", f"{stats['Std Dev']:.2f}")
    with col4:
        st.metric("Range", f"{stats['Minimum']:.2f} - {stats['Maksimum']:.2f}")
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Jumlah Data", f"{stats['Jumlah Data']}")
    with col6:
        st.metric("Data Hilang", f"{stats['Data Hilang']}")

# Statistik lengkap semua polutan
st.subheader('Statistik Lengkap Semua Polutan')
stats_all = data[numeric_columns].describe().round(3)
st.dataframe(stats_all)

# Analisis tren waktu jika ada timestamp dan polutan terpilih
if timestamp_col and selected_pollutant:
    st.header('📈 Analisis Tren Waktu')
    
    # Pastikan tidak ada nilai NaN di kolom timestamp
    trend_data = data[[timestamp_col, selected_pollutant]].dropna()
    
    if len(trend_data) > 0:
        # Pilihan agregasi
        aggregation = st.selectbox('Tingkat Agregasi', ['Harian', 'Mingguan', 'Bulanan'])
        freq_map = {'Harian': 'D', 'Mingguan': 'W', 'Bulanan': 'M'}
        
        try:
            # Set index untuk grouping
            trend_data = trend_data.set_index(timestamp_col)
            
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
                        f"{last_value:.2f}"
                    )
                with col2:
                    st.metric(
                        f"Perubahan Tren",
                        f"{trend_change:.1f}%",
                        delta_color="inverse" if trend_change > 0 else "normal"
                    )
                
                # Plot tren
                fig = px.line(aggregated_data, x=timestamp_col, y=selected_pollutant,
                             title=f'Tren {selected_pollutant} ({aggregation})',
                             labels={timestamp_col: 'Waktu', selected_pollutant: 'Konsentrasi'})
                st.plotly_chart(fig)
                
                # Tampilkan data agregat
                with st.expander("Lihat Data Agregat Tren"):
                    st.dataframe(aggregated_data)
            else:
                st.warning("Tidak ada data yang cukup untuk analisis tren.")
        except Exception as e:
            st.error(f"Error dalam analisis tren: {str(e)}")
    else:
        st.warning("Tidak ada data yang valid untuk analisis tren.")
else:
    if not timestamp_col:
        st.warning("⚠️ Tidak ditemukan kolom timestamp untuk analisis tren waktu.")

# Analisis polutan utama
st.header('🥇 Polutan Utama')
pollutant_means = data[numeric_columns].mean().sort_values(ascending=False)

col1, col2 = st.columns(2)
with col1:
    st.subheader('Ranking Polutan')
    for i, (pollutant, value) in enumerate(pollutant_means.items()):
        st.write(f"{i+1}. **{pollutant}**: {value:.2f}")

with col2:
    fig_bar = px.bar(x=pollutant_means.index, y=pollutant_means.values,
                    title='Rata-rata Konsentrasi Polutan',
                    labels={'x': 'Polutan', 'y': 'Konsentrasi Rata-rata'})
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar)

# Analisis distribusi
st.header('📊 Distribusi Polutan')
if selected_pollutant:
    fig_hist = px.histogram(data, x=selected_pollutant,
                           title=f'Distribusi {selected_pollutant}',
                           labels={selected_pollutant: 'Konsentrasi', 'count': 'Frekuensi'})
    st.plotly_chart(fig_hist)

# Analisis korelasi jika ada multiple polutan
if len(numeric_columns) > 1:
    st.header('🔗 Korelasi Antar Polutan')
    
    correlation_matrix = data[numeric_columns].corr()
    
    fig_corr = px.imshow(correlation_matrix,
                        title='Matriks Korelasi Antar Polutan',
                        color_continuous_scale='RdBu_r',
                        aspect="auto",
                        zmin=-1, zmax=1)
    st.plotly_chart(fig_corr)

# Download data hasil analisis
st.header('📥 Ekspor Data')
if st.button('Download Statistik Deskriptif sebagai CSV'):
    stats_all.to_csv('statistik_polutan.csv')
    st.success('File statistik_polutan.csv berhasil diunduh!')

# Footer
st.markdown('---')
st.caption('Dashboard Analisis Kualitas Udara | Data sumber: main_data.csv | Dibuat dengan Streamlit')
