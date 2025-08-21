import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# Judul dashboard
st.title('Dashboard Kualitas Udara')
st.markdown('---')

# Load data
@st.cache_data
def load_data():
    # Ganti dengan path yang sesuai atau URL jika data berada di lokasi lain
    try:
        data = pd.read_csv('main_data.csv')
        # Pastikan kolom timestamp dalam format datetime
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    except FileNotFoundError:
        st.error("File main_data.csv tidak ditemukan.")
        return pd.DataFrame()

data = load_data()

if data.empty:
    st.warning("Tidak ada data yang dapat dimuat. Pastikan file main_data.csv ada di direktori yang benar.")
else:
    # Sidebar untuk filter
    st.sidebar.header('Filter Data')
    
    # Pilih polutan
    pollutant_options = [col for col in data.columns if col not in ['timestamp', 'location', 'station']]
    selected_pollutant = st.sidebar.selectbox('Pilih Polutan', pollutant_options)
    
    # Filter tanggal jika tersedia
    if 'timestamp' in data.columns:
        min_date = data['timestamp'].min()
        max_date = data['timestamp'].max()
        date_range = st.sidebar.date_input(
            "Rentang Tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    
    # Tampilkan statistik deskriptif
    st.header('Statistik Deskriptif Polutan')
    st.write(data[pollutant_options].describe())
    
    # Visualisasi tren waktu
    st.header('Tren Kualitas Udara dari Waktu ke Waktu')
    
    if 'timestamp' in data.columns and selected_pollutant in data.columns:
        # Agregasi data harian
        daily_data = data.groupby(pd.Grouper(key='timestamp', freq='D'))[selected_pollutant].mean().reset_index()
        
        fig = px.line(daily_data, x='timestamp', y=selected_pollutant, 
                     title=f'Tren {selected_pollutant} Harian')
        st.plotly_chart(fig)
        
        # Agregasi data bulanan
        monthly_data = data.groupby(pd.Grouper(key='timestamp', freq='M'))[selected_pollutant].mean().reset_index()
        
        fig2 = px.line(monthly_data, x='timestamp', y=selected_pollutant, 
                      title=f'Tren {selected_pollutant} Bulanan')
        st.plotly_chart(fig2)
    
    # Analisis polutan utama
    st.header('Polutan Utama')
    pollutant_means = data[pollutant_options].mean().sort_values(ascending=False)
    fig3 = px.bar(x=pollutant_means.index, y=pollutant_means.values,
                 labels={'x': 'Polutan', 'y': 'Rata-rata Konsentrasi'},
                 title='Rata-rata Konsentrasi Polutan')
    st.plotly_chart(fig3)

# Catatan penting
st.sidebar.markdown('---')
st.sidebar.info(
    "Pastikan semua dependensi sudah terinstall dengan menjalankan: "
    "`pip install -r requirements.txt`"
)
