# Streamlit Dashboard Sederhana: Analisis Kualitas Udara
# Jalankan lokal:  
# 1) Buat/aktifkan venv (opsional)  
# 2) pip install -r requirements.txt  
# 3) streamlit run streamlit_air_quality_dashboard.py

# ====== IMPORTS ======
import io
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ====== PAGE CONFIG ======
st.set_page_config(
    page_title="Dashboard Kualitas Udara",
    page_icon="🫧",
    layout="wide",
)

# ====== UTILS ======
COMMON_DATE_COLS = [
    "datetime", "timestamp", "date", "time", "waktu", "tanggal",
]
COMMON_POLLUTANTS = [
    "PM2.5", "PM25", "PM_2_5", "PM10", "SO2", "NO2", "CO", "O3", "NH3", "H2S",
]

@st.cache_data(show_spinner=False)
def load_data(uploaded_file: io.BytesIO) -> pd.DataFrame:
    """Load CSV/XLSX to DataFrame with best-effort datetime parsing."""
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Format file tidak didukung. Gunakan CSV atau Excel.")
        return pd.DataFrame()
    return df

@st.cache_data(show_spinner=False)
def make_sample_data(n_days: int = 120, freq: str = "H", seed: int = 7) -> pd.DataFrame:
    """Generate contoh data kualitas udara untuk uji dashboard."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp.today().normalize() - pd.Timedelta(days=n_days), periods=n_days * (24 if freq == "H" else 1), freq=freq)
    base = np.linspace(0, 1, len(idx))
    df = pd.DataFrame({
        "timestamp": idx,
        "PM2.5": 35 + 10*np.sin(2*np.pi*base*3) + rng.normal(0, 5, len(idx)),
        "PM10": 60 + 12*np.cos(2*np.pi*base*2) + rng.normal(0, 7, len(idx)),
        "SO2": 12 + 3*np.sin(2*np.pi*base*1.5) + rng.normal(0, 1.5, len(idx)),
        "NO2": 25 + 6*np.cos(2*np.pi*base*1.2) + rng.normal(0, 2.5, len(idx)),
        "CO": 0.9 + 0.2*np.sin(2*np.pi*base*4) + rng.normal(0, 0.08, len(idx)),
        "O3": 40 + 8*np.sin(2*np.pi*base*0.8) + rng.normal(0, 3, len(idx)),
    })
    # pastikan tidak ada negatif
    for c in df.columns:
        if c != "timestamp":
            df[c] = df[c].clip(lower=0)
    return df


def find_datetime_col(df: pd.DataFrame) -> str | None:
    if df.empty:
        return None
    # 1) dtype datetime dulu
    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    if dt_cols:
        return dt_cols[0]
    # 2) nama umum
    lower_map = {c.lower(): c for c in df.columns}
    for key in COMMON_DATE_COLS:
        if key in lower_map:
            return lower_map[key]
    # 3) coba parse kolom pertama yang bisa di-parse
    for c in df.columns:
        try:
            _ = pd.to_datetime(df[c], errors="raise")
            return c
        except Exception:
            continue
    return None


def auto_pollutant_cols(df: pd.DataFrame, dt_col: str | None) -> list[str]:
    if df.empty:
        return []
    candidates = []
    # 1) nama umum
    lower_map = {c.lower(): c for c in df.columns}
    for key in COMMON_POLLUTANTS:
        if key.lower() in lower_map:
            candidates.append(lower_map[key.lower()])
    # 2) semua kolom numerik selain datetime
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if dt_col in numeric_cols:
        numeric_cols.remove(dt_col)
    # gabung, pertahankan urutan dan unik
    seen = set()
    result = []
    for c in candidates + numeric_cols:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def robust_clip(df: pd.DataFrame, cols: list[str], iqr_mult: float = 3.0) -> pd.DataFrame:
    """Winsorize sederhana berbasis IQR untuk mengurangi outlier ekstrem."""
    dfc = df.copy()
    for c in cols:
        q1 = dfc[c].quantile(0.25)
        q3 = dfc[c].quantile(0.75)
        iqr = q3 - q1
        low = q1 - iqr_mult * iqr
        high = q3 + iqr_mult * iqr
        dfc[c] = dfc[c].clip(lower=low, upper=high)
    return dfc


def descriptive_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame()
    agg = df[cols].agg(["mean", "median", "std", "min", "max", "count"]).T
    agg.rename(columns={
        "mean": "Mean",
        "median": "Median",
        "std": "StdDev",
        "min": "Min",
        "max": "Max",
        "count": "N",
    }, inplace=True)
    # tambahkan Missing
    agg["Missing"] = df.shape[0] - agg["N"].astype(int)
    return agg


def resample_df(df: pd.DataFrame, dt_col: str, cols: list[str], freq: str) -> pd.DataFrame:
    tmp = df[[dt_col] + cols].copy()
    tmp[dt_col] = pd.to_datetime(tmp[dt_col], errors="coerce")
    tmp = tmp.dropna(subset=[dt_col])
    tmp = tmp.sort_values(dt_col)
    tmp = tmp.set_index(dt_col)
    # gunakan mean per periode
    out = tmp.resample(freq).mean(numeric_only=True)
    out.index.name = dt_col
    return out.reset_index()


def trend_summary(df_r: pd.DataFrame, dt_col: str, cols: list[str]) -> pd.DataFrame:
    rows = []
    if df_r.empty or len(df_r) < 2:
        return pd.DataFrame(rows)
    # ordinal time untuk slope
    t0 = df_r[dt_col].min()
    t_ord = (df_r[dt_col] - t0).dt.total_seconds() / (24*3600)  # hari

    for c in cols:
        series = df_r[c].astype(float)
        if series.dropna().shape[0] < 2:
            continue
        # slope (unit per hari)
        try:
            slope, intercept = np.polyfit(t_ord[series.notna()], series.dropna(), 1)
        except Exception:
            slope, intercept = np.nan, np.nan
        first = series.iloc[0]
        last = series.iloc[-1]
        pct_change = (last - first) / first * 100 if pd.notna(first) and first != 0 else np.nan
        direction = "Naik" if pct_change > 0 else ("Turun" if pct_change < 0 else "Stagnan")
        rows.append({
            "Polutan": c,
            "Rata-rata": series.mean(),
            "Median": series.median(),
            "Slope_per_hari": slope,
            "%Perubahan (first→last)": pct_change,
            "Arah": direction,
        })
    return pd.DataFrame(rows)


def download_csv_button(df: pd.DataFrame, filename: str, label: str):
    if df is None or df.empty:
        return
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ====== SIDEBAR ======
st.sidebar.title("⚙️ Pengaturan")

uploaded = st.sidebar.file_uploader("Upload dataset (CSV/XLSX)", type=["csv", "xlsx", "xls"])
use_sample = st.sidebar.toggle("Gunakan contoh data", value=True if uploaded is None else False)

raw_df = load_data(uploaded) if uploaded is not None else (make_sample_data() if use_sample else pd.DataFrame())

if raw_df.empty:
    st.info("Silakan upload file CSV/Excel atau aktifkan \"Gunakan contoh data\" di sidebar.")
    st.stop()

# pilih kolom tanggal & polutan
dt_guess = find_datetime_col(raw_df)
dt_col = st.sidebar.selectbox("Kolom tanggal/waktu", options=list(raw_df.columns), index=(list(raw_df.columns).index(dt_guess) if dt_guess in raw_df.columns else 0))

pollutant_guess = auto_pollutant_cols(raw_df, dt_guess)
pollutant_cols = st.sidebar.multiselect(
    "Pilih kolom polutan",
    options=[c for c in raw_df.columns if c != dt_col],
    default=pollutant_guess[:5] if pollutant_guess else [],
)

freq_label = st.sidebar.selectbox("Agregasi Tren", options=["Harian", "Bulanan", "Tahunan"], index=1)
freq_map = {"Harian": "D", "Bulanan": "MS", "Tahunan": "YS"}
freq = freq_map[freq_label]

roll_window = st.sidebar.number_input("Rolling window (periode)", min_value=1, value=3, step=1)
clip_outliers = st.sidebar.toggle("Kurangi outlier ekstrem (IQR winsorize)", value=False)

# ====== MAIN ======
st.title("🫧 Dashboard Kualitas Udara — Analisis Deskriptif & Tren")
st.caption("Upload dataset Anda (atau gunakan contoh). Pilih kolom tanggal & polutan di sidebar. Dashboard ini akan menghitung statistik deskriptif (mean, median, std, min/max) serta memvisualisasikan tren harian/bulanan/tahunan.")

with st.expander("🔎 Pratinjau Data", expanded=False):
    st.write(raw_df.head(20))

# pembersihan dasar
work_df = raw_df.copy()
work_df[dt_col] = pd.to_datetime(work_df[dt_col], errors="coerce")
work_df = work_df.dropna(subset=[dt_col])
work_df = work_df.sort_values(dt_col)

if clip_outliers and pollutant_cols:
    work_df = robust_clip(work_df, pollutant_cols)

# ============= PERTANYAAN 1: Statistik Deskriptif =============
st.subheader("Pertanyaan 1: Statistik Deskriptif Polutan")
if not pollutant_cols:
    st.warning("Pilih minimal satu kolom polutan di sidebar.")
else:
    stats_df = descriptive_stats(work_df, pollutant_cols)
    st.dataframe(stats_df.style.format({
        "Mean": "{:.3f}", "Median": "{:.3f}", "StdDev": "{:.3f}", "Min": "{:.3f}", "Max": "{:.3f}",
        "N": "{:.0f}", "Missing": "{:.0f}",
    }))
    download_csv_button(stats_df.reset_index().rename(columns={"index": "Polutan"}), "descriptive_stats.csv", "⬇️ Unduh Statistik Deskriptif")

# ============= PERTANYAAN 2: Tren Kualitas Udara =============
st.subheader("Pertanyaan 2: Tren Konsentrasi dari Waktu ke Waktu")
if pollutant_cols:
    rdf = resample_df(work_df, dt_col, pollutant_cols, freq)
    # rolling mean
    roll_df = rdf.copy()
    for c in pollutant_cols:
        roll_df[c] = roll_df[c].rolling(window=roll_window, min_periods=1).mean()

    # pilih polutan untuk grafik
    sel_pol = st.multiselect("Pilih polutan untuk grafik", pollutant_cols, default=pollutant_cols[: min(3, len(pollutant_cols))])

    tab1, tab2 = st.tabs(["📈 Tren (Mean)", "📉 Tren (Rolling Mean)"])

    with tab1:
        if sel_pol:
            plot_df = rdf.melt(id_vars=[dt_col], value_vars=sel_pol, var_name="Polutan", value_name="Konsentrasi")
            fig = px.line(plot_df, x=dt_col, y="Konsentrasi", color="Polutan", markers=True)
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            download_csv_button(rdf[[dt_col] + sel_pol], f"tren_{freq_label.lower()}.csv", "⬇️ Unduh Data Tren")
        else:
            st.info("Pilih polutan untuk menampilkan grafik.")

    with tab2:
        if sel_pol:
            plot_df2 = roll_df.melt(id_vars=[dt_col], value_vars=sel_pol, var_name="Polutan", value_name="Konsentrasi (Rolling)")
            fig2 = px.line(plot_df2, x=dt_col, y="Konsentrasi (Rolling)", color="Polutan")
            fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Pilih polutan untuk menampilkan grafik.")

    # ringkasan tren dan arah
    rdf[dt_col] = pd.to_datetime(rdf[dt_col])
    tsum = trend_summary(rdf, dt_col, pollutant_cols)
    if not tsum.empty:
        st.markdown("**Ringkasan Tren per Polutan** (berdasarkan agregasi yang dipilih)")
        st.dataframe(tsum.style.format({
            "Rata-rata": "{:.3f}", "Median": "{:.3f}", "Slope_per_hari": "{:.4f}", "%Perubahan (first→last)": "{:.2f}%",
        }))

        # simpulan keseluruhan
        dir_counts = tsum["Arah"].value_counts()
        naik = int(dir_counts.get("Naik", 0))
        turun = int(dir_counts.get("Turun", 0))
        stagnan = int(dir_counts.get("Stagnan", 0))
        st.success(
            f"**Simpulan singkat**: Dari {len(tsum)} polutan terpilih, {naik} menunjukkan kecenderungan naik, {turun} turun, dan {stagnan} relatif stagnan pada tingkat agregasi **{freq_label.lower()}**."
        )

# ====== FOOTER ======
st.divider()
st.caption(
    "Catatan: Slope dihitung via regresi linear sederhana (unit per hari) pada seri agregat. %Perubahan dihitung dari periode pertama ke terakhir. Gunakan opsi winsorize untuk mereduksi pengaruh outlier ekstrem."
)
