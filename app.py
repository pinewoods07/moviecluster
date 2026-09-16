import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# 페이지 기본 설정
# ---------------------------
st.set_page_config(
    page_title="영화 유형 나누기",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 영화 유형 나누기")
st.caption("스크린 수, 누적 관객, 10위권 유지 일수, 롱런 지수를 활용해 영화를 세 유형으로 묶어봅니다.")

# ---------------------------
# 데이터 불러오기
# ---------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/kobis_movies.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df

df_raw = load_data()

# ---------------------------
# 파생 변수 생성
# ---------------------------
df = df_raw.copy()

# 첫 주 관객이 0이거나 결측인 행 제거를 위해 우선 숫자형 변환
for col in ["first_scrn", "total_audi", "days_in_top10", "first_week_audi"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 네 속성 관련 결측치 및 first_week_audi==0 제거
before_count = len(df)
df = df.dropna(subset=["first_scrn", "total_audi", "days_in_top10", "first_week_audi"])
df = df[df["first_week_audi"] != 0]

# 로그 변환 (0 이하 값 방지를 위해 log10(x+1) 대신, 값이 0 이하인 경우도 걸러줌)
df = df[(df["first_scrn"] > 0) & (df["total_audi"] > 0)]

df["log_first_scrn"] = np.log10(df["first_scrn"])
df["log_total_audi"] = np.log10(df["total_audi"])

# 롱런 지수: 누적 관객 / 첫 주 관객, 20 초과시 20으로 자름
df["long_run_index"] = df["total_audi"] / df["first_week_audi"]
df["long_run_index"] = df["long_run_index"].clip(upper=20)

after_count = len(df)

# ---------------------------
# 속성 선택 UI
# ---------------------------
st.subheader("1️⃣ 군집에 사용할 속성 선택")

feature_map = {
    "스크린 수(로그)": "log_first_scrn",
    "누적 관객(로그)": "log_total_audi",
    "10위권 유지 일수": "days_in_top10",
    "롱런 지수": "long_run_index"
}

selected_labels = st.multiselect(
    "두 개 이상 선택하세요.",
    options=list(feature_map.keys()),
    default=list(feature_map.keys())
)

if len(selected_labels) < 2:
    st.warning("⚠️ 속성을 두 개 이상 선택해야 군집화를 진행할 수 있습니다.")
    st.stop()

selected_cols = [feature_map[label] for label in selected_labels]

st.write(f"전체 편수: **{before_count}편** → 사용 가능한 편수(결측치·조건 제외 후): **{after_count}편**")

# ---------------------------
# 표준화 및 K-평균 군집화
# ---------------------------
X = df[selected_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)
df["cluster_raw"] = cluster_labels

# ---------------------------
# 누적 관객 평균 기준으로 묶음 이름 재배정 (㉮ ㉯ ㉰)
# ---------------------------
cluster_order = (
    df.groupby("cluster_raw")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

symbol_map = {cluster_order[0]: "㉮", cluster_order[1]: "㉯", cluster_order[2]: "㉰"}
df["cluster"] = df["cluster_raw"].map(symbol_map)

symbol_order = ["㉮", "㉯", "㉰"]

# ---------------------------
# 2D 산점도
# ---------------------------
st.subheader("2️⃣ 2차원 산점도")

col1, col2 = st.columns(2)
with col1:
    x_axis_label = st.selectbox("가로축 속성", options=selected_labels, index=0, key="x2d")
with col2:
    remaining_labels = [l for l in selected_labels if l != x_axis_label]
    y_axis_label = st.selectbox("세로축 속성", options=remaining_labels, index=0, key="y2d")

x_col = feature_map[x_axis_label]
y_col = feature_map[y_axis_label]

fig2d = px.scatter(
    df,
    x=x_col,
    y=y_col,
    color="cluster",
    category_orders={"cluster": symbol_order},
    hover_name="movieNm",
    labels={x_col: x_axis_label, y_col: y_axis_label, "cluster": "유형"},
    title=f"{x_axis_label} vs {y_axis_label}"
)
fig2d.update_traces(marker=dict(size=8, opacity=0.75))
st.plotly_chart(fig2d, use_container_width=True)

# ---------------------------
# 3D 산점도
# ---------------------------
st.subheader("3️⃣ 3차원 산점도")

if len(selected_labels) < 3:
    st.info("ℹ️ 3차원 산점도를 그리려면 속성을 세 개 이상 선택해야 합니다.")
else:
    col3, col4, col5 = st.columns(3)
    with col3:
        x3_label = st.selectbox("X축 속성", options=selected_labels, index=0, key="x3d")
    with col4:
        y3_label = st.selectbox("Y축 속성", options=selected_labels, index=1, key="y3d")
    with col5:
        z3_label = st.selectbox("Z축 속성", options=selected_labels, index=2, key="z3d")

    x3_col = feature_map[x3_label]
    y3_col = feature_map[y3_label]
    z3_col = feature_map[z3_label]

    fig3d = px.scatter_3d(
        df,
        x=x3_col,
        y=y3_col,
        z=z3_col,
        color="cluster",
        category_orders={"cluster": symbol_order},
        hover_name="movieNm",
        labels={x3_col: x3_label, y3_col: y3_label, z3_col: z3_label, "cluster": "유형"},
        title=f"{x3_label} · {y3_label} · {z3_label}"
    )
    fig3d.update_traces(marker=dict(size=3, opacity=0.75))
    st.plotly_chart(fig3d, use_container_width=True)

# ---------------------------
# 묶음별 통계표
# ---------------------------
st.subheader("4️⃣ 묶음별 편수와 평균값 (원래 단위)")

summary = df.groupby("cluster").agg(
    편수=("movieNm", "count"),
    평균_스크린수=("first_scrn", "mean"),
    평균_누적관객=("total_audi", "mean"),
    평균_10위권일수=("days_in_top10", "mean"),
    평균_롱런지수=("long_run_index", "mean"),
).reindex(symbol_order)

summary = summary.round(2)
st.dataframe(summary, use_container_width=True)

# ---------------------------
# 묶음별 상위 5편
# ---------------------------
st.subheader("5️⃣ 묶음별 누적 관객 상위 5편")

cols = st.columns(3)
for i, symbol in enumerate(symbol_order):
    with cols[i]:
        st.markdown(f"### {symbol} 유형")
        top5 = (
            df[df["cluster"] == symbol]
            .sort_values("total_audi", ascending=False)
            .head(5)[["movieNm", "total_audi"]]
        )
        top5 = top5.reset_index(drop=True)
        top5.index = top5.index + 1
        top5.columns = ["영화 제목", "누적 관객"]
        st.dataframe(top5, use_container_width=True)
