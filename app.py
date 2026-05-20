from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(page_title="ROI + PLS + DD-SIMCA", layout="wide")

DATA_DIR = Path("dados")
DATA_DIR.mkdir(exist_ok=True)

ROI_FILE = DATA_DIR / "roi_dados.csv"

st.title("ROI RGB/HSV/Gray + PLS + DD-SIMCA")

uploaded_files = st.file_uploader(
    "Carregue imagens",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

def rgb_to_hsv_array(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(float)
    hsv[:, :, 0] *= 2
    hsv[:, :, 1] = hsv[:, :, 1] / 255 * 100
    hsv[:, :, 2] = hsv[:, :, 2] / 255 * 100
    return hsv

def extract_features(rgb_roi):
    R = rgb_roi[:,:,0].ravel()
    G = rgb_roi[:,:,1].ravel()
    B = rgb_roi[:,:,2].ravel()

    hsv = rgb_to_hsv_array(rgb_roi)

    H = hsv[:,:,0].ravel()
    S = hsv[:,:,1].ravel()
    V = hsv[:,:,2].ravel()

    Gray = 0.299*R + 0.587*G + 0.114*B

    return {
        "R_mean": np.mean(R),
        "G_mean": np.mean(G),
        "B_mean": np.mean(B),
        "H_mean": np.mean(H),
        "S_mean": np.mean(S),
        "V_mean": np.mean(V),
        "Gray_mean": np.mean(Gray),
    }

if uploaded_files:

    file = st.selectbox(
        "Imagem",
        uploaded_files,
        format_func=lambda x: x.name
    )

    img_pil = Image.open(file)
    rgb = np.array(img_pil)

    st.image(img_pil)

    st.subheader("ROI")

    x = st.number_input("x", 0, rgb.shape[1]-1, 50)
    y = st.number_input("y", 0, rgb.shape[0]-1, 50)
    w = st.number_input("largura", 1, rgb.shape[1], 100)
    h = st.number_input("altura", 1, rgb.shape[0], 100)

    roi = rgb[y:y+h, x:x+w]

    st.image(roi, caption="ROI")

    sample = st.text_input("Amostra", value=file.name)
    concentration = st.number_input("Concentração")

    if st.button("Salvar ROI"):

        feats = extract_features(roi)

        row = pd.DataFrame([{
            "DataHora": datetime.now(),
            "Imagem": file.name,
            "Amostra": sample,
            "Concentracao": concentration,
            **feats
        }])

        if ROI_FILE.exists():
            old = pd.read_csv(ROI_FILE)
            row = pd.concat([old, row])

        row.to_csv(ROI_FILE, index=False)

        st.success("ROI salva")

st.header("PLS")

if ROI_FILE.exists():

    df = pd.read_csv(ROI_FILE)

    st.dataframe(df)

    vars_x = st.multiselect(
        "Variáveis X",
        [
            "R_mean","G_mean","B_mean",
            "H_mean","S_mean","V_mean",
            "Gray_mean"
        ],
        default=["R_mean","G_mean","B_mean"]
    )

    n_comp = st.slider("LVs",1,5,2)

    if st.button("Rodar PLS"):

        X = df[vars_x].values
        y = df["Concentracao"].values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        pls = PLSRegression(n_components=n_comp)

        loo = LeaveOneOut()

        ycv = cross_val_predict(
            pls,
            Xs,
            y,
            cv=loo
        )

        pls.fit(Xs,y)

        ycal = pls.predict(Xs).ravel()

        r2 = r2_score(y,ycv)
        rmse = np.sqrt(mean_squared_error(y,ycv))

        st.write("R² CV:", r2)
        st.write("RMSECV:", rmse)

        pred = pd.DataFrame({
            "Real": y,
            "Predito": ycv
        })

        st.dataframe(pred)

st.header("DD-SIMCA")

if ROI_FILE.exists():

    df = pd.read_csv(ROI_FILE)

    if "Classe" not in df.columns:
        st.warning("Adicione coluna Classe no CSV")
