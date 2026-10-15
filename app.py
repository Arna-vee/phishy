"""Demo UI:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from phishy.predict import Detector

st.set_page_config(page_title="Phishy", page_icon="🎣")
st.title("🎣 Phishy - URL phishing detector")


@st.cache_resource
def load():
    return Detector()


try:
    det = load()
except FileNotFoundError:
    st.error("No trained model found. Run `python -m phishy.train ...` first.")
    st.stop()

tab_single, tab_batch, tab_about = st.tabs(["Check a URL", "Batch (CSV)", "Model info"])

with tab_single:
    url = st.text_input("Paste a URL", placeholder="http://paypal-secure-login.xyz/verify")
    if st.button("Check") and url.strip():
        r = det.predict(url.strip())
        if r["verdict"] == "Phishing":
            st.error(f"Phishing - {r['probability']:.1%}")
        else:
            st.success(f"Legitimate - phishing probability {r['probability']:.1%}")
        st.progress(min(max(r["probability"], 0.0), 1.0))
        if r["top_features"] is not None:
            st.caption("Why? (positive = pushes towards phishing)")
            st.dataframe(r["top_features"], hide_index=True)

with tab_batch:
    up = st.file_uploader("CSV with a 'url' column", type="csv")
    if up:
        urls = pd.read_csv(up)["url"].dropna().astype(str).tolist()
        res = det.score(urls)
        st.dataframe(res)
        st.download_button("Download results", res.to_csv(index=False), "results.csv")

with tab_about:
    st.json(det.report)
    st.write("**Limitation:** the model sees URL structure only - no page content, "
             "domain age or reputation.")
