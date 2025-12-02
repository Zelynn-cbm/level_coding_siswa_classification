import joblib
import pandas as pd
import streamlit as st

model = joblib.load('model.joblib')

st.title("Machine Learning Pemrediksi Level Coding Siswa")
st.markdown("Aplikasi Pemrediksi Level Coding Siswa Berdasarkan Data Siswa")

hours_coding_daily = st.slider('Hours Coding Daily',1,6,3)
preferred_language = st.pills("Preferred Language",['Python', 'C++', 'Java'],default='Python')
typing_speed = st.slider("Typing Speed",20,70,35)
import_usage = st.pills("Import Usage", ['Yes','No'],default='Yes')
oop_usage = st.pills("OOP Usage", ['Yes','No'],default='No')

if st.button("Prediksi", type='primary'):
	data_baru = pd.DataFrame([[hours_coding_daily ,preferred_language ,typing_speed ,import_usage ,oop_usage]],
                         columns=['hours_coding_daily', 'preferred_language', 'typing_speed','import_usage', 'oop_usage'])

	prediksi = model.predict(data_baru)[0]
	presentase = max(model.predict_proba(data_baru)[0])
	st.success(f'Model memprediksi **{prediksi}** dengan tingkat keyakinan **{presentase*100.:f}%**')
	st.balloons()

st.divider()
st.caption("Dibuat Oleh Maulana Elvano")