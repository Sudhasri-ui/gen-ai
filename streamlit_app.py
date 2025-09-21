import streamlit as st

st.title("Misinformation Detector")

text = st.text_area("Enter the text or URL you want to check:")

if st.button("Check"):
    st.write("This is where your prediction will show.")
