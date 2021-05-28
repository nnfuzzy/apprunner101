import streamlit as st


def _max_width_(value=900):
    max_width_str = f"max-width: {value}px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )