import streamlit as st

def initialize_session_state():
    defaults = {
        'start_camera': False,
        'incident_count': 0,
        'last_incident_time': 0,
        'incidents': [],
        'color': 'green'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
