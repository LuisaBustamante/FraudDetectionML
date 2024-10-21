import streamlit as st
from auto_detection import auto_detection_page
from report import generate_anomaly_report

def dashboard_page():

    # Organizando las opciones en un sidebar
    st.sidebar.title('Opciones')
    selection = st.sidebar.radio('Ir a:', ['Detección ML', 'Reporte', 'Cerrar Sesión'])

    if selection == 'Detección ML':
        auto_detection_page()
    elif selection == 'Reporte':
        generate_anomaly_report()
    elif selection == 'Cerrar Sesión':
        st.write("Sesión cerrada")
        st.session_state.logged_in = False  
        st.warning("Sesión cerrada. Por favor, inicia sesión nuevamente [Ir a la página de inicio](/?state=Inicio)")
