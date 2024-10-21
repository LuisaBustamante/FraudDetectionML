import streamlit as st
from datetime import datetime, timedelta
from database import verify_user_credentials

def login_page():
    MAX_LOGIN_ATTEMPTS = 3
    LOCKOUT_DURATION_MINUTES = 5  # Duración inicial de bloqueo temporal en minutos
    MAX_LOCKOUT_DURATION_MINUTES = 60  # Duración máxima de bloqueo temporal en minutos

    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = 0
    if 'last_failed_attempt' not in st.session_state:
        st.session_state.last_failed_attempt = None

    st.title("Inicio de Sesión")
    username = st.text_input("Usuario")
    password = st.text_input("Contraseña", type="password")

    if st.button("Iniciar Sesión"):
        if username != "" and password != "":
            if st.session_state.login_attempts < MAX_LOGIN_ATTEMPTS:
                if verify_user_credentials(username, password):
                    st.session_state.logged_in = True
                    st.success("Inicio de sesión exitoso")
                    st.session_state.login_attempts = 0
                    st.rerun()
                    st.session_state.redirect_to = "/?state=Dashboard"
                else:
                    st.error("Credenciales incorrectas. Por favor, inténtalo de nuevo")
                    st.session_state.login_attempts += 1
                    st.session_state.last_failed_attempt = datetime.now()
                    if st.session_state.login_attempts >= MAX_LOGIN_ATTEMPTS:
                        st.session_state.lockout_duration = min(LOCKOUT_DURATION_MINUTES * (2 ** (st.session_state.login_attempts - MAX_LOGIN_ATTEMPTS)), MAX_LOCKOUT_DURATION_MINUTES)
                        st.error(f"Demasiados intentos fallidos. Por favor, inténtalo nuevamente después de {st.session_state.lockout_duration} minutos.")
            else:
                if st.session_state.last_failed_attempt is not None:
                    if (datetime.now() - st.session_state.last_failed_attempt).total_seconds() / 60 >= st.session_state.lockout_duration:
                        st.session_state.login_attempts = 0
                    else:
                        remaining_lockout_time = int(st.session_state.lockout_duration - (datetime.now() - st.session_state.last_failed_attempt).total_seconds() / 60)
                        st.error(f"Demasiados intentos fallidos. Por favor, inténtalo nuevamente después de {remaining_lockout_time} minutos.")
                else:
                    st.error("Demasiados intentos fallidos. Por favor, inténtalo nuevamente.")
        else:
            st.warning("Por favor, complete todos los campos")

    if st.button("Registrarse"):
        st.session_state.redirect_to = "/register"
        st.rerun()
