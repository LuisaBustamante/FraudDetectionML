import streamlit as st
from database import insert_user
import re

def is_password_secure(password):
    # Validar si la contraseña cumple con criterios de seguridad
    if (
        len(password) < 8 or  # Longitud mínima de 8 caracteres
        not re.search("[a-z]", password) or  # Al menos una letra minúscula
        not re.search("[A-Z]", password) or  # Al menos una letra mayúscula
        not re.search("[0-9]", password) or  # Al menos un dígito
        not re.search("[!@#$%^&*()_+{}[\]:;<>,.?/~]", password)  # Al menos un carácter especial
    ):
        return False
    return True

def register_page():
    st.title("Registro")
    new_username = st.text_input("Nuevo Usuario")
    new_password = st.text_input("Nueva Contraseña", type="password")
    confirm_password = st.text_input("Confirmar contraseña", type="password")

    if st.button("Registrarse"):
        if (
            len(new_username) >= 5
            and len(new_password) >= 8
            and new_username.strip() != "" 
            and new_password.strip() != ""
            and new_password == confirm_password
            and is_password_secure(new_password)
            and new_password.lower() != new_username.lower()  # Verificar que la contraseña no sea igual al nombre de usuario
        ):
            insert_user(new_username, new_password)
            st.success("Registro exitoso")
            st.session_state.logged_in = True
            st.success("Inicio de sesión exitoso")
            st.session_state.redirect_to = "/?state=Dashboard"  
            st.rerun()  
        else:
            if len(new_username) < 5:
                st.warning("El usuario debe tener al menos 5 caracteres")
            elif len(new_password) < 8:
                st.warning("La contraseña debe tener al menos 8 caracteres y contener al menos una letra mayúscula, una letra minúscula, un dígito y un carácter especial")
            elif new_username.strip() == "":
                st.warning("Por favor, introduzca un usuario")
            elif new_password.strip() == "":
                st.warning("Por favor, introduzca una contraseña")
            elif new_password != confirm_password:
                st.warning("Las contraseñas no coinciden")
            elif not is_password_secure(new_password):
                st.warning("La contraseña no cumple con los criterios de seguridad")
            elif new_password.lower() == new_username.lower():
                st.warning("La contraseña no puede ser igual al nombre de usuario")
    
    if st.session_state.redirect_to == "/?state=Dashboard":
        st.experimental_reroute(st.session_state.redirect_to)
    
    # Agregar texto para regresar al inicio de sesión
    st.markdown("Ya tienes una cuenta? [Inicia sesión aquí](/?state=Inicio)")
