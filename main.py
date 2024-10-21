#main.py

import streamlit as st
from pages import login_page
from registration import register_page
from dashboard import dashboard_page
from database import verify_user_credentials

def main():
    if 'redirect_to' not in st.session_state:  
        st.session_state.redirect_to = ""

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.redirect_to == "/register":
        register_page()
    elif st.session_state.logged_in or verify_user_credentials('username', 'password'):
        st.session_state.logged_in = True  # Ensure logged_in is True
        dashboard_page()
    else:
        login_page()

if __name__ == "__main__":
    main()
