from translator_model import generate_response
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout='wide',page_title="Translator", page_icon="🌍")
st.title("🌍 AI-Powered Translator")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def add_message(role,message):
    st.session_state.chat_history.append({'role':role, 'message':message})

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])

user_input = st.chat_input("Ask me anything...")


# if button:
if user_input:
    add_message('User', user_input)

    response = generate_response(user_input)
    add_message('Assistant',response)

    with st.chat_message("Assistant"):
        st.markdown(response)