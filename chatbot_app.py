import streamlit as st
import requests

# Replace with your actual Serpstack API key
SERPSTACK_API_KEY = "a8101a5c7e5fcada7b154ccf31bac85c"

def call_serpstack_search(query):
    url = "https://api.serpstack.com/search"
    params = {
        "access_key": SERPSTACK_API_KEY,
        "query": query
    }
    try:
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception:
        return None

def summarize_search_results(search_json):
    if not search_json or 'organic_results' not in search_json:
        return "Sorry, I couldn't find any information on that."

    top_results = search_json['organic_results'][:3]
    summaries = []
    for i, res in enumerate(top_results, 1):
        title = res.get('title', 'No title')
        snippet = res.get('snippet', '')
        url = res.get('url', '')
        summaries.append(f"{i}. {title}\n{snippet}\nLink: {url}")
    return "\n\n".join(summaries)

st.set_page_config(page_title="Chatbot - Plant Disease", layout="centered")

st.title("💬 Plantelligence")

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

def display_chat():
    for sender, message in st.session_state['chat_history']:
        if sender == "User":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Plantelligence:** {message}")

# User Input Form to persist input box after submission
with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input("Ask me about plant diseases or remedies:")
    submit = st.form_submit_button("Send")

if submit and user_input:
    st.session_state['chat_history'].append(("User", user_input))
    search_response = call_serpstack_search(user_input)
    bot_reply = summarize_search_results(search_response)
    st.session_state['chat_history'].append(("Bot", bot_reply))

display_chat()
