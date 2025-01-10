import streamlit as st
import requests
import time
import os

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

USER = "user"
ASSISTANT = "assistant"

# Display previous messages
for msg in st.session_state.messages:
        if msg["role"] == USER:
            st.chat_message(USER).write(msg["content"])
        else:
            st.chat_message(ASSISTANT).write(msg["content"])     
        

# Input box for user prompt
prompt: str = st.chat_input("Type your message here...")

if prompt:
    # Display the user's message
    st.chat_message(USER).write(prompt)
    # Save user's message to session state
    st.session_state.messages.append({"role": USER, "content": prompt})

    # Make a POST request to get the assistant's response
    try:
        response = requests.post(
            "http://127.0.0.1:9000/query", 
            json={"query": prompt}  # Assuming the API expects a JSON payload
        )
        response.raise_for_status()  # Check if the request was successful
        assistant_reply = response.json()
    except requests.RequestException as e:
        assistant_reply = [f"Error: {e}"]

    # Display the assistant's response
    for i in assistant_reply:
        st.chat_message(ASSISTANT).write(i)
        # Save assistant's response to session state
        st.session_state.messages.append({"role": ASSISTANT, "content": i})

prompt=None

   

# Continuously call /rl and display response if it's a string
rl_response_area = st.empty()  # This will create a placeholder that updates dynamically

status_file = os.path.join(os.path.dirname(os.path.abspath(__name__)), "llm", "LLM", "LLM", "status_retrive.txt")

while True:
    try:
        # Make the GET request to /rl endpoint
        with open(status_file, "r", encoding="utf-8") as file:
            signal = file.read()
        # requests.get("http://127.0.0.1:9000/reset")
        # If the response is a string, display it as a message
        if signal!="null" and signal.strip():  # Check if the response is not empty
            rl_message = f"{signal} has reached its target"
              # Update the message in Streamlit
            st.chat_message(ASSISTANT).write(rl_message)
            st.session_state.messages.append({"role": ASSISTANT, "content": rl_message})
            requests.get("http://127.0.0.1:9000/reset")
        

  
          # Delay before the next GET request (adjust as needed)
    except requests.RequestException as e:
            print(e)

