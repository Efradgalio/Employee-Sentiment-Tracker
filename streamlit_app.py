import time
import random
import streamlit as st



responses = [
            "Thank you, your feedback will be processed and review by our HR team.",
            "Your Welcome, if you want to submit the feedbacks again, please refresh the website.",
           ]


# Initialize session state if not already done
if 'step' not in st.session_state:
    st.session_state.step = 0

def get_next_response():
    # Get the next response based on the current step
    response = responses[st.session_state.step]
    st.session_state.step = (st.session_state.step + 1) % len(responses)

    for word in response.split():
        yield word + " "
        time.sleep(0.05)

st.title("Sparkbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message('assistant'):
    st.markdown("""Hello! Welcome to the Employee Feedback Chatbot. Please provide your feebacks
                 or complaints.""")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(get_next_response())

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
