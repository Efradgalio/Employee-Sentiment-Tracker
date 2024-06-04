### STILL ERROR --> Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details.

from openai import OpenAI
import streamlit as st
import os

st.title('Spark Bot')

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
client = OpenAI()

if 'openai_model' not in st.session_state:
    st.session_state['openai_model'] = 'gpt-3.5-turbo'

# Initialize chat history
## Session state feature is allow you to remember the chat history interaction
if 'messages' not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# React to user input
prompt = st.chat_input('What is up?')
if prompt:
    # Display user message in chat message container
    with st.chat_message('user'):
        st.markdown('prompt')

    # Add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Create chat message container for assistant (CODE FOR OPEN AI)
    with st.chat_message('assistant'):
        message_placeholder = st.empty()
        full_response = ''
        # Calling openai API
        for response in client.chat.completions.create(
            model=st.session_state['openai_model'],
            messages=[
                {'role': m['role'], 'content': m['content']}
                for m in st.session_state.messages
            ],
            # Get chat-gpt response and simulation typing effect
            stream=True,
        ):
            full_response += response.choice[0].delta.get('content', '')
            message_placeholder.markdown(full_response+ '| ')
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({'role': 'assistant', 'content': full_response})

