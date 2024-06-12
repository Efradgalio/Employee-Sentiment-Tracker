import time
import random
import json
import data_preprocessing

import streamlit as st

from streamlit_option_menu import option_menu


st.set_page_config(layout='wide', page_title='Spark Employee Dashboard')

with st.sidebar:
    selected = option_menu(
        menu_title = 'Main Menu',
        options = ['Sparkbot', 'Employee Tracker', 'Contact Us'],
        icons=['robot', 'bar-chart', 'envelope'],
        menu_icon = 'app-indicator',
        default_index=0,
        # orientation='horizontal'
    )

if selected == 'Sparkbot':

    responses = [
                "Thank you, your feedback will be processed and review by our HR team.",
                "Your Welcome, if you want to submit the feedbacks again, please refresh the website."
            ]

    # Define the message limit
    MESSAGE_LIMIT = 1

    # Initialize session state if not already done
    if 'step' not in st.session_state:
        st.session_state.step = 0

    # Initialize session state for the message count
    if 'message_count' not in st.session_state:
        st.session_state.message_count = 0


    def send_message():
        st.session_state.message_count += 1

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


    if st.session_state.message_count < MESSAGE_LIMIT:
        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Access user feedbacks
            user_employee_feedbacks = st.session_state.messages[-1]
            # Save the last message to a JSON file
            file_path = 'user_employee_feedbacks/user_employee_feedbacks.json'
            with open(file_path, 'r+') as f:
                all_user_response = json.load(f)
                all_user_response['user_responses'].append(user_employee_feedbacks)
                f.seek(0)
                json.dump(all_user_response, f, indent=4)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = st.write_stream(get_next_response())

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            send_message()
    else:
        st.write("Your Feedbacks is already being processed. Please check in the Employee Tracker Menu to see more.")

        # Read User Response
        file_path = 'user_employee_feedbacks/user_employee_feedbacks.json'
        test_data = data_preprocessing.processing(file_path)
        st.session_state.test_data = test_data


if selected == 'Employee Tracker':
    st.title(f'Employee Tracker')

    tab_titles = ['Overview', 'Sentiment Analysis', 'Topic Modelling']
    tabs = st.tabs(tab_titles)

    st.table(st.session_state.test_data)

if selected == 'Contact Us':
    st.header(":mailbox: Get In Touch With Us!")


    contact_form = """
    <form action="https://formsubmit.co/efradgamer@gmail.COM" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here"></textarea>
        <button type="submit">Send</button>
    </form>
    """

    st.markdown(contact_form, unsafe_allow_html=True)

    # Use Local CSS File
    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


    local_css("style/style.css")
