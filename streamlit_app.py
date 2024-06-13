import time
import random
import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
import plotly.express as px

from datetime import datetime, date

from streamlit_option_menu import option_menu

import data_preprocessing

path = os.getcwd()
TOPIC_MODELING_FOLDER = 'topic_modeling'
MODEL_NAME = 'final_xgboost_sg_tuned.joblib'

# Load XGBoost Model for Topic Predictions
model = joblib.load(os.path.join(TOPIC_MODELING_FOLDER, MODEL_NAME))

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
            # To get when user input the feedbacks
            today = datetime.now()

            # Format the date
            created_at = today.strftime("%d/%m/%Y %H:%M:%S")

            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt, 'created_at': created_at})

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
            st.session_state.messages.append({"role": "assistant", "content": response, 'created_at': created_at})

            send_message()
    else:
        st.write("Your Feedbacks is already being processed. Please check in the Employee Tracker Menu to see more.")

        # Read User Response
        file_path = 'user_employee_feedbacks/user_employee_feedbacks.json'
        test_data = data_preprocessing.processing(file_path)


        # Vectorize User Response
        test_data_vectorized = data_preprocessing.embed_text(list(test_data['user_employee_feedbacks']))

        # Predicted Topic
        predicted_topic = model.predict(test_data_vectorized)

        st.session_state.test_data = test_data


if selected == 'Employee Tracker':
    st.title(f'Employee Tracker')

    tab_titles = ['Overview', 'Sentiment Analysis', 'Topic Modelling']
    tab1, tab2, tab3 = st.tabs(tab_titles)

    data_path = './dataset/Capgemini_Employee_Reviews_from_AmbitionBox.csv'
    # Load the data
    data = pd.read_csv(data_path)

    with tab1:
        # Count the total employees
        data['Unique_Helper_Col'] = data['Title'] + data['Place'] + data['Job_type']

        # Initialize 'test_data' in session state if it doesn't exist
        if 'test_data' not in st.session_state:
            st.session_state['test_data'] = None

        if st.session_state.test_data is None:

            col1, col2, col3 = st.columns(3)

            with col1:     
                st.metric(
                    label='Total Employees',
                    value=data.Title.nunique()
                )

            # Read User Response
            file_path = 'user_employee_feedbacks/user_employee_feedbacks.json'
            test_data = data_preprocessing.processing(file_path)
           
            # Count Today's Total Feedbacks
            test_data['created_at'] = pd.to_datetime(test_data['created_at'])
            today = date.today()

            today_total_feedbacks = test_data[test_data['created_at'].dt.date == today].shape[0]

            with col2:
                st.metric(
                label="Total Today's Feedbacks",
                value=today_total_feedbacks
                )

            with col3:
                st.metric(
                    label='Avg Overall Rating',
                    value=data.Overall_rating.mean().round(2)
                )

            st.write("List of Today's Feedbacks")
            st.table(test_data.loc[test_data['created_at'].dt.date == today, ['created_at', 'content']].to_dict(orient='records'))# Calculate counts of each place

            ########################################## PLACE BARCHART #####################################################
            place_counts = data['Place'].value_counts().reset_index()
            place_counts.columns = ['Place', 'Count']

            # Sort by count and select top 10
            top_places = place_counts.head(10)

            # Create a countplot using Plotly Express
            fig = px.bar(top_places, x='Place', y='Count', title='Top 10 Places by Count')

            # Add data labels (annotations)
            fig.update_traces(texttemplate='%{y}', textposition='outside')

            # Customize layout
            fig.update_layout(
                xaxis_title='Place',
                yaxis_title='Count',
                height=550
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            ########################################## END OF PLACE BARCHART #####################################################
            ########################################## DEPARTMENT BARCHART #####################################################


            # Calculate counts of each Department
            place_counts = data['Department'].value_counts().reset_index()
            place_counts.columns = ['Department', 'Count']

            # Sort by count and select top 10
            top_places = place_counts.head(10)

            # Create a countplot using Plotly Express
            fig = px.bar(top_places, x='Department', y='Count', title='Top 10 Department by Count')

            # Add data labels (annotations)
            fig.update_traces(texttemplate='%{y}', textposition='outside')

            # Customize layout
            fig.update_layout(
                xaxis_title='Department',
                yaxis_title='Count',
                height=550
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            ########################################## END OF PLACE BARCHART #####################################################
   
        else:
            col1, col2, col3 = st.columns(3)
            test_data = st.session_state.test_data
            # Count Today's Total Feedbacks
            test_data['created_at'] = pd.to_datetime(test_data['created_at'])
            today = date.today()
            today_total_feedbacks = test_data[test_data['created_at'].dt.date == today].shape[0]


            with col1:     
                st.metric(
                    label='Total Employees',
                    value=data.Title.nunique()
                )

            with col2:
                st.metric(
                label="Total Today's Feedbacks",
                value=today_total_feedbacks
                )

            with col3:
                st.metric(
                    label='Avg Overall Rating',
                    value=data.Overall_rating.mean().round(2)
                )

            st.write("List of Today's Feedbacks")
            st.table(test_data.loc[test_data['created_at'].dt.date == today, ['created_at', 'content']].to_dict(orient='records'))

            ########################################## PLACE BARCHART #####################################################
            place_counts = data['Place'].value_counts().reset_index()
            place_counts.columns = ['Place', 'Count']

            # Sort by count and select top 10
            top_places = place_counts.head(10)

            # Create a countplot using Plotly Express
            fig = px.bar(top_places, x='Place', y='Count', title='Top 10 Places by Count')

            # Add data labels (annotations)
            fig.update_traces(texttemplate='%{y}', textposition='outside')

            # Customize layout
            fig.update_layout(
                xaxis_title='Place',
                yaxis_title='Count',
                height=550
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            ########################################## END OF PLACE BARCHART #####################################################
            ########################################## DEPARTMENT BARCHART #####################################################


            # Calculate counts of each Department
            place_counts = data['Department'].value_counts().reset_index()
            place_counts.columns = ['Department', 'Count']

            # Sort by count and select top 10
            top_places = place_counts.head(10)

            # Create a countplot using Plotly Express
            fig = px.bar(top_places, x='Department', y='Count', title='Top 10 Department by Count')

            # Add data labels (annotations)
            fig.update_traces(texttemplate='%{y}', textposition='outside')

            # Customize layout
            fig.update_layout(
                xaxis_title='Department',
                yaxis_title='Count',
                height=550
            )

            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True)
            ########################################## END OF PLACE BARCHART #####################################################

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
