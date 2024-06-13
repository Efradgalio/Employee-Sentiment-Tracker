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
from bert_sentiment import BertSentimentAnalyzer

import data_preprocessing

path = os.getcwd()
TOPIC_MODELING_FOLDER = 'topic_modeling'
MODEL_NAME = 'final_xgboost_sg_tuned.joblib'
SENTIMEN_MODEL_NAME = 'habibul08/employee-sentiment-tracker'

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

        bert_sentiment = BertSentimentAnalyzer(SENTIMEN_MODEL_NAME)
        user_predictions = bert_sentiment.tokenize_predict(test_data.iloc[-1]['user_responses_cleaned'])

        # Determine sentiment
        sentiment = "Positive" if user_predictions.argmax() == 1 else "Negative"

        previously_test_data = pd.read_csv('./user_employee_feedbacks/user_feedbacks.csv')
        test_data['sentiment'] = previously_test_data.sentiment

        # Input the data into final test
        test_data.loc[test_data.shape[0]-1, 'sentiment'] = sentiment

        test_data.to_csv('./user_employee_feedbacks/user_feedbacks.csv')

        # Vectorize User Response
        # test_data_vectorized = data_preprocessing.embed_text(test_data['user_responses_cleaned'][-1])

        # Predicted Topic
        # predicted_topic = model.predict(test_data_vectorized)
        # test_data['topic'] = predicted_topic
        st.session_state.test_data = test_data


if selected == 'Employee Tracker':
    st.title(f'Employee Tracker')

    tab_titles = ['Overview', 'Sentiment Analysis', 'Topic Modelling']
    tab1, tab2, tab3 = st.tabs(tab_titles)

    data_path = './dataset/data_employee_topic.csv'
    # Load the data
    data = pd.read_csv(data_path)


    ## SIMPLE PREPROCESSING TO FIX DATE ##
    data.Date = data.Date.str.replace(' ', '-')
    data.Date = data.Date.str.replace('0', '1')
    data.Date = data.Date.str.replace('0', '1')
    data.Date = data.Date.str.replace('2017', '17')
    data.Date = data.Date.str.replace('2018', '18')
    data.Date = data.Date.str.replace('2019', '19')
    data.Date = data.Date.str.replace('2020', '20')
    data.Date = data.Date.str.replace('2021', '21')
    data.Date = data.Date.str.replace('2022', '22')
    data.Date = data.Date.str.replace('2023', '23')
    data.Date = pd.to_datetime(data.Date)
    data.Date = data.Date.astype(str)
    data.Date = data.Date.str.replace('2123', '2023')
    data.Date = data.Date.str.replace('2122', '2022')
    data.Date = data.Date.str.replace('2121', '2021')
    data.Date = data.Date.str.replace('2120', '2020')
    data.Date = data.Date.str.replace('2119', '2019')
    data.Date = data.Date.str.replace('2118', '2018')
    data.Date = data.Date.str.replace('2117', '2017')
    data.Date = pd.to_datetime(data.Date)

    three_months_ago = pd.Timestamp(data.sort_values(by='Date',ascending=False).Date[0]) - pd.DateOffset(months=3)\
    # Filter the DataFrame to include only rows from the last three months
    filtered_data = data[data['Date'] >= three_months_ago]

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

            test_data = pd.read_csv('./user_employee_feedbacks/user_feedbacks.csv')
            test_data = data_preprocessing.processing_csv(test_data)
           
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
            st.table(test_data.loc[test_data['created_at'].dt.date == today, ['created_at', 'content', 'sentiment']].to_dict(orient='records'))# Calculate counts of each place
            st.subheader('Topic Distribution by Place & Departments for the past 3 months')

            ########################################## PLACE BARCHART #####################################################
            # Get counts of Place, Category, and Binary
            place_category_binary_counts = filtered_data.groupby(['Place', 'Topic', 'Sentiment']).size().reset_index(name='Count')

            # Sort by count and select top 10 places
            top_places = place_category_binary_counts.groupby('Place').sum().reset_index().sort_values(by='Count', ascending=False).head(10)['Place']
            top_place_category_binary_counts = place_category_binary_counts[place_category_binary_counts['Place'].isin(top_places)]

            # Create a grouped bar chart using Plotly Express
            fig = px.bar(top_place_category_binary_counts, x='Place', y='Count', color='Topic',
                        facet_col='Sentiment', title='Top 10 Places by Count, Topic, and Sentimen')

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
            # Get counts of Department, Category, and Binary
            department_category_binary_counts = filtered_data.groupby(['Department', 'Topic', 'Sentiment']).size().reset_index(name='Count')

            # Sort by count and select top 10 departments
            top_departments = department_category_binary_counts.groupby('Department').sum().reset_index().sort_values(by='Count', ascending=False).head(10)['Department']
            top_department_category_binary_counts = department_category_binary_counts[department_category_binary_counts['Department'].isin(top_departments)]

            # Define custom color sequence
            custom_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

            # Create a grouped bar chart using Plotly Express
            fig = px.bar(top_department_category_binary_counts, x='Department', y='Count', color='Topic',
                        facet_col='Sentiment', title='Top 10 Department by Count, Topic, and Sentimen',
                        color_discrete_sequence=custom_colors)

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
            ########################################## END OF DEPARTMENT BARCHART #####################################################
   
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
            st.table(test_data.loc[test_data['created_at'].dt.date == today, ['created_at', 'content', 'sentiment']].to_dict(orient='records'))


            st.subheader('Topic Distribution by Place & Departments for the past 3 months')
             ########################################## PLACE BARCHART #####################################################
            # Get counts of Place, Category, and Binary
            place_category_binary_counts = filtered_data.groupby(['Place', 'Topic', 'Sentiment']).size().reset_index(name='Count')

            # Sort by count and select top 10 places
            top_places = place_category_binary_counts.groupby('Place').sum().reset_index().sort_values(by='Count', ascending=False).head(10)['Place']
            top_place_category_binary_counts = place_category_binary_counts[place_category_binary_counts['Place'].isin(top_places)]

            # Create a grouped bar chart using Plotly Express
            fig = px.bar(top_place_category_binary_counts, x='Place', y='Count', color='Topic',
                        facet_col='Sentiment', title='Top 10 Places by Count, Topic, and Sentimen')

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
            # Get counts of Department, Category, and Binary
            department_category_binary_counts = filtered_data.groupby(['Department', 'Topic', 'Sentiment']).size().reset_index(name='Count')

            # Sort by count and select top 10 departments
            top_departments = department_category_binary_counts.groupby('Department').sum().reset_index().sort_values(by='Count', ascending=False).head(10)['Department']
            top_department_category_binary_counts = department_category_binary_counts[department_category_binary_counts['Department'].isin(top_departments)]

            # Define custom color sequence
            custom_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

            # Create a grouped bar chart using Plotly Express
            fig = px.bar(top_department_category_binary_counts, x='Department', y='Count', color='Topic',
                        facet_col='Sentiment', title='Top 10 Department by Count, Topic, and Sentimen',
                        color_discrete_sequence=custom_colors)

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
            ########################################## END OF DEPARTMENT BARCHART #####################################################
   

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
