import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

class BertSentimentAnalyzer:
    def __init__(self, model_name):
        # Load the model and tokenizer from Hugging Face or local
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def convert_to_list(self, dataframe, column_name):
        """Convert DataFrame column to list of strings."""
        return dataframe[column_name].astype(str).tolist()

    def tokenize_predict(self, texts):
        """Tokenize texts and obtain BERT embeddings and predictions."""
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        predictions = outputs.logits
        return predictions

    def compute_percentage(self, predictions):
        """Compute positive and negative sentiment percentages."""
        softmax_output = torch.softmax(predictions, dim=1)
        positive_percentage = softmax_output[:, 1].tolist()
        negative_percentage = (1 - softmax_output[:, 1]).tolist()  # Since it's a binary classification
        return positive_percentage, negative_percentage

    def tokenize_predict_and_calculate(self, texts):
        """Combines tokenization, prediction, and percentage calculation."""
        predictions = self.tokenize_predict(texts)
        positive_percentage, negative_percentage = self.compute_percentage(predictions)
        sentiments = ["Positive" if pred.argmax() == 1 else "Negative" for pred in predictions]
        return sentiments, positive_percentage, negative_percentage

    def execute_prediction(self, dataframe, text_column):
        """Execute the prediction and append results to the DataFrame."""
        texts = self.convert_to_list(dataframe, text_column)
        sentiments, positive_percentages, negative_percentages = self.tokenize_predict_and_calculate(texts)
        dataframe['sentiment'] = sentiments
        dataframe['positive_percentage'] = positive_percentages
        dataframe['negative_percentage'] = negative_percentages
        return dataframe

    def predict_manual_input(self, text):
        """Predict sentiment for a manually inputted text."""
        predictions = self.tokenize_predict([text])
        positive_percentage, negative_percentage = self.compute_percentage(predictions)
        sentiment = "Positive" if predictions.argmax() == 1 else "Negative"
        return {
            "sentiment": sentiment,
            "positive_percentage": positive_percentage[0],
            "negative_percentage": negative_percentage[0]
        }

    def save_csv(self, dataframe, output_path):
        """Save DataFrame with predictions to a CSV file."""
        dataframe.to_csv(output_path, index=False)
        print(f'The output saved on {output_path}')

    def save_excel(self, dataframe, output_path):
        """Save DataFrame with predictions to a xlsx file."""
        dataframe.to_excel(output_path, index=False)
        print(f'The output saved on {output_path}')

    def save_json(self, dataframe, output_path):
        """Save DataFrame with predictions to a json file."""
        dataframe.to_json(output_path, index=False)
        print(f'The output saved on {output_path}')

