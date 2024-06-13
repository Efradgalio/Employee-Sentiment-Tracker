import pandas as pd
import torch
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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

    def execute_predict_manual(self, text):
        """Predict sentiment for a manually inputted text."""
        predictions = self.tokenize_predict([text])
        positive_percentage, negative_percentage = self.compute_percentage(predictions)
        sentiment = "Positive" if predictions.argmax() == 1 else "Negative"
        return {
            "sentiment": sentiment,
            "positive_percentage": positive_percentage[0],
            "negative_percentage": negative_percentage[0]
        }

    def evaluate(self, dataframe, text_column, true_labels):
        """Evaluate model's predictions against true labels."""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before evaluation.")

        texts = self.convert_to_list(dataframe, text_column)
        predictions = self.tokenize_predict(texts)
        self.predicted_labels = [pred.argmax().item() for pred in predictions]
        self.true_labels = true_labels

        # Classification report
        report = classification_report(true_labels, self.predicted_labels, target_names=['Negative', 'Positive'])
        print("Classification Report:")
        print(report)

        # Confusion Matrix
        self.visualize_confusion_matrix()
        plt.savefig('confusion_matrix.png')  # Save the confusion matrix plot

        # ROC-AUC Curve
        self.visualize_roc_auc()
        plt.savefig('roc_auc_curve.png')  # Save the ROC-AUC curve plot

        # Return evaluation metrics
        return report

    def visualize_roc_auc(self, save_path=None):
        """Visualize the ROC-AUC curve."""
        if not self.true_labels or not self.predicted_labels:
            raise ValueError("True labels and predicted labels must be available for ROC-AUC visualization.")

        roc_auc = roc_auc_score(self.true_labels, self.predicted_labels)
        fpr, tpr, _ = roc_curve(self.true_labels, self.predicted_labels)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-AUC Curve')
        plt.legend(loc='lower right')
        if save_path:
            plt.savefig(save_path)  # Save the ROC-AUC curve plot to specified path
        plt.show()

    def visualize_confusion_matrix(self, save_path=None):
        """Visualize the confusion matrix."""
        if not self.true_labels or not self.predicted_labels:
            raise ValueError("True labels and predicted labels must be available for confusion matrix visualization.")

        cm = confusion_matrix(self.true_labels, self.predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        if save_path:
            plt.savefig(save_path)  # Save the confusion matrix plot to specified path
        plt.show()

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

