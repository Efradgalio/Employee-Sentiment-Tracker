'''
This code is example for use bert_sentiment.py module
'''

from bert_sentiment import BertSentimentAnalyzer
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification


model_name = 'habibul08/employee-sentiment-tracker'  
analyzer = BertSentimentAnalyzer(model_name)

# test 1: predict from dataset
# Load dataset
df = pd.read_excel('inference100data.xlsx')

# Execute prediction
result_df = analyzer.execute_prediction(df, 'text')

# Save results
analyzer.save_excel(result_df, 'latest_inference_output_with_predictions.xlsx')
print(result_df.head())
analyzer.evaluate()
# Evaluate the model's predictions on the loaded dataset and save graphics
analyzer.evaluate()

# Save ROC-AUC curve to a specific path
analyzer.visualize_roc_auc(save_path='roc_auc_curve.png')

# Save confusion matrix to a specific path
analyzer.visualize_confusion_matrix(save_path='confusion_matrix.png')

# test2: predict manually

user_text = ( 
    "best workplace!"
    )
result = analyzer.execute_predict_manual(user_text)
print(user_text, result) 
# Evaluate the model's predictions on the loaded dataset and save graphics
analyzer.evaluate()

