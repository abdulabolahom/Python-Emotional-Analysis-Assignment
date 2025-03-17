# YouTube Comment Sentiment Analysis with BERT and TextBlob

This project performs sentiment analysis on YouTube comments using a combination of a fine-tuned BERT model and TextBlob, with additional preprocessing and visualization.

## Project Overview

This Jupyter Notebook contains code for:

1.  **Fine-tuning a BERT model for sentiment analysis:**
    * Training a BERT-based sequence classification model on a custom dataset.
    * Saving and loading the fine-tuned model from Google Drive.
2.  **Fetching YouTube comments:**
    * Using the YouTube Data API to retrieve comments from specified video IDs.
    * Storing the comments in a Pandas DataFrame.
3.  **Preprocessing text data:**
    * Cleaning and normalizing the text using techniques like lowercasing, handling contractions, removing URLs, special characters, and stopwords.
    * Lemmatization.
4.  **Sentiment analysis with TextBlob:**
    * Using TextBlob to determine the sentiment polarity of comments.
    * Visualizing the sentiment distribution using pie charts.
    * Including like counts in the textblob analysis.
5.  **Sentiment Analysis with a Lexicon based model:**
    * Utilizing the Vader lexicon to determine sentiment.
    * Visualizing the results.
6.  **Applying the fine-tuned BERT model for predictions:**
    * Loading the saved model and tokenizer.
    * Performing sentiment analysis on the fetched YouTube comments.
    * Displaying the predictions and confidence scores.

## Prerequisites

Before running the code, ensure you have the following:

* A Google account (for Google Colab and Google Drive).
* A YouTube Data API key.
* The following Python libraries installed:
    * Pandas
    * PyTorch
    * Transformers (Hugging Face)
    * Scikit-learn
    * Google API Client
    * TextBlob
    * Matplotlib
    * NLTK
    * Contractions
    * Autocorrect
    * Joblib

You can install the required libraries using pip:

```bash
pip install pandas torch transformers scikit-learn google-api-python-client textblob matplotlib nltk contractions autocorrect joblib
Setup
Clone the repository:
Clone this repository to your local machine or Google Drive.
Set up Google Colab:
Open the Jupyter Notebook in Google Colab.
Mount your Google Drive to Colab to access your datasets and save the trained model.
Place your excel files into the google drive location specified in the notebook.
YouTube Data API key:
Replace "AIzaSyAM07vB7FSGs46coHuu1wv3i5prszm8e54" with your actual YouTube Data API key.
Datasets:
Place your training, validation, and YouTube comments datasets in the specified Google Drive locations.
Ensure the excel files are formatted correctly, with the correct column names.
Usage
Run the notebook cells sequentially:
Execute each cell in the Jupyter Notebook in order.
The notebook is divided into sections, so that each section performs a specific task.
View the results:
The sentiment analysis results, visualizations, and downloaded CSV file will be available in your Google Drive or local downloads folder.
Adjust parameters:
Modify parameters like the number of training epochs, batch size, and video IDs as needed.
The file paths to the excel files can be changed.
Key Files
YouTube_Sentiment_Analysis.ipynb: The main Jupyter Notebook containing the code.
Segment1.xlsx: Training dataset.
Segment6.xlsx: Training dataset for the pretrained model.
Validation.xlsx: Validation dataset.
YouTubeComments.csv: CSV file containing the fetched YouTube comments and sentiment analysis results.
/content/drive/MyDrive/AI and Data Science/Model: location where the model is saved.
/content/drive/MyDrive/AI and Data Science/Model/label_encoder.pkl: Location where the label encoder is saved.
Notes
The code assumes that your datasets have columns named "Text" and "Sentiment_3". Adjust the column names if necessary.
The YouTube Data API has usage quotas. Be mindful of these quotas when fetching comments.
The API key has been removed from the provided readme, and should be replaced with your own key.
The location of the model and excel files can be changed.
Author
Abdulrahman Aboluhom

License
MIT License
