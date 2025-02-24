
# Fine-tune XLM-RoBERTa for YouTube Comment Sentiment Analysis 

This project focuses on fine-tuning a multilingual sentiment analysis model, `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`, specifically for YouTube comments. By training on a dataset of approximately 1 million YouTube comments with sentiment labels, we aimed to create a model that is more accurate and nuanced in understanding the sentiment expressed in online video discussions.
## Table of Contents

- [Project Overview](#project-overview)
- [Code Structure](#code-structure)
  - [Cleaning Notebook](#cleaningipynb-data-cleaning-and-preprocessing)
  - [Fine-tuning Notebook](#finetuneipynb-model-fine-tuning)
  - [Testing Notebook](#testipynb-model-testing-and-comparison)
- [Datasets](#datasets)
- [Hugging Face Model Hub](#hugging-face-model-hub)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Model Card (Summary)](#model-card-summary)
- [Author](#author)

## Project Overview

The original `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` model is trained primarily on Twitter data. While effective, it may not be perfectly optimized for the nuances, tone, slang of sentiment expression found in YouTube comments, which can differ in length, style, and context compared to tweets.

This project addresses this gap by fine-tuning the base model on a large dataset of YouTube comments. The result is a sentiment analysis model that demonstrates improved performance on YouTube-specific text, particularly in identifying neutral and mixed sentiments.

## Code Structure

This repository contains three main Jupyter Notebooks detailing the project workflow:

1.  **`cleaning.ipynb`**: Data Cleaning and Preprocessing
    *   This notebook focuses on preparing the YouTube comments dataset (`youtube_comments.csv`) for fine-tuning.
    *   **Key steps include:**
        *   Loading and exploring the dataset.
        *   Analyzing comment lengths and token counts to understand data characteristics.
        *   Removing duplicate comments to ensure data quality.
        *   **Link and Timestamp Removal:** All URLs and timestamps within the comment text were removed to focus the model on textual sentiment cues.
        *   Tokenizing comments using the `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` tokenizer.
        *   Further analysis of token counts post-cleaning.
        *   Saving cleaned and preprocessed data to `df_preprocessed.csv` and other intermediate files for subsequent steps.

2.  **`finetune.ipynb`**: Model Fine-tuning
    *   This notebook details the fine-tuning process of the `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` model.
    *   **Key steps include:**
        *   Loading the preprocessed dataset (`comments.csv`, which is assumed to be `df_preprocessed.csv` from cleaning step).
        *   Splitting the dataset into training and validation sets (90/10 split).
        *   Mapping sentiment labels (Negative, Neutral, Positive) to numerical labels (0, 1, 2).
        *   Tokenizing the comment text using the pre-trained tokenizer with truncation and padding to a maximum length of 64 tokens.
        *   Setting up the fine-tuning Trainer with:
            *   `CustomTrainer` class with label smoothing in loss computation.
            *   `TrainingArguments` defining output directory, evaluation strategy, batch sizes, learning rate, and other hyperparameters.
            *   `EarlyStoppingCallback` to prevent overfitting.
        *   Defining `compute_metrics` function to evaluate accuracy.
        *   Training the model and saving the fine-tuned model and tokenizer to `./youtube_sentiment_model_final`.
    *   **Performance:** The fine-tuned model achieved an accuracy of approximately **80.17%** on the YouTube comments validation dataset, demonstrating a significant improvement over the base model's reported accuracy of **69.3%** when fine-tuned on tweets.

3.  **`test.ipynb`**: Model Testing and Comparison
    *   This notebook evaluates the performance of the fine-tuned model and compares it against the original `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` model.
    *   **Key steps include:**
        *   Loading both the fine-tuned model and the original base model along with their respective tokenizers.
        *   Defining a list of example YouTube comments representing various sentiments and nuances.
        *   Implementing a `predict_sentiment` function to get sentiment predictions from both models.
        *   Comparing the predictions of both models on the example comments.
    *   **Results Summary:**
        *   Both models showed agreement on clearly negative comments.
        *   The **fine-tuned model** demonstrated **improved performance** in identifying **neutral sentiments** in comments with mixed feedback, where the base model often incorrectly predicted negative sentiment.
        *   The fine-tuned model is better at understanding **nuanced sentiment** in YouTube comments, especially constructive or mixed feedback.
        *   The original base model, primarily trained on tweets, showed limitations in accurately identifying neutral sentiment in YouTube comment contexts and sometimes leaned towards negative sentiment even in balanced comments.
        *   The original base model has a reported accuracy of approximately **69.3%** on tweet sentiment analysis tasks.

## Datasets

The original dataset of YouTube comments with sentiment labels. Due to data privacy considerations, this file is not included in the repository. You can download a similar dataset from [Kaggle: YouTube Comments Sentiment Dataset](https://www.kaggle.com/datasets/amaanpoonawala/youtube-comments-sentiment-dataset). 

## Hugging Face Model Hub

The fine-tuned model is also available on the Hugging Face Model Hub for easy access and usage: [AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual](https://huggingface.co/AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual). You can directly load and use the model from Hugging Face using the provided link in your code, as demonstrated in `test.ipynb`.

## Dependencies

*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `transformers` (`datasets`, `AutoTokenizer`, `AutoModelForSequenceClassification`, `TrainingArguments`, `Trainer`, `EarlyStoppingCallback`)
*   `scikit-learn` (`accuracy_score`)
*   `torch`

## Usage

To use the fine-tuned model for sentiment prediction on new YouTube comments, you can refer to the `test.ipynb` notebook for an example.  Load the `youtube_sentiment_model_final` model (or directly from Hugging Face Hub) and tokenizer and use the `predict_sentiment` function.

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_path = "AmaanP314/youtube-xlm-roberta-base-sentiment-multilingual"
model_finetuned = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer_finetuned = AutoTokenizer.from_pretrained(model_path)

labels = ["Negative", "Neutral", "Positive"]

def predict_sentiment(model, tokenizer, comments):
    inputs = tokenizer(comments, return_tensors="pt", truncation=True, padding=True, max_length=64)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_indices = torch.argmax(logits, dim=-1).tolist()
    return [labels[idx] for idx in predicted_class_indices]

comments_to_predict = [
    "This video is amazing!",
    "This video is OK",
    "This is terrible."
]

predictions = predict_sentiment(model_finetuned, tokenizer_finetuned, comments_to_predict)
print(predictions) # Output: ['Positive', 'Neutral', 'Negative']
```
## Model Card (Summary)

*   **Base Model:** `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`
*   **Fine-tuning Dataset:** Approximately 1 million YouTube comments with sentiment labels, available on [Kaggle: YouTube Comments Sentiment Dataset](https://www.kaggle.com/datasets/amaanpoonawala/youtube-comments-sentiment-dataset).
*   **Metrics:**
    *   **Fine-tuned Model Accuracy (YouTube Comments Validation Set):** ~80.17%
    *   **Base Model Accuracy (Reported on Tweet Sentiment Task):** ~69.3%
*   **Intended Use:** Sentiment analysis of YouTube comments, understanding viewer feedback on video content.
*   **Limitations:** The model is fine-tuned specifically for YouTube comments and may not perform as well on other types of text. The dataset and specific cleaning/preprocessing steps influence the model's performance.

## Author

[Amaan Poonawala](https://www.linkedin.com/in/amaan-poonawala/)
