# 🚩 Fake News vs Real News LLM Classifier

A Streamlit web app for classifying news headlines as "Fake" or "Real" using OpenAI GPT models with few-shot prompting. Built for fast, interactive classification with single or batch inputs.

## Features

- **LLM-Powered Classification**: Uses OpenAI GPT models (3.5-turbo, GPT-4) with few-shot prompting
- **Few-Shot Learning**: Automatically creates balanced few-shot examples from training data
- **Configurable Prompting**: Adjustable number of examples per class in the prompt
- **Multiple Model Support**: Choose between different OpenAI models
- **Interactive Web Interface**: User-friendly Streamlit app for real-time classification
- **Batch Processing**: Support for multiple upload methods with cost estimation:
  - Text files (one sentence per line)
  - CSV files (with 'sentence' column)
  - Manual text input
- **LLM Evaluation**: Performance testing on sample test set with API cost tracking
- **Downloadable Results**: Export batch classification results as CSV

## Setup

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Get OpenAI API Key:**

   - Sign up at https://platform.openai.com
   - Create an API key in your dashboard
   - Add credits to your account

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## Data

This app uses data from kaggle: https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection/data?select=true.csv
Csv files were converted into a single json for simplicity.

- `data/fake_news_dataset.json`

Each file contains JSON arrays with objects having:

- `item_id`: Unique identifier
- `headline`: News headline content
- `gold_label`: Either "Fake" or "Real"

## Usage

1. **API Setup**: Enter your OpenAI API key in the sidebar and select a model
2. **Prepare Few-Shot Examples**: Click "Prepare Few-Shot Examples" to create balanced training examples
3. **Single Classification**: Enter text and click "Classify Text" for individual predictions
4. **Batch Classification**: Upload files or enter multiple sentences (with cost estimates)
5. **LLM Evaluation**: Test performance on a sample of the test set

## Model Details

- **Algorithm**: Few-shot prompting with OpenAI GPT models
- **Prompt Engineering**: Structured prompts with clear task definitions and balanced examples
- **Few-Shot Examples**: Configurable number of examples per class (3-20)
- **Temperature**: Low temperature (0.1) for consistent predictions
- **Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrix on test samples

## Cost Considerations

- **Single Classification**: ~$0.001 per text
- **Batch Processing**: Cost scales linearly with number of texts
- **Model Evaluation**: ~$0.05-0.20 for 50-item test sample
- **Model Choice**: GPT-3.5-turbo is most cost-effective, GPT-4 is more accurate but expensive

## Example Usage

### Single Text Classification

```
Input: "U.S. looks to Facebook, private groups to battle online extremism."
Output: ✅ Real (Confidence: 0.95)
```

### Few-Shot Prompt Structure

```
You are a text classifier that categorizes headlines as either "Fake" or "Real".

Fake: Misleading, false, or sensational headlines
Real: Verified, factual, or reliable headlines

Examples:
Text: "Aliens Invade Washington DC Overnight"
Classification: Fake

Text: "NASA Confirms Water on Mars"
Classification: Real

Text: "[Your headline here]"
Classification:
```

### Batch Classification

Upload files with cost estimates shown before processing. Results include confidence scores and are downloadable as CSV.
