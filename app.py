import streamlit as st
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import plotly.express as px
import openai
import random
import time
from typing import List, Dict

def load_data():
    """Load and combine data from both consensus files."""
    data_files = [
        "data/Consensus items : Group 1 - Red Flag vs Green Flag.json",
        "data/Consensus items: Group 2 - Red Flag vs Green Flag.json"
    ]
    
    combined_data = []
    
    for file_path in data_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined_data.extend(data)
        except FileNotFoundError:
            st.error(f"Data file not found: {file_path}")
            return pd.DataFrame()
        except json.JSONDecodeError:
            st.error(f"Invalid JSON in file: {file_path}")
            return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(combined_data)
    
    # Filter out 'Neither' labels for binary classification
    df = df[df['gold_label'].isin(['Red Flag', 'Green Flag'])]
    
    return df

def create_balanced_split(df, test_size=0.3, random_state=42):
    """Create a balanced train-test split for few-shot examples and evaluation."""
    X = df['sentence']
    y = df['gold_label']
    
    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def create_few_shot_examples(X_train, y_train, n_examples_per_class=10):
    """Create balanced few-shot examples for the LLM prompt."""
    df_train = pd.DataFrame({'sentence': X_train, 'label': y_train})
    
    few_shot_examples = []
    
    # Get examples for each class
    for label in ['Red Flag', 'Green Flag']:
        class_examples = df_train[df_train['label'] == label].sample(
            n=min(n_examples_per_class, len(df_train[df_train['label'] == label])),
            random_state=42
        )
        
        for _, row in class_examples.iterrows():
            few_shot_examples.append({
                'sentence': row['sentence'],
                'label': row['label']
            })
    
    # Shuffle the examples
    random.shuffle(few_shot_examples)
    return few_shot_examples

def build_few_shot_prompt(few_shot_examples: List[Dict], target_text: str) -> str:
    """Build a few-shot prompt for the LLM."""
    
    prompt = """You are a text classifier that categorizes sentences as either "Red Flag" or "Green Flag".

Red Flag: Sentences that contain concerning, negative, problematic, or potentially harmful content. This includes violence, inappropriate behavior, disturbing themes, offensive language, or anything that would be considered problematic.

Green Flag: Sentences that contain neutral, positive, educational, or harmless content. This includes normal descriptions, factual information, positive interactions, or everyday activities.

Here are some examples:

"""
    
    # Add few-shot examples
    for example in few_shot_examples:
        prompt += f'Text: "{example["sentence"]}"\nClassification: {example["label"]}\n\n'
    
    # Add the target text
    prompt += f'Text: "{target_text}"\nClassification:'
    
    return prompt

def predict_with_llm(client, few_shot_examples: List[Dict], texts: List[str], model_name: str = "gpt-3.5-turbo") -> List[Dict]:
    """Make predictions using LLM with few-shot prompting."""
    if isinstance(texts, str):
        texts = [texts]
    
    results = []
    
    for text in texts:
        try:
            # Build the prompt
            prompt = build_few_shot_prompt(few_shot_examples, text)
            
            # Make API call
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful text classifier. Respond with exactly 'Red Flag' or 'Green Flag' only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            prediction = response.choices[0].message.content.strip()
            
            # Clean up the prediction
            if "Red Flag" in prediction:
                prediction = "Red Flag"
                confidence = 0.85  # Mock confidence since LLM doesn't give probabilities
                red_flag_prob = 0.85
                green_flag_prob = 0.15
            elif "Green Flag" in prediction:
                prediction = "Green Flag"
                confidence = 0.85
                red_flag_prob = 0.15
                green_flag_prob = 0.85
            else:
                # Default to Green Flag if unclear
                prediction = "Green Flag"
                confidence = 0.50
                red_flag_prob = 0.50
                green_flag_prob = 0.50
            
            result = {
                'text': text,
                'prediction': prediction,
                'red_flag_prob': red_flag_prob,
                'green_flag_prob': green_flag_prob,
                'confidence': confidence
            }
            
            results.append(result)
            
            # Small delay to avoid rate limits
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Error processing text: {e}")
            # Return default result on error
            result = {
                'text': text,
                'prediction': "Green Flag",
                'red_flag_prob': 0.50,
                'green_flag_prob': 0.50,
                'confidence': 0.50
            }
            results.append(result)
    
    return results

def evaluate_llm_model(client, few_shot_examples: List[Dict], X_test, y_test, model_name: str = "gpt-3.5-turbo"):
    """Evaluate the LLM model on test set."""
    # Sample a smaller subset for evaluation to save costs
    test_sample_size = min(50, len(X_test))
    test_indices = random.sample(range(len(X_test)), test_sample_size)
    
    X_test_sample = [X_test.iloc[i] for i in test_indices]
    y_test_sample = [y_test.iloc[i] for i in test_indices]
    
    # Get predictions
    predictions = predict_with_llm(client, few_shot_examples, X_test_sample, model_name)
    y_pred = [pred['prediction'] for pred in predictions]
    
    return y_test_sample, y_pred

def main():
    st.set_page_config(
        page_title="Red Flag vs Green Flag LLM Classifier",
        page_icon="🚩",
        layout="wide"
    )
    
    st.title("🚩 Red Flag vs Green Flag LLM Many-Shot Classifier")
    st.markdown("*Powered by OpenAI GPT with Few-Shot Learning*")
    st.markdown("---")
    
    # API Configuration
    st.sidebar.header("🔧 API Configuration")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")
    model_choice = st.sidebar.selectbox("Model", ["gpt-4.1-nano"], index=0)
    
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        return
    
    # Initialize OpenAI client
    try:
        client = openai.OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        return
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    if df.empty:
        st.error("No data available. Please check the data files.")
        return
    
    # Sidebar with dataset info
    st.sidebar.header("📊 Dataset Information")
    st.sidebar.metric("Total Examples", len(df))
    
    class_counts = df['gold_label'].value_counts()
    for label, count in class_counts.items():
        st.sidebar.metric(f"{label} Examples", count)
    
    # Configuration
    st.sidebar.header("⚙️ Model Configuration")
    n_examples_per_class = st.sidebar.slider("Examples per class in prompt", min_value=3, max_value=20, value=10, 
                                            help="Number of examples for each class (Red/Green Flag) to include in the few-shot prompt")
    test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=30) / 100
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🤖 LLM Setup")
        
        if st.button("🔄 Prepare Few-Shot Examples", type="primary"):
            with st.spinner("Preparing few-shot examples..."):
                # Create train-test split
                X_train, X_test, y_train, y_test = create_balanced_split(df, test_size=test_size)
                
                # Create few-shot examples
                few_shot_examples = create_few_shot_examples(X_train, y_train, n_examples_per_class)
                
                # Store in session state
                st.session_state.client = client
                st.session_state.few_shot_examples = few_shot_examples
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.model_name = model_choice
                
                st.success(f"✅ Few-shot setup complete!")
                
                # Show few-shot examples info
                st.info(f"""
                **Few-Shot Examples**: {len(few_shot_examples)} total
                - {len([ex for ex in few_shot_examples if ex['label'] == 'Red Flag'])} Red Flag examples
                - {len([ex for ex in few_shot_examples if ex['label'] == 'Green Flag'])} Green Flag examples
                
                **Test Set**: {len(X_test)} examples  
                - Red Flag: {sum(y_test == 'Red Flag')}
                - Green Flag: {sum(y_test == 'Green Flag')}
                """)
                
                # Show sample few-shot examples
                with st.expander("🔍 View Sample Few-Shot Examples"):
                    for i, example in enumerate(few_shot_examples[:6]):  # Show first 6
                        if example['label'] == 'Red Flag':
                            st.error(f"**{example['label']}**: {example['sentence'][:100]}...")
                        else:
                            st.success(f"**{example['label']}**: {example['sentence'][:100]}...")
    
    with col2:
        st.header("🎯 Make Predictions")
        
        if 'few_shot_examples' not in st.session_state:
            st.warning("⚠️ Please prepare few-shot examples first!")
        else:
            # Single text prediction
            st.subheader("Single Text Classification")
            user_text = st.text_area("Enter text to classify:", 
                                    placeholder="Type or paste a sentence here...")
            
            if st.button("🔍 Classify Text") and user_text:
                with st.spinner("Classifying with LLM..."):
                    results = predict_with_llm(
                        st.session_state.client, 
                        st.session_state.few_shot_examples, 
                        [user_text],
                        st.session_state.model_name
                    )
                    result = results[0]
                
                # Display prediction with confidence
                if result['prediction'] == 'Red Flag':
                    st.error(f"🚩 **Red Flag** (Confidence: {result['confidence']:.2f})")
                else:
                    st.success(f"✅ **Green Flag** (Confidence: {result['confidence']:.2f})")
                
                # Probability breakdown
                col_rf, col_gf = st.columns(2)
                with col_rf:
                    st.metric("Red Flag Probability", f"{result['red_flag_prob']:.3f}")
                with col_gf:
                    st.metric("Green Flag Probability", f"{result['green_flag_prob']:.3f}")
            
            st.markdown("---")
            
            # Batch upload
            st.subheader("📤 Batch Upload & Classification")
            st.warning("⚠️ Note: LLM classification incurs API costs. Use small batches for testing.")
            
            # File upload methods
            upload_method = st.radio("Choose upload method:", ["Text File", "CSV File", "Manual Input"])
            
            if upload_method == "Text File":
                uploaded_file = st.file_uploader("Upload a text file (one sentence per line)", 
                                                type=['txt'])
                if uploaded_file is not None:
                    texts = uploaded_file.read().decode('utf-8').strip().split('\\n')
                    texts = [t.strip() for t in texts if t.strip()]
                    
                    st.info(f"Found {len(texts)} texts. Estimated cost: ~${len(texts) * 0.001:.3f}")
                    
                    if st.button("🔍 Classify Batch (Text File)"):
                        process_batch_llm(texts)
            
            elif upload_method == "CSV File":
                uploaded_file = st.file_uploader("Upload a CSV file with a 'sentence' column", 
                                                type=['csv'])
                if uploaded_file is not None:
                    try:
                        csv_df = pd.read_csv(uploaded_file)
                        if 'sentence' in csv_df.columns:
                            texts = csv_df['sentence'].dropna().tolist()
                            st.info(f"Found {len(texts)} sentences. Estimated cost: ~${len(texts) * 0.001:.3f}")
                            
                            if st.button("🔍 Classify Batch (CSV)"):
                                process_batch_llm(texts)
                        else:
                            st.error("CSV file must contain a 'sentence' column")
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")
            
            else:  # Manual Input
                manual_texts = st.text_area("Enter multiple sentences (one per line):", 
                                           height=150,
                                           placeholder="Sentence 1\\nSentence 2\\nSentence 3...")
                
                if st.button("🔍 Classify Batch (Manual)") and manual_texts:
                    texts = [t.strip() for t in manual_texts.strip().split('\\n') if t.strip()]
                    st.info(f"Processing {len(texts)} texts. Estimated cost: ~${len(texts) * 0.001:.3f}")
                    process_batch_llm(texts)

def process_batch_llm(texts):
    """Process a batch of texts using LLM and display results."""
    if not texts:
        st.warning("No texts to classify!")
        return
    
    with st.spinner(f"Classifying {len(texts)} texts with LLM..."):
        results = predict_with_llm(
            st.session_state.client,
            st.session_state.few_shot_examples,
            texts,
            st.session_state.model_name
        )
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    st.subheader("📊 Batch Results Summary")
    col1, col2, col3 = st.columns(3)
    
    red_count = sum(1 for r in results if r['prediction'] == 'Red Flag')
    green_count = len(results) - red_count
    avg_confidence = np.mean([r['confidence'] for r in results])
    
    with col1:
        st.metric("🚩 Red Flags", red_count)
    with col2:
        st.metric("✅ Green Flags", green_count)
    with col3:
        st.metric("📈 Avg Confidence", f"{avg_confidence:.3f}")
    
    # Detailed results table
    st.subheader("📋 Detailed Results")
    
    # Prepare display dataframe
    display_df = pd.DataFrame({
        'Text': [r['text'][:100] + '...' if len(r['text']) > 100 else r['text'] for r in results],
        'Prediction': results_df['prediction'],
        'Confidence': results_df['confidence'].round(3),
        'Red Flag Prob': results_df['red_flag_prob'].round(3),
        'Green Flag Prob': results_df['green_flag_prob'].round(3)
    })
    
    # Color code the predictions
    def color_predictions(row):
        if row['Prediction'] == 'Red Flag':
            return ['background-color: #ffebee'] * len(row)
        else:
            return ['background-color: #e8f5e8'] * len(row)
    
    st.dataframe(
        display_df.style.apply(color_predictions, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"llm_classification_results_{len(results)}_items.csv",
        mime="text/csv"
    )

# Show LLM evaluation if available
if 'few_shot_examples' in st.session_state:
    st.markdown("---")
    st.header("📊 LLM Performance Evaluation")
    
    with st.expander("🧪 Evaluate LLM on Test Set", expanded=False):
        st.warning("⚠️ This will make API calls to evaluate performance. Estimated cost: ~$0.05-0.20")
        
        if st.button("🔬 Run LLM Evaluation"):
            with st.spinner("Evaluating LLM performance on test set..."):
                try:
                    y_test_sample, y_pred = evaluate_llm_model(
                        st.session_state.client,
                        st.session_state.few_shot_examples,
                        st.session_state.X_test,
                        st.session_state.y_test,
                        st.session_state.model_name
                    )
                    
                    # Store evaluation results
                    st.session_state.y_test_sample = y_test_sample
                    st.session_state.y_pred = y_pred
                    
                    # Calculate accuracy
                    accuracy = accuracy_score(y_test_sample, y_pred)
                    st.success(f"✅ LLM Evaluation Complete! Test Accuracy: {accuracy:.3f}")
                    
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
    
    # Show evaluation results if available
    if 'y_test_sample' in st.session_state and 'y_pred' in st.session_state:
        st.subheader("📈 Evaluation Results")
        
        # Classification report
        try:
            report = classification_report(
                st.session_state.y_test_sample, 
                st.session_state.y_pred, 
                output_dict=True
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Classification Metrics")
                metrics_df = pd.DataFrame(report).transpose()
                st.dataframe(metrics_df.round(3))
            
            with col2:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state.y_test_sample, st.session_state.y_pred)
                
                fig = px.imshow(
                    cm,
                    text_auto=True,
                    labels={'x': 'Predicted', 'y': 'Actual'},
                    x=['Green Flag', 'Red Flag'],
                    y=['Green Flag', 'Red Flag'],
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating evaluation metrics: {e}")
            
        # Show sample predictions
        st.subheader("🔍 Sample Predictions")
        sample_df = pd.DataFrame({
            'Actual': st.session_state.y_test_sample[:10],
            'Predicted': st.session_state.y_pred[:10],
            'Text': [text[:80] + "..." if len(text) > 80 else text 
                    for text in st.session_state.X_test.iloc[:10]]
        })
        
        # Color code correct/incorrect predictions
        def highlight_predictions(row):
            if row['Actual'] == row['Predicted']:
                return ['background-color: #d4edda'] * len(row)  # Light green for correct
            else:
                return ['background-color: #f8d7da'] * len(row)  # Light red for incorrect
        
        st.dataframe(
            sample_df.style.apply(highlight_predictions, axis=1),
            use_container_width=True
        )

if __name__ == "__main__":
    main()