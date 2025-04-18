import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import time

# Function to read a CoNLL‑style TSV file (one token and tag per line; blank lines separate sentences)
def read_conll_file(filepath):
    sentences = []
    words, tags = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append((words, tags))
                    words, tags = [], []
            else:
                parts = line.split("\t")
                if len(parts) != 2:
                    raise ValueError(f"Line does not have exactly two columns: {line}")
                word, tag = parts
                words.append(word)
                tags.append(tag)
        if words:
            sentences.append((words, tags))
    return sentences

# Function to convert a list of tags into BIO format.
def bio_encode_tags(tags):
    bio_tags = []
    prev_tag = "O"
    for tag in tags:
        if tag == "O":
            bio_tags.append("O")
            prev_tag = "O"
        else:
            # Start a new entity if the previous tag was different (or O)
            if prev_tag != tag:
                bio_tags.append("B-" + tag)
            else:
                bio_tags.append("I-" + tag)
            prev_tag = tag
    return bio_tags

# Function to build a DataFrame from sentences and optionally apply BIO encoding.
def build_dataframe_from_sentences(sentences, apply_bio=True):
    data = []
    for words, tags in sentences:
        if apply_bio:
            tags = bio_encode_tags(tags)
        data.append({"words": words, "labels": tags})
    return pd.DataFrame(data)

# Define the file paths to your TSV files
train_file = "/raid/home/ganeshb/.nve/train.tsv"  
test_file = "/raid/home/ganeshb/.nve/test.tsv"   

# Read and process the data
train_sentences = read_conll_file(train_file)
test_sentences = read_conll_file(test_file)

train_df = build_dataframe_from_sentences(train_sentences, apply_bio=False)
test_df = build_dataframe_from_sentences(test_sentences, apply_bio=False)

print("Number of training sentences:", len(train_df))
print("Number of testing sentences:", len(test_df))
print("\nSample training row:\n", train_df.iloc[0])

# Extract unique tags
unique_tags = sorted(set(tag for tags in train_df["labels"] for tag in tags))
print("\nUnique Tags:", unique_tags)

# Create tag mappings
tag2id = {tag: i for i, tag in enumerate(unique_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}
num_labels = len(unique_tags)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define models to compare with "Fast" tokenizers
models_to_compare = {
    "BERT": "bert-base-uncased",
    "RoBERTa": "roberta-base",
    "DistilBERT": "distilbert-base-uncased",
    "XLM-RoBERTa": "xlm-roberta-base",
    "DeBERTa": "microsoft/deberta-base",
    "DistilRoBERTa": "distilroberta-base"
}

# Alternative tokenization approach for NER task that doesn't require word_ids()
def tokenize_and_align_labels_manual(examples, tokenizer, model_name):
    tokenized_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    
    for i, (words, labels) in enumerate(zip(examples["words"], examples["labels"])):
        # Map words to tokens and keep track of word boundaries
        tokens = []
        word_ids = []
        token_labels = []
        
        for word_idx, (word, label) in enumerate(zip(words, labels)):
            # Get the token IDs for this word
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                # Handle empty token result by using a space
                word_tokens = tokenizer.tokenize(" " + word)
                if not word_tokens:
                    # Skip if still empty
                    continue
            
            # Add the tokens for this word
            tokens.extend(word_tokens)
            
            # Add word_id for the first token of the word
            word_ids.append(len(tokens) - len(word_tokens))
            
            # Add the label for this word
            token_labels.append(tag2id[label])
        
        # Convert tokens to IDs
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Add special tokens
        if model_name.startswith("roberta") or model_name.startswith("xlm-roberta") or model_name.startswith("distilroberta"):
            input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
            word_offset = 1  # RoBERTa-like models have a leading BOS token
        else:
            input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
            word_offset = 1  # BERT-like models have a leading CLS token
            
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Create token-aligned labels
        aligned_labels = [-100] * len(input_ids)  # Initialize with padding label
        for idx, word_pos in enumerate(word_ids):
            aligned_labels[word_pos + word_offset] = token_labels[idx]
        
        # Pad or truncate sequences to max_length
        max_length = 128
        if len(input_ids) > max_length:
            # Truncate
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            aligned_labels = aligned_labels[:max_length]
        else:
            # Pad
            padding_length = max_length - len(input_ids)
            input_ids = input_ids + [tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            aligned_labels = aligned_labels + [-100] * padding_length
            
        tokenized_inputs["input_ids"].append(input_ids)
        tokenized_inputs["attention_mask"].append(attention_mask)
        tokenized_inputs["labels"].append(aligned_labels)
    
    return tokenized_inputs

# Convert DataFrames to Hugging Face Datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Compute metrics function for evaluation
def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)
    
    true_preds = []
    true_labels = []
    for pred_seq, label_seq in zip(preds, labels):
        for p_val, l_val in zip(pred_seq, label_seq):
            if l_val != -100:  # Ignore special tokens
                true_preds.append(p_val)
                true_labels.append(l_val)
    
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, true_preds, average='weighted', labels=np.arange(num_labels))
    accuracy = (np.array(true_preds) == np.array(true_labels)).mean() if true_labels else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Function to train and evaluate a specific model
def train_and_evaluate_model(model_name, model_path, train_dataset, test_dataset):
    print(f"\n{'=' * 50}")
    print(f"Training {model_name} model")
    print(f"{'=' * 50}")
    
    try:
        # Load tokenizer (using standard AutoTokenizer)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Process datasets using the manual alignment approach
        def process_dataset(dataset):
            tokenized_dataset = {
                "input_ids": [],
                "attention_mask": [],
                "labels": []
            }
            
            for i in range(len(dataset)):
                example = dataset[i]
                tokens = tokenize_and_align_labels_manual(
                    {"words": [example["words"]], "labels": [example["labels"]]}, 
                    tokenizer, 
                    model_path
                )
                tokenized_dataset["input_ids"].append(tokens["input_ids"][0])
                tokenized_dataset["attention_mask"].append(tokens["attention_mask"][0])
                tokenized_dataset["labels"].append(tokens["labels"][0])
            
            return Dataset.from_dict(tokenized_dataset)
        
        processed_train_dataset = process_dataset(train_dataset)
        processed_test_dataset = process_dataset(test_dataset)
        
        # Load model
        model = AutoModelForTokenClassification.from_pretrained(
            model_path, 
            num_labels=num_labels
        ).to(device)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./results_{model_name.lower().replace('-', '_')}",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,  # Reduced epochs for faster comparison
            weight_decay=0.01,
            logging_dir=f"./results_{model_name.lower().replace('-', '_')}/logs",
            save_strategy="no",
            report_to="none"
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_train_dataset,
            eval_dataset=processed_test_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train the model and measure time
        start_time = time.time()
        trainer.train()
        training_time = time.time() - start_time
        
        # Evaluate the model
        eval_results = trainer.evaluate()
        
        # Add training time to results
        eval_results["training_time"] = training_time
        
        print(f"\n{model_name} Evaluation Results:", eval_results)
        return eval_results, model, tokenizer
    
    except Exception as e:
        print(f"Error training {model_name}: {e}")
        return {"error": str(e)}, None, None

# Store results
results = {}
trained_models = {}

# Train and evaluate each model
for model_name, model_path in tqdm(models_to_compare.items(), desc="Training models"):
    try:
        # Check if model exists on HuggingFace
        try:
            AutoTokenizer.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading model {model_name} ({model_path}): {e}")
            results[model_name] = {"error": str(e)}
            continue
            
        results[model_name], model, tokenizer = train_and_evaluate_model(
            model_name, model_path, train_dataset, test_dataset
        )
        if model is not None and "error" not in results[model_name]:
            trained_models[model_name] = (model, tokenizer)
    except Exception as e:
        print(f"Unexpected error training {model_name}: {e}")
        results[model_name] = {"error": str(e)}

# Filter successful models
successful_models = {
    model_name: result for model_name, result in results.items() 
    if "error" not in result
}

# Create a DataFrame for the results if we have any successful models
if successful_models:
    results_df = pd.DataFrame({
        model_name: {
            "Accuracy": results[model_name].get("eval_accuracy", 0),
            "Precision": results[model_name].get("eval_precision", 0),
            "Recall": results[model_name].get("eval_recall", 0),
            "F1 Score": results[model_name].get("eval_f1", 0),
            "Training Time (s)": results[model_name].get("training_time", 0)
        }
        for model_name in models_to_compare.keys() if model_name in successful_models
    }).T

    # Print the results in a fancy table
    print("\n" + "=" * 80)
    print("Model Performance Comparison")
    print("=" * 80)
    print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))
    print("=" * 80)

    # Visualize the results if we have successful models
    plt.figure(figsize=(16, 12))

    # Plot Accuracy, Precision, Recall, and F1 in a subplot
    plt.subplot(2, 1, 1)
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_data = results_df[metrics_to_plot]
    sns.heatmap(metrics_data, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5, cbar_kws={"label": "Score"})
    plt.title("Model Performance Metrics", fontsize=16)
    plt.ylabel("Model", fontsize=14)

    # Plot Training Time in a separate subplot
    plt.subplot(2, 1, 2)
    time_data = results_df[['Training Time (s)']]
    sns.barplot(x=time_data.index, y='Training Time (s)', data=time_data.reset_index(), palette="viridis")
    plt.title("Training Time Comparison", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel("Time (seconds)", fontsize=14)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("model_comparison_results.png", dpi=300)
    plt.show()

    # Export results to CSV
    results_df.to_csv("ner_model_comparison_results.csv")
    print("\nResults saved to 'ner_model_comparison_results.csv' and visualization saved to 'model_comparison_results.png'")
else:
    print("\nNo models were successfully trained. Cannot create comparison table or visualization.")

# Check if there were any model training errors
failed_models = [model_name for model_name in models_to_compare.keys() 
                if model_name in results and "error" in results[model_name]]
if failed_models:
    print("\nModels that failed to train:")
    for model_name in failed_models:
        print(f"- {model_name}: {results[model_name]['error']}")

# Function to predict with a specific model
def predict_tokens_with_model(tokens, model, tokenizer, model_name):
    # Process tokens using our manual tokenization method
    processed_input = tokenize_and_align_labels_manual(
        {"words": [tokens], "labels": [["O"] * len(tokens)]},
        tokenizer,
        model_name
    )
    
    # Convert to tensors
    input_ids = torch.tensor(processed_input["input_ids"]).to(device)
    attention_mask = torch.tensor(processed_input["attention_mask"]).to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2).cpu().numpy()[0]
    
    # Map predictions back to tokens (ignoring special tokens)
    predicted_tags = []
    current_token_idx = 0
    
    for token in tokens:
        # Find the corresponding prediction for this token
        # (simplification: we take the first token's prediction for each word)
        while current_token_idx < len(predictions) and processed_input["labels"][0][current_token_idx] == -100:
            current_token_idx += 1
            
        if current_token_idx < len(predictions):
            tag_id = predictions[current_token_idx]
            predicted_tags.append(id2tag[tag_id])
            current_token_idx += 1
        else:
            predicted_tags.append("O")  # Default if we run out of predictions
    
    return list(zip(tokens, predicted_tags))

# Example usage with the best model
test_sentence = ["2", "tomatoes", ",", "cut", "into", "1/2-inch", "pieces"]

if successful_models and trained_models:
    try:
        # Find the best model based on F1 score
        best_model_name = results_df['F1 Score'].idxmax()
        if best_model_name in trained_models:
            model, tokenizer = trained_models[best_model_name]
            print(f"\nMaking predictions with the best model ({best_model_name}):")
            predictions = predict_tokens_with_model(test_sentence, model, tokenizer, models_to_compare[best_model_name])
            for word, label in predictions:
                print(f"{word}: {label}")
    except Exception as e:
        print(f"Error making predictions: {e}")
        
