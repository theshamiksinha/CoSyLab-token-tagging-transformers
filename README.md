# 🍳 CoSyLab NER Transformer Benchmark

Implementation and benchmarking of transformer-based Named Entity Recognition (NER) models for cooking instructions.  
Developed at **Complex Systems Laboratory (CoSy Lab), IIIT-Delhi**, this project explores the performance of various transformer architectures on a custom BIO-encoded recipe dataset.

---

## 📁 Repository Structure

```
├── code_model.py                                # Core implementation: preprocessing, training, evaluation, prediction
├── train.tsv                                    # BIO-encoded training dataset (CoNLL-style)
├── test.tsv                                     # BIO-encoded testing dataset
├── report.pdf                                   # Full project report with results and analysis
```

---

## 🧠 Project Overview

This project focuses on extracting structured information from cooking instructions using NER. It identifies entities such as:

- `QUANTITY`
- `UNIT`
- `NAME` (Ingredient name)
- `STATE` (Preparation state like chopped, diced)
- and others (e.g., FORM, SIZE, DRY/FRESH)

Our objectives included:
- Implementing and benchmarking transformer-based NER models
- Studying the impact of BIO encoding on model performance
- Comparing model accuracy, training efficiency, and suitability for deployment

---

## ⚙️ Methodology

### Dataset
- Two datasets: an initial 8K phrase dataset and an extended 10K+10K version
- Each token is labeled using BIO format (`B-UNIT`, `I-STATE`, etc.)
- Format: `.tsv`, one token-tag pair per line, sentences separated by blank lines

### Model Architectures
Evaluated models include:
- **BERT**, **DistilBERT**, **BioBERT**, **BERT-large-cased**
- **RoBERTa-base**, **RoBERTa-large**, **DistilRoBERTa**
- **XLM-RoBERTa**, **ALBERT**, **DeBERTa**
- **spaCy transformer**

### Implementation
- Tokenization & label alignment using custom `word_ids` logic
- BIO encoding performed via custom function
- Fine-tuning via Hugging Face `Trainer`
- Evaluation metrics: Precision, Recall, F1 Score, Accuracy

---

## 📊 Key Results (Cleaned BIO-Encoded Dataset)

| Model           | Accuracy | Precision | Recall | F1 Score | Training Time (s) |
|----------------|----------|-----------|--------|----------|--------------------|
| BERT           | 0.9179   | 0.9191    | 0.9179 | 0.9181   | 125.76             |
| RoBERTa        | 0.9170   | 0.9179    | 0.9170 | 0.9172   | 126.66             |
| DistilBERT     | 0.9140   | 0.9144    | 0.9140 | 0.9140   | 67.24              |
| XLM-RoBERTa    | 0.9174   | 0.9184    | 0.9174 | 0.9174   | 142.86             |
| DeBERTa        | 0.9170   | 0.9176    | 0.9170 | 0.9170   | 160.27             |
| DistilRoBERTa  | 0.9100   | 0.9109    | 0.9100 | 0.9100   | 69.67              |

*RoBERTa and DeBERTa achieved consistent high scores; DistilBERT provided a fast, lightweight alternative.*

---

## 📚 Dataset Format

CoNLL-style BIO-encoded `.tsv`:

```
2	B-QUANTITY
lbs	B-UNIT
pork	B-NAME
,	O
cut	B-STATE
into	I-STATE
1-inch	I-STATE
cubes	I-STATE

<blank line separates sentences>
```

---

## 🔄 How to Run

### 1. Install Dependencies
```bash
pip install transformers datasets scikit-learn matplotlib seaborn torch
```

### 2. Train & Evaluate Models
```bash
python code_model.py
```
Outputs:
- CSV file with evaluation metrics
- PNG heatmap + bar chart comparing model performance

---

## 📄 Report

Full details, including experimental setup, results, training analysis, and key findings, are available in:

[`NER_for_cooking_intructions_FINAL_REPORT.pdf`](./NER_for_cooking_intructions_FINAL_REPORT.pdf)

---

## 👍 Acknowledgments

Conducted as part of Winter Semester 2025 Independent Project at [CoSy Lab, IIIT-Delhi]  
**Supervisor**: Dr. Ganesh Bagler

**Team Members:**
- Vansh Yadav (2022559)  
- Tejus Madan (2022540)  
- Shamik Sinha (2022468)

---

## 📄 License

This project is made available for academic and non-commercial research purposes.
