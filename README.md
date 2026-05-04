# 📰 Multi-Class Text Classification for News Headlines

> A comprehensive NLP project implementing and comparing multiple machine learning and deep learning models for automated news headline classification.

## 🎯 Project Overview

This project focuses on building and evaluating various machine learning and deep learning models to classify news headlines into multiple categories. The study compares different text representation techniques (TF-IDF and Word2Vec), preprocessing strategies, and neural network architectures to identify the most effective approach for text classification.

**Course:** CSE 440 - Group 14, Section 04

## ✨ Key Features

- **Multiple Classification Models:**
  - Logistic Regression (with TF-IDF)
  - Deep Neural Networks
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - SimpleRNN
  - Bidirectional variants of RNN architectures

- **Text Representation Techniques:**
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - Skip-gram Word2Vec embeddings

- **Preprocessing Strategies:**
  - No preprocessing
  - Extreme preprocessing
  - Optimized preprocessing pipeline

- **Comprehensive Evaluation:**
  - Accuracy metrics
  - Macro F1-scores
  - Confusion matrices
  - Performance analysis across preprocessing strategies

## 📊 Results Highlights

| Model               | Representation | Preprocessing | Accuracy   | Macro F1-Score |
| ------------------- | -------------- | ------------- | ---------- | -------------- |
| Logistic Regression | TF-IDF         | Optimum       | **91.58%** | **0.9155**     |
| Logistic Regression | TF-IDF         | Extreme       | 91.48%     | 0.9145         |
| Deep Neural Network | TF-IDF         | No            | 90.65%     | 0.9064         |
| Bidirectional LSTM  | Word2Vec       | Extreme       | 90.70%     | 0.9064         |
| Bidirectional GRU   | Word2Vec       | Optimum       | 90.47%     | 0.9040         |

**Best Performing Model:** Logistic Regression with TF-IDF representation and optimized preprocessing, achieving **91.58% accuracy**.

## 📁 Project Structure

```
NLP PROJECT/
├── README.md
├── Group14_Section04_CSE440_Multi_Class_Text_Classification_Project.ipynb
├── Dataset/
│   ├── Training_data_14.csv          # Training dataset
│   └── Test_data.csv                 # Test dataset
└── Test Results CSV/
    ├── final_test_results.csv        # Complete results summary
    └── validation_experiment_log.csv # Validation logs
```

## 🛠️ Technologies & Libraries

```python
# Core Libraries
- Python 3.x
- Jupyter Notebook

# Data Processing
- pandas: Data manipulation
- numpy: Numerical computations

# NLP Libraries
- NLTK: Natural Language Toolkit
  - Tokenization
  - Stopword removal
  - Stemming & Lemmatization
- Gensim: Word embeddings
  - Word2Vec models

# Machine Learning
- scikit-learn
  - TF-IDF Vectorization
  - Logistic Regression
  - Model evaluation metrics
  - Train-test splitting

# Deep Learning
- PyTorch
  - LSTM, GRU, SimpleRNN layers
  - Bidirectional RNNs
  - Custom neural architectures

# Visualization
- matplotlib: Plotting
- seaborn: Statistical graphics
- wordcloud: Text visualization
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed on your system.

### Installation

1. **Clone or download the repository:**

   ```bash
   # Navigate to the project directory
   cd "NLP PROJECT"
   ```

2. **Install required packages:**

   ```bash
   pip install nltk gensim wordcloud pandas numpy matplotlib seaborn scikit-learn torch
   ```

3. **Download NLTK resources:**
   The notebook automatically handles this, but you can manually download if needed:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('punkt_tab')
   nltk.download('omw-1.4')
   ```

### Running the Project

1. **Open Jupyter Notebook:**

   ```bash
   jupyter notebook Group14_Section04_CSE440_Multi_Class_Text_Classification_Project.ipynb
   ```

2. **Execute cells sequentially:**
   - Start with environment setup
   - Load and explore the datasets
   - Run preprocessing pipelines
   - Train and evaluate models
   - Review comprehensive results and visualizations

## 📋 Preprocessing Strategies

### 1. **No Preprocessing**

- Raw text without modifications
- Baseline for comparison

### 2. **Extreme Preprocessing**

- HTML entity decoding
- Lowercasing
- Special character removal
- Tokenization
- Stopword removal
- Stemming

### 3. **Optimum Preprocessing**

- HTML entity decoding
- Lowercasing
- Selective character removal
- Tokenization
- Stopword removal
- Lemmatization (more sophisticated than stemming)

## 🔍 Data Overview

- **Training Dataset:** `Training_data_14.csv`
  - Contains news headlines and their corresponding categories
  - Format: [News Headline, News Topic]

- **Test Dataset:** `Test_data.csv`
  - Validation and test data for model evaluation
  - Same format as training data

- **Sample Categories:** World News, Business, Technology, Politics, Sports, Entertainment, etc.

## 📈 Experimentation & Validation

The project conducts **30 comprehensive experiments** (RUN_01 to RUN_30) exploring:

- Different model architectures
- Multiple text representation methods
- Various preprocessing strategies
- Impact on accuracy and computational efficiency

All results are logged in `Test Results CSV/final_test_results.csv` with:

- Model name
- Text representation technique
- Preprocessing strategy applied
- Test accuracy
- Macro F1-score
- Runtime in seconds

## 🎓 Key Insights

1. **Logistic Regression outperforms deep learning models** for this task
2. **Optimized preprocessing significantly improves performance** over extreme preprocessing
3. **TF-IDF representation achieves better results** than Word2Vec for this dataset
4. **Bidirectional RNNs improve performance** compared to unidirectional variants
5. **Model simplicity and efficiency matter** - faster training with comparable or better accuracy

## 📊 Visualization Features

The notebook includes comprehensive visualizations:

- **EDA Plots:**
  - Distribution of news categories
  - Text length analysis
  - Word frequency histograms
  - Word clouds by category

- **Model Performance:**
  - Confusion matrices
  - Accuracy comparisons
  - F1-score distributions
  - Runtime analysis

- **Feature Analysis:**
  - Top words per category
  - TF-IDF term importance
  - Word2Vec embeddings visualization

## 💡 How to Extend This Project

1. **Implement additional models:**
   - Transformer-based models (BERT, DistilBERT)
   - Attention mechanisms
   - Ensemble methods

2. **Enhance preprocessing:**
   - Custom domain-specific tokenization
   - Contextual word embeddings
   - Advanced NLP techniques

3. **Increase dataset:**
   - Collect more training samples
   - Include additional news categories
   - Multi-lingual support

4. **Production deployment:**
   - Build REST API for predictions
   - Create web interface
   - Containerize with Docker

## 📝 Project Details

- **Model Count:** 6 unique architectures
- **Total Experiments:** 30 runs
- **Best Accuracy:** 91.58%
- **Evaluation Metrics:** Accuracy, Macro F1-Score, Runtime
- **Total Categories:** Multiple news classifications

## 🔧 Performance Optimization

The project tracks:

- **Training efficiency:** Runtime per model
- **Memory usage:** GPU/CPU utilization
- **Accuracy trade-offs:** Performance vs. speed
- **Reproducibility:** Fixed random seeds for all experiments

## 📧 Contact & Questions

For questions or contributions regarding this project, please refer to the course materials or contact your instructor.

---

<div align="center">

**Made with ❤️ for CSE 440 - Multi-Class Text Classification**

_A collaborative project exploring machine learning and NLP techniques_

</div>
