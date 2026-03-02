# AI Sentiment Analyzer (CLI)

A simple command-line sentiment analysis tool built while learning **Python** and **Hugging Face Transformers**.

This project analyzes the sentiment of user-provided text using a pretrained transformer model and also exposes some of the internal steps used by the model such as tokenization and token IDs.

The goal of this project is to understand **how transformer models process text**, not just how to use them.

---

## Features

* Interactive CLI sentiment analysis
* Uses a pretrained transformer model for classification
* Displays tokenized text used by the model
* Shows token IDs that are fed into the neural network
* Displays special tokens `[CLS]` and `[SEP]`
* Prints sentiment prediction and confidence score
* Handles empty input validation
* Supports `exit` command to terminate the program

---

## Model Used

This project uses the Hugging Face model:

distilbert-base-uncased-finetuned-sst-2-english

This model is a **DistilBERT** variant fine-tuned on the **Stanford Sentiment Treebank (SST-2)** dataset for binary sentiment classification.

DistilBERT is a smaller and faster version of BERT that retains most of its performance while being significantly more efficient.

---

## Example

```
Enter text: this is a good learning

Tokens:
['[CLS]', 'this', 'is', 'a', 'good', 'learning', '[SEP]']

Token IDs:
[101, 2023, 2003, 1037, 2204, 4083, 102]

Sentiment: POSITIVE
Confidence: 97.82 %
```

---

## Installation

Clone the repository:

```
git clone https://github.com/your-username/ai-sentiment-analyzer.git
cd ai-sentiment-analyzer
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Run the Program

```
python sentiment.py
```

The program will start an interactive CLI.

Example:

```
Enter text: I love learning AI
```

To exit the program:

```
exit
```

---

## Project Structure

```
ai-sentiment-analyzer
│
├── sentiment.py
├── requirements.txt
└── README.md
```

---

## What I Learned

### Python

* loops (`while`)
* conditionals (`if`)
* user input handling
* basic CLI program structure
* working with lists and dictionaries

### Transformers / NLP

* using Hugging Face pipelines
* tokenization
* token IDs
* special tokens `[CLS]` and `[SEP]`
* pretrained model inference

### Machine Learning Concepts

* sentiment classification
* subword tokenization
* transformer-based NLP models
* confidence scores and probabilities

---

## Future Improvements

* sentence-level sentiment analysis
* support for multiple transformer models
* attention visualization
* convert the CLI tool into a small web application
* experiment with other NLP tasks such as summarization

