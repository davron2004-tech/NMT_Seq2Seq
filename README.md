# Uzbek-English Neural Machine Translation (Seq2Seq with Attention)

This repository contains an implementation of a **sequence-to-sequence (Seq2Seq)** model with **attention**, designed for **translating sentences between Uzbek and English** (in both directions).

The architecture is inspired by the 2015 paper:
📄 [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) by Luong et al.

---

## 🚀 Features

- Encoder-decoder model with **LSTM** layers
- **Luong-style attention mechanism** (global attention)
- Vocabulary size: **50,000**
- Embedding dimension: **1000**
- Hidden state dimension: **1000**
- Trained on **50,000 Uzbek-English parallel sentences**
- Word-level tokenization
- Built with **PyTorch**
- Achieved **BLEU score ~22** for both Uzbek→English and English→Uzbek translation tasks

---

## 📚 Dataset

We use the bilingual dataset:
🔗 [SlimOrca-Dedup-English-Uzbek](https://huggingface.co/datasets/MLDataScientist/SlimOrca-Dedup-English-Uzbek)

Each entry in the dataset is a sentence pair with translations between English and Uzbek.

---

## 🧠 Model Architecture

- **Encoder:** LSTM that encodes the source sentence
- **Decoder:** LSTM with attention and input-feeding
- **Attention Layer:** Dot-product attention (Luong-style global attention)
- **Output Layer:** Concatenated decoder + context → Linear → Softmax

---

## 🏋️ Training

- Optimizer: `Adam`
- Loss function: `CrossEntropyLoss` with masking for padded tokens
- Batch size: configurable
- Training data size: ~50,000 samples
- Token `<eos>` used for padding

---

## 📊 Evaluation

- Evaluation metric: **BLEU score**
- Average BLEU on validation set (~64 samples per direction):
  - **Uzbek → English:** ~22
  - **English → Uzbek:** ~22

---



## 🌌 GUI

![Gradio web app]()
