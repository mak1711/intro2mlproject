# intro2mlproject
This project focuses on **converting natural speech into executable commands** for a quadruped robot.  
It uses machine learning and natural language understanding to interpret spoken instructions (e.g., *“stand up”*, *“go forward”*) and map them to robot control actions.

---

## 📁 Project Overview
The project aims to bridge the gap between **human speech** and **robotic control** by training models that classify voice commands into predefined intents.  
This forms the foundation for enabling intuitive human–robot interaction.
---

## 🚀 How to Use

1. **Upload the dataset** file to your **Google Drive**.  
2. **Open** the Jupyter notebooks in **Google Colab**.  
3. **Mount your Drive** inside Colab and update the dataset path if needed.  
4. **Run all cells** to preprocess data, train models, and evaluate results.

---

## 🧠 Models Used
- Logistic Regression (baseline)  
- LSTM and 1D CNN for intent classification  
- Transformer-based models: **BERT**, **DistilBERT**, **RoBERTa**

---

## 📊 Dataset
- **Total examples:** 56,250  
- **Intent classes:** 9 (each with 6,250 examples)  
- **Language:** English  

The dataset was **generated using ChatGPT**, allowing full flexibility to modify or expand it — for example, adding new intents or incorporating multiple languages if needed.

---

## 🛠 Requirements
- Python 3.10+  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- TensorFlow / PyTorch  
- transformers (Hugging Face)

---

## 🎯 Goal
To develop a system that can accurately interpret spoken commands and translate them into robot control actions — a crucial step toward **natural voice-controlled quadruped robots**.

---
