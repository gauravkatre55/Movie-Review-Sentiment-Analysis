# 🎬 Movie Review Sentiment Analysis using SimpleRNN

## 📌 Project Overview

This project is a Deep Learning-based Movie Review Sentiment Analysis application that predicts whether a movie review expresses a **Positive** or **Negative** sentiment.

The project uses a **Simple Recurrent Neural Network (SimpleRNN)** with an Embedding layer to understand sequential text data and classify movie reviews.

A user-friendly web application is developed using **Streamlit**, allowing users to enter a movie review and get an instant sentiment prediction.

## 🎯 Project Objectives

- Understand Natural Language Processing (NLP) concepts.
- Convert text reviews into numerical sequences.
- Apply padding to make input sequences uniform.
- Build and train a SimpleRNN model.
- Predict positive and negative movie sentiments.
- Deploy the trained model using Streamlit.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| NumPy | Numerical Operations |
| TensorFlow | Deep Learning Framework |
| Keras | Neural Network Development |
| SimpleRNN | Sentiment Classification |
| NLP | Text Processing |
| Streamlit | Web Application |
| Git & GitHub | Version Control |

## 📂 Project Structure

```text
Movie-Review-Sentiment-Analysis/
│
├── app.py                  # Streamlit Web Application
├── train.py                # Model Training
├── prediction.py           # Sentiment Prediction
├── rnn_model.keras         # Trained RNN Model
├── requirements.txt        # Project Dependencies
├── README.md               # Project Documentation

```

## 📊 Dataset

This project uses the **IMDB Movie Reviews Dataset**, available through TensorFlow/Keras.

Dataset Details:

- Training Reviews: 25,000
- Testing Reviews: 25,000
- Total Reviews: 50,000
- Sentiment Classes: Positive and Negative

The dataset contains movie reviews with corresponding sentiment labels.

Dataset is loaded using:

```python
from tensorflow.keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=10000
)
```

## ⚙️ Project Workflow

### 1. Data Loading
Load the IMDB movie review dataset using Keras.

### 2. Text Preprocessing
- Convert reviews into numerical sequences.
- Use the top 10,000 most frequent words.
- Apply padding to maintain a fixed sequence length.

### 3. Model Architecture

The SimpleRNN model consists of:

| Layer | Description |
|---|---|
| Input | Accepts numerical review sequences |
| Embedding | Converts word indexes into dense vectors |
| SimpleRNN | Learns sequential patterns in text |
| Dense | Produces sentiment probability |
| Sigmoid | Classifies positive or negative sentiment |

### 4. Model Training

The model is compiled using:

- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Evaluation Metric: Accuracy
- Early Stopping: To help prevent overfitting

### 5. Sentiment Prediction

The trained model receives a new movie review, processes the text, and predicts its sentiment.

### 6. Web Application

Streamlit provides an interactive interface where users can enter movie reviews and view prediction results.

## 🧠 Model Architecture

```text
Movie Review
     |
     v
Text Preprocessing
     |
     v
Numerical Encoding
     |
     v
Padding Sequences
     |
     v
Embedding Layer
     |
     v
SimpleRNN Layer
     |
     v
Dense Layer
     |
     v
Sigmoid Activation
     |
     v
Sentiment Prediction
     |
     v
Positive / Negative
```

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/gauravkatre55/Movie-Review-Sentiment-Analysis.git
```

### Step 2: Navigate to Project Folder

```bash
cd Movie-Review-Sentiment-Analysis
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 💬 Example Predictions

**Positive Review:**

"This movie was absolutely fantastic. The acting was brilliant, and the storyline was amazing. I really enjoyed watching it."

Expected sentiment: Positive

**Negative Review:**

"This was one of the worst movies I have ever watched. The story was boring, the acting was terrible, and I wasted my time."

Expected sentiment: Negative

*Note: Actual predictions depend on the trained model.*

## 📈 Key Learnings

Through this project, I learned:

- Natural Language Processing fundamentals.
- Text encoding and sequence padding.
- Word Embeddings.
- Recurrent Neural Networks (RNN).
- Model training and evaluation.
- Early Stopping.
- Building interactive applications using Streamlit.
- Saving and loading trained Deep Learning models.

## 🔮 Future Improvements

- Implement LSTM and GRU models.
- Compare SimpleRNN, LSTM, and GRU performance.
- Improve text preprocessing.
- Experiment with different sequence lengths and embedding dimensions.
- Deploy the application online.

## 👨‍💻 Author

**Gaurav Katre**

- GitHub: https://github.com/gauravkatre55
- LinkedIn: https://www.linkedin.com/in/gauravkatre/

---

⭐ If you find this project useful, consider giving it a star!
