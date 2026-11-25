# 🎬 Movie Review Sentiment Analysis using SimpleRNN

An NLP and Deep Learning project that classifies movie reviews as
**Positive** or **Negative** using a Simple Recurrent Neural Network
(SimpleRNN). A Streamlit web application allows users to enter a review
and view the model's prediction.

## 🚀 Live Demo

> Add your Streamlit Community Cloud app URL here after deployment.

## 📌 Project Overview

The goal of this project is to build a sentiment analysis system that
can identify whether a movie review expresses a positive or negative
opinion.

The project uses the IMDB movie review dataset provided by
TensorFlow/Keras. Reviews are represented as integer sequences, padded
to a consistent length, and passed to an embedding layer and SimpleRNN
classifier.

## ✨ Features

-   Movie review sentiment classification: Positive or Negative
-   Text preprocessing using the IMDB word-index mapping
-   Sequence padding and truncation
-   Word embedding representation
-   SimpleRNN-based binary classification
-   Streamlit interface for interactive predictions
-   Prediction score and predicted-class confidence display

## 🧠 Model Architecture

The model workflow:

``` text
Movie Review
     ↓
Lowercase and split into words
     ↓
IMDB word-index conversion
     ↓
Sequence padding / truncation
     ↓
Embedding Layer
     ↓
SimpleRNN Layer
     ↓
Dense Layer (Sigmoid)
     ↓
Positive / Negative
```

The model architecture used in the training script should match the
saved `rnn_model.keras` file. The Streamlit app is configured for the
following input settings:

  Setting                                               Value
  ------------------------- ---------------------------------
  Dataset                                  IMDB movie reviews
  Vocabulary limit                                     10,000
  Maximum sequence length                                 200
  Padding                                         Pre-padding
  Truncation                                   Pre-truncation
  Output                      Binary sentiment classification

## 📂 Project Structure

``` text
Movie-Review-Sentiment-Analysis/
│
├── app.py
├── train.py
├── prediction.py
├── rnn_model.keras
├── requirements.txt
├── README.md
└── .gitignore
```

If your repository uses different filenames, update this structure to
match the files actually committed.

## 🛠️ Technologies Used

-   Python
-   TensorFlow
-   Keras
-   NumPy
-   Streamlit
-   Natural Language Processing (NLP)
-   Simple Recurrent Neural Network (SimpleRNN)

## 📊 Dataset

This project uses the IMDB movie review dataset available through
`tensorflow.keras.datasets.imdb`.

The standard dataset contains: - 25,000 training reviews - 25,000
testing reviews - Binary labels: positive and negative

The model uses a vocabulary limit of 10,000 words.

## ⚙️ Installation and Setup

### 1. Clone the repository

``` bash
git clone https://github.com/gauravkatre55/Movie-Review-Sentiment-Analysis.git
```

### 2. Open the project directory

``` bash
cd Movie-Review-Sentiment-Analysis
```

### 3. Install dependencies

``` bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit application

``` bash
streamlit run app.py
```

The app will open in your browser. Enter a movie review and select
**Analyze Sentiment**.

## 🧪 Example Reviews

**Positive example**

``` text
This movie was fantastic. The acting was brilliant, the story was engaging,
and I enjoyed every moment.
```

**Negative example**

``` text
This was a boring movie with weak acting and a disappointing story.
I did not enjoy it.
```

These are sample inputs for testing. Actual predictions depend on the
trained model.

## 📈 Model Evaluation

After training, evaluate the model on the held-out test set. Add your
actual results here:

  Metric          Result
  --------------- --------------------
  Test loss       Add measured value
  Test accuracy   Add measured value

Do not enter estimated or assumed metrics. Use the values printed by
your model's `model.evaluate(x_test, y_test)`.

## 🌐 Deployment

This application can be deployed using Streamlit Community Cloud:

1.  Push the project to GitHub.
2.  Open [Streamlit Community Cloud](https://share.streamlit.io/).
3.  Connect your GitHub account.
4.  Select this repository and choose `app.py` as the main file.
5.  Deploy the application.

Make sure `rnn_model.keras` is available to the deployed app. If the
model file is too large for GitHub's normal file-size limits, use Git
LFS or an appropriate model-hosting service and update the app to load
it from there.

## ⚠️ Important Notes

-   The preprocessing used during prediction must match the
    preprocessing used during training.
-   Keep the same IMDB word-index convention, vocabulary limit, sequence
    length, padding, and truncation settings.
-   The displayed confidence is derived from the model's sigmoid output
    and is not necessarily a calibrated probability.
-   This is an educational project and its predictions may be incorrect,
    especially for sarcasm, mixed opinions, slang, or reviews with
    complex context.

## 👨‍💻 Author

**Gaurav Katre**

-   GitHub: [@gauravkatre55](https://github.com/gauravkatre55)
-   LinkedIn: [Gaurav Katre](https://www.linkedin.com/in/gauravkatre/)

------------------------------------------------------------------------

⭐ If you find this project useful, feel free to explore the repository
and share feedback.
