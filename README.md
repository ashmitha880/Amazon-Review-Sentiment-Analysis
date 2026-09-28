# Amazon Review Sentiment Analysis

A machine learning project that classifies **Amazon Alexa product reviews as Positive or Negative** using Natural Language Processing (NLP) and machine learning.

The project follows a complete text-classification workflow:

**Data Loading → Data Cleaning → Exploratory Data Analysis → Train/Test Split → TF-IDF → Model Comparison → Cross-Validation → Hyperparameter Tuning → Final Model → Model Saving → Prediction**

The final model uses **TF-IDF feature extraction with a Linear Support Vector Machine (Linear SVM)**.

---

## 1. Project Overview

Customer reviews contain useful information about how users perceive a product. In this project, Amazon Alexa product reviews are analyzed and classified into two feedback classes:

* **Positive → 1**
* **Negative → 0**

The project demonstrates how text data can be converted into numerical features using **TF-IDF** and then used to train machine learning classification models.

The notebook includes:

* Data loading and inspection
* Missing-value handling
* Review cleaning
* Duplicate investigation
* Exploratory Data Analysis
* Class-distribution analysis
* Train/test splitting
* TF-IDF feature extraction
* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine
* Stratified K-Fold Cross-Validation
* Hyperparameter tuning using GridSearchCV
* Final model pipeline
* Model serialization using Joblib
* Testing on new reviews
* Decision-function score analysis

---

## 2. Dataset

The project uses the **Amazon Alexa Reviews dataset** stored in:

```text
amazon_alexa.tsv
```

The original dataset contains **3150 reviews** and 5 columns:

| Column             | Description                           |
| ------------------ | ------------------------------------- |
| `rating`           | Product rating from 1 to 5            |
| `date`             | Date on which the review was recorded |
| `variation`        | Alexa product variation               |
| `verified_reviews` | Text of the customer review           |
| `feedback`         | Binary feedback label                 |

### Example

```text
rating: 5
variation: Charcoal Fabric
verified_reviews: Love my Echo!
feedback: 1
```

---

## 3. Target Variable

The target variable is:

```text
feedback
```

It contains two classes:

| Feedback | Meaning  |
| -------: | -------- |
|      `0` | Negative |
|      `1` | Positive |

An important characteristic of this dataset is that the `feedback` label is derived from the product rating. Therefore, the target represents **rating-based feedback**, rather than an independently human-annotated sentiment label.

---

## 4. Data Cleaning

The dataset was first inspected for its structure, data types, and missing values.

Initially:

```text
3150 rows
5 columns
```

There was one missing value in:

```text
verified_reviews
```

The missing review was removed.

The review text was then stripped of leading and trailing whitespace, and empty reviews were removed.

After cleaning:

```text
3070 reviews
5 columns
```

The notebook also investigates duplicate review texts and checks whether duplicate reviews contain conflicting feedback labels.

### Duplicate Investigation

The project checks:

* Number of duplicate review texts
* Whether duplicate reviews have different labels
* Remaining duplicate review texts

However, duplicates were **investigated rather than removed** in the current notebook.

---

## 5. Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the dataset before training the models.

The analysis includes:

### Feedback Distribution

After cleaning:

```text
Positive (1): 2833
Negative (0): 237
```

This shows that the dataset is **imbalanced**, with considerably more positive reviews than negative reviews.

Approximate distribution:

| Class    | Reviews | Percentage |
| -------- | ------: | ---------: |
| Positive |    2833 |     92.28% |
| Negative |     237 |      7.72% |

### Rating Distribution

| Rating | Number of Reviews |
| -----: | ----------------: |
|      1 |               146 |
|      2 |                91 |
|      3 |               140 |
|      4 |               447 |
|      5 |              2246 |

The distribution shows that **5-star reviews dominate the dataset**.

The notebook also visualizes the feedback distribution using Seaborn.

---

## 6. Train-Test Split

The cleaned review text is used as the input feature:

```python
X = data["verified_reviews"]
```

The feedback column is used as the target:

```python
y = data["feedback"]
```

The dataset is divided into:

* **80% training data**
* **20% testing data**

A stratified split is used so that the class proportions remain approximately consistent between the training and testing sets.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

The resulting split contains:

```text
Training samples: 2456
Testing samples: 614
```

---

## 7. TF-IDF Feature Extraction

Machine learning algorithms cannot directly work with raw text.

Therefore, the review text is converted into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The project uses:

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)
```

### Parameters

| Parameter      |    Value | Purpose                                    |
| -------------- | -------: | ------------------------------------------ |
| `max_features` |     5000 | Limits the vocabulary to 5000 features     |
| `ngram_range`  | `(1, 2)` | Uses unigrams and bigrams                  |
| `sublinear_tf` |   `True` | Applies logarithmic term-frequency scaling |

Therefore, the model can learn from both individual words and two-word combinations.

For example:

```text
"good"
"product"
"good product"
```

can all become TF-IDF features.

The resulting feature matrices contain:

```text
Training: (2456, 5000)
Testing:  (614, 5000)
```

---

# 8. Machine Learning Models

Three classification approaches are explored in the notebook.

## 8.1 Logistic Regression

The first model is Logistic Regression.

```python
LogisticRegression(
    class_weight="balanced",
    random_state=42,
    max_iter=1000
)
```

`class_weight="balanced"` is used because the dataset contains substantially more positive reviews than negative reviews.

### Test Performance

Accuracy:

```text
92.83%
```

Classification results:

| Class         | Precision |   Recall |       F1 |
| ------------- | --------: | -------: | -------: |
| Negative      |      0.52 |     0.77 |     0.62 |
| Positive      |      0.98 |     0.94 |     0.96 |
| **Macro Avg** |  **0.75** | **0.85** | **0.79** |

Confusion Matrix:

```text
[[ 36  11]
 [ 33 534]]
```

---

## 8.2 Multinomial Naive Bayes

The second model is Multinomial Naive Bayes.

```python
MultinomialNB()
```

### Test Performance

Accuracy:

```text
92.35%
```

However, the model predicted **no negative reviews** in the test set.

Classification results:

| Class         | Precision |   Recall |       F1 |
| ------------- | --------: | -------: | -------: |
| Negative      |      0.00 |     0.00 |     0.00 |
| Positive      |      0.92 |     1.00 |     0.96 |
| **Macro Avg** |  **0.46** | **0.50** | **0.48** |

Confusion Matrix:

```text
[[  0  47]
 [  0 567]]
```

This illustrates why accuracy alone can be misleading when working with an imbalanced dataset.

---

## 8.3 Linear Support Vector Machine

A Linear SVM is also used for text classification.

The model is implemented using:

```python
LinearSVC(
    C=1,
    class_weight="balanced",
    random_state=42
)
```

The SVM uses `class_weight="balanced"` to account for the class imbalance.

---

# 9. Cross-Validation

To obtain a more reliable estimate of model performance, **5-fold Stratified Cross-Validation** is used.

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The following metrics are evaluated:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1

Macro averaging gives equal importance to both positive and negative classes, which is particularly useful for this imbalanced dataset.

### Logistic Regression Cross-Validation

| Metric          |  Score |
| --------------- | -----: |
| Accuracy        | 0.9365 |
| Macro Precision | 0.7743 |
| Macro Recall    | 0.8257 |
| Macro F1        | 0.7963 |

The notebook also performs the same cross-validation procedure for the Linear SVM.

---

# 10. Hyperparameter Tuning

The Linear SVM is further optimized using **GridSearchCV**.

The hyperparameter tuned is:

```text
C
```

The following values are tested:

```python
C = [0.01, 0.1, 1, 10, 100]
```

The search uses:

```python
scoring="f1_macro"
```

This is important because the dataset is imbalanced and Macro F1 evaluates the performance of both classes equally.

### Best Parameter

The grid search selected:

```text
C = 1
```

Best cross-validation Macro F1:

```text
0.8167
```

---

# 11. Tuned Linear SVM Performance

The tuned Linear SVM is evaluated on the held-out test set.

### Test Accuracy

```text
96.09%
```

### Classification Report

| Class            | Precision |   Recall |       F1 |
| ---------------- | --------: | -------: | -------: |
| Negative         |      0.77 |     0.70 |     0.73 |
| Positive         |      0.98 |     0.98 |     0.98 |
| **Macro Avg**    |  **0.87** | **0.84** | **0.86** |
| **Weighted Avg** |  **0.96** | **0.96** | **0.96** |

### Confusion Matrix

```text
[[ 33  14]
 [ 10 557]]
```

This means:

* 33 negative reviews were correctly classified.
* 14 negative reviews were classified as positive.
* 10 positive reviews were classified as negative.
* 557 positive reviews were correctly classified.

---

# 12. Final Model

The final model combines TF-IDF and Linear SVM into a single Scikit-learn Pipeline.

```python
final_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("svm", LinearSVC(
        C=1,
        class_weight="balanced",
        random_state=42
    ))
])
```

The advantage of using a Pipeline is that the text transformation and prediction model are kept together.

The final model therefore performs:

```text
Raw Review
     ↓
TF-IDF
     ↓
Linear SVM
     ↓
Positive / Negative
```

### Final Test Result

```text
Test Accuracy: 96.09%
Macro F1:      0.86
```

---

# 13. Model Saving

The trained pipeline is saved using Joblib:

```python
joblib.dump(final_model, "sentiment_model.pkl")
```

This allows the trained model to be reused later without retraining it.

The saved model contains both:

* TF-IDF vectorization
* Linear SVM classifier

Therefore, new raw review text can be directly passed to the saved pipeline.

---

# 14. Testing on New Reviews

The saved model is loaded using:

```python
loaded_model = joblib.load("sentiment_model.pkl")
```

Example reviews are then passed to the model.

Example:

```text
"I absolutely love this product, it works perfectly!"
→ Positive

"This product is terrible and stopped working after one day."
→ Negative

"The product is okay, nothing special."
→ Positive
```

The project also tests several additional reviews, including:

```text
"This is a very bad product"
"This product is terrible"
"I hate this product"
"Worst product ever"
"This product is amazing"
"I absolutely love this product"
```

The model's `decision_function()` is also used to inspect the prediction score.

A positive score corresponds to the Positive class prediction, while a negative score corresponds to the Negative class prediction.

---

# 15. Project Workflow

```text
                    Amazon Alexa Reviews
                            │
                            ▼
                    Data Loading
                            │
                            ▼
                    Data Inspection
                            │
                            ▼
                     Data Cleaning
                            │
                            ▼
                         EDA
                            │
                            ▼
                    Train/Test Split
                            │
                            ▼
                         TF-IDF
                            │
                            ▼
                ┌───────────┼───────────┐
                │           │           │
                ▼           ▼           ▼
           Logistic      Naive Bayes   Linear SVM
          Regression
                │           │           │
                └───────────┼───────────┘
                            ▼
                    Model Evaluation
                            │
                            ▼
                  Stratified 5-Fold CV
                            │
                            ▼
                    Hyperparameter
                        Tuning
                            │
                            ▼
                    Final Linear SVM
                            │
                            ▼
                   Save Model (.pkl)
                            │
                            ▼
                  Predict New Reviews
```

---

# 16. Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib

### Machine Learning

* TF-IDF
* Logistic Regression
* Multinomial Naive Bayes
* Linear Support Vector Machine
* Stratified K-Fold Cross-Validation
* GridSearchCV

### NLP

* Text feature extraction
* Unigrams
* Bigrams
* TF-IDF representation

---

# 17. Project Structure

A typical project structure is:

```text
Amazon-Review-Sentiment-Analysis/
│
├── amazon_alexa.tsv
├── sentiment_analysis.ipynb
├── sentiment_model.pkl
├── app.py
├── requirements.txt
└── README.md
```

### File Description

| File                       | Description                            |
| -------------------------- | -------------------------------------- |
| `amazon_alexa.tsv`         | Amazon Alexa review dataset            |
| `sentiment_analysis.ipynb` | Complete data analysis and ML workflow |
| `sentiment_model.pkl`      | Trained TF-IDF + Linear SVM pipeline   |
| `app.py`                   | Streamlit application for predictions  |
| `requirements.txt`         | Python dependencies                    |
| `README.md`                | Project documentation                  |

---

# 18. Streamlit Application

The trained model can be integrated into a Streamlit web application to allow users to enter a review and receive a sentiment prediction.

The application uses the saved:

```text
sentiment_model.pkl
```

instead of retraining the model every time.

The basic application workflow is:

```text
User enters review
        ↓
Streamlit Application
        ↓
Saved ML Pipeline
        ↓
TF-IDF Transformation
        ↓
Linear SVM
        ↓
Sentiment Prediction
```

---

# 19. How to Run the Project

## Clone the Repository

```bash
git clone <your-repository-url>
cd Amazon-Review-Sentiment-Analysis
```

## Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run the Notebook

Open:

```text
sentiment_analysis.ipynb
```

using Jupyter Notebook or VS Code.

Run the notebook to:

1. Load the dataset
2. Clean the data
3. Perform EDA
4. Train the models
5. Perform cross-validation
6. Tune the Linear SVM
7. Train the final pipeline
8. Save `sentiment_model.pkl`

## Run the Streamlit Application

```bash
streamlit run app.py
```

---

# 20. Key Results

| Model                   | Test Accuracy | Macro F1 |
| ----------------------- | ------------: | -------: |
| Logistic Regression     |        92.83% |     0.79 |
| Multinomial Naive Bayes |        92.35% |     0.48 |
| Tuned Linear SVM        |    **96.09%** | **0.86** |

The results show that the Linear SVM pipeline achieved the strongest test performance among the evaluated models in this project, particularly when considering the Macro F1 score for the imbalanced dataset.

---

# 21. Key Learnings

This project demonstrates several important concepts in machine learning and NLP:

* How to inspect and clean real-world text data
* How class imbalance affects classification performance
* Why accuracy alone may not be sufficient for imbalanced datasets
* How TF-IDF converts text into numerical features
* How unigrams and bigrams can capture useful text patterns
* How different classification algorithms perform on text data
* How Stratified K-Fold Cross-Validation can provide more robust evaluation
* How GridSearchCV can be used for hyperparameter tuning
* Why Macro F1 is useful for imbalanced classification
* How to build an end-to-end Scikit-learn Pipeline
* How to save and reuse a trained machine learning model
* How model decision scores can be inspected using `decision_function()`

---

# 22. Limitations

There are several limitations to consider:

1. The dataset is highly imbalanced toward positive reviews.
2. The target label is derived from product ratings rather than independently annotated sentiment.
3. Duplicate reviews are investigated but are not removed in the current notebook.
4. Some reviews can be difficult to classify correctly because sentiment may depend on context or sarcasm.
5. The model is trained specifically on Amazon Alexa reviews and may not generalize equally well to reviews from completely different products or domains.

---

# 23. Future Improvements

Possible improvements include:

* More advanced text preprocessing
* Stop-word analysis
* Lemmatization
* Experimenting with word and character n-grams
* Additional machine learning models
* More extensive hyperparameter tuning
* Threshold analysis for SVM decision scores
* Handling duplicate reviews more explicitly
* Testing on an independent external dataset
* Trying transformer-based NLP models such as BERT
* Improving the Streamlit interface
* Deploying the application online

---

## Conclusion

This project implements an end-to-end **Amazon Alexa review sentiment classification system** using classical NLP and machine learning techniques.

TF-IDF is used to transform review text into numerical features, while several classification models are evaluated. Cross-validation and hyperparameter tuning are used to improve the evaluation process.

The final **TF-IDF + Linear SVM** pipeline achieves:

```text
96.09% Test Accuracy
0.86 Macro F1
```

The trained pipeline is saved with Joblib and can be reused to classify new Amazon Alexa reviews through a Streamlit application.
