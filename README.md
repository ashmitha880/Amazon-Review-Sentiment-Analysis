# Amazon Review Sentiment Analysis

A machine learning project that classifies Amazon Alexa product reviews as positive or negative using TF-IDF text features and a Linear Support Vector Machine (Linear SVM).

The project includes data cleaning, exploratory data analysis, text feature extraction, model comparison, hyperparameter tuning, model evaluation, and a Streamlit web application for making predictions on new reviews.

---

## 1. Project Overview

The goal of this project is to build a text classification model that predicts the sentiment of an Amazon Alexa product review.

The dataset contains customer reviews along with ratings and a feedback label.

The workflow followed in this project is:

**Data → Cleaning → EDA → Train/Test Split → TF-IDF → Model Training → Evaluation → Hyperparameter Tuning → Final Model → Streamlit App**

---

## 2. Dataset

The dataset used in this project is `amazon_alexa.tsv`.

It contains the following columns:

| Column | Description |
|---|---|
| `rating` | Product rating from 1 to 5 |
| `date` | Date of the review |
| `variation` | Alexa product variation |
| `verified_reviews` | Text of the customer review |
| `feedback` | Binary feedback label |

The original dataset contains 3150 reviews.

### Important Note About the Target Label

The `feedback` label is derived from the product rating:

- Ratings 1–2 → Negative (`0`)
- Ratings 3–5 → Positive (`1`)

Therefore, the target represents rating-derived feedback rather than independently annotated human sentiment.

---

## 3. Data Preprocessing

The following preprocessing steps were performed:

- Checked the structure and data types of the dataset
- Checked for missing values
- Removed reviews with missing text
- Removed empty/whitespace-only reviews
- Checked for duplicate review texts
- Removed duplicate reviews
- Verified that duplicate reviews did not have conflicting labels

After preprocessing, the dataset contained **2299 unique reviews**.

### Class Distribution

- Positive: 2094
- Negative: 205

The dataset is therefore imbalanced, with substantially more positive reviews than negative reviews.

---

## 4. Exploratory Data Analysis

Exploratory analysis was performed to understand the dataset before model training.

The analysis included:

- Feedback distribution
- Rating distribution
- Relationship between rating and feedback
- Review characteristics
- Class distribution after cleaning

The analysis showed that the feedback labels are strongly related to the rating values, as expected from the label construction.

---

## 5. Machine Learning Approach

### Train-Test Split

The cleaned dataset was divided into training and testing sets using an 80:20 split.

Stratified splitting was used to preserve the class distribution in both sets.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)