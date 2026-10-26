# CampusPulse AI — Comprehensive Viva Voce Questions & Answers

This guide is designed to help students and beginners confidently ace project viva, code walkthroughs, and academic presentations for the **CampusPulse AI: Smart Student Feedback Analyzer** project.

---

## 1. Project Concept & Objectives

### Q1: What is CampusPulse AI, and what problem does it solve?
**Answer:**  
CampusPulse AI is an end-to-end Natural Language Processing (NLP) system designed for higher education institutions. Each semester, colleges collect hundreds or thousands of open-ended student evaluations across departments. Manually reading, tagging, and synthesizing this text is slow, subjective, and labor-intensive.  
CampusPulse AI automates this process by:
1. Detecting emotional valence (**Sentiment Classification**: Positive, Neutral, Negative).
2. Identifying the campus topic (**Category Classification**: Teaching, Labwork, Examination, Library Facilities, Extracurricular, etc.).
3. Providing an executive analytics dashboard with real-time prediction and dataset exploration for college administrators.

---

### Q2: Why is feedback analysis framed as two separate classification tasks instead of one?
**Answer:**  
Sentiment and category represent two orthogonal dimensions of student feedback:
- **Sentiment** captures *how* the student feels (polarity: positive, neutral, negative).
- **Category** captures *what* area the student is discussing (topic/domain).  
Training separate classifiers allows each model to optimize its decision boundaries for that specific target without combinatorial label explosion.

---

## 2. Text Preprocessing & Feature Engineering

### Q3: What preprocessing steps did you apply to the raw student feedback?
**Answer:**  
Our preprocessing function (`clean_text`) applies:
1. **Lowercasing**: Unifies case differences so that words like `"Good"`, `"GOOD"`, and `"good"` are recognized as the exact same token.
2. **URL & Email Filtering**: Strips out hyperlinks or email strings that add noise.
3. **Punctuation & Symbol Cleaning**: Cleans extraneous punctuation while maintaining word apostrophes.
4. **Preservation of Negation**: Unlike aggressive stopword removers that drop words like `"not"` or `"never"`, we preserve negations because `"not good"` has the opposite sentiment of `"good"`.
5. **Whitespace Normalization**: Removes excessive spaces and tabs to produce standard token streams.

---

### Q4: What is TF-IDF, and how does it represent text numerically?
**Answer:**  
TF-IDF stands for **Term Frequency-Inverse Document Frequency**:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
- **Term Frequency (TF)** measures how frequently a term appears in a specific document.
- **Inverse Document Frequency (IDF)** penalizes terms that appear across almost all documents (e.g., `"the"`, `"is"`, `"student"`), giving higher weights to terms that are distinct and informative.

In this project, we also use **sublinear TF scaling** ($1 + \log(\text{TF})$) to prevent high frequency terms from dominating, and set `min_df=2` to eliminate one-off typos.

---

### Q5: Why did you use both unigrams and bigrams (`ngram_range=(1, 2)`)?
**Answer:**  
Unigrams (single words like `"good"`) miss context. If a student writes `"not good"`, a unigram model only sees the word `"not"` and the word `"good"`, which can mislead the classifier into thinking it's positive.  
By including **bigrams** (two consecutive words), the model captures phrases like `"not good"`, `"very helpful"`, `"too rushed"`, and `"lab equipment"`, significantly improving sentiment and topic accuracy.

---

### Q6: What is Data Leakage, and how did your pipeline prevent it?
**Answer:**  
Data leakage occurs when information from outside the training dataset (such as test or validation data) is used to create or tune features. For instance, if you fit a TF-IDF vectorizer on the *entire dataset* before splitting, the IDF weights and vocabulary leak test set statistics into the training phase.  
We prevented data leakage by:
1. Splitting the dataset into train (80%) and test (20%) partitions first.
2. Fitting the `TfidfVectorizer` **only** on `X_train`.
3. Transforming `X_test` using the vocabulary learned strictly from `X_train`.
4. Wrapping this process in scikit-learn's `Pipeline` object.

---

## 3. Machine Learning Algorithms & Model Comparison

### Q7: How does Multinomial Naive Bayes (MNB) work for text classification?
**Answer:**  
Multinomial Naive Bayes is a probabilistic classifier based on Bayes' Theorem:
$$P(c \mid d) \propto P(c) \prod_{i=1}^{n} P(w_i \mid c)$$
It makes the "naive" conditional independence assumption that all words in a document are conditionally independent given the class.  
We apply **Laplace smoothing** (`alpha=0.5`) to prevent zero-probability issues when encountering words in the test set that did not appear in a particular class during training.

---

### Q8: How does Logistic Regression work for multi-class classification?
**Answer:**  
Logistic Regression uses a linear combination of input features mapped through a softmax function to output calibrated probabilities across multiple classes:
$$P(Y = c \mid x) = \frac{e^{w_c^T x}}{\sum_{j=1}^K e^{w_j^T x}}$$
It optimizes a cross-entropy loss function with L2 regularization to prevent overfitting on high-dimensional sparse TF-IDF vectors.

---

### Q9: What is Class Imbalance, and how did you handle it?
**Answer:**  
In our dataset, positive feedback significantly outnumbers neutral and negative feedback (reflecting real-world student surveys).  
We handled this by:
1. Using **Stratified Splitting** (`stratify=y`) so that the train and test sets have the identical class proportions.
2. Enabling **`class_weight='balanced'`** in Logistic Regression, which automatically adjusts weights inversely proportional to class frequencies:
   $$w_c = \frac{N}{K \times n_c}$$
3. Evaluating performance using **Macro-F1 score** rather than basic accuracy.

---

### Q10: Why is Accuracy misleading, and why is Macro-F1 our primary metric?
**Answer:**  
If a dataset is 80% positive, a naive model that predicts "positive" for every feedback will achieve 80% accuracy, while having 0% precision and recall on negative and neutral classes.  
- **Macro-F1** calculates the unweighted mean of the F1 scores across all classes:
  $$\text{Macro-F1} = \frac{F1_{\text{pos}} + F1_{\text{neu}} + F1_{\text{neg}}}{3}$$
It treats minority and majority classes equally, ensuring that poor performance on negative feedback directly lowers the overall score.

---

## 4. Evaluation & Results Interpretation

### Q11: What is a Confusion Matrix, and how do you read it?
**Answer:**  
A confusion matrix is a grid showing True Labels on the vertical axis and Predicted Labels on the horizontal axis:
- **Diagonal cells** indicate correct predictions (True Positives for each class).
- **Off-diagonal cells** indicate misclassifications (False Positives and False Negatives).  
In our dashboard and report, we plot this matrix using a heatmap, allowing examiners and administrators to immediately see which categories get confused (e.g., neutral feedback being classified as positive).

---

### Q12: Which model performed better between Naive Bayes and Logistic Regression?
**Answer:**  
**Logistic Regression** achieved the higher Macro-F1 score. While Naive Bayes provides a fast baseline, its naive independence assumption and vulnerability to class imbalance cause it to over-predict the majority class. Logistic Regression with `class_weight='balanced'` effectively adjusted penalty boundaries, yielding superior recall on minority neutral and negative feedback.

---

## 5. System Architecture & Deployment

### Q13: Why did you use `joblib` instead of retraining inside the web app?
**Answer:**  
In a production-ready application, model training and model serving must be decoupled:
- Retraining upon every web request would cause high latency, memory spikes, and non-deterministic behavior.
- `joblib` allows fast binary serialization of trained scikit-learn pipelines (vectorizer + classifier).
- The Streamlit web application loads the serialized `.joblib` file once into cache (`@st.cache_resource`), allowing instant sub-millisecond inference for live student comments.

---

### Q14: What are the main limitations of this NLP system?
**Answer:**  
1. **Bag-of-Words Limitation**: TF-IDF models word occurrences and local bigrams, but lacks deep syntactic understanding and long-range semantic context.
2. **Sarcasm and Ambiguity**: Sarcastic comments (e.g., *"Great, another 3-hour theoretical lecture!"*) may be classified as positive due to positive keywords like "Great".
3. **Dataset Volume**: The dataset has 724 records, which is sufficient for linear baselines but would benefit from tens of thousands of records for broader domain generalization.

---

### Q15: What would be the next steps to enhance CampusPulse AI?
**Answer:**  
1. **Transformer Encoders**: Fine-tuning pre-trained transformer models (such as BERT or RoBERTa) for contextual embeddings.
2. **Aspect-Based Sentiment Analysis (ABSA)**: Decomposing multi-sentence reviews to extract separate sentiments for different aspects (e.g., *"Teacher was great [Positive], but lab equipment was broken [Negative]"*).
3. **Integration**: Connecting the system to university Learning Management Systems (Canvas, Moodle, Blackboard) via automated REST APIs.
