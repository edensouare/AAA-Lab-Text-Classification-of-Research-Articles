# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
nltk.download('punkt_tab') # Add this line to download the missing resource
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB

# Import Dataset

df = pd.read_csv('/content/Systematic Review about AI - Duplicate check.csv')
display(df.head())

df.shape

df.info()

df['Keywords(SSCI)'].value_counts()

# Preprocessing
# Subtask: Cleaning and preprocessing the text data. Removing special characters, numbers, stop words, and performing lemmatization.


# Combining 'Title' and 'Abstract' columns for text analysis
df['text'] = df['Title'] + ' ' + df['Abstract'].fillna('')

# Converting text to lowercase
df['text'] = df['text'].str.lower()

# Removing special characters and numbers
df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Tokenize the text
df['text'] = df['text'].apply(lambda x: word_tokenize(x))

# Remove stop words
stop_words = set(stopwords.words('english'))
df['text'] = df['text'].apply(lambda x: [word for word in x if word not in stop_words])

# Lemmatization (kind of optional, but can improve results)
lemmatizer = WordNetLemmatizer()
df['text'] = df['text'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

# Join the tokens back into a string for feature extraction
df['text'] = df['text'].apply(lambda x: ' '.join(x))

display(df[['Title', 'Abstract', 'text']].head())

# Feature Extraction
## Subtask: Convert the text data into numerical features that can be used by the machine learning model. Using TfidfVectorizer

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000) # max_features are adjustable but i chose 5000

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

# Display the shape of the TF-IDF matrix
print("Shape of TF-IDF matrix:", tfidf_matrix.shape)

# Split the dataset
# Subtask: Split the dataset into training and testing sets.

#Using these target variables/key words
search_terms_ai = ["artificial intelligence", "ai", "chatbot", "llm", "large language model", "conversational agent"]
search_terms_addiction = ["addiction", "overuse", "overreliance", "excessive use", "dependence", "problematic use"]

def contains_keywords(text, keywords_list):
    if isinstance(text, str):
        return any(word in text.lower() for word in keywords_list)
    return False

df['useful'] = df.apply(lambda row: contains_keywords(row['Title'], search_terms_ai) and contains_keywords(row['Abstract'], search_terms_addiction), axis=1)

X = tfidf_matrix
y = df['useful']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Build and Train the Model
# Subtask: Choosing a suitable machine learning model for text classification (like Logistic Regression, Naive Bayes, or SVM) and train it on the training data

from sklearn.naive_bayes import MultinomialNB

# Initializing and training the Multinomial Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the Model
# Subtask: Evaluating the performance of the trained model on the testing data using metrics (accuracy, precision, recall, F1-score).

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(confusion)

# Implement Filtering
# Subtask: Create a function to filter articles based on keywords using the trained model

# Use the trained model to predict 'useful' articles on the entire dataset
df['predicted_useful'] = model.predict(tfidf_matrix)

# Filter the DataFrame to keep only the articles predicted as useful
useful_articles = df[df['predicted_useful'] == True]

# Display the number of useful articles found and the first few rows
print(f"Number of useful articles found: {len(useful_articles)}")
display(useful_articles[['Title', 'Abstract', 'predicted_useful']].head())

# MAIN Task: Now I'm going to refine the text classification model
# Explore different models

# Subtask: Trying to train and evaluate other text classification models to improve accuracy, such as Logistic Regression, Support Vector Machines (SVM), and RandomForestClassifier (used in https://www.analyticsvidhya.com/blog/2021/12/text-classification-of-news-articles/)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

# --- Logistic Regression ---
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# --- Support Vector Classifier (using LinearSVC for efficiency) ---
svc_model = LinearSVC(random_state=42)
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)

# --- Random Forest Classifier ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Logistic Regression predictions:")
print(y_pred_lr)
print("\nSVC predictions:")
print(y_pred_svc)
print("\nRandom Forest predictions:")
print(y_pred_rf)

#Now that the models have been trained and predictions made, I will evaluate the performance of each model using accuracy, classification report, and confusion matrix, and print the results.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluate Logistic Regression
print("--- Logistic Regression Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr)}")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lr))

# Evaluate SVC (LinearSVC)
print("\n--- SVC (LinearSVC) Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svc)}")
print("Classification Report:")
print(classification_report(y_test, y_pred_svc))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svc))

# Evaluate Random Forest
print("\n--- Random Forest Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# Hyperparameter tuning

# Subtask: For the most promising models (Random Forest and SVC), perform hyperparameter tuning to find the optimal settings that maximize performance.
# So based on the previous evaluation, the Random Forest and LinearSVC models performed the best. I will now perform hyperparameter tuning for both models using `GridSearchCV` to find the optimal parameters. I will define a parameter grid for each model and use 'f1_weighted' as the scoring metric, as it's a good choice for imbalanced datasets.

from sklearn.model_selection import GridSearchCV

# --- Hyperparameter Tuning for LinearSVC ---
param_grid_svc = {
    'C': [0.1, 1, 10],
    'loss': ['hinge', 'squared_hinge']
}

grid_search_svc = GridSearchCV(LinearSVC(random_state=42), param_grid_svc, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search_svc.fit(X_train, y_train)

print("Best parameters for LinearSVC:", grid_search_svc.best_params_)
print("Best f1_weighted score for LinearSVC:", grid_search_svc.best_score_)


# --- Hyperparameter Tuning for RandomForestClassifier ---
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search_rf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search_rf.fit(X_train, y_train)

print("\nBest parameters for RandomForestClassifier:", grid_search_rf.best_params_)
print("Best f1_weighted score for RandomForestClassifier:", grid_search_rf.best_score_)

# Select the best model for text classification
#Compare the f1_weighted scores from the hyperparameter tuning results to determine the best model.

# Compare the best f1_weighted scores
best_score_svc = grid_search_svc.best_score_
best_score_rf = grid_search_rf.best_score_

print(f"Best f1_weighted score for LinearSVC: {best_score_svc}")
print(f"Best f1_weighted score for RandomForestClassifier: {best_score_rf}")

if best_score_rf > best_score_svc:
    best_model_type = "RandomForestClassifier"
    best_score = best_score_rf
else:
    best_model_type = "LinearSVC"
    best_score = best_score_svc

print(f"\nBased on f1_weighted score, the best model is: {best_model_type} with a score of {best_score}")

# Retrain Random Forest on the training dataset
# Subtask: Retrain the selected best model (RandomForestClassifier) using the optimal hyperparameters on the entire dataset (training and testing combined) to utilize all available data for the final model.

# Best parameters for RandomForestClassifier from previous step: {'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}
final_model = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42)

# Train the model on the training dataset
final_model.fit(X_train, y_train)

print("Final model trained successfully on the training dataset.")

# Final evaluation
# Subtask: Evaluate the final model on the test set to ensure it generalizes well.

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test data using the final model
y_pred_final = final_model.predict(X_test)

# Evaluate the final model
accuracy_final = accuracy_score(y_test, y_pred_final)
report_final = classification_report(y_test, y_pred_final)
confusion_final = confusion_matrix(y_test, y_pred_final)

print(f"Final Model Accuracy on Test Set: {accuracy_final}")
print("\nFinal Model Classification Report on Test Set:")
print(report_final)
print("\nFinal Model Confusion Matrix on Test Set:")
print(confusion_final)

# Implement filtering with Random Forest model
# Subtask: Update the filtering step to use the best-performing model (`final_model`) to identify relevant articles from the entire dataset.

# Use the trained final model to predict useful articles on the entire dataset and filter the DataFrame to display the results.

# Use the trained final_model to predict 'useful' articles on the entire dataset
df['predicted_useful_final'] = final_model.predict(tfidf_matrix)

# Filter the DataFrame to keep only the articles predicted as useful by the final model
useful_articles_final = df[df['predicted_useful_final'] == True]

# Display the number of useful articles found and the first few rows
print(f"Number of useful articles found by the final model: {len(useful_articles_final)}")
display(useful_articles_final[['Title', 'Abstract', 'predicted_useful_final']].head())

# Manual Review of Predicted Articles
# Subtask: Display a sample of articles predicted as 'useful' and 'not useful' for manual review.

# Display a sample of articles predicted as useful
print("Sample of articles predicted as USEFUL:")
display(useful_articles_final[['Title', 'Abstract', 'predicted_useful_final']].sample(5)) # Displaying 5 random samples


# Filter for articles predicted as not useful
not_useful_articles_final = df[df['predicted_useful_final'] == False]

# Display a sample of articles predicted as not useful
print("\nSample of articles predicted as NOT USEFUL:")
display(not_useful_articles_final[['Title', 'Abstract', 'predicted_useful_final']].sample(5)) # Displaying 5 random samples

# Extract Useful Articles
# Subtask:Extract the articles predicted as 'useful' into a separate DataFrame and save it for further analysis.

# The useful articles are already in the 'useful_articles_final' DataFrame
# Display the first few rows of the useful articles DataFrame
print("First 5 rows of the useful articles DataFrame:")
display(useful_articles_final.head())

# Save the useful articles DataFrame to a new CSV file
output_filename = "useful_articles_for_analysis.csv"
useful_articles_final.to_csv(output_filename, index=False)

print(f"\nUseful articles saved to '{output_filename}' for further analysis.")

# Summary:
# Data Analysis Key Findings

"""* Initially, Logistic Regression, SVC (LinearSVC), and Random Forest models were trained and evaluated. Random Forest showed the highest initial accuracy at 0.801, compared to SVC at 0.796 and Logistic Regression at 0.786.
* Hyperparameter tuning was performed on LinearSVC and RandomForestClassifier using a 5-fold cross-validation and 'f1\_weighted' scoring.
* The best parameters for LinearSVC were found to be `{'C': 1, 'loss': 'squared_hinge'}` with a best 'f1\_weighted' score of approximately 0.7619.
* The best parameters for RandomForestClassifier were `{'max_depth': None, 'min_samples_split': 5, 'n_estimators': 200}` with a best 'f1\_weighted' score of approximately 0.7677.
* Based on the 'f1\_weighted' scores from hyperparameter tuning, RandomForestClassifier was selected as the best model.
* The final RandomForestClassifier model, trained with optimal hyperparameters on the **training dataset**, achieved an accuracy of **0.818** on the **test set**.
* The final model identified **1386 articles as useful** from the entire dataset.

### Insights or Next Steps

* The initial observation of perfect accuracy on the test set was due to training the model on the full dataset and evaluating on a subset of it. The corrected evaluation on a held-out test set provides a more realistic measure of the model's generalization capability.
* The final model shows improved performance metrics for identifying useful articles compared to the initial Naive Bayes model.
* With the relevant articles identified, the next step should involve further analysis of the content of these 1386 articles to extract specific information or themes related to AI and addiction, fulfilling the original research goal.

There are still about 1,386 articles, now going to exclude the articles that have the word "dependence" as a keyword from the 1,386 and see how many are left
"""

# Define the addiction keywords
search_terms_addiction = ["addiction", "overuse", "overreliance", "excessive use", "dependence", "problematic use"]

# Filter out articles where "dependence" is the ONLY keyword from search_terms_addiction in the Abstract
def contains_only_dependence(abstract, keywords_list):
    if not isinstance(abstract, str):
        return False
    abstract_lower = abstract.lower()
    found_keywords = [keyword for keyword in keywords_list if keyword in abstract_lower]
    return 'dependence' in found_keywords and len(found_keywords) == 1

filtered_useful_articles = useful_articles_final[
    ~useful_articles_final['Abstract'].fillna('').apply(lambda x: contains_only_dependence(x, search_terms_addiction))
]

# Display the number of articles remaining after filtering
print(f"Number of useful articles remaining after refined filtering: {len(filtered_useful_articles)}")

# Display a sample of the filtered useful articles
print("\nSample of useful articles after refined filtering:")
display(filtered_useful_articles[['Title', 'Abstract', 'Keywords', 'predicted_useful_final']].sample(5))

# Save the filtered useful articles DataFrame to a new CSV file
output_filename_filtered = "filtered_useful_articles.csv"
filtered_useful_articles.to_csv(output_filename_filtered, index=False)

print(f"Filtered useful articles saved to '{output_filename_filtered}' for further analysis.")

# Define the addiction keywords again to use the same logic for filtering out
search_terms_addiction = ["addiction", "overuse", "overreliance", "excessive use", "dependence", "problematic use"]

# Function to identify articles where "dependence" is the ONLY keyword from search_terms_addiction in the Abstract
def contains_only_dependence(abstract, keywords_list):
    if not isinstance(abstract, str):
        return False
    abstract_lower = abstract.lower()
    found_keywords = [keyword for keyword in keywords_list if keyword in abstract_lower]
    return 'dependence' in found_keywords and len(found_keywords) == 1

# Filter the useful_articles_final DataFrame to get the articles that were filtered OUT
dependence_only_articles = useful_articles_final[
    useful_articles_final['Abstract'].fillna('').apply(lambda x: contains_only_dependence(x, search_terms_addiction))
]

# Display the number of articles in this new file
print(f"Number of articles where 'dependence' was the only addiction keyword in Abstract: {len(dependence_only_articles)}")

# Display a sample of these articles
print("\nSample of articles with 'dependence' as the only addiction keyword in Abstract:")
display(dependence_only_articles[['Title', 'Abstract', 'Keywords', 'predicted_useful_final']].sample(5))

# Save these articles to a new CSV file
output_filename_dependence_only = "dependence_only_articles.csv"
dependence_only_articles.to_csv(output_filename_dependence_only, index=False)

print(f"\nArticles with 'dependence' as the only addiction keyword in Abstract saved to '{output_filename_dependence_only}'.")
