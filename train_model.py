import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Load new dataset
df = pd.read_csv("data/cyberbullying_dataset_120.csv")

# Preprocessing
X = df['tweet_text']
y = df['label']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize
tfidf = TfidfVectorizer()
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/cyberbully_model.pkl", "wb"))
pickle.dump(tfidf, open("model/vectorizer.pkl", "wb"))


from sklearn.metrics import classification_report, accuracy_score

# Predict on test data
y_pred = model.predict(X_test_vec)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.2f}")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Cyberbullying", "Cyberbullying"]))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ["Not Cyberbullying", "Cyberbullying"]

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.show()

from sklearn.metrics import precision_recall_fscore_support

precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, labels=model.classes_)

metrics_df = pd.DataFrame({
    'Precision': precision,
    'Recall': recall,
    'F1-Score': fscore,
    'Support': support
}, index=["Not Cyberbullying", "Cyberbullying"])

plt.figure(figsize=(8, 4))
sns.heatmap(metrics_df.iloc[:, :3], annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Precision, Recall, F1-Score Heatmap")
plt.tight_layout()
plt.savefig("model/classification_metrics_heatmap.png")
plt.show()

from sklearn.metrics import roc_curve, auc

y_prob = model.predict_proba(X_test_vec)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.savefig("model/roc_curve.png")
plt.show()

