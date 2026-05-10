import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# ──────────────────────────────────────────────
# 1. Generate a realistic dataset
# ──────────────────────────────────────────────

spam_emails = [
    "Congratulations! You have won a $1,000 Walmart gift card. Click here to claim now!",
    "URGENT: Your account has been compromised. Verify your details immediately.",
    "Free iPhone 15 giveaway! Limited time offer. Click the link below to enter.",
    "You have been selected for a cash prize of $5000. Reply with your bank details.",
    "Make money fast! Work from home and earn $500 daily. No experience needed.",
    "Buy cheap meds online. No prescription required. 90% discount today only!",
    "Your PayPal account is suspended. Login immediately to restore access.",
    "Hot singles in your area are waiting to meet you. Join for free today!",
    "Lose 30 pounds in 30 days with this miracle pill. Order now!",
    "You are a winner! Claim your free vacation package worth $2500 now.",
    "Exclusive deal: Rolex watches at 95% off. Limited stock available!",
    "Dear friend, I am a Nigerian prince and need your help transferring funds.",
    "Earn $1000 per day from home. No skills required. Start today!",
    "Your email has won the international lottery. Send us your details to claim.",
    "Click here for free adult content. No credit card required!",
    "Urgent notice: Your bank account will be closed unless you verify now.",
    "Get rich quick with this amazing investment opportunity. 500% returns!",
    "You have a pending package delivery. Confirm your address to receive it.",
    "Increase your credit score overnight. Guaranteed results!",
    "Free gift cards available. Take a quick survey and claim yours!",
    "WINNER! You have been randomly selected to receive $10,000. Claim now!",
    "Best prices on Viagra and Cialis. Order online, fast delivery!",
    "Your computer is infected with a virus. Call our toll-free number now!",
    "Double your Bitcoin in 24 hours. Guaranteed! Send 0.1 BTC now.",
    "Cheap insurance rates! Save 80% today. Click here for a free quote.",
    "You owe taxes. Pay immediately to avoid arrest. Call this number now.",
    "Free weight loss program! Lose 20 pounds without diet or exercise.",
    "Claim your $500 Amazon voucher. You have been selected as a lucky winner.",
    "Mega sale! All brand name products at 90% discount. Shop now!",
    "Your subscription is expiring. Renew now to avoid service interruption.",
    "Make thousands weekly stuffing envelopes from home. No experience needed.",
    "Congratulations! Your email was selected for our monthly prize draw.",
    "Online pharmacy: No prescription needed. Order medications at low prices.",
    "Your social security number has been suspended. Call immediately.",
    "Investment opportunity of a lifetime! Returns of 1000% in 6 months.",
    "You have unclaimed funds. Contact us immediately to release your money.",
    "Hot stock tips! This stock will rise 300% next week. Buy now!",
    "Free casino credits! Sign up now and get $500 bonus with no deposit.",
    "Urgent: IRS notice. You owe back taxes. Avoid legal action. Pay now.",
    "Exclusive member offer: Buy one get ten free. Today only!",
]

ham_emails = [
    "Hi, can we reschedule our meeting to 3pm tomorrow? Let me know if that works.",
    "Please find attached the report you requested for the Q3 analysis.",
    "Just wanted to remind you about the team lunch scheduled for Friday at noon.",
    "The project deadline has been moved to next Monday. Please update your tasks.",
    "Can you review the pull request I submitted earlier today? Thanks!",
    "I will be working from home tomorrow. Available on Slack if needed.",
    "Happy birthday! Hope you have a wonderful day with your family.",
    "Here are the meeting notes from today's standup. Action items are highlighted.",
    "Could you send me the presentation slides before the client call at 4pm?",
    "The server maintenance is scheduled for this Saturday from 2am to 6am.",
    "Your order has been shipped. Expected delivery is in 3-5 business days.",
    "Please review and sign the attached contract by end of business Friday.",
    "The quarterly budget report is ready for your review. Let me know your feedback.",
    "Can we set up a quick call this week to discuss the project requirements?",
    "Your flight booking is confirmed. Check-in opens 24 hours before departure.",
    "Thanks for your application. We would like to schedule an interview next week.",
    "The new software update is available. Please install it at your earliest convenience.",
    "Reminder: Your dentist appointment is tomorrow at 10:30am.",
    "I have reviewed your proposal and have a few questions. Can we talk tomorrow?",
    "The team has completed the sprint. Here is the summary of what was accomplished.",
    "Your library books are due in 3 days. Please return or renew them online.",
    "We have received your refund request and will process it within 5 business days.",
    "The annual performance review forms are due by the end of this month.",
    "Please join the video call using the link shared in the calendar invite.",
    "I wanted to follow up on the email I sent last week regarding the proposal.",
    "The conference registration is now open. Early bird pricing ends this Friday.",
    "Your subscription renewal is coming up. No action needed if you wish to continue.",
    "The new intern will be joining our team on Monday. Please make them feel welcome.",
    "Could you please update the project tracker with your current task status?",
    "The client has approved the design mockups. We can proceed to development.",
    "I am sending over the invoice for the services provided last month.",
    "Your password was successfully changed. If this was not you, contact support.",
    "The weekly newsletter is here! Check out this week's top articles and updates.",
    "Reminder: Submit your timesheet by 5pm today to ensure timely payroll processing.",
    "We are pleased to inform you that your loan application has been approved.",
    "The study group will meet in the library at 6pm on Wednesday.",
    "Your professor has posted new lecture notes on the student portal.",
    "Can you help me debug this function? I think there is an issue with the loop.",
    "The hackathon results are out. Our team came in second place. Great work everyone!",
    "Looking forward to seeing you at the graduation ceremony next week.",
]

# Create DataFrame
emails = spam_emails + ham_emails
labels = ['spam'] * len(spam_emails) + ['ham'] * len(ham_emails)

df = pd.DataFrame({'email': emails, 'label': labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print("=" * 60)
print("       SPAM EMAIL CLASSIFIER — Sajjal Naeem")
print("=" * 60)
print(f"\nDataset size: {len(df)} emails")
print(f"Spam emails : {len(df[df['label'] == 'spam'])}")
print(f"Ham emails  : {len(df[df['label'] == 'ham'])}")

# ──────────────────────────────────────────────
# 2. Preprocessing & Feature Extraction
# ──────────────────────────────────────────────

X = df['email']
y = df['label'].map({'spam': 1, 'ham': 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=3000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ──────────────────────────────────────────────
# 3. Train Models
# ──────────────────────────────────────────────

print("\n--- Training Models ---\n")

# Model 1: Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_preds = nb_model.predict(X_test_tfidf)
nb_accuracy = accuracy_score(y_test, nb_preds)

# Model 2: Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)
lr_preds = lr_model.predict(X_test_tfidf)
lr_accuracy = accuracy_score(y_test, lr_preds)

print(f"Naive Bayes Accuracy      : {nb_accuracy * 100:.2f}%")
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")

# Use the best model
best_model = lr_model if lr_accuracy >= nb_accuracy else nb_model
best_name = "Logistic Regression" if lr_accuracy >= nb_accuracy else "Naive Bayes"
best_preds = lr_preds if lr_accuracy >= nb_accuracy else nb_preds

print(f"\nBest Model: {best_name}")

# ──────────────────────────────────────────────
# 4. Evaluation
# ──────────────────────────────────────────────

print("\n--- Classification Report ---\n")
print(classification_report(y_test, best_preds, target_names=['Ham', 'Spam']))

print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, best_preds)
print(f"True Negatives  (Ham correctly identified)  : {cm[0][0]}")
print(f"False Positives (Ham misclassified as Spam) : {cm[0][1]}")
print(f"False Negatives (Spam misclassified as Ham) : {cm[1][0]}")
print(f"True Positives  (Spam correctly identified) : {cm[1][1]}")

# ──────────────────────────────────────────────
# 5. Save Model
# ──────────────────────────────────────────────

with open('spam_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("\n✓ Model saved as spam_model.pkl")

# ──────────────────────────────────────────────
# 6. Predict on new emails
# ──────────────────────────────────────────────

def predict_email(email_text):
    email_tfidf = vectorizer.transform([email_text])
    prediction = best_model.predict(email_tfidf)[0]
    probability = best_model.predict_proba(email_tfidf)[0]
    label = "SPAM 🚨" if prediction == 1 else "HAM ✅"
    confidence = max(probability) * 100
    return label, confidence

print("\n--- Testing on New Emails ---\n")
test_emails = [
    "Congratulations! You won a free iPhone. Click here to claim your prize now!",
    "Hey, are you coming to the study session tomorrow at 5pm in the library?",
    "Your account will be suspended unless you verify your information immediately.",
    "The project report is due on Friday. Let me know if you need any help.",
]

for email in test_emails:
    label, confidence = predict_email(email)
    print(f"Email : {email[:60]}...")
    print(f"Result: {label} (Confidence: {confidence:.1f}%)\n")
