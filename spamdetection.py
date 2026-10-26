import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data = pd.read_csv("spam.csv", sep="\t", names=["Category", "Message"])
messages = data["Message"]
labels = data["Category"]

mess_train, mess_test, lab_train, lab_test = train_test_split(
    messages,
    labels,
    test_size=0.20,
    random_state=42
)

cv = CountVectorizer(stop_words="english")
features = cv.fit_transform(mess_train)

model = MultinomialNB()
model.fit(features, lab_train)

features_test = cv.transform(mess_test)
predictions = model.predict(features_test)
accuracy = accuracy_score(lab_test, predictions)


def predict(text):
    input_text = cv.transform([text])
    result = model.predict(input_text)
    return result[0]


st.title("📧 Email Spam Detection")
st.write("Enter an email or message to check whether it is Spam or Not Spam.")
st.info(f"Model Accuracy: {accuracy * 100:.2f}%")

user_message = st.text_area(
    "Enter your email/message:",
    height=150
)

if st.button("Check Message"):
    if user_message.strip() == "":
        st.warning("Please enter a message.")
    else:
        result = predict(user_message)
        if result == "spam":
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM")