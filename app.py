import streamlit as st
import joblib

# Load the saved model and vectorizer
svm = joblib.load('svm_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Predict function
def predict(text):
    # Vectorize the input text
    tfidf_X = tfidf_vectorizer.transform([text])

    # Make prediction
    prediction = svm.predict(tfidf_X)

    return prediction[0]

def main():
    st.title("Political Misinformation Detection")
    st.write("This app predicts whether a news headline is REAL or FAKE")

    # Text input for user
    user_input = st.text_area("Enter a news headline:")

    if st.button("Predict"):
        if user_input:
            prediction = predict(user_input)
            result = "FAKE" if prediction == 1 else "REAL"
            if result == "FAKE":
                st.markdown(f'<p style="color:red;">The news is {result}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p style="color:green;">The news is {result}</p>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a news headline")

if __name__ == "__main__":
    main()
