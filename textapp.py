import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
import re

# Load the emotion analysis model and vectorizer
emotion_model = pickle.load(open('text_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Define preprocessing functions
def clean_punctuations(text):
    """
    Remove punctuations from the given text.
    """
    return re.sub(r'[^\w\s]', '', text)

def cleaning_repeating_char(text):
    """
    Clean repeating characters from the given text.
    """
    return re.sub(r'(.)\1+', r'\1', text)

def cleaning_URLs(text):
    """
    Remove URLs from the given text.
    """
    return re.sub(r'http\S+', '', text)

def cleaning_numbers(text):
    """
    Remove numbers from the given text.
    """
    return re.sub(r'\d+', '', text)

def stemming_on_text(text):
    """
    Perform stemming on the given text.
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in text]

# Define Indian laws with sections for negative comments
negative_comment_laws = {
    "racist": "Indian Penal Code (IPC), Section 153A: Promoting enmity between different groups on grounds of religion, race, etc.",
    "rapist": "Indian Penal Code (IPC), Section 376: Punishment for rape.",
    "hate": "No specific law/regulation mentioned for comments containing the word 'hate'.",
    "abuse": "Indian Penal Code (IPC), Section 498A: Husband or relative of husband of a woman subjecting her to cruelty.",
    "violence": "No specific law/regulation mentioned for comments containing the word 'violence'.",
    "discrimination": "No specific law/regulation mentioned for comments containing the word 'discrimination'.",
    "bigotry": "No specific law/regulation mentioned for comments containing the word 'bigotry'.",
    "assault": "Indian Penal Code (IPC), Section 354: Assault or criminal force to woman with intent to outrage her modesty.",
    "obscene": "Indian Penal Code (IPC), Section 294: Punishment for obscene acts or words in public.",
    "bad": "No specific law/regulation mentioned for comments containing the word 'bad'.",
    "vulgar": "No specific law/regulation mentioned for comments containing the word 'vulgar'.",
    "offensive": "No specific law/regulation mentioned for comments containing the word 'offensive'.",
    "harassment": "No specific law/regulation mentioned for comments containing the word 'harassment'.",
    "threat": "No specific law/regulation mentioned for comments containing the word 'threat'.",
    "bullying": "No specific law/regulation mentioned for comments containing the word 'bullying'.",
    "cyberbullying": "No specific law/regulation mentioned for comments containing the word 'cyberbullying'.",
    "women abuse": "Various laws including Indian Penal Code (IPC), Protection of Women from Domestic Violence Act, 2005.",
    "gender violence": "Various laws including Indian Penal Code (IPC), Protection of Women from Domestic Violence Act, 2005."
}

# Sensitive words associated with negative sentiment
negative_sensitive_words = ["racist", "rapist", "hate", "abuse", "violence", "discrimination", "bigotry", 
                            "assault", "obscene", "bad", "vulgar", "offensive", "harassment", "threat", 
                            "bullying", "cyberbullying", "women abuse", "gender violence"]

# Streamlit app
st.title("Text Emotion Analysis App")

# User input
user_input = st.text_area("Enter text:")

if user_input:
    # Preprocess the input
    user_input = user_input.lower()
    user_input = clean_punctuations(user_input)
    user_input = cleaning_repeating_char(user_input)
    user_input = cleaning_URLs(user_input)
    user_input = cleaning_numbers(user_input)
    user_input = ' '.join(stemming_on_text(sent_tokenize(user_input)))

    # Vectorize the input
    user_input_vectorized = vectorizer.transform([user_input])

    # Make prediction
    predicted_emotion = emotion_model.predict(user_input_vectorized)

    # Adjust prediction based on sensitive words
    for word in negative_sensitive_words:
        if word in user_input:
            predicted_emotion = [0]  # Manually set as negative sentiment
            break

    # Debugging: Print the predicted emotion
    print("Predicted Emotion:", predicted_emotion)

    # Display the result
    st.markdown("---")
    st.header("Analysis Result")
    if predicted_emotion[0] == 1:  # Assuming 1 corresponds to positive class
        st.success("Predicted Emotion: Positive comment")
        st.write("Relevant laws/regulations for positive comments:")
        st.write("No specific law/regulation mentioned for positive comments.")
    else:
        st.warning("Predicted Emotion: Negative comment")
        st.write("Relevant laws/regulations for negative comments:")
        relevant_laws = []
        for word in negative_sensitive_words:
            if word in user_input:
                relevant_laws.append(negative_comment_laws[word])
        if relevant_laws:
            for law in relevant_laws:
                st.write(law)
        else:
            st.write("No specific law/regulation mentioned.")
