import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    # Tokenize the text and convert to lowercase
    tokens = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english') + list(punctuation))
    tokens = [token for token in tokens if token not in stop_words]

    # Return the preprocessed text as a single string
    return ' '.join(tokens)


def calculate_cosine_similarity(text1, text2):
    # Preprocess the texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)

    # Create the count vectorizer and fit it with the processed texts
    vectorizer = CountVectorizer().fit([processed_text1, processed_text2])

    # Transform the processed texts to TF vectors
    tf_vectors = vectorizer.transform([processed_text1, processed_text2]).toarray()

    # Calculate the cosine similarity between the TF vectors
    similarity = cosine_similarity([tf_vectors[0]], [tf_vectors[1]])

    return similarity[0][0]


# summarize:
text1 = """Mark-up / Mark-down: The firm's mark-up or mark-down on your transaction is measured from the prevailing market price for the fixed income security. The mark-up or mark-down may not be the same as the firm's profit or loss on the transaction and can be impacted by various factors."""

# Actual:
text2 = """Mark-up / Mark-down: Fixed income securities (bonds) are traded over the counter (OTC) and not through an exchange like equity securi- ties. The firm's mark-up or mark-down on your transaction is measured from the prevailing market price for the fixed income security. The prevailing market price is determined by application of a series of factors prescribed by industry regulators (FINRA and MSRB) and may not be the same as the firm's cost of sourcing, or its proceeds from selling, the security. As such, industry participants may come to different determinations concerning the prevailing market price of any given fixed income security at any point in time. In addition, the mark-up or mark-down is not necessarily the same as the firm's profit or loss on the transaction, which may be impacted by a number of other factors, such as out-of-pocket costs, namely regulatory fees and hedging costs, and intervening market movements, among others factors, that do not impact the prevailing market price and/or the calculated mark-up or mark-down. The firm has engaged a third party vendor to calculate the prevailing market price in accordance with FINRA and MSRB rules."""

# wrong for MF:
text3 = """Forward Priced Mutual Fund:
The price and share totals shown above reflect the price received for the trade date. Due to market operating hour differences, the trade
date may be after the date your order was placed with us for certain funds. Contact us for additional
information. We are providing
information to all offshore investors, to advise that there is a key investor information document (KIID) available for each fund offered by
offshore (non-us domiciled) investment companies regulated as undertaking for collective investments in transferable securities (UCITs).
The KIID contains essential information and key facts about a UCITS fund aimed at helping investors make informed investment
decisions about whether the fund meets their needs. Please read the KIID carefully before you invest. You can access a repository of
KIID documents at the following location www.morganstanley.com/offshoremutualfunds/kiidrepository. For UCITS mutual funds and the following web location https://www.morganstanley.com/disclaimers/etfkiidrepository
For UCITS exchange-traded funds. For further information about the fund, please refer to the fund's prospectus."""

similarity_score1 = calculate_cosine_similarity(text1, text2)
similarity_score2 = calculate_cosine_similarity(text1, text3)
print(f"Cosine similarity for text1 and text2: {similarity_score1}")
print(f"Cosine similarity for text1 and text3: {similarity_score2}")
