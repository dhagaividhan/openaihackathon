import openai
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity

# Set your OpenAI API key
openai.api_key = "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"

# Function to calculate similarity score between two sentences
def calculate_similarity(sentence1, sentence2):
    resp = openai.Embedding.create(
        input=[sentence1, sentence2],
        engine="text-similarity-davinci-001"
    )

    embeddings = [item['embedding'] for item in resp['data']]
    similarity_score = np.dot(embeddings[0], embeddings[1])

    return similarity_score

# Example sentences
sentence1 = "this is tiger"
sentence2 = "this is white tiger"
sentence3 = "this is ice cream"
sentence4 = "Disclaimer: The following sample disclaimer is intended to provide general information and should not be considered as legal advice. It is essential to consult with a legal professional to ensure compliance with applicable laws and regulations.This trade confirmation is for informational purposes only and does not constitute an offer, solicitation, or recommendation to buy or sell any fixed income security bond. The information provided herein is based on sources believed to be reliable, but we do not guarantee its accuracy, completeness, or timeliness.Investing in fixed income securities involves risks, including the potential loss of principal. The past performance of any investment is not indicative of future results. The value of fixed income securities may fluctuate based on changes in interest rates, credit quality, and market conditions.This trade confirmation is not intended to be relied upon as tax, legal, or investment advice. You should consult with your own advisors regarding the specific implications of any investment decision.We do not make any warranties, expressed or implied, regarding the performance, suitability, or fitness for a particular purpose of any fixed income security bond mentioned in this trade confirmation.By accepting this trade confirmation, you acknowledge and understand that you are solely responsible for making your own investment decisions and that you have conducted your own research and due diligence.This trade confirmation is subject to the terms and conditions outlined in our client agreement. Any unauthorized use, disclosure, or distribution of this trade confirmation is prohibited.Please note that this sample disclaimer is provided as a starting point and may need to be tailored to your specific business and legal requirements. It is recommended to seek professional advice to ensure accuracy and compliance with applicable laws and regulations."
sentence5 = "Disclaimer:  Fixed income securities (bonds) are traded over the counter (OTC) and not through an exchange like equity securities.The firm's mark-up or mark-down on your transaction is measured from the 'prevailing market price' for the fixed income security. The prevailing market price is determined by application of a series of factors prescribed by industry regulators (FINRA and MSRB) and may not be the same as the firm's cost of sourcing, or its proceeds from selling, the security. As such, industry participants may come to different determinations concerning the prevailing market price of any given fixed income security at any point in time. In addition, the mark-up or mark-down is not necessarily the same as the firm's profit or loss on the transaction, which may be impacted by a number of other factors, such as out-of-pocket costs, namely regulatory fees and hedging costs, and intervening market movements, among others factors, that do not impact the prevailing market price and/or the calculated mark-up or mark-down. The firm has engaged a third party vendor to calculate the prevailing market price in accordance with FINRA and MSRB rules."

original_disclaimer = """
Disclaimer: 

This trade confirmation of a fixed income security bond is provided for informational purposes only and should not be considered as investment advice or a recommendation to buy or sell securities. The information contained in this confirmation is based on sources believed to be reliable, but no representation or warranty, expressed or implied, is made regarding its accuracy, completeness, or suitability for any particular purpose.

The value of fixed income securities can fluctuate and may be affected by changes in interest rates, creditworthiness of issuers, market conditions, and other factors. Past performance is not indicative of future results. Investors should carefully consider their own investment objectives, risk tolerance, and financial situation before making any investment decisions.

The bond described in this trade confirmation may be subject to certain risks, including but not limited to credit risk, interest rate risk, and liquidity risk. Investors should carefully review the bond's prospectus, offering memorandum, and any other relevant documents to understand the specific risks associated with the security.

This trade confirmation is not intended to be a legal, tax, or accounting advice, and investors are encouraged to consult with their own advisors regarding such matters.

By accepting this trade confirmation, the recipient acknowledges and agrees that neither the sender nor any affiliated party shall be liable for any loss or damages arising from the use of this information. The recipient also agrees to hold harmless the sender and any affiliated party from any claims, actions, or liabilities arising from the recipient's reliance on the information provided herein.

This trade confirmation is confidential and intended solely for the use of the recipient. Any unauthorized use or dissemination of this information is strictly prohibited.

Please note that this disclaimer is provided as a general template and may need to be customized to accurately reflect the specific circumstances and requirements of the trade confirmation.
"""

# Provided Disclaimer for Forward Priced Mutual Fund (provided earlier)
provided_disclaimer = """
Disclosures: Forward Priced Mutual Fund: The price and share totals shown above reflect the price received for the trade date. Due to market operating hour differences, the trade date may be after the date your order was placed with us for certain funds. Contact us for additional information. We are providing information to all offshore investors, to advise that there is a key investor information document (KIID) available for each fund offered by offshore (non-us domiciled) investment companies regulated as undertaking for collective investments in transferable securities (UCITs). The KIID contains essential information and key facts about a UCITS fund aimed at helping investors make informed investment decisions about whether the fund meets their needs. Please read the KIID carefully before you invest. You can access a repository of KIID documents at the following location www.companyabc.com offshoremutualfunds/kiidrepository. For UCITS mutual funds and the following web location https://www.companyabc.com/disclaimers/etfkiidrepository For UCITS exchange-traded funds. For further information about the fund, please refer to the fund's prospectus.
"""

# # Calculate similarity between sentence1 and sentence2
# similarity_score = calculate_similarity(original_disclaimer, provided_disclaimer)

# print("Similarity score:", similarity_score)

import openai
import numpy as np

# Set your OpenAI API key
openai.api_key = "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"

# Function to calculate similarity score between two sentences
def calculate_similarity(sentence1, sentence2):
    resp = openai.Embedding.create(
        input=[sentence1, sentence2],
        engine="text-similarity-davinci-001"
    )

    embeddings = [item['embedding'] for item in resp['data']]
    # similarity_score = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
    
    norm_embeddings_a = embeddings[0] / np.linalg.norm(embeddings[0])
    norm_embeddings_b = embeddings[1] / np.linalg.norm(embeddings[1])

# Calculate cosine similarity using the dot product of normalized embeddings
    cosine_similarity = np.dot(norm_embeddings_a, norm_embeddings_b)

    return cosine_similarity

# Example sentences
sentence1 = "this is tiger"
sentence2 = "this is white tiger"
sentence3 = "this is ice cream"
sentence4 = "the cat was sittinig on the mat"
sentence5 = "the dog was lyiinig on the carpet"

# Calculate similarity between sentence1 and sentence2
similarity_score = calculate_similarity(sentence1, sentence2)

print("Similarity score for some: ", similarity_score)
