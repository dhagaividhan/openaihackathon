import openai, numpy as np, os

os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")

disc1="1. Regulatory disclaimer: The confirmation statement should include a disclaimer stating that the investment is subject to regulatory guidelines and requirements, and the transaction has been executed in accordance with applicable laws and regulations. " \
      "2. Mutual Fund disclosure: The confirmation should clearly state that the investment is in a mutual fund and provide the full name of the fund, in this case, " \
      "3. Investment objective and risk disclosure: The confirmation should provide a brief description of the mutual fund's investment objective and disclose the associated risks. This may include information about the volatility, market risks, and potential loss of principal associated with investing in small-cap funds." \
      "4. Performance disclosure: The confirmation should disclose that past performance is not necessarily indicative of future results and that the investment value can fluctuate." \
      "5. Legal and tax implications: The confirmation statement should state that clients should seek legal and tax advice regarding the investment, as the purchase of mutual fund units may have legal and tax implications based on the investor's unique circumstances." \
      "6. Fees and expenses: The confirmation should clearly disclose any fees, expenses, or charges associated"

disc2="Advisory account conversion notice we are please to announce a change to your advisory mutual fund share class holdings. As you may already know, we have developed " \
      "and automated process to convert your current mutual fund share classes into advisory share classes of the same funds. Advisory share classes typically have lower expense ratios." \
      "The mutual fund share classes to which your mutual fund positions have converted will now receive all of the advisory services detailed in your advisory client " \
      "agreement and will be subject to the feed outlined in that agreement. On this confirmation, you will see the mutual fund conversions that have taken place in your account, " \
      "pursuant to the terms of your advisory agreement. If you have any question, please contact us"

disc3="Conversion Terms and Conditions: Details about the specific terms and conditions of the conversion, including any eligibility criteria and requirements." \
      "Prospectus or Offering Document: Information about the investment product's objectives, strategies, risks, fees, and expenses." \
      "The prospectus is a legal document that provides comprehensive information about the fund." \
      "Conversion Process: Explanation of how the conversion transaction will be executed, including any necessary forms or procedures." \
      "Tax implications: Disclosure of any potential tax consequences related to the conversion, as tax treatment may vary depending on the type of the account and the specific transaction." \
      "Fees and Expenses: Information about any fees or changes associated with the conversion, such as conversion fees or account maintenance fees." \
      "Risks: Disclosure of the risks associated with the conversion, as well as risks related to the investment product itself." \
      "Investment Objectives and Strategies: Explanation of the investment objectives and strategies of the destination fund (the fund you are converting into)." \
      "Past Performance: Historical performance data of both the source fund (the fund you are converting from) and the destination fund, as well as the usual disclaimer that past" \
      "performance does not guarantee future results. Regulatory and Legal Information: Disclosures about the compliance with relevant laws and regulations."

resp = openai.Embedding.create(
    input=[disc1, disc2, disc3],
    engine="text-similarity-davinci-001")

embedding_a = resp['data'][0]['embedding']
embedding_b = resp['data'][1]['embedding']
embedding_c = resp['data'][2]['embedding']

similarity_score_ab = np.dot(embedding_a, embedding_b)
similarity_score_cb = np.dot(embedding_c, embedding_b)
print("similarity score of disc1 and disc2: " + str(similarity_score_ab))
print("similarity score of disc3 and disc2: " + str(similarity_score_cb))

#similarity score of disc1 and disc2: 0.7331801313683288
#similarity score of disc3 and disc2: 0.7704865640846361
