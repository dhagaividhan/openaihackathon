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

resp = openai.Embedding.create(
    input=[disc1, disc2],
    engine="text-similarity-davinci-001")

embedding_a = resp['data'][0]['embedding']
embedding_b = resp['data'][1]['embedding']

similarity_score = np.dot(embedding_a, embedding_b)

print(similarity_score)