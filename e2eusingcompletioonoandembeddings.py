import requests
import openai
import numpy as np

def call_openai_chat_api(messages, api_key):
    api_endpoint = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        'model': 'gpt-3.5-turbo',
        'messages': messages,
        'temperature' : 0.7
    }

    try:
        response = requests.post(api_endpoint, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def main():
    # Replace 'YOUR_API_KEY' with your actual API key from OpenAI
    api_key = 'sk-'

    # Initialize the conversation with a system message and a user message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    while True:
        # Get user input for the next message
        user_input = input("You: ")

        # Break the loop if the user wants to end the conversation
        if user_input.lower() in ['exit', 'quit', 'stop']:
            break

        user_input.split

        if "text:" in user_input:
            user_input = user_input.replace("text:","")
            messages.append({"role": "user", "content": user_input})
            response_data = call_openai_chat_api(messages, api_key)
            if response_data and 'choices' in response_data and len(response_data['choices']) > 0:
                generated_text = response_data['choices'][0]['message']['content']
                print("Chatbot:", generated_text)
                messages.append({"role": "assistant", "content": generated_text})

        if "similarity_score" in user_input:
            # print(f'Assistance data : ', messages[2]['content'])
            similarity_score = calculate_similarity(user_input, messages[2]['content'])
            print("Chatbot:", similarity_score)
        

openai.api_key = "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"

def calculate_similarity(sentence1, sentence2):
    # newAddon = "Disclaimer:  Fixed income securities (bonds) are traded over the counter (OTC) and not through an exchange like equity securities.The firm's mark-up or mark-down on your transaction is measured from the ‘prevailing market price’ for the fixed income security.The prevailing market price is determined by application of a series of factors prescribed by industry regulators (FINRA and MSRB) and may not be the same as the firm's cost of sourcing, or its proceeds from selling, the security.As such, industry participants may come to different determinations concerning the prevailing market price of any given fixed income security at any point in time.In addition, the mark-up or mark-down is not necessarily the same as the firm's profit or loss on the transaction, which may be impacted by a number of other factors, such as out-of-pocket costs, namely regulatory fees and hedging costs, and intervening market movements, among others factors, that do not impact the prevailing market price and or the calculated mark-up or mark-down.The firm has engaged a third party vendor to calculate the prevailing market price in accordance with FINRA and MSRB rules."
    newAddon = "Disclaimer: Forward Priced Mutual Fund: The price and share totals shown above reflect the price received for the trade date. Due to market operating hour differences, the trade date may be after the date your order was placed with us for certain funds. Contact us for additional information. We are providing information to all offshore investors, to advise that there is a key investor information document (KIID) available for each fund offered by offshore (non-us domiciled) investment companies regulated as undertaking for collective investments in transferable securities (UCITs). The KIID contains essential information and key facts about a UCITS fund aimed at helping investors make informed investment decisions about whether the fund meets their needs. Please read the KIID carefully before you invest. You can access a repository of KIID documents at the following location www.companyabc.com offshoremutualfunds/kiidrepository. For UCITS mutual funds and the following web location https://www.companyabc.com/disclaimers/etfkiidrepository For UCITS exchange-traded funds. For further information about the fund, please refer to the fund's prospectus."
    #  finalText = newAddon+sentence2
    print(f'comparing ===== from completiion =====', newAddon)
    print(f'data ===== from user input  =====', sentence2)
    resp = openai.Embedding.create(
        input=[newAddon, sentence2],
        engine="text-similarity-davinci-001"
    )

    embeddings = [item['embedding'] for item in resp['data']]
    similarity_score = np.dot(embeddings[0], embeddings[1])

    return similarity_score

if __name__ == "__main__":
    main()
