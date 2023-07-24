import openai, os, sys, numpy as np
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI

st.set_page_config(page_title='Confirms Gen AI', page_icon='🤖', layout='centered', initial_sidebar_state='auto')

os.environ["OPENAI_API_KEY"] = ''
openai.api_key = os.getenv("OPENAI_API_KEY")

sample_disclosure_intial_str=""
actual_disclosure_str=""

def sample_disclosure(product_type):
    # Instantiate LLM model
    # llm = OpenAI(model_name="text-davinci-003")
    # Prompt
    template = "Give me some sample disclaimers or disclosures in paragraphs to be printed in a trade confirmation for a {topic}?"
    prompt = PromptTemplate(input_variables=["topic"], template=template)
    prompt_query = prompt.format(topic=product_type)

    print(prompt_query)

    # read txt
    loader = DirectoryLoader(".", glob="MF*.txt")
    index = VectorstoreIndexCreator().from_loaders([loader])

    # Run LLM model
    first_response = index.query(prompt_query, llm=ChatOpenAI())
    # Print results
    return first_response

st.title("✅📰 Confirms Disclosure Relevance Evaluator")
with st.form("myform2"):
    product_type = st.text_input("Product Type", placeholder="Mutual Fund")
    actual_disclosure_str = st.text_area("Enter Your Actual Disclosure", placeholder="Actual Disclosure")
    submitted = st.form_submit_button("Submit")
    if submitted:
        sample_disclosure_intial_str = sample_disclosure(product_type)
        st.markdown(":green[Your Sample Disclosure is: ]" + sample_disclosure_intial_str)
        st.markdown(":green[Your Actual Disclosure is: ]" + actual_disclosure_str)

        # write your code to create a prompt of text 1 and text 2
        # can you provide the relevance of text 2 as compared to text 1 below? Only respond as relevant, not relevent or partially relevant. text 1 - {sampleDiscText} and text 2 - {actual_disclosure}

        # template2 = "can you provide the relevance of text 2 as compared to text 1 below? Only respond as relevant, not relevent or partially relevant. \n \ntext 1 - {sampleDiscText} \n \ntext 2 - {actual_disclosure}"
        jinja2_template = "can you provide the relevance of text 2 as compared to text 1 below? Only respond as relevant, irrelevant or partially relevant. \n \ntext 1 - {{sampleDiscText}} \n \ntext 2 - {{actual_disclosure}}"
        prompt = PromptTemplate.from_template(jinja2_template, template_format="jinja2")

        prompt_query = prompt.format(sampleDiscText=sample_disclosure_intial_str, actual_disclosure=actual_disclosure_str)
        st.info(":blue[This should be your prompt: ]"+prompt_query)

# resp = openai.Embedding.create(
#     input=[str(sample_disclosure_intial_str), str(actual_disclosure_str)],
#     engine="text-similarity-davinci-001")
#
# embedding_a = resp['data'][0]['embedding']
# embedding_b = resp['data'][1]['embedding']
#
# similarity_score_ab = np.dot(embedding_a, embedding_b)
# print("similarity score of disc1 and disc2:  " + str(similarity_score_ab))

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant",
                                     "content": "Hey, I can help you provide the relevancy of trade confirmation disclaimer or disclosure by comparing two disclosure texts."
                                                "Please enter the actual disclosure that was printed in the trade confirmation document."}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():

        # getting user prompt and appending in the chatbox message display session state st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # open ai connection for completion model
        response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
        msg = response.choices[0].message

        # appending the response back to session state st.session_state.messages.append(msg)
        st.session_state.messages.append(msg)
        st.chat_message("assistant").write(msg.content)
        # st.chat_message("assistant").write(msg.content + ". Similarity Score is: " + str(similarity_score_ab))
