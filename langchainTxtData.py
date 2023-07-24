import openai, os, sys
import constants
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = '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'
openai.api_key = os.getenv("OPENAI_API_KEY")

query = sys.argv[1]
print(query)

#loader = TextLoader("Disclosures.txt")
loader = DirectoryLoader(".", glob="*.txt")
index = VectorstoreIndexCreator().from_loaders([loader])
print(index.query(query, llm=ChatOpenAI()))
#print(index.query(query))