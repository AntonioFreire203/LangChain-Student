from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain


load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini")

# Exemplo com uma variável
template = "Me conte uma piada sobre {input}"
prompt = PromptTemplate(input_variables=["input"], template=template)
chain = LLMChain(llm=llm, prompt=prompt)
print(chain.invoke({"input": "um papagaio"}))

# Exemplo com múltiplas variáveis
template_multi = "Me conte uma piada sobre {input} em {language}"
prompt_multi = PromptTemplate(input_variables=["input", "language"], template=template_multi)
chain_multi = LLMChain(llm=llm, prompt=prompt_multi)
print(chain_multi.invoke({"input": "um papagaio", "language": "alemão"}))