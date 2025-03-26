from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains.llm import LLMChain


from langchain.prompts import PromptTemplate

load_dotenv(find_dotenv())


llm =ChatOpenAI(model = 'gpt-4o-mini')
template_prompt = 'Me conta uma piada de portugues'

prompt_template = PromptTemplate(input_variables=["input"],template = template_prompt)
chain = LLMChain(llm=llm,prompt=prompt_template)

# print(chain.invoke({'input':''}))

# 📌 EXPLICAÇÕES DETALHADA
# Executa uma sequência básica: recebe input → aplica o prompt → envia ao LLM → retorna resposta


from langchain.callbacks.base import BaseCallbackHandler

class MyCustomHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs) -> None:
        print(f"RESPONSE: {response}")

handler = MyCustomHandler()
config = {'callbacks': [handler]}

chain.invoke(input="rabit", config=config)
