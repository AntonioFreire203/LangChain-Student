from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


# Criando o modelo de chat
llm = ChatOpenAI()

# Exemplo simples com apenas uma pergunta
resposta = llm.invoke("Me conta uma piada")
print(resposta.content)

# Exemplo com contexto: definindo o comportamento do assistente
mensagens = [
    SystemMessage(content="Você é um assistente divertido que adora contar piadas em português."),
    HumanMessage(content="Conta uma piada sobre portugues de portugal.")
]

resposta_com_contexto = llm.invoke(mensagens)
print(resposta_com_contexto.content)
