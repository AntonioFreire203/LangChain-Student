

from langchain_openai import ChatOpenAI
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

TEMPLATE = """
Interprete o texto e classifique:
- sentimento (positivo, neutro, negativo)
- assunto (uma palavra)

Formato JSON:
sentiment: {sentiment}
subject: {subject}

Texto: {input}


"""


# Criando o modelo de chat
llm = ChatOpenAI()

examples = [
    {"text": "O atendimento foi ótimo!", "response": "sentiment: positivo\nsubject: atendimento"},
    {"text": "O produto estava danificado.", "response": "sentiment: negativo\nsubject: loja"}
]

example_prompt = PromptTemplate(
    input_variables=["text", "response"],
    template="Texto: {text}\n{response}"
)

prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Texto: {input}",
    input_variables=["input"]
)



resposta = llm.invoke(prompt.format(input='O camisa que comprei veio com defeito na manga da camisa.'))
print(resposta.content)
