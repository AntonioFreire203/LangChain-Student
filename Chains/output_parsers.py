from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import SystemMessagePromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
load_dotenv(find_dotenv())


TEMPLATE = """
Interprete o texto e avalie o conteúdo abaixo:

sentiment: o texto tem sentimento positivo, neutro ou negativo? (Obrigatório)
subject: qual é o assunto principal do texto? Use UMA palavra. Use 'None' se não for possível identificar.
price: quanto o cliente pagou? Use 'None' se o preço não for mencionado.

Formate a saída como JSON com as seguintes chaves:
sentiment
subject
price

Texto: {input}
"""



llm = ChatOpenAI()
prompt_template = ChatPromptTemplate.from_template(template=TEMPLATE)
chain = LLMChain(llm=llm, prompt=prompt_template)

# Exemplo de uso do chain
response = chain.invoke({"input": "I ordered pizza salami from the restaurant Bellavista. It was ok, but the dough could have been a bit more crisp."})
print(response)


# Utilização de Output Parsers com Schema


# Definindo schemas esperados
response_schemas = [
    ResponseSchema(name="sentiment", description="Sentimento: positivo, neutro ou negativo."),
    ResponseSchema(name="subject", description="Assunto principal do texto. Use apenas uma palavra."),
    ResponseSchema(name="price", description="Preço pago pelo cliente. Use None se não houver valor.", type="float")
]

# Criando o parser estruturado
parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()
print("Instruções de formatação:\n", format_instructions)


# Template com instruções + parser



prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "Interprete o texto e avalie-o.\n"
            "sentiment: o texto tem sentimento positivo, neutro ou negativo?\n"
            "subject: qual é o assunto do texto? Use UMA palavra.\n"
            "Retorne APENAS o JSON, sem interpretações extras!\n"
            "Texto: {input}\n"
            "{format_instructions}"
        )
    ],
    input_variables=["input"],
    partial_variables={"format_instructions": format_instructions}
)

# Formata a entrada para o modelo
_input = prompt.format_prompt(input="I ordered pizza salami from the restaurant Bellavista. It was ok, but the dough could have been a bit more crisp.")
output = llm.invoke(_input.to_messages())
print("\nResposta bruta do modelo:\n", output.content)

# Faz o parsing do JSON gerado pelo LLM
json_output = parser.parse(output.content)
print("\nResposta estruturada:\n", json_output)

# Acessa valores individuais
print("\nSentimento identificado:", json_output.get("sentiment"))





# ✅ StructuredOutputParser:
# Usa os schemas definidos para criar um parser que converte a resposta do modelo em dicionário Python.
#   - Método: `get_format_instructions()` retorna instruções de como o LLM deve formatar a saída.
#   - Método: `parse()` recebe a string gerada pelo modelo e converte em JSON (dict).

