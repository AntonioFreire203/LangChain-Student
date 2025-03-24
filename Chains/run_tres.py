from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Inicializa o modelo de linguagem
llm = ChatOpenAI(model="gpt-4o-mini")

# Define o template do prompt com variáveis dinâmicas
prompt_template = PromptTemplate(
    input_variables=["assunto", "idioma"],
    template="Conte uma piada sobre {assunto} em {idioma}."
)

# Cria a cadeia LLMChain
chain = LLMChain(llm=llm, prompt=prompt_template)

# Executa a cadeia com valores específicos para as variáveis
resultado = chain.invoke({"assunto": "papagaio", "idioma": "português"})
print(resultado)