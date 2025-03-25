from langchain.chains import SequentialChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv,find_dotenv
load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini")

# Chain 1: Escreve uma resenha
class ReviewChain():
    def __init__(self, prompt, llm):

        self.llm = llm
        self.prompt = prompt
        self.chain = LLMChain(llm=llm,prompt=prompt,output_key="review")

template_review = "Você pediu {dish_name} e sua experiência foi {experience}. Escreva uma resenha:"
prompt_review = PromptTemplate.from_template(template_review)
review_chain = ReviewChain(llm, prompt_review)

# Chain 2: Comentário com base na resenha
template_comment = "Dada a seguinte resenha do restaurante: {review}, escreva um comentário de acompanhamento:"
prompt_comment = PromptTemplate.from_template(template_comment)
comment_chain = ReviewChain(llm, prompt_comment)

# Chain 3: Resumo do comentário
template_summary = "Resuma o seguinte comentário em uma frase curta:\n\n{comment}"
prompt_summary = PromptTemplate.from_template(template_summary)
summary_chain = ReviewChain(llm, prompt_summary)

# Chain 4: Tradução para alemão
template_translation = "Traduza o seguinte resumo para o alemão:\n\n{summary}"
prompt_translation = PromptTemplate.from_template(template_translation)


# Chain sequencial completa
chain_completo = SequentialChain(
    chains=[review_chain, comment_chain, summary_chain],
    input_variables=["dish_name", "experience"],
    output_variables=["review", "comment", "summary", ],
    verbose=True
)

resultado = chain_completo.invoke({"dish_name": "Pizza de Salame", "experience": "Foi horrível!"})
print(resultado)