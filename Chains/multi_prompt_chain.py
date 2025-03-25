# exemplo_03_multi_prompt_chain.py
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini")

# Templates personalizados para análise de sentimentos
templates = {
    "positivo": "Você é uma IA que foca nos aspectos positivos. Texto:\n{input}",
    "neutro": "Você é uma IA com perspectiva neutra. Texto:\n{input}",
    "negativo": "Você é uma IA que identifica aspectos negativos. Texto:\n{input}"
}

prompt_infos = [
    {"name": nome, "description": f"Análise com viés {nome}", "prompt_template": tpl}
    for nome, tpl in templates.items()
]

# Criação dos chains de destino
destination_chains = {}
for info in prompt_infos:
    prompt = PromptTemplate(template=info["prompt_template"], input_variables=["input"])
    destination_chains[info["name"]] = LLMChain(llm=llm, prompt=prompt)

# Template do roteador
destinations = "\n".join([f"{p['name']}: {p['description']}" for p in prompt_infos])
router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations)
router_prompt = PromptTemplate(template=router_template, input_variables=["input"], output_parser=RouterOutputParser())

# Criação do roteador
router_chain = LLMRouterChain.from_llm(llm, router_prompt)

# MultiPromptChain com fallback padrão "neutro"
multi_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=destination_chains,
    default_chain=destination_chains["neutro"],
    verbose=True
)

multi_chain.invoke({"input": "Pedi uma pizza de salame por R$39,99 e estava maravilhosa!"})
