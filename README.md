# LangChain-Student

consultar a documentação oficial do LangChain : https://python.langchain.com/api_reference/langchain/

📚langchain
### Descrição:

Este módulo centraliza os principais componentes da biblioteca LangChain, incluindo agentes, chains, LLMs, embeddings, ferramentas e callbacks.

🧠 **langchain.agents**

Fornece funcionalidades relacionadas à criação, execução e gerenciamento de agentes.

**langchain.agents.agent**
Função principal: Agent

    Representa a base para agentes que interagem com ferramentas para atingir objetivos complexos.

**langchain.agents.agent_types**

Define os diferentes tipos de agentes disponíveis:

    zero-shot-react-description

    chat-zero-shot-react-description

    conversational-react-description

    structured-chat-zero-shot-react-description

**langchain.agents.load_tools**

Utilitários para carregar ferramentas externas para agentes.

    load_tools(tool_names: List[str], llm: BaseLLM)

**langchain.agents.mrkl**

Implementação do agente MRKL (Modular Reasoning, Knowledge and Language).

**langchain.agents.react.base**

Implementação do agente baseado no padrão ReAct (Reasoning + Acting).

**langchain.agents.tools**

Módulo para criação e registro de ferramentas customizadas utilizadas por agentes.

**langchain.agents.utils**

Funções auxiliares para formatação de inputs/outputs de agentes.

### 🔁 langchain.callbacks
Callbacks são usados para rastrear eventos durante a execução de chains, agentes ou LLMs.
    CallbackManager(...)
