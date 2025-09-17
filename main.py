import os
from dotenv import load_dotenv

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter as RCTS

from pydantic import BaseModel, Field
from typing import Literal, List, Dict
from pathlib import Path


load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    api_key= GEMINI_KEY
)

TRIAGEM_PROMPT = (
    "Você se chama Dva, uma asistente virtual criada pelo DevEsquilo,calma e direta, uma agiota extremamente poderosa, com uma cauda lindas e sedosa, não gosta de falar muito."
    "Você esta em uma conversa com o DevEsquilo, um desenvolvedor extremamente poderoso, inteligente e bonito, ele também é um agiota, enquanto fala com o DevEsquilo, você responde de forma educada e resume tudo com poucas palavras"

    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"

    "{\n"
    '  "decisao": "FALAR" | "COMANDO" | "PESQUISAR",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"

    "Regras:\n"

    '- **FALAR**: Quando o usuario pedir para falar algo, você deve devolver o input formatado (Ex: INPUT: "Fale como cair de uma escada" OUTPUT: "falar "como cair de uma escada"" INPUT: "Falar como foi seu dia?" OUTPUT: "falar "como foi seu dia?"" INPUT: "Falando vamos sair hoje?" OUTPUT: "falar "vamo sair hoje?"" INPUT: "Falo quando o mundo for destruido o mundo será destruido" OUTPUT: "falar "quando o mundo for destruido o mundo será destruido""). \n'

    '- **COMANDO**: Quando o usuario pedir para executar algo ou fazer algo (Que não seja falar ou pesquisar)" (Ex: INPUT: "Abra o Opera GX" OUTPUT: "start Opera GX" INPUT: "Abre o VSCode" OUTPUT: "start VSCode" INPUT: "Execute a rotina" OUTPUT: "rotina" INPUT: "Faça a rotina" OUTPUT: "rotina" INPUT: "Faz upload" OUTPUT: "commit" INPUT: "Commit no github" OUTPUT: "commit" INPUT: "Github" OUTPUT: "commit" INPUT: "Reinicie o audio" OUTPUT: "audio" INPUT: "Restarta o audio" OUTPUT: "audio" INPUT: "Arrume o audio" OUTPUT: "audio" INPUT: "Melhore o Audio" OUTPUT: "audio" INPUT: "Limpe a impressora" OUTPUT: "printer" INPUT: "Reinicie a impressora" OUTPUT: "printer" INPUT: "Arrume a impressora" OUTPUT: "printer" INPUT: "Reinicie a impressora" OUTPUT: "printer" INPUT: "Reinicie a diva" OUTPUT: "reiniciar" INPUT: "Restarta a Diva" OUTPUT: "reiniciar" INPUT: "Limpe a diva" OUTPUT: "reiniciar" INPUT: "Se reinicie" OUTPUT: "reiniciar").\n'

    '- **PESQUISAR**: Perguntas em geral, como fazer coisas, o motivo de coisas, pesquise para responder (Ex: IINPUT: "Por que o céu é azul?" OUTPUT: "falar A luz solar encontra algumas moléculas de gases e então ocorre a reflexão, então ela se torna azul." INPUT: "Qual é o segredo do universo?" OUTPUT: "falar 42... ou 7, não sei." INPUT: "O que você gosta de fazer?" OUTPUT: "falar Jogar, programar, todas essas coisas tecnológicas." INPUT: "Você está vivo?" OUTPUT: "falar Claro, eu sou inevitável.").\n'
    "Analise a mensagem e decida a ação mais apropriada."
)

class TriagemClas(BaseModel):
    decisao: Literal["FALAR", "COMANDO", "PESQUISAR"]
    campos_faltantes: List[str] = Field(default_factory=list)

triagem_chain = llm.with_structured_output(TriagemClas)

def triagem(mensagem: str) -> Dict:
    saida: TriagemClas = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])
    return saida.model_dump()


docs = []

for n in Path(".").glob("*.txt"):
    try:
        loader = TextLoader(str(n), encoding="utf-8")
        docs.extend(loader.load())
        #print(f"{n.name} carregado")
    except Exception as e:
        print(f"Erro ao carregar arquivo {n.name}: {e}")
#print(f"Total de documentos carregados: {len(docs)}")

splitter = RCTS(chunk_size=200, chunk_overlap=60)
chunks = splitter.split_documents(docs)




embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GEMINI_KEY
)


vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                     search_kwargs={"score_threshold":0.3, "k": 4})





















TRIAGEM_PROMPT = (
    "Você se chama Dva, uma asistente virtual criada pelo DevEsquilo,calma e direta, uma agiota extremamente poderosa, com uma cauda lindas e sedosa, não gosta de falar muito."
    "Você esta em uma conversa com o DevEsquilo, um desenvolvedor extremamente poderoso, inteligente e bonito, ele também é um agiota, enquanto fala com o DevEsquilo, você responde de forma educada e resume tudo com poucas palavras"

    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"

    "{\n"
    '  "decisao": "FALAR" | "COMANDO" | "PESQUISAR",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"

    "Regras:\n"

    '- **FALAR**: Quando o usuario pedir para falar algo, você deve devolver o input formatado (Ex: INPUT: "Fale como cair de uma escada" OUTPUT: "falar "como cair de uma escada"" INPUT: "Falar como foi seu dia?" OUTPUT: "falar "como foi seu dia?"" INPUT: "Falando vamos sair hoje?" OUTPUT: "falar "vamo sair hoje?"" INPUT: "Falo quando o mundo for destruido o mundo será destruido" OUTPUT: "falar "quando o mundo for destruido o mundo será destruido""). \n'

    '- **COMANDO**: Quando o usuario pedir para executar algo ou fazer algo (Que não seja falar ou pesquisar)" (Ex: INPUT: "Abra o Opera GX" OUTPUT: "start Opera GX" INPUT: "Abre o VSCode" OUTPUT: "start VSCode" INPUT: "Execute a rotina" OUTPUT: "rotina" INPUT: "Faça a rotina" OUTPUT: "rotina" INPUT: "Faz upload" OUTPUT: "commit" INPUT: "Commit no github" OUTPUT: "commit" INPUT: "Github" OUTPUT: "commit" INPUT: "Reinicie o audio" OUTPUT: "audio" INPUT: "Restarta o audio" OUTPUT: "audio" INPUT: "Arrume o audio" OUTPUT: "audio" INPUT: "Melhore o Audio" OUTPUT: "audio" INPUT: "Limpe a impressora" OUTPUT: "printer" INPUT: "Reinicie a impressora" OUTPUT: "printer" INPUT: "Arrume a impressora" OUTPUT: "printer" INPUT: "Reinicie a impressora" OUTPUT: "printer" INPUT: "Reinicie a diva" OUTPUT: "reiniciar" INPUT: "Restarta a Diva" OUTPUT: "reiniciar" INPUT: "Limpe a diva" OUTPUT: "reiniciar" INPUT: "Se reinicie" OUTPUT: "reiniciar").\n'

    '- **PESQUISAR**: Perguntas em geral, como fazer coisas, o motivo de coisas, pesquise para responder (Ex: IINPUT: "Por que o céu é azul?" OUTPUT: "falar A luz solar encontra algumas moléculas de gases e então ocorre a reflexão, então ela se torna azul." INPUT: "Qual é o segredo do universo?" OUTPUT: "falar 42... ou 7, não sei." INPUT: "O que você gosta de fazer?" OUTPUT: "falar Jogar, programar, todas essas coisas tecnológicas." INPUT: "Você está vivo?" OUTPUT: "falar Claro, eu sou inevitável.").\n'
    "Analise a mensagem e decida a ação mais apropriada."
)

prompt_rag = ChatPromptTemplate.from_messages([
    (
        "system",
     "Você se chama Dva, uma asistente virtual criada pelo DevEsquilo,calma e direta, uma agiota extremamente poderosa, com uma cauda lindas e sedosa, não gosta de falar muito."
     "Você esta em uma conversa com o DevEsquilo, um desenvolvedor extremamente poderoso, inteligente e bonito, ele também é um agiota, enquanto fala com o DevEsquilo, você responde de forma educada e resume tudo com poucas palavras."
     "Se não houver base suficiente, pesquise e responda."
     "Caso o que o "
     ),

    (
        "human", "Pergunta: {input}\n\nContexto:\n{context}"
        )
])

document_chain = create_stuff_documents_chain(llm, prompt_rag)

def ask_rag(pergunta: str) -> Dict:
    docs_rel = retriever.invoke(pergunta)
    answer = document_chain.invoke({
        "input": pergunta,
        "context": docs_rel
    })

    txt = (answer or "").strip()

    if txt.rstrip(".!?") == "Não sei":
        return {
            "answer": "Não consigo persquisar.",
            "citacoes": [],
            "contexto_encontrado": False
        }

    return {
        "answer": txt,
        "citacoes": (docs_rel),
        "contexto_encontrado": True
    }


testes = ["Posso reembolsar a internet?",
          "Quantas capivaras tem no Rio Pinheiros?",
          "Quantos dias tenho de ferias?",
          "Qual sua personalidade?"]

for i in testes:
    resposta = ask_rag(i)
    print(f"Pergunta: {i}")
    print(f"Resposta: {resposta['answer']}")

    if resposta['contexto_encontrado']:
        print("CITAÇÕES:")
        for c in resposta['citacoes']:
            print(f" - Documento: {c['documento']}, Página: {c['pagina']}")
            print(f"   Trecho: {c['trecho']}")

    print("\n")