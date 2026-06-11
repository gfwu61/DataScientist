# Beispiel Frage
# wie lang ist die Kündigungsfrist für mich wenn ich 4 Jahre in der Wohnung schon lebe?
# PS C:\0_DA\AI_Practice\11_RAG\law> streamlit run 05_0_rag_chroma_api_st.py


import os
os.environ["LANGCHAIN_OPENAI_TCP_KEEPALIVE"] = "0"
import socket
import urllib3
import ssl
import httpx
import streamlit as st  # Neu für die Web-Oberfläche
from typing import List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ============================================================
# AUTOMATISCHE PROXY- & VPN-ERKENNUNG (NUR EINMALIG AUSFÜHREN)
# ============================================================
@st.cache_resource
def initialize_network_and_clients():
    """Prüft die Umgebung und erstellt die HTTPX-Clients einmalig."""
    def check_if_px_is_running():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect(("127.0.0.1", 3128))
            s.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            return False

  
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
# Initialisiere die Verbindungsparameter für LangChain
    custom_http_client = None
    custom_http_async_client = None

    if check_if_px_is_running():
        os.environ["HTTP_PROXY"] = "http://127.0.0.1:3128"
        os.environ["HTTPS_PROXY"] = "http://127.0.0.1:3128"
        os.environ["CURL_CA_BUNDLE"] = ""
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        px_transport = httpx.HTTPTransport(proxy="http://127.0.0.1:3128", verify=ssl_context)
        client = httpx.Client(mounts={"http://": px_transport, "https://": px_transport})
        async_client = httpx.AsyncClient(mounts={"http://": px_transport, "https://": px_transport})
    else:
        proxy_url = "http://rb-proxy-de.bosch.com:8080"
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        bosch_transport = httpx.HTTPTransport(proxy=proxy_url, verify=ssl_context)
        client = httpx.Client(mounts={"http://": bosch_transport, "https://": bosch_transport}, timeout=60.0)
        async_client = httpx.AsyncClient(mounts={"http://": bosch_transport, "https://": bosch_transport}, timeout=60.0)
        
    return client, async_client

# Clients holen
custom_http_client, custom_http_async_client = initialize_network_and_clients()

# ============================================================
# CONFIG & LANGCHAIN INITIALISIERUNG
# ============================================================
load_dotenv()
PERSIST_DIRECTORY = "chroma_legal_rag"

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    http_client=custom_http_client,
    http_async_client=custom_http_async_client
)

llm_query = ChatOpenAI(
    model="gpt-5.4-nano",  # Korrigiertes offizielles Modell-Beispiel
    temperature=0.1,
    http_client=custom_http_client,
    http_async_client=custom_http_async_client
)

llm_answer = ChatOpenAI(
    model="gpt-5.4-mini", 
    temperature=0,
    http_client=custom_http_client,
    http_async_client=custom_http_async_client
)

vector_store = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embeddings
)

# ============================================================
# KETTEN & LOGIK (Kopiert aus deinem Original-Code)
# ============================================================
class QueryExtraction(BaseModel):
    search_query: str = Field(description="Optimierte, suchbare juristische Kernphrase.")
    paragraph_filter: List[str] = Field(default_factory=list, description="Liste reiner Paragraphen-Nummern.")

query_prompt = ChatPromptTemplate.from_messages([
    ("system", "Du bist ein präziser juristischer Such-Assistent.Extrahiere die suchbare Kernphrase.\n"
     "Erkennst du konkrete Paragraphen-Nennungen (z.B. '§ 556d' oder 'Paragraph 573'), extrahiere NUR die reine Nummer/Ziffer (z.B. '556d', '573') in die Liste paragraph_filter."
    ),
    ("human", "{question}")
])
query_chain = query_prompt | llm_query.with_structured_output(QueryExtraction)


def retrieve_documents(input_data: dict):
    question = input_data["question"] if isinstance(input_data, dict) else input_data
    try:
        extracted = query_chain.invoke({"question": question})
        search_phrase = extracted.search_query if extracted.search_query else question
        p_filters = extracted.paragraph_filter if extracted.paragraph_filter else []
    except Exception as e:
        print(f"  ⚠ Query Extraction fehlgeschlagen: {e}")
        search_phrase = question
        p_filters = []
        
    print(f"\n🔍 [Query Analyse] Suchphrase: '{search_phrase}' | Aktive Filter-§: {p_filters}")
    search_kwargs = {"k": 5}
    has_filter = False

    if p_filters:
        clean_filters = [str(p).replace("§", "").strip() for p in p_filters if p]
        if clean_filters:
            has_filter = True
            if len(clean_filters) == 1:
                search_kwargs["filter"] = {"paragraph": clean_filters[0]}
            else:
                search_kwargs["filter"] = {"$or": [{"paragraph": p} for p in clean_filters]}

    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
    docs = retriever.invoke(search_phrase)

    if not docs and has_filter:
        print("  🔀 [Fallback] Keine Dokumente mit Metadaten-Filter gefunden. Starte ungefilterte Vektorsuche...")
        fallback_kwargs = {"k": 3}
        fallback_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs=fallback_kwargs)
        docs = fallback_retriever.invoke(search_phrase)

    return {"question": question, "docs": docs}


answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Beantworte die juristische Frage des Nutzers AUSSCHLIESSLICH basierend auf dem bereitgestellten Kontext. "
        "Wenn der Kontext die Antwort nicht hergibt, sage das sachlich. "
        "Gehe strukturiert vor, nenne immer die exakte Quelle (§, Absatz, Nummer) und bleibe rechtssicher.\n\n"
        "Kontext:\n{context}"
    ),
    ("human", "{question}")
])

def generate_answer(input_data: dict):
    if not isinstance(input_data, dict) or "docs" not in input_data:
        return {"answer": "Fehler in der Verarbeitungskette.", "docs": []}

    question = input_data["question"]
    docs = input_data["docs"]

    if not docs:
        return {"answer": "Ich konnte keine passenden Dokumente finden.", "docs": []}
    
    context_str = "\n\n".join([doc.page_content for doc in docs])
    prompt = answer_prompt.format(context=context_str, question=question)
    response = llm_answer.invoke(prompt)
    
    # KORREKTUR: Wir geben jetzt die Antwort UND die Dokumente als Dictionary zurück
    return {"answer": response.content, "docs": docs}

# Pipeline Definition bleibt gleich, reicht aber jetzt das Dictionary weiter
rag_chain = (
    {"question": RunnablePassthrough()}
    | RunnableLambda(retrieve_documents)
    | RunnableLambda(generate_answer)
)



# ============================================================
# STREAMLIT UI INTERFACE
# ============================================================
st.set_page_config(page_title="Legal RAG System", page_icon="⚖")
st.title("⚖ Legal RAG System")
st.caption("Rechtssichere Antworten basierend auf deiner Chroma-Datenbank")

# Chat-Verlauf im Session-State initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []

# Alten Chat-Verlauf inklusive Quellen anzeigen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Wenn Quellen für diese Nachricht existieren, zeige sie im Expander an
        if "docs" in message and message["docs"]:
            with st.expander("📚 Gefundene Gesetzestexte & Quellen"):
                for i, doc in enumerate(message["docs"]):
                    paragraph_info = doc.metadata.get('paragraph', 'Unbekannt')
                    st.markdown(f"**Quelle {i+1}: Paragraph {paragraph_info}**")
                    st.caption(doc.page_content)

# Hauptfunktion für das UI-Handling
async def handle_ui():
    if user_input := st.chat_input("Ihre juristische Frage..."):
        # Nutzerwunsch im UI anzeigen
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Durchsuche Datenbank und generiere Antwort..."):
                try:
                    # Kette liefert nun das Dictionary
                    result = await rag_chain.ainvoke(user_input)
                    answer_text = result["answer"]
                    source_docs = result["docs"]
                    
                    # 1. Antwort anzeigen
                    st.markdown(answer_text)
                    
                    # 2. Quellen im ausklappbaren Menü anzeigen
                    if source_docs:
                        with st.expander("📚 Gefundene Gesetzestexte & Quellen"):
                            for i, doc in enumerate(source_docs):
                                # Holt die Paragraphen-Nummer aus den Chroma-Metadaten
                                paragraph_info = doc.metadata.get('paragraph', 'Unbekannt')
                                st.markdown(f"**Quelle {i+1}: Paragraph {paragraph_info}**")
                                st.caption(doc.page_content)
                    
                    # Im Verlauf speichern, damit es beim Reload nicht verschwindet
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer_text,
                        "docs": source_docs
                    })
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Fehler bei der Verarbeitung: {e}")

# Starte den asynchronen Streamlit-Zyklus
asyncio.run(handle_ui())

