import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
import gc

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

st.title("🦜🔗 Welcome to Paperbot")

uploaded_file = st.sidebar.file_uploader("Choose a file")
with st.sidebar:
    clear_chat = st.button('Clear chat')
if clear_chat:
    st.session_state.messages = []
    gc.collect()


if uploaded_file is not None:
    ### 논문 삽입 ###
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    file_path = "./temp.pdf"
    with open(file_path, "wb") as f:
        f.write(bytes_data)
    loader = PyPDFLoader(file_path)
    docs = loader.load()



    ### 대형언어모델, 메모리, 파서 설정 ###
    llm = ChatOllama(model="llama3.2")
    embed_model = "snowflake-arctic-embed2"        


    ### 리트리버 생성 ###
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=OllamaEmbeddings(model=embed_model))

    retriever = vectorstore.as_retriever()



    ### 리트리버툴 구축 ###
    tool = create_retriever_tool(
        retriever,
        "paper_retriever",
        "Search for information about a paper. For any questions about the paper, you must use this tool!",
    )
    tools = [tool]

    # Get the prompt to use - you can modify this!
    prompt = hub.pull("hwchase17/openai-functions-agent")
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    agent_with_chat_history = RunnableWithMessageHistory(
        agent_executor,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    config = {"configurable": {"session_id": "abc123"}}
    

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

chat_instruction = "What is up?" 

if prompt := st.chat_input(chat_instruction):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        response = agent_with_chat_history.invoke(
            {"input": prompt},
            config=config,
        )
        st.markdown(response['output'])
        
    st.session_state.messages.append({"role": "assistant", "content": response['output']})

    
