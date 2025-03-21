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
    parser = StrOutputParser()
    llm = ChatOllama(model="Llama3.2-Korean")
    memory = MemorySaver()
    embed_model = "snowflake-arctic-embed2"        


    ### 리트리버 생성 ###
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OllamaEmbeddings(model=embed_model))

    retriever = vectorstore.as_retriever()



    ### 리트리버툴 구축 ###
    tool = create_retriever_tool(
        retriever,
        "paper_retriever",
        "Searches and returns excerpts from the Artificial Intelligence paper.",
    )
    tools = [tool]


    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    config = {"configurable": {"thread_id": "abc123"}}



def generate_response(input_text):
    for s in agent_executor.stream(
        {"messages": HumanMessage(content=input_text)}, config=config
    ):
        pass
        # print(s)
        

    # print(f"User: {user_input}")
    s = s['agent']['messages'][0]
    st.info(parser.invoke(s))


### 메인 화면에서 챗봇 구동 ###
with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "I will summarize a paper in this page, present" +
        "the contents and explain the insights from it." +
        "In order to make 2 or 3 slides, give me some key" +
        "points of the paper. Especially, elaborate the method."
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
