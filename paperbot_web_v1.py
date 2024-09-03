import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader


st.title("🦜🔗 Welcome to Paperbot")

uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    ### 논문 삽입 ###
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    file_path = "./example_data/temp.pdf"
    with open(file_path, "wb") as f:
        f.write(bytes_data)
    loader = PyPDFLoader(file_path)
    docs = loader.load()



    ### 대형언어모델, 메모리, 파서 설정 ###
    parser = StrOutputParser()
    llm = ChatOllama(model="llama3.1")
    memory = MemorySaver()
    embed_model = "nomic-embed-text"        


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
        "What is annotation?"
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        generate_response(text)
