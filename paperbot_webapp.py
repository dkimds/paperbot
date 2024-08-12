import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader


st.title("🦜🔗 Welcome to Paperbot")

openai_api_key = st.sidebar.text_input("Open AI Key", type="password")
uploaded_file = st.sidebar.file_uploader("Choose a file")
### 논문 삽입 ###

if (openai_api_key is not None) and (uploaded_file is not None):
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    file_path = "./example_data/temp.pdf"
    with open(file_path, "wb") as f:
        f.write(bytes_data)
    loader = PyPDFLoader(file_path)
    docs = loader.load()



    ### 대형언어모델, 메모리, 파서 설정 ###
    parser = StrOutputParser()
    memory = SqliteSaver.from_conn_string(":memory:")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
    # llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=openai_api_key)
        


    ### 리트리버 생성 ###
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(api_key=openai_api_key))

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
    st.info(s)


### 메인 화면에서 챗봇 구동 ###
with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What is annotation?"
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your Open AI key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text)
