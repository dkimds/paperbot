# "Write a file" with the following source-codes

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


### 논문 삽입 ###
file_path = "./example_data/2408.00714v1.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()



### 대형언어모델, 메모리, 파서 설정 ###
parser = StrOutputParser()
llm = ChatOllama(model="llama3")
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



### 터미널에서 챗봇 구동 ###
print("Welcome to the paperbot. If you want to quit, please enter 'exit'.")
config = {"configurable": {"thread_id": "abc123"}}

while True:
    # Input
    user_input = input("User: ")

    # 종료 입력하면 대화 종료
    if user_input.lower() == "exit":
        print("Thank you.")
        break

    # 응답 생성 및 출력

    for s in agent_executor.stream(
        {"messages": HumanMessage(content=user_input)}, config=config
    ):
        pass
        # print(s)
        

    # print(f"User: {user_input}")
    s = s['agent']['messages'][0]
    print(f"Assitant: {parser.invoke(s)}")
    print("----")
