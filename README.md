# 딥러닝 기반 논문 질의응답 챗봇
## 프로젝트 개요
- 이름: 페이퍼봇

- RAG을 활용하여 논문 내용을 기반으로 질의응답을 수행할 수 있는 챗봇 시스템을 개발하고, 연구자들이 논문 정보를 효과적으로 활용할 수 있도록 에이전트를 적용

- **기술스택**:python, LangChain, Ollama

## 설치 및 실행
1. **필수 요구사항**: python 3.11 이상, pip
2. **설치**
```bash
git clone https://github.com/dkimds/paperbot.git
cd paperbot
pip install -r requirement.txt
```

## 사용 방법
- Main function
```python
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
```
- Console
```bash
python paperbot.py
```
- Web
```bash
export LANGSMITH_API_KEY="..."
streamlit run paperbot_web.py
```
- Deployment
```bash
ngrok http --domain=violently-well-rabbit.ngrok-free.app 8501
```
## 라이선스
MIT License