# 딥러닝 기반 논문 질의응답 챗봇
## 프로젝트 개요
- 이름: 페이퍼봇

- RAG을 활용하여 논문 내용을 기반으로 질의응답을 수행할 수 있는 챗봇 시스템을 개발하고, 연구자들이 논문 정보를 효과적으로 활용할 수 있도록 에이전트를 적용

- **기술스택**:python, LangChain, Ollama

## 설치 및 실행
1. **필수 요구사항**: python 3.11 이상, pip
2. **설치**
```bash
git clone https://github.com/username/AI-master.git
cd AI-master
pip install -r requirement.txt
```

## 사용 방법
- Console
```bash
cd project
python paperbot.py
```
- Web
```
streamlit run paperbot_web.py
```
- Deployment
```
ngrok http --domain=violently-well-rabbit.ngrok-free.app 8501
```
## 라이선스
MIT License