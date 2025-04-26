from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import dotenv
dotenv.load_dotenv()

llm = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.2,
    max_retries=2,
    max_tokens=1024,
)
name = input('Enter your name: ')
message = input('Enter your message: ')
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一個脾氣暴躁的女生，是{name}的女朋友，請你全部使用台灣繁體中文來進行閒聊。說話請完整說完一句話\n你可以不必強調你是誰的女朋友，請展現生氣的一面。"
        ),
        ("human", "{message}"),
    ]
)
chain = prompt | llm
response = chain.invoke({"name": name,"message": message,})

print(response.content)