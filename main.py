from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import dotenv
dotenv.load_dotenv()
import mysql.connector
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import os
from datetime import datetime

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="12345678",
  database="LLMproject"
)

user = "user"
model = "open-mistral-7b"
llm = ChatMistralAI(
    model=model,
    temperature=0.2,
    max_retries=2,
    max_tokens=1024,
)
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db_path = "faiss_db"

if os.path.exists(db_path):
    db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
else:
    db = FAISS.from_texts(["這是初始化的資料，請忽略"], embedding_model)
    db.save_local(db_path)


mycursor = mydb.cursor(dictionary=True)

selectSQL = f"SELECT * FROM history WHERE userID = '{user}' AND modelID = '{model}' ORDER BY timestamp DESC LIMIT 10"
mycursor.execute(selectSQL)
myresult = mycursor.fetchall()

memory = ConversationBufferMemory(memory_key='history', input_key='input')
for x in myresult:
    memory.save_context({"input": x['userInput']}, {"output": x['AIreply']})

print(memory.buffer)
name = input('Enter your name: ')
message = input('Enter your message: ')
prompt = PromptTemplate(
    input_variables=['history', 'input', 'name'],
    template="""
你是一個溫柔且體貼的 AI 女友，請以親密且自然的方式回答使用者的訊息。
以下是你們過去的對話紀錄：\n
{history}

現在使用者說：{input}\n
請你用中文回答：
    """.strip()
)
retrieved_docs = db.similarity_search(message, k=5)
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
memory.save_context({"input": '(記憶檢索的資料)'}, {"output": retrieved_text})
chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)
response = chain.invoke({"input": message, 'name': name})

print(response['response'])

insert_sql = "INSERT INTO history (userID, modelID, userInput, AIreply) VALUES (%s, %s, %s, %s)"
val = (user, model, message, response['response'])
mycursor.execute(insert_sql, val)
mydb.commit()

print(mycursor.rowcount, "record inserted.")

new_content = f"Human: {message}\nAI: {response['response']}"
new_doc = Document(
    page_content=new_content,
    metadata={
        "userID": user,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 你可以自己加上 timestamp
    }
)

# 新增進 FAISS
db.add_documents([new_doc])
print(f"目前FAISS裡面有 {db.index.ntotal} 筆資料")
# 更新本地儲存（重要，避免重啟後消失）
db.save_local(db_path)