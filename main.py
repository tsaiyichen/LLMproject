from langchain_mistralai import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import dotenv
dotenv.load_dotenv()
import mysql.connector

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="12345678",
  database="LLMproject"
)

print(mydb)

user = "user"
model = "open-mistral-7b"
llm = ChatMistralAI(
    model=model,
    temperature=0.2,
    max_retries=2,
    max_tokens=1024,
)
mycursor = mydb.cursor(dictionary=True)

selectSQL = f"SELECT * FROM history WHERE userID = '{user}' AND modelID = '{model}' ORDER BY timestamp DESC LIMIT 10"
print(selectSQL)
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