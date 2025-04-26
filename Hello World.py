import os
from mistralai import Mistral
from dotenv import main
main.load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

model = "pixtral-12b-2409"
client = Mistral(api_key=api_key)

chat_response = client.chat.complete(
model = model,
messages = [
{
"role": "user",
"content": "你會說中文嗎？",
},
]
)
print(chat_response.choices[0].message.content)
'''
import numpy as np
np.random.seed(42)
sentence = "The New York Rangers beat the Pittsburgh Penguins last night" # This is your input sentence
words = sentence.lower().split()
int2word = {k: v for (k, v) in enumerate(set(words))} # get integer of every word (same word, same index)
word2int = {v: k for (k, v) in int2word.items()} #give every word there integer in index order.
vocab_size = len(word2int)
embedding_dim = 3 #This is d_k, you can dertermine dimension
embedding_layer = np.random.randn(vocab_size, embedding_dim) # here we use random values, to give every words an vector value
tokens = [word2int[w] for w in words]
# Here we don't use any embedding model, just transfer them into numeric value
embeddings = np.asarray([embedding_layer[idx] for idx in tokens])
print(sentence)
print(tokens)
print(embeddings)
# weights to calculate (Q, K, V) # Here we use random value to give them weights
w_q = np.random.random((embedding_dim, 3))
w_k = np.random.random((embedding_dim, 3))
w_v = np.random.random((embedding_dim, 3))
# calculate (Q, K, V), each as a seperate linear transform of the same input
Q = embeddings @ w_q
K = embeddings @ w_k
V = embeddings @ w_v
print("Embeddings")
print(embeddings)
print("Query")
print(Q)
print("Keys")
print(K)
print("Values")
print(V)
def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    x = np.exp(x - np.amax(x, axis=axis, keepdims=True))
    return x / np.sum(x, axis=axis, keepdims=True)
# calculate attention scores as dot product between Q and K
scores = Q @ K.T # (n x n) matrix
print(scores.shape)
# divide by dimensionality of K, and pass through softmax operation
scores = softmax(scores / K.shape[1]**0.5, axis=1)
# multiple attention scores with our valuese (V). this tells us how much to "attend" to our values
attention_output = scores @ V
print("Attention Output:")
print(attention_output)
def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    scores = np.dot(q, k.T)
    scores = softmax(scores / k.shape[1]**0.5, axis=1)
    return np.dot(scores, v)

num_heads = 2 # Here, we use 2 heads
# split each of Q, K, V into 'num_heads' chunks
# in reality, Q, K, V are projected 'num_heads' times,
# with each having a dimensionality of d_K / num_heads
Q_heads = np.array_split(Q, num_heads)
K_heads = np.array_split(K, num_heads)
V_heads = np.array_split(V, num_heads)
mha = [] # multi_headed_attention
for q, k, v in zip(Q_heads, K_heads, V_heads):
    mha.append(attention(q, k, v))

mha = np.concatenate(mha)

print(f"Multi-Head Attention With {num_heads} Heads:")
print(mha)
'''