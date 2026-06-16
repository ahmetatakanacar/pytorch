# attention
import numpy as np

d = {
      "apple": 10,
      "banana": 5,
      "chair": 2
}

query = "fruit"

def softmax(x):
      return np.exp(x) / np.sum(np.exp(x))

def get_word_vector(word, d_k=8):
    return np.random.normal(size=(d_k,))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def attention(q, K, v):
    return softmax(q @ K.T) @ v

def kv_lookup(query, keys, values):
    return attention(
        q = get_word_vector(query),
        K = np.array([get_word_vector(key) for key in keys]),
        v = values,
    )

print(kv_lookup("fruit", ["apple", "banana", "chair"], [10, 5, 2]))

d = {
    "apple": [0.9, 0.2, -0.5, 1.0],
    "banana": [1.2, 2.0, 0.1, 0.2],
    "chair": [-1.2, -2.0, 1.0, -0.2]
}

def softmax(x):
    # assumes x is a matrix and we want to take the softmax along each row
    # (which is achieved using axis=-1 and keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

def attention(Q, K, V):
    # assumes Q is a matrix of shape (n_q, d_k)
    # assumes K is a matrix of shape (n_k, d_k)
    # assumes v is a matrix of shape (n_k, d_v)
    # output is a matrix of shape (n_q, d_v)
    d_k = K.shape[-1]
    return softmax(Q @ K.T / np.sqrt(d_k)) @ V