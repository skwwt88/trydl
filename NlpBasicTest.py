from keras.preprocessing.text import *
from keras.preprocessing.sequence import skipgrams

text = ""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

word2id = tokenizer.word_index
id2word = {v:k for k, v in word2id.items()}

print word2id

print word2id["love"]
print id2word[3]

print text_to_word_sequence("love green")