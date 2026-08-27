import sys
import numpy as np 
import nltk 
from nltk.tokenize import word_tokenize 
import spacy 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Dense 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences 
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")
try:
      nltk.data.find("tokenizers/punkt")
except LookupError:
      nltk.download("punkt", quiet=True)
try:
      nlp = spacy.load("en_core_web_sm")
except OSError as error:
      raise RuntimeError(
            "Install the spaCy model with: python -m spacy download en_core_web_sm"
      ) from error
corpus = "One disadvantage of using 'Best Of' samping is that it may lead to limited exploration of the model's knowledge and creativity. By focusing on the most probable next words, the model might generate responses that are safe and conventional, potentially missing out on more diverse and innovative outputs. The lack of exploration could result in repetitive or less imaginative responses, especially in situations where novel and unconventional ideas are desired.To address this limitation, other sampling strategies like temperature-based sampling or top-p (nucleus) sampling can be employed to introduce more randomness and encourage the model to explore a broader range of possibilities. However, it's essential to carefully balance exploration and exploitation based on the specific requirements of the task or application." 
tokens = word_tokenize(corpus) 
lemmatized_tokens = [token.lemma_ for token in nlp(corpus)] 
all_tokens = tokens + lemmatized_tokens 
tokenizer = Tokenizer() 
tokenizer.fit_on_texts([" ".join(all_tokens)]) 
total_words = len(tokenizer.word_index) + 1 
input_sequences = [] 
token_list = tokenizer.texts_to_sequences([" ".join(all_tokens)])[0]
for i in range(1, len(token_list)):
      input_sequences.append(token_list[:i + 1])
max_sequence_length = max(len(seq) for seq in input_sequences) 
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre') 
X, y = input_sequences[:, :-1], input_sequences[:, -1] 
y = np.array(y) 
model = Sequential() 
model.add(Embedding(total_words, 100, input_length=max_sequence_length-1)) 
model.add(LSTM(100)) 
model.add(Dense(total_words, activation='softmax')) 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) 
model.fit(X, y, epochs=10, verbose=1)