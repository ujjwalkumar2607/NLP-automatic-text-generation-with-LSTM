#importing necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np

#instantiating the tokenizer
tokenizer = Tokenizer()

#storing the training sequence in a single string variable
sentence = "the scope of deep learning has been increasing at an exponential rate \n the reason deep learning has bloomed is hidden in the fact that their exists a vast number of applications in todays world that we take for granted \n from using hey siri on our iphone (trigger word detection) \n to using automatic replys on our gmail/linkedin (sentiment analysis) \n deep learning has automated our world without us even realising \n the world needs deep learning to sustain as it has become necessary"
corpus = sentence.lower().split("\n") #converting the sentence to lowercase and storing each word as a separate iterable string in a list

tokenizer.fit_on_texts(corpus) #creates tokens of each words as a dictionary with key being word and the value beings its token
total_words = len(tokenizer.word_index) + 1 #calculating total number of words in the initial sentence

print(tokenizer.word_index)
print(total_words)

input_sequences = [] #training features (xs) will be a list

for line in corpus:
	token_list = tokenizer.texts_to_sequences([line])[0] #converts each sentence as its tokenized equivalent
	#print(token_list)
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1] #generating n gram sequences: 1st two words in the token_list is one sequence, next two words are another sequence and so on
		#print(n_gram_sequence)
		input_sequences.append(n_gram_sequence) #appending each n gram sequence to the list of our features (xs)
		#print(input_sequences)
#print("The training features are:\n",input_sequences)	

# pad sequences 
max_sequence_len = max([len(x) for x in input_sequences]) #calculating the length of the longest sequence in our training features (xs)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')) #pre-pading each value of the input_sequence
#print(input_sequences)
# create predictors and label
xs, labels = input_sequences[:,:-1],input_sequences[:,-1] #creating xs and their labels using numpy slicing
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words) #creating one hot encoding values of each labels to make our ys

print(tokenizer.word_index)

model = Sequential() #creating a sequential model
  model.add(Embedding(total_words, 64, input_length=max_sequence_len-1)) #adding an embedding layer with 64 as the embedding dimension
  model.add(Bidirectional(LSTM(20))) #adding 20 LSTM units
  model.add(Dense(total_words, activation='softmax')) #creating a dense layer with 54 output units (total_words) with softmax activation
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) #compiling the model with adam optimiser
  history = model.fit(xs, ys, epochs=500, verbose=1) #training for 500 epochs

#plotting the training accuracy of the model
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.xlabel("Number of Epochs")
plt.ylabel('Training accuracy per epochs')
plt.show()

#predicting the next word using an initial sentence
seed_text = "scope of artificial intelligence"
next_words = 5
  
for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([seed_text])[0] #converting our seed_text to tokens and excluding the out of vcabulary words
	token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre') #padding the seed_text
	predicted = model.predict_classes(token_list, verbose=0) #predicting the token of the next word using our trained model
	output_word = "" #initialising output word as blank at the beginning
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word #converting the token back to the corresponding word and storing it in the output_word
			break
	seed_text += " " + output_word
print(seed_text)
