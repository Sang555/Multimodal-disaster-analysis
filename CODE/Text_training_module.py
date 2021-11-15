### This is the LSTM training module ###

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
import numpy as np
import pandas as pd
from matplotlib import pyplot

data = pd.read_csv('C:/Users/Swars/Desktop/FYP/FYP/processed_req_offer.csv')
column = data.label

for i in range(len(column)):
    if data.label[i]=='request':
        data.label[i]=0
    elif data.label[i]=='offer':
        data.label[i]=1
    elif data.label[i]=='RO':
        data.label[i]=2
    

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['tweettext'])
vocabulary_size=len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(data['tweettext'])
#print sequences[0]


#TRAINING USING GLOVE PRETRAINED
embeddings_index= dict()
f=open('C:/Users/Swars/Desktop/FYP/glove.6B.100d.txt',encoding='utf8')
for line in f:
		values=line.split()
		word=values[0]
		coefs=np.asarray(values[1:],dtype='float32')
		embeddings_index[word]=coefs
f.close()

embedding_matrix=np.zeros((vocabulary_size,100))
for word, i in tokenizer.word_index.items():
	embedding_vector=embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i]=embedding_vector

df = pad_sequences(sequences, maxlen=30)
model = Sequential()
model.add(Embedding(vocabulary_size, 100,weights=[embedding_matrix], input_length=30,trainable=False))
model.add(Bidirectional(LSTM(100,return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
model.add(Bidirectional(LSTM(50, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history=model.fit(df, np.array(data.label), validation_split=0.2, epochs=20)
model.summary()

X=['hi hike is donating rs 10 lacs towards kashmir flood relief go to menu gt rewards gt donate for kashmir and make it count kashmirflood','india offers help to pakistan on floods']
tokenizer.fit_on_texts(X)
sequence = tokenizer.texts_to_sequences(X)
#print sequences[0]
seq = pad_sequences(sequence, maxlen=30)
Y=model.predict_classes(seq)
print (Y)
print ("Training completed 100%")
#score=model.evaluate(xtest,ytest,batchsize)
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()
#plot_model(model, to_file='model_plot1.png', show_shapes=True, show_layer_names=True)