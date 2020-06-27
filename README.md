# NLP-automatic-text-generation-with-LSTM
The model was given an initial data sequence to learn from and this was:
"the scope of deep learning has been increasing at an exponential rate \n the reason deep learning has bloomed is hidden in the fact that their exists a vast number of applications in todays world that we take for granted \n from using hey siri on our iphone (trigger word detection) \n to using automatic replys on our gmail/linkedin (sentiment analysis) \n deep learning has automated our world without us even realising \n the world needs deep learning to sustain as it has become necessary"
The data sequence contained 59 unique words which were stored in a list and later tokens were generated for each of them.

To this, a seed text was give to predict the next 5 words, the seed given was:
"scope of artificial intelligence"
Note that the the model has never previously seen the word "artifical intelligence" before and does not even know its meaning, to this the model was able to predict the next 5 words as:
"scope of artificial intelligence learning has automated our world"
The sentence although is not 100% correct gramatically but we can somewhat infer the root meaning if it which matters the most.


