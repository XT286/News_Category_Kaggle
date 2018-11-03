# News_Category_Kaggle
This is from a Kaggle Competition. 

Link to the Competition: https://www.kaggle.com/rmisra/news-category-dataset/home

The python code uses NLP to train a Text CNN model in order to categorize news based on their headlines and a short description of the contents. For the NLP we used the GloVe embedding to construct our embedding matrix.

We further train our model using a bidirectional GRU and a bidirectional LSTM. This noteboook only shows the first 10 epoch trained in the three Neural Network models, we can conclude a few observations: 1) speed-wise: CNN fastest, BiGRU slower, BiLSTM slowest (This is not directly relates to the number of parameters but to the number of computation needed for gradient descent); 2) performance: BiLSTM > BiGRU > CNN (It is not obvious from the first 10 epoch, but one will observe a difference when we train using 15 or more epoch. For instance, the BiGRU improves the validation accuracy from 58.15% (CNN) to 63.77%.)

In general, the performance of these methods are not of satifaction. I am currently implementing LSTM with attention and hopefully this can give us another boost. Stay tuned!
