# News_Category_Kaggle
This is from a Kaggle Competition. 

Link to the Competition: https://www.kaggle.com/rmisra/news-category-dataset/home

The python code uses NLP to train a Text CNN model in order to categorize news based on their headlines and a short description of the contents. For the NLP we used the GloVe embedding to construct our embedding matrix.

We further train our model using a bidirectional GRU and a bidirectional LSTM. This noteboook only shows the first 10 epoch trained in the three Neural Network models, we can conclude a few observations: 1) speed-wise: CNN fastest, BiGRU slower, BiLSTM slowest (This is not directly relates to the number of parameters but to the number of computation needed for gradient descent); 2) performance: BiLSTM > BiGRU > CNN (It is not obvious from the first 10 epoch, but one will observe a difference when we train using 15 or more epoch. For instance, the BiGRU improves the validation accuracy from 58.15% (CNN) to 63.77%.)

In general, the performance of these methods are not of satifaction. I am currently implementing LSTM with attention and hopefully this can give us another boost. Stay tuned!

Update (11/4): I have uploaded a Python notebook on LSTM with Attention. The validation accuracy was better than the other RNN models but it then very quickly stablized around 61%. The trainning accuracy kept increasing and soon crosses out the validation accuracy. This is unexpected and it left room for improvement.

Update (11/7): I finally run the four models (CNN, BiGRU, LSTM, LSTM with Attention) on a GPU. This allows me to run them for 20 epoches each and get more comprehensive results. I uploaded four plots and the result (after 20 epoches) for each is as follows:
CNN:                 loss: 1.5442 - acc: 0.5690 - val_loss: 1.4906 - val_acc: 0.5901
BiLSTM:              loss: 1.5175 - acc: 0.5720 - val_loss: 1.2830 - val_acc: 0.6296
LSTM with Attention: loss: 0.4631 - acc: 0.8523 - val_loss: 1.8956 - val_acc: 0.5911
BiGRU:               
Remark: We can observe that (from the plots) the validation accuracy of LSTM with Attention actually decline gradually after the 6th epoch, while the training acc keep climbing up. This is an obvious overfitting. Looking forward to fix this!
