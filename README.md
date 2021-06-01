# Multilabel Classification of 7 Different Cells: 

- I used Pytorch to build a CNN model to identify the seven different data cells, namely: "red blood cell", "difficult", "gametocyte", "trophozoite", "ring", "schizont", "leukocyte". My best model on the seventh day resulted in a 10.31149292 loss.

- A very low learning rate meant that the model did not learn the data at all, therefore I lowered the learning rate from 5e-3 to 0.001. I also used a small number of epochs in order to avoid overfitting of data. A small batch size was used to avoid memory issues, and lastly I used a drop out rate of 0.05 to get an improved loss value.

- I chose to use an Adam optimizer and BCE loss function for training the model.
