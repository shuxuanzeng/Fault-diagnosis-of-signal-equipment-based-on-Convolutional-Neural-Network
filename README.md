# Fault-diagnosis-of-signal-equipment-based-on-Convolutional-Neural-Network
As an important part of the railway operation system, the safe and reliable operation of
signaling equipment directly affects the safety and transportation efficiency of trains. Traditional
signal equipment fault diagnosis relies on the personal experience and knowledge of on-site
maintenance personnel, diagnosis efficiency is low, easy to cause judgment errors, with the
development of intelligent fault diagnosis methods, the use of new technology to establish an
efficient and reliable fault diagnosis system to promote the rapid development and safe operation
of the railway is of great significance.
In this thesis, a fault diagnosis of signal equipment based on convolutional neural network
is proposed, taking the turnout as the research object, directly extracting the fault features from
the original time series signal, and then using the gram angle field to convert it into a two-
dimensional image, and adjusting the parameters and structure of the convolutional neural
network through training and testing to determine the network model. The specific work is as
follows:
The first is the analysis of the action current curve and data preprocessing. After introducing
the structure of the turnout and the working principle of ZDJ9 switch machine, the normal
operation of the turnout and several common fault conditions are summarized, and the curve is
denoised and normalized.
Secondly, a fault diagnosis method based on LeNet-5 is proposed. The LeNet-5 model is
used to summarize the basic principles of convolutional neural networks, and the one-
dimensional turnout simulation data is converted into two-dimensional images and then fed into
the convolutional neural network for training, and a slight overfitting is found. Later, the
classification accuracy was improved by using Dropout optimization, which proved the
feasibility of the optimization method.
Thirdly, a fault diagnosis method based on ResNet50d is proposed. For the two-dimensional
image data, by referring to the ResNet model structure, the network is deepened, and the residual
structure is used to extract the features of the turnout data for training the network model.
Compared with the LeNet-5 model, the high accuracy of ResNet50d indicates that the residual
network plays a role in the complex working environment of turnouts, which can improve the
efficiency of turnout fault diagnosis.
Finally, the results show that both models have obvious effects on diagnosis. It shows that
the convolutional neural network can use multiple convolutional layers and the nonlinear


Reference: https://blog.csdn.net/ECHOSON/article/details/141068765?spm=1001.2014.3001.5502
activation function of each layer to automatically extract features from input samples, greatly
reduce the complexity of the network, have strong classification ability, and overcome the
weakness of incomplete feature extraction by artificial use of expert knowledge, which has
important application value in ensuring the safe and reliable operation of railways.
Keyword: Convolutional Neural Network; fault diagnosis; action current curves; turnout;
feature extraction
