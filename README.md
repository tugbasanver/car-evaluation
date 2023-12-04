# car-evaluation
A comparative research on neural networks for a multiclass classification problem 

Car Evaluation: A Comparative Research on Neural Networks for a Multiclass Classification Problem 


1.	Introduction:
   
Classification is one of the most researched areas to acquire successful solutions for decision making systems. A classification problem for a decision making consists of a dataset which has many attributes and a target to classify correctly. For many classification problems, the target variable which has to be classified is binary, it means those are basically “Yes/No” type of decision making problems[1][2]. However, some classification problems have a target variable which has more than two categories, those problems called as multiclass classification problems. In this paper, a critically evaluated comparison of the performances of two given models which will be trying to solve a decision making problem which is “Car Evaluation” based on  6 specifications of a given car was derived from a simple hierarchical decision model. In this manner, Multilayer Perceptron(MLP) model which is a type of Artificial Neural Network(ANN) and Support Vector Machine(SVM) will be researched and compared.  

2.	Data set:
   
Car Evaluation Dataset from UCI Machine Learning Repository is used to analyse and solve the problem[3]. The dataset has 1728 values and 7 attributes. The attributes are following concept structure:
Buying price(buying), price of the maintenance(maint), number of doors(doors), capacity in terms of persons to carry(persons), the size of luggage boot(lug_boot), estimated safety of the car(safety) and car acceptability(decision), the target variable. Almost all variables are categorical type as seen below:
buying: vhigh, high, med, low.
maint: vhigh, high, med, low.
doors: 2, 3, 4, 5more.
persons: 2, 4, more.
lug_boot: small, med, big.
safety: low, med, high
decision: unacc, acc, good, vgood
At the beginning, the data is split into train and test data sets. It is significantly essential to keep test data set untouched without any effect of encoding or oversampling techniques. Those techniques may cause data leakage between train and test data sets if the data set is not split before those processes[4].
As almost all variables are categorical, it is crucial to encode all variables to numerical values to conduct a proper classification modelling. OneHotEncoder function in Python is used to encode predictor variable into binary vector variables and LabelEncoder in Python is used to encode target variable into numerical labels. 
For a multiclass classification problem, it is significant to figure out the distribution of the target variable among the different classes. If the classification categories are not approximately equally represented in a dataset it means it may cause an imbalanced classification which is that there are too few examples of the minority class for a model to effectively learn the decision boundary. One way to solve this problem is to oversample the examples in the minority class. Among several oversampling techniques, SMOTE approach works effectively by drawing a line between the examples in the feature space and drawing a new sample at a point along that line.is chosen to rebalance the training data[5]. When the distribution of the target variable of this given dataset is analysed, it is undoubtedly clear to figure out it is imbalanced as seen in the Figure 1. The percentage of the labels in the target variable is ‘unacceptable’ with 68.7%, ‘acceptable’ with 23.5%, ‘good’ with 4.0% and ‘very good’ with 3.8, respectively.

3.	Neural Network Models:

a)	Multilayer Perceptron(MLP)

Multilayer perceptron (MLP) model is a subclass of feedforward artificial neural network (ANN). As an artificial neural network, the multilayer perceptron does not need to make no assumptions beforehand based on the data distribution to conduct a prediction for unseen data, unlike other statistical techniques. For this reason, it is highly attractive technique comparing other predictive models. The multilayer perceptron contains a system of several layers of simple interconnected neurons. The input layer passes the input vector to the network. A multilayer perceptron may have one or more hidden layers and an output layer. Multilayer perceptrons fully connected, all layers are connected each other in several ways. Neurons are connected by weights and input/output vectors by some calculated activation functions[6]. Multilayer Perceptron algorithms are widely used in multiclass decision making classification problems. 
Analysing the advantages of MLP structure, it implements the global approximation strategy usually uses significantly small number of hidden neurons. However, the test data sets need more hidden units and it is needed to make the number of the hidden units small to control complexity of the algorithm counts as a disadvantage of MLP structures[7].

b)	Support Vector Machines(SVM)

SVM is a supervised machine learning model which helps to solve classification or regression problems. SVM tries to maximise the separation boundaries between data points based on the labels with selected kernel function and penalising parameters. Support Vector Machines were designed for binary classification and do not natively support classification tasks with more than two classes. To adopt this modelling to multiclass classification problems, there are two approaches called as One-vs-Rest(also called as One-vs-All) and One-vs-One. In this paper, One-vs-Rest approach is chosen. This approach offers splitting the multi-class dataset into multiple binary classification problems. A binary classifier is then trained on each binary classification problem and predictions are made based on the model which is the most successful[8].
SVMs acquire the optimum separating which is very efficient on unseen data points and it makes it the best classifier in most cases, as a strong advantage. For another pro, complexity of SVM does not depend on the data dimensionality. On the other side the SVM uses local approximation strategy with large number of hidden units, as a con. The one of the main advantage of SVM is it reduces the number of operations in the learning problem as leading to the quadratic optimization task[7]. It makes SVM algorithm is usually much quicker as it is seen in this paper’s research.
	

4.	Methodology and Architecture

As a beginning methodology, the original data set split into 80% training data and 20% test data and test data is kept as an Excel sheet to make the further reference loading easy. After encoding relevant variables into numerical form, SMOTE approach is used to rebalance the training data as mentioned in the Data Set part. 
a)	MLP Methodology, Architecture and Parameter Selection: 
For model selections, some approaches are considered. For MLP model, the first step is deciding hidden layer number and sizes. According to previous researches and approaches, just one hidden layer may be considered as ideal small to medium level of a classification problem[9]. Since the data set has only 6 predictor variables, 4 labels in the target variable and 1700 samples, it is decided that only one hidden layer may be good choice to avoid complexity of the neural network, and a consequently probable overfitting. Deciding hidden layer size, since too few neurons in the hidden layers will cause underfitting and too many neurons cause overfitting, hidden layer size options reduced to 11, 21 and 42 considering encoded input size of the training data is 21. 
As a feedforwarding function for the MLP model, tanh() function which calculates the hyperbolic tangent of the elements of input and is convenient for classification problems is used. Also softmax() function which is a generalization output a multiclass categorical probability distribution is used as a last activation function. 
Additionally, dropout method which is a regularisation which employs training neural networks with different architectures in parallel is used as a parameter. Dropout adding noise to the training process. It creaks the process when layers adapt to correct mistakes from prior layers, thus the model becomes more robust and avoids overfitting the training data. As dropout values, the most appropriate range is between 0,5 to 0,8[10]. Therefore, these two values are applied during experimental works. 
As a criterion, Early Stopping technique is considered which can stop training large nets when they have learned models similar to those learned by smaller nets. It causes much less loss in generalization performance with excess capacity using early stopping criterion[11].
When looking other hyperparameters, the learning rate, a configurable hyperparameter which controls how instantly the model is adapted to the problem, is chosen. According to some researches, learning rate may become the most important hyperparameter in the neural networks[12]. Larger learning rates may cause instantaneous changes and require fewer training epochs, therefore model can converge too quickly. On the other hand smaller values require more training epochs for the smaller changes made to the weights, and too small values may cause the process to get stuck. Hence, three learning rate which are 0.1, 0.01 and 0.001 are considered for experimental results. 
Looking for another hyperparameter, neural network optimisers which are algorithms to update the various parameters that can reduce the loss in much less effort, several options are discussed. Most popular optimisation choices are momentum, Adam and SGD(Stochastic gradient descent) for MLP models. Adam optimiser, which calculates individual adaptive learning rates and momentum value for different parameters from estimates of first and second moments of the gradients, is highly efficient and successful on train a deep or complex neural network with fast convergence[14]. Beside this, SGD optimiser are usually much more reliant on a robust initialization to achieve to find a minimum, but it might take significantly longer than Adam[15]. These two optimisers are chosen to compare for the experiment. 
In order to compare all those experimental choices and find the best hyperparameters efficiently and rapidly, grid search technique is used. Stratified cross validation approach using 3 folds is considered to make an efficient and robust fitting with higher accuracy and specificity for the models on the grid search. Also, maximum epoch number for each fold is set to 50 to avoid longer training times and overcome possible overfitting problem which may occur with too many epochs[16].
b)	SVM Methodology, Architecture and Parameter Selection:
As mentioned above, to adopt SVM into the multiclass classification problem, OnevsRestClassifier technique is used as a SVM classifier for this work. SVM has not artifical neural network structure but it has own type of kernel parameters to optimise the algorithm. Considered SVM hyperparameters are C, gamma and kernel type.
 Parameter C represents the error penalty for misclassification for SVM. It controls the trade-off between smoother hyperplane and misclassifications. It is significant to avoid the overfitting with allowing some misclassification. Thus, choosing right C value can be complicated. Higher value may cause higher penalty which may trigger overfitting. 0.05, 0.1,0.5 and 1, 5 values are tried in this experiment. 
Parameter gamma controls margin softness or hardness, which can control overfitting directly. Chosing soft margin which means lower values of gamma instead of a hard margin can help to avoid overfitting because the higher the gamma, the higher the hyperplane tries to match the training data[8]. 5, 1, 0.5, 0.05 and 0.01 are considered.
Kernel choice is other significant parameters in SVM as the kernel-function defines the hyperplane chosen for the problem. Since our problem is not much complicated, simpler kernels may be the best fit. Thus, linear and polynomial are chosen. 
To pack all of these, grid search methodology with stratified cross validation approach using 3 folds is conducted for SVM like in the MLP model.

5.	Results, Findings & Evaluation

a)	Model Selection, Fittings and Evaluation for MLP
After grid search for the MLP model, the best models have around 97% accuracy with the learning rate of 0.01, dropout value of 0.5, hidden layer size of 11 and Adam optimiser. Also, it can be seen as mean time score for the fittings are good, it means the best models can run significantly fast. It was total 36 model fits for MLP modelling, the best 20 fits order by mean test score can be seen in figure 2 below. Test score through grid search fluctuated the range of 97% to 27%, which is quite poor. It indicates unstable performance of the MLP model.    

 
After choosing best parameters for the estimator, it is meaningful to analyse the training versus validation loss curves of the selected model to understand the performance of the fitting. As it is seen Figure 3, the shape of the loss curves through epochs shows the model is perfectly fit. After the last cross validation fit with the best estimator, it is achieved 0.97 accuracy, which is high enough. 
In the testing part the best model is loaded and fitted the unseen test data and achieved 0.93 accuracy score which is quite reliant considering 0.97 training accuracy score. 


The Receiver Operating Characteristic (ROC) curve is a standard technique for summarising classifier performance over a range of between true positive and false positive error rates. The Area Under the Curve (AUC) is an accepted traditional performance metric for a ROC curve[17]. As it is seen as in Figure 5, True Positive Rate is significantly high and close to 1. It is indicator if a robust prediction.

b)	Model Selection, Fittings and Evaluation for SVM
After grid search for the SVM model, the best models have around 99.9% accuracy with the C parameter of 1, gamma parameter of 0.5 and polynomial kernel type as it can be seen as in Figure 6. There are 51 fits in total but only best 17 ones are showed in the figure. Test score through grid search changes the range of 99% to 85%, which is not poor. It shows the robustness of the SVM model.   
To analyse the learning habit of the training algorithm for chosen SVM model, learning curves for the scores through training examples, scalability of the model considering fitting times through training examples and performance of the model based on score through fitting times are plotted. According to those plots, selected SVM model shows very high performance in the training set as seen as in Figure 7. However, it is crucial to show a similar performance in the test data to prove the high prediction performance of the model.  


When the selected model for SVM is loaded and fitting in the test data, it shows a roaring performance of accuracy as 99%. Also confusion matrix and ROC-AUC curve can be seen as in Figure 8 and 9, respectively. The selected model predicts almost all unseen data correctly.



6.	Conclusions, Discussion and Future Works

This study aimed to compare model performances of the two selected models which are Multilayer Perceptron and Support Vector Machines. It is reviewed this aim with prediction qualities such as accuracy and Roc-auc values derived from the ROC-AUC curve. The selected SVM model has 99.9% accuracy on both training and testing datasets, which means predicted almost every single unseen data correctly. The selected MLP model has a very good 97% accuracy on training data but slightly lower 93% accuracy for the unseen test data. Additionally, average fitting time for each fit is lower for MLP model than SVM model, however overall SVM grid search is undoubtedly faster than MLP grid search with a breakthrough 35 second comparing 19 minutes for MLP. As a hypothesis statement, the SVM model has better performance for multiclass data of this research on both accuracy and fitting time than the MLP model. 
This study shows how grid search with selected number of folded cross validation eases the model selection for both models. Also the importance of balanced data set through target labels and its comparison to the imbalanced data set led this project to importance of oversampling techniques. In this manner, SMOTE oversampling technique is reviewed and its efficiency is proved. 
As a discussion, the selection of the data set for this project may be simpler than it is needed for an artificial neural network, which is a bit “data hungry”. MLP can achieve a better result if the data set has more sample. 
For a future work, more hidden layer and different optimiser parameters such as Adagrad or Nadam and weight decay can be analysed for MLP models to improve model performance considering accuracy, Roc curve and fitting times.

7.	References

1.	M. Bohanec and V. Rajkovic: Knowledge acquisition and explanation for multi-attribute decision making. In 8th Intl Workshop on Expert Systems and their Applications, Avignon, France. pages 59-78, 1988.
2.	F. Thabtah, P. Cowling and Y. Peng, "MCAR: multi-class classification based on association rule," The 3rd ACS/IEEE International Conference onComputer Systems and Applications, 2005., Cairo, Egypt, 2005, pp. 33-, doi: 10.1109/AICCSA.2005.1387030.
3.	Car Evaluation Data Set, https://archive.ics.uci.edu/ml/datasets/car+evaluation
4.	https://towardsdatascience.com/avoid-data-leakage-split-your-data-before-processing-a7f172632b00
5.	Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. Journal of artificial intelligence research, 16, 321-357.
6.	M.W Gardner, S.R Dorling, Artificial neural networks (the multilayer perceptron)—a review of applications in the atmospheric sciences, Atmospheric Environment, Volume 32, Issues 14–15, 1998, Pages 2627-2636,ISSN 1352-2310.
7.	E.A. Zanaty, Support Vector Machines (SVMs) versus Multilayer Perception (MLP) in data classification, Egyptian Informatics Journal, Volume 13, Issue 3, 2012, Pages 177-183, ISSN 1110-8665.
8.	Noble, W. S. (2006). What is a support vector machine?. Nature biotechnology, 24(12), 1565-1567.
9.	https://www.heatonresearch.com/2017/06/01/hidden-layers.html
10.	Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, Ruslan Salakhutdinov(2014). Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
11.	Caruana, R., Lawrence, S. and Giles, L., 2000, November. Overfitting in neural nets: Backpropagation, conjugate gradient, and early stopping.
12.	Ian Goodfellow, Yoshua Bengio, Aaron Courville, Deep Learning (Adaptive Computation and Machine Learning series), 2016.
13.	CS231n: Convolutional Neural Networks for Visual Recognition-Stanford University- Spring 2021. Web link: https://cs231n.github.io/neural-networks-3/
14.	Sebastian Ruder (2016). An overview of gradient descent optimisation algorithms. arXiv preprint arXiv:1609.04747
15.	Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
16.	Murphy, K. P. (2012). Machine Learning: A Probabilistic Perspective. United Kingdom: MIT Press.
17.	Duda, Hart, & Stork, 2001; Bradley, 1997; Lee, 2000.



