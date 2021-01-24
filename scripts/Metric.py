import numpy as np
from forward_model import forward_model

class evaluation_metrics:
    '''
    "This class contains functions that can calculate
     the evaluation metrices
     using the labels and the predictions of the model"
    '''
    def __init__(self, labels, input, parameters):
        '''
        "constractor of evaluation_metrics"

        :type input:matrix[number_of_examples * inputsize] input of first layer
        :param input: WX+b

        :type labels:matrix[number_of_examples * inputsize]
        :param labels: true classes

        :type parameters:dict
        :param parameters: weights and biases
        '''
        self.input = input
        self.Y_true = labels
        self.parameters=parameters

    def confusionMatrix(self):
        '''
        "function returns the confusion matrix that is used to calculate FP , TP and FN."

        :return: confusion matrix to calculate accuracy, Precision, Recall, F1_score
        '''
        y_hat=[]
        predictions,packet_of_packets=forward_model().forward_model(self.input,self.parameters)
        predictions=predictions.T
        for i in range(predictions.shape[0]):
            max=np.argmax(predictions[i])
            y_hat.append(max)
        classes = set(self.Y_true[0])
        number_of_classes = len(classes)
        conf_matrix = pd.DataFrame(
            np.zeros((number_of_classes, number_of_classes), dtype=int),
            index=classes,
            columns=classes)
        for i, j in zip(self.Y_true[0], y_hat):
            conf_matrix.loc[i, j] += 1
        return conf_matrix.values, conf_matrix

    def TP(self):
        '''
        "True Positive : Number of times the model predicts positive and the actual label is positive"

        :return:True Positive -> list of diagonal of confusionMatrix
        '''
        values, cm = self.confusionMatrix()
        return np.diag(cm)

    def FP(self):
        '''
        "False Positive : Number of times the model predicts positive and the actual label is negative"

        :return: False Positive -> (summation of row of confusionMatrix) - list of True Positive
        '''
        values, cm = self.confusionMatrix()
        return np.sum(cm, axis=0) - self.TP()

    def FN(self):
        '''
        "False Negative - Number of times the model predicts negative and the actual label is positive"

        :return: False Negative -> (summation of column of confusionMatrix) - list of True Positive
        '''
        values, cm = self.confusionMatrix()
        return np.sum(cm, axis=1) - self.TP()

    def Accuracy(self, data_size):
        '''
        "calculate accuracy of training model"
        :param data_size: datasize of dataset(mnist, cifar-10)
        :return: accuracy of training model -> summation of True Positive / data_size
        '''
        return np.sum(self.TP()/data_size)

    def Precision(self):
        '''
        "calculate Precision of training model : What proportion of positive identifications was actually correct?"

        :return:mean of (True Positive / True Positive + False Positive )
        '''
        return np.mean(self.TP() / (self.TP() + self.FP()))

    def Recall(self):
        '''
        "calculate Recall of training model : What proportion of actual positives was identified correctly?"
        :return: mean(True Positive / True Positive + False Negative)
        '''
        return np.mean(self.TP() / (self.TP() + self.FN()))

    def F1_score(self):
        '''
        "calculate F1_score of training model"
        :return: if(True Positive > 0) {2* (Precision * Recall / Precision + Recall)} else {0}
        '''
        if self.TP() > 0:
            return 2 * ((self.Precision() * self.Recall()) / (self.Precision() + self.Recall()))
        else:
            return 0



