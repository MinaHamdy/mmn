User guide  
==========
* **get started**
 
     1) pip install TensorFlow-El-8lbah==0.0.1.

     2) take instance from dataset class specifying dataset_name and path to folder containing  training dataset and testing dataset  and our package will do all work form you and give you training examples , training labels , testing examples and testing labels. To make your labels one hot and other are zeros to give propalistic cost. 
        
        Example : 

	* dataset = **dataset** ('mnist', r'C:\Users\FacultyStudent\PycharmProjects\final_NN')
	* train_X, train_Y, test_X, test_Y = dataset. **get_dataset()**
	* train_Y  =  dataset. **labels_to_onehot(train_Y)**

     3) you should adjust your list of layers dimension and list of activations functions  
        
        Example: 

        * **layers_dimensions** = [train_X.T.shape[0], 128, train_Y.shape[0]]
        * **activation_functions** = ["NONE", "sigmoid", "sigmoid"]
        * **NONE**: refers to input layer which does not need an activation function.


     4) Batch training:

	then make instance from training_model and give it:

        * **X**: The input of the first layer
        * **Y**: the True labels of the training examples
        * **layers_dimenstions** : the number of hidden units in each hidden layer
        * **activation_functions** : the type of activation function in each hidden layer
        * **alpha** : learning rate
        * **no_of_iterations** : number of iterations
        * **print_cost** : boolean variable , put it with ( True ) value if you want to print the cost every 10 iterations
        * **lambd** : regularization parameter
        * **return** : The Trained Parameters for a certain model trained on a certain dataset

        Example: 
 
	model = training_model(train_X.T, train_Y, layers_dimensions, activation_functions,alpha = 0.01 , no_of_iterations = 100,print_cost = 1,lambd = 0.1).


     5)	Mini_batch training:

        You can create batches as you like by using random_mini_batches(X,Y, mini_batch_size):
         * **X** : The whole Training Set AKA all the training examples
         * **Y** : The whole Training examples' labels
         * **mini_batch_size** : The wanted mini_batch size for the training examples
         * **return** : a list of mini batches after division

        Example:

        training_minibatches = random_mini_batches(train_X.T, train_Y,mini_batch_size=1024).
        
        then make instance from **training_model** and give it 
         * **minibatches** : list of mini_batches after dividing the training set to mini_batches
         * **activations** : the type of activation function in each layer
         * **alpha** : learning rate
         * **no_of_iterations** : number of iterations
         * **print_cost** : boolean variable , put it with ( True ) value if you want to print the cost every 10 iterations
         * **lambd** : regularization parameter
         * **momentum** : boolean : put it with ( True ) value if you want to apply the momentum Gradient descent
         * **beta** : momentum parameter
         * **ADAM** : boolean : put it with a ( True ) value if you want to apply ADAM optimization

        Example:

        training_model(training_minibatches,activation_functions,alpha=0.007 , no_of_iterations=90, print_cost = True, lambd = 0 ,momentum=0,beta=0.9,ADAM=1)

     6)	then you need to train your model on mnist dataset, you can use **train()** .
        It will return your training parameters(weights, biases).
        
        Example:
	
        Training_parameters = model. **train()** .

     7)	You can save your training parameters in packle file to revisit it again.it will return the saved packle file name.
        
        Example:

	File_name = Model. **save(Trainig_parameters)** .
	
     8)	You can use our evaluation metrics function to check model’s confusionMatrix, accuracy, Precision, Recall, F1_score.
        You should make instance from evaluation_metrics and give it labels, training examples, training parameters, activation functions.Then:

         * you can use **confusionMatrix()** , it will return two values a visualized matrix and matrix values.
	 * you can use **Accuracy(dataset_size)** to calculate accuracy of your model.
	 * you can use **Precision()** to calculate Precision of your model.
	 * you can use **Recall()** to calculate Recall of your training model.
	 * you can use **F1_score()** to calculateF1_Score of your training model.
	
     9)	During training you can visualize  real time learning curve.

     10) Thank you for using our package.

