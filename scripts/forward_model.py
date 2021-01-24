from layers import layers
from forward_prop import forward_prop
class forward_model():
    '''
    " Class of Applying Forward propagation process . "
    '''

    def forward_model (self,X,parameters):
        '''
        :param X: The input to the current layer as to start the forward propagation process
        :param parameters: Weights and biases of the current layer
        :return : - Y_hat ( predictions resulted from the current layer's forward propagation )
                  - packet of packets : Tuple of 2 elements which will be used in backward propagation :
                     1- linear packer : contains ( input , weights , biases ) of the current layer
                     2- activation packet : contains ( Z ) which is the input to the activation function
        '''
        packet_of_packets = []
        A = X # as for the first time A will equal to the input layer
        #print(type(parameters))
        L = len(parameters) // 2  # number of layers in the neural network
        #print(L)
        test_forward=forward_prop()
        for l in range(1, L):
            A_prev = A
            A, packet = test_forward.activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation_type="sigmoid")
            packet_of_packets.append(packet)

        prediction, packet = test_forward.activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation_type="softmax")
        packet_of_packets.append(packet)



        return prediction, packet_of_packets


'''
dimensions = [3,5,1]
test_model=forward_model(dimensions)
AL , caches = test_model.forward_model([1,5,4])
print(AL)
print("###########################")
print(caches)
'''