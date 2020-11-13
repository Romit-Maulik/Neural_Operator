import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

import tensorflow as tf
tf.random.set_seed(10)

from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
import numpy as np
np.random.seed(10)

# Plotting
import matplotlib.pyplot as plt

# Iterative operator layer
class operator_layer(Layer):
    def __init__(self,num_iterations):
        super(operator_layer, self).__init__()

        self.num_iterations = num_iterations

        def build(self,input_shape):
            # Because call will need this
            self.samples = input_shape[0]
            self.lifted_dimension = input_shape[1] # samples x lifted_dimension

            # Add the affine transform weight
            # self.w_list = []
            # for i in range(self.num_iterations):
            #     self.w_list.append(self.add_weight(shape=(self.lifted_dimension,self.lifted_dimension),initializer='xavier',trainable=True,name='w'))

            self.w = self.add_weight(shape=(self.lifted_dimension,self.lifted_dimension),initializer='xavier',trainable=True,name='w')
            self.b = self.add_weight(shape=(self.lifted_dimension,),initializer='xavier',trainable=True,name='w')

            # Define the Fourier space weight
            self.r_phi = self.add_weight(shape=(self.lifted_dimension,self.lifted_dimension),initializer='xavier',trainable=True,name='r_phi')

            # # Add the regular global kernel
            # self.kw = self.add_weight(shape=(self.samples,self.samples),initializer='xavier',trainable=True,name='kw')
            # self.kb = self.add_weight(shape=(self.samples,),initializer='xavier',trainable=True,name='kb')

        def call(self,inputs):
            result = tf.identity(inputs) # Make a copy for the result

            for iteration in range(self.num_iterations):
                
                # Global operations
                # # Regular kernel
                # temp = tf.nn.swish(tf.matmul(self.kw,result)+self.kb)

                # Fourier kernel
                for dim in range(self.lifted_dimension):
                    temp = tf.transpose(tf.identity(result)) # lifted_dimension x samples
                    # Fourier kernel
                    temp[dim:dim+1,:] = tf.matmul(tf.signal.fft(self.r_phi),tf.signal.fft(temp[dim:dim+1,:]))

                # # Linear operation (of periodic function) in Fourier space
                # temp = tf.matmul(tf.signal.fft(self.r_phi),temp)

                for dim in range(self.lifted_dimension):
                    temp[dim:dim+1,64:] = 0.0 # Truncation
                    temp = tf.signal.ifft(temp[dim:dim+1,:])
                
                # Local operations
                temp = tf.transpose(temp)
                for sample in range(self.samples):
                    result[sample] = tf.nn.swish(temp[sample]+tf.matmul(result[sample:sample+1,:],self.w)+self.b)

                # Normalization
                result[sample] = result[sample]/tf.math.reduce_sum(result)

            return result


#Build the model which does basic map of inputs to coefficients
class neural_operator(Model):
    def __init__(self,ip_data,op_data):
        super(neural_operator, self).__init__()

        # Data sets
        self.ip_data = ip_data
        self.op_data = op_data

        # Constants
        self.num_fields = np.shape(ip_data)[0]

        # Define lifting operator
        self.lifting_layer_0 = tf.keras.layers.Dense(100,activation='swish') # Lift to high dimensional space 'v'
        self.lifting_layer_1 = tf.keras.layers.Dense(100,activation='linear') # Lift to high dimensional space 'v'

        # Iterative update layer
        num_iterations = 10
        self.operator_layer = operator_layer(num_iterations)

        # Define projection operator
        self.projection_layer_0 = tf.keras.layers.Dense(100,activation='swish')
        self.projection_layer_1 = tf.keras.layers.Dense(1,activation='linear')

        # Train operation
        self.train_op = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Running the model - applied to every point in domain
    def call(self,X):
        hh = self.lifting_layer_0(X)
        hh = self.lifting_layer_1(hh)

        hh = self.operator_layer(hh)

        hh = self.projection_layer_0(hh)
        hh = self.projection_layer_1(hh)
        return hh
    
    # Regular MSE
    def get_loss(self,X,Y):
        op=self.call(X)
        return tf.reduce_mean(tf.math.square(op-Y))

    # get gradients - regular
    def get_grad(self,X,Y):
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            L = self.get_loss(X,Y)
            g = tape.gradient(L, self.trainable_variables)
        return g

    # perform gradient descent - regular
    def network_learn(self,x,y):
        g = self.get_grad(x,y)
        self.train_op.apply_gradients(zip(g, self.trainable_variables))

    # Training the model
    def train_model(self):
        plot_iter = 0
        stop_iter = 0
        patience = 10
        best_train_loss = 999999.0 # Some large number

        idx = np.arange(self.num_fields) 
        np.random.shuffle(idx)
        self.ip_data = self.ip_data[idx]
        self.op_data = self.op_data[idx]
       
        for i in range(1000): # Number of training epochs
            # Training loss
            print('Training iteration:',i)

            train_loss = 0.0
            for sample in range(self.num_fields):
                input_field = self.ip_data[sample:sample+1,:].reshape(-1,1)
                output_field = self.op_data[sample:sample+1,:].reshape(-1,1)
                self.network_learn(input_field,output_field)

                train_loss = train_loss + self.get_loss(input_field,output_field).numpy()

            train_loss = train_loss/self.num_fields
            
            # Check early stopping criteria
            if train_loss < best_train_loss:
                
                print('Improved training loss from:',best_train_loss,' to:', train_loss)                
                best_train_loss = train_loss

                self.save_weights('./checkpoints/my_checkpoint')
                
                stop_iter = 0
            else:
                print('Training loss (no improvement):',train_loss)
                stop_iter = stop_iter + 1

            if stop_iter == patience:
                break

    def test_model(self,test_data_ip,test_data_op):

        self.load_weights(dir_path+'/checkpoints/my_checkpoint') # Load pretrained model

        num_test_fields = test_data_ip.shape[0]
        input_field = test_data_ip[0:1,:].reshape(-1,1)

        predictions = []
        for sample in range(num_test_fields):
                output_field = self.call(input_field).numpy()

                plt.figure()
                plt.plot(output_field[:,0],label='Predicted')
                plt.plot(test_data_op[sample,:],label='True')
                plt.legend()
                plt.ylim((0,0.5))
                plt.savefig('Snapshot_'+str(sample)+'.png')
                plt.close()
                # exit()

                # Get prediction
                predictions.append(output_field[:,0])
                # input_field = output_field.copy()
                input_field = test_data_op[sample,:].reshape(-1,1)

        return np.asarray(predictions)


if __name__ == '__main__':
    print('Neural operator learning in tensorflow')

    # burgers_train = np.load('Burgers_Snapshots_Train.npy')

    # train_ip = []
    # train_op = []
    # for i in range(9):
    #     train_ip.append(burgers_train[i*100:(i+1)*100-1])
    #     train_op.append(burgers_train[i*100+1:(i+1)*100])

    # train_ip = np.asarray(train_ip).reshape(-1,128)
    # train_op = np.asarray(train_op).reshape(-1,128)

    # np.save('Train_data_ip.npy',train_ip)
    # np.save('Train_data_op.npy',train_op)


    # train_data_ip =  np.load('KS_Snapshots_0.npy')[:-1]
    # train_data_ip =  np.concatenate((train_data_ip,np.load('KS_Snapshots_1.npy')[:-1]),axis=0)
    # train_data_ip =  np.concatenate((train_data_ip,np.load('KS_Snapshots_2.npy')[:-1]),axis=0)
    # train_data_ip =  np.concatenate((train_data_ip,np.load('KS_Snapshots_3.npy')[:-1]),axis=0)
    # train_data_ip =  np.concatenate((train_data_ip,np.load('KS_Snapshots_4.npy')[:-1]),axis=0)
    # train_data_ip =  np.concatenate((train_data_ip,np.load('KS_Snapshots_5.npy')[:-1]),axis=0)
    # train_data_ip =  np.concatenate((train_data_ip,np.load('KS_Snapshots_6.npy')[:-1]),axis=0)
    # train_data_ip =  np.concatenate((train_data_ip,np.load('KS_Snapshots_7.npy')[:-1]),axis=0)
    # train_data_ip =  np.concatenate((train_data_ip,np.load('KS_Snapshots_9.npy')[:-1]),axis=0)
    # train_data_ip =  np.concatenate((train_data_ip,np.load('KS_Snapshots_10.npy')[:-1]),axis=0)

    # train_data_op =  np.load('KS_Snapshots_0.npy')[1:]
    # train_data_op =  np.concatenate((train_data_op,np.load('KS_Snapshots_1.npy')[1:]),axis=0)
    # train_data_op =  np.concatenate((train_data_op,np.load('KS_Snapshots_2.npy')[1:]),axis=0)
    # train_data_op =  np.concatenate((train_data_op,np.load('KS_Snapshots_3.npy')[1:]),axis=0)
    # train_data_op =  np.concatenate((train_data_op,np.load('KS_Snapshots_4.npy')[1:]),axis=0)
    # train_data_op =  np.concatenate((train_data_op,np.load('KS_Snapshots_5.npy')[1:]),axis=0)
    # train_data_op =  np.concatenate((train_data_op,np.load('KS_Snapshots_6.npy')[1:]),axis=0)
    # train_data_op =  np.concatenate((train_data_op,np.load('KS_Snapshots_7.npy')[1:]),axis=0)
    # train_data_op =  np.concatenate((train_data_op,np.load('KS_Snapshots_9.npy')[1:]),axis=0)
    # train_data_op =  np.concatenate((train_data_op,np.load('KS_Snapshots_10.npy')[1:]),axis=0)

    # np.save('Train_data_ip.npy',train_data_ip)
    # np.save('Train_data_op.npy',train_data_op)

    # test_data_ip =  np.load('KS_Snapshots_8.npy')[:-1]
    # test_data_op =  np.load('KS_Snapshots_8.npy')[1:]

    # np.save('Test_data_ip.npy',test_data_ip)
    # np.save('Test_data_op.npy',test_data_op)

    
    train_data_ip = np.load('Train_data_ip.npy')[:100:10]
    train_data_op = np.load('Train_data_op.npy')[:100:10]

    test_data_ip = np.load('Train_data_ip.npy')[:100:10]
    test_data_op = np.load('Train_data_op.npy')[:100:10]

    my_model = neural_operator(train_data_ip,train_data_op)
    # my_model.train_model()

    predictions = my_model.test_model(test_data_ip,test_data_op)

    # plt.figure()
    # plt.plot(test_data_op[4,:],label='True')
    # plt.plot(predictions[4,:],label='Predicted')
    # plt.legend()
    # plt.show()