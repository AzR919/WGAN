"""
10-707 Assignment 3
Problem 1: WGAN
TA in charge: Amartya Basu, Tanya Marwah.

IMPORTANT:
    DO NOT change any function signatures
    If you do need to add something to the function signatures, please add
        them towards the end and provide them with a default value. Make sure
        that the default value is able to pass the local and auto-grader test.
"""
from utils import *
import progressbar

class Constants:
    """
    Recommended hyperparameters.
    Feel free to add/remove/modify these values to run your code.
    However, keep a copy of the original values as they will be used as
    reference for the local and autograder tests.

    In order to access any hyperparameter from the Constants class, just call Constants.xxx.
    """
    num_epochs = 100 # number of epochs
    num_iterate = 100 # number of iterations per epoch.
    batch_size = 100
    learning_rate = 0.00005
    n_critic = 5
    c = 0.01
    beta = 0.9 # RMSProp Parameter
    latent= 100 # latent space dimension
    discriminator_hidden_sizes = [784, 512, 256, 1]
    generator_hidden_sizes = [latent, 256, 512, 1024, 784]
    eps = 1e-8



def weights_init(inp_dim, out_dim):
    """
    Function for weights initialization
    :param inp_dim: Input dimension of the weight matrix
    :param out_dim: Output dimension of the weight matrix
    """
    b = np.sqrt(6)/np.sqrt(inp_dim+out_dim)
    return np.random.uniform(-b,b,(inp_dim,out_dim))


def biases_init(dim):
    """
    Function for biases initialization
    :param dim: Dimension of the biases vector
    """
    return np.zeros(dim).astype(np.float32)


class Discriminator(object):
    def __init__(self):
        """
        Initialize your weights, biases and anything you need here.
        Please follow this naming convention for your variables (helpful for tests.py and autograder)
            Variable name for weight matrices: W0, W1, ...
            Variable name for biases: b0, b1, ...
        """
        
        self.W0 = weights_init(Constants.discriminator_hidden_sizes[0], Constants.discriminator_hidden_sizes[1])
        self.W1 = weights_init(Constants.discriminator_hidden_sizes[1], Constants.discriminator_hidden_sizes[2])
        self.W2 = weights_init(Constants.discriminator_hidden_sizes[2], Constants.discriminator_hidden_sizes[3])

        self.b0 = biases_init(Constants.discriminator_hidden_sizes[1])
        self.b1 = biases_init(Constants.discriminator_hidden_sizes[2])
        self.b2 = biases_init(Constants.discriminator_hidden_sizes[3])

        self.a1 = LRelu("wgan")
        self.a2 = LRelu("wgan")

        self.vW0 = np.zeros_like(self.W0)
        self.vW1 = np.zeros_like(self.W1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb0 = np.zeros_like(self.b0)
        self.vb1 = np.zeros_like(self.b1)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, x):
        """
        Forward pass for discriminator
        :param x: Input for the forward pass with shape (batch_size, 28, 28, 1)
        :return Output of the discriminator with shape (batch_size, 1)
        NOTE: Given you are using linear layers, you will have to appropriately reshape your data.
        """
        
        self.l0 = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        self.l1 = self.a1(self.l0 @ self.W0 + self.b0)
        self.l2 = self.a2(self.l1 @ self.W1 + self.b1)
        self.l3 = self.l2 @ self.W2 + self.b2

        return self.l3


    def backward(self, logit, inp, image_type, update_params=True):
        """
        Backward pass for discriminator. 
        Implement RMSProp for the gradient update.
        Use/set the Learning Rate in the Constants class.

        :param logit: logit value with shape (batch_size, 1)
        :param inp: input image with shape (batch_size, 28, 28, 1). This parameter might not be required, depending on your implementation.
        :image_type: Integer value -1 or 1 depending on whether it is a real or fake image.
            image_type will ensure that the gradients are taken in the right direction.
        """

        dW2 = (np.sum(self.l2, axis=0).reshape(self.W2.shape[0], self.W2.shape[1]))/logit.shape[0]
        db2 = 1

        da2 = self.a2.backward(np.ones_like(logit) @ self.W2.T)/logit.shape[0]
        dW1 = self.l1.T @ da2
        db1 = np.sum(da2, axis=0)

        da1 = self.a1.backward(da2 @ self.W1.T)
        dW0 = self.l0.T @ da1
        db0 = np.sum(da1, axis=0)

        if update_params:

            self.vW2 = Constants.beta * self.vW2 + (1-Constants.beta) * np.square(dW2)
            self.W2 -= image_type * Constants.learning_rate * ( dW2 / (np.sqrt(self.vW2) + Constants.eps) )

            self.vb2 = Constants.beta * self.vb2 + (1-Constants.beta) * np.square(db2)
            self.b2 -= image_type * Constants.learning_rate * ( db2 / (np.sqrt(self.vb2) + Constants.eps) )

            self.vW1 = Constants.beta * self.vW1 + (1-Constants.beta) * np.square(dW1)
            self.W1 -= image_type * Constants.learning_rate * ( dW1 / (np.sqrt(self.vW1) + Constants.eps) )

            self.vb1 = Constants.beta * self.vb1 + (1-Constants.beta) * np.square(db1)
            self.b1 -= image_type * Constants.learning_rate * ( db1 / (np.sqrt(self.vb1) + Constants.eps) )
            
            self.vW0 = Constants.beta * self.vW0 + (1-Constants.beta) * np.square(dW0)
            self.W0 -= image_type * Constants.learning_rate * ( dW0 / (np.sqrt(self.vW0) + Constants.eps) )

            self.vb0 = Constants.beta * self.vb0 + (1-Constants.beta) * np.square(db0)
            self.b0 -= image_type * Constants.learning_rate * ( db0 / (np.sqrt(self.vb0) + Constants.eps) )

        return da1 @ self.W0.T
            


    def weight_clipping(self):
        """
        Implement weight clipping for discriminator's weights and biases.
        Set/use the value defined in the Constants class.
        """
        self.W0 = np.clip(self.W0, -1 * Constants.c, Constants.c)
        self.W1 = np.clip(self.W1, -1 * Constants.c, Constants.c)
        self.W2 = np.clip(self.W2, -1 * Constants.c, Constants.c)
        self.b0 = np.clip(self.b0, -1 * Constants.c, Constants.c)
        self.b1 = np.clip(self.b1, -1 * Constants.c, Constants.c)
        self.b2 = np.clip(self.b2, -1 * Constants.c, Constants.c)

    def __call__(self, x):
        """
        Do not change/remove.
        """
        return self.forward(x)


class Generator(object):

    def __init__(self):
        """
        Initialize your weights, biases and anything you need here.
        Please follow this naming convention for your variables (helpful for tests.py and autograder)
            Variable name for weight matrices: W0, W1, ...
            Variable name for biases: b0, b1, ...
        """
    
        self.W0 = weights_init(Constants.generator_hidden_sizes[0], Constants.generator_hidden_sizes[1])
        self.W1 = weights_init(Constants.generator_hidden_sizes[1], Constants.generator_hidden_sizes[2])
        self.W2 = weights_init(Constants.generator_hidden_sizes[2], Constants.generator_hidden_sizes[3])
        self.W3 = weights_init(Constants.generator_hidden_sizes[3], Constants.generator_hidden_sizes[4])

        self.b0 = biases_init(Constants.generator_hidden_sizes[1])
        self.b1 = biases_init(Constants.generator_hidden_sizes[2])
        self.b2 = biases_init(Constants.generator_hidden_sizes[3])
        self.b3 = biases_init(Constants.generator_hidden_sizes[4])

        self.a1 = LRelu("wgan")
        self.a2 = LRelu("wgan")
        self.a3 = LRelu("wgan")
        self.a4 = Tanh()

        self.vW0 = np.zeros_like(self.W0)
        self.vW1 = np.zeros_like(self.W1)
        self.vW2 = np.zeros_like(self.W2)
        self.vW3 = np.zeros_like(self.W3)
        self.vb0 = np.zeros_like(self.b0)
        self.vb1 = np.zeros_like(self.b1)
        self.vb2 = np.zeros_like(self.b2)
        self.vb3 = np.zeros_like(self.b3)

    def forward(self, z):
        """
        Forward pass for generator
        :param z: Input for the forward pass with shape (batch_size, 100)
                 Output of the final linear layer will be of shape (batch_size, 784)
        :return Output of the generator with shape (batch_size, 28, 28, 1)
        """
        self.l0 = z
        self.l1 = self.a1(self.l0 @ self.W0 + self.b0)
        self.l2 = self.a2(self.l1 @ self.W1 + self.b1)
        self.l3 = self.a3(self.l2 @ self.W2 + self.b2)
        self.l4 = self.a4(self.l3 @ self.W3 + self.b3)

        return self.l4.reshape(z.shape[0], 28, 28, 1)

    def backward(self, fake_logit, fake_input, discriminator):
        """
        Backward pass for generator
        Implement RMSProp for the gradient update.
        Use/set the Learning Rate in the Constants class.

        :param fake_logit: Logit output from the discriminator with shape (batch_size, 1)
        :param fake_input: Fake images generated by the generator with shape (batch_size, 28, 28, 1) -- may or may not be required depending upon your implementation.
        :param discriminator: discriminator object
        NOTE: In order to perform backward, you may or may not (based on your implementation) need to call the backward function in the discriminator.
              In such an event, make sure that the Discriminator weights are *not* being updated for this particular call.
        """
        da4 = self.a4.backward(discriminator.backward(fake_logit, self.l4, -1, update_params=False))
        dW3 = self.l3.T @ da4
        db3 = np.sum(da4, axis=0)

        da3 = self.a3.backward(da4 @ self.W3.T)
        dW2 = self.l2.T @ da3
        db2 = np.sum(da3, axis=0)

        da2 = self.a2.backward(da3 @ self.W2.T)
        dW1 = self.l1.T @ da2
        db1 = np.sum(da2, axis=0)

        da1 = self.a1.backward(da2 @ self.W1.T)
        dW0 = self.l0.T @ da1
        db0 = np.sum(da1, axis=0)

        self.vW3 = Constants.beta * self.vW3 + (1-Constants.beta) * np.square(dW3)
        self.W3 += Constants.learning_rate * ( dW3 / (np.sqrt(self.vW3) + Constants.eps) )

        self.vb3 = Constants.beta * self.vb3 + (1-Constants.beta) * np.square(db3)
        self.b3 += Constants.learning_rate * ( db3 / (np.sqrt(self.vb3) + Constants.eps) )

        self.vW2 = Constants.beta * self.vW2 + (1-Constants.beta) * np.square(dW2)
        self.W2 += Constants.learning_rate * ( dW2 / (np.sqrt(self.vW2) + Constants.eps) )

        self.vb2 = Constants.beta * self.vb2 + (1-Constants.beta) * np.square(db2)
        self.b2 += Constants.learning_rate * ( db2 / (np.sqrt(self.vb2) + Constants.eps) )
        
        self.vW1 = Constants.beta * self.vW1 + (1-Constants.beta) * np.square(dW1)
        self.W1 += Constants.learning_rate * ( dW1 / (np.sqrt(self.vW1) + Constants.eps) )

        self.vb1 = Constants.beta * self.vb1 + (1-Constants.beta) * np.square(db1)
        self.b1 += Constants.learning_rate * ( db1 / (np.sqrt(self.vb1) + Constants.eps) )

        self.vW0 = Constants.beta * self.vW0 + (1-Constants.beta) * np.square(dW0)
        self.W0 += Constants.learning_rate * ( dW0 / (np.sqrt(self.vW0) + Constants.eps) )

        self.vb0 = Constants.beta * self.vb0 + (1-Constants.beta) * np.square(db0)
        self.b0 += Constants.learning_rate * ( db0 / (np.sqrt(self.vb0) + Constants.eps) )
        

    def __call__(self, z):
        """
        Do not change/remove.
        """
        return self.forward(z)

class WGAN(object):

    def __init__(self, generator, discriminator, numbers):
        """
        Initialize the GAN with your discriminator, generator and anything you need here.
        Feel free to change/modify your function signatures here, this will *not* be autograded.
        """
        
        self.gen = generator
        self.dis = discriminator
        self.num = numbers


    def linear_interpolation(self):
        """
        Generate linear interpolation between two data points. 
        (See example on Lec15 slides page 29
        and feel free to modify the function signature. This function is not graded by autograder.)
        """
        raise NotImplementedError

    def train(self, data):
        """
        Determine how to train the WGAN. 
        You can use the "img_tile" functions in utils.py to visualize your generated batch.

        If you make changes to the vizualization procedure for the fake batches, 
        please include them in your utils.py file.
        """

        dis_losses = []
        
        for epoch in progressbar.progressbar(range(Constants.num_epochs)):

            random_X = data[0].copy()
            np.random.shuffle(random_X)

            dis_n_losses = []

            for i in range(Constants.n_critic):

                x_batch = random_X[i*100 : (i+1)*100]
                z_batch = np.random.normal(0, 1, (Constants.batch_size, 100))

                x_logit = self.dis.forward(x_batch)
                self.dis.backward(x_logit, x_batch, +1)

                f_batch = self.gen.forward(z_batch)
                f_logit = self.dis.forward(f_batch)
                self.dis.backward(f_logit, f_batch, +1)
                
                self.dis.weight_clipping()

                dis_n_losses.append(np.sum(x_logit)-np.sum(f_logit)/Constants.batch_size)

            dis_losses.append(sum(dis_n_losses)/len(dis_n_losses))
            
            z_batch = np.random.normal(0, 1, (Constants.batch_size, 100))

            f_batch = self.gen.forward(z_batch)
            f_logit = self.dis.forward(f_batch)
            
            self.gen.backward(f_logit, f_batch, self.dis)

            img_tile(f_batch, "./gen_i", epoch, True)


        plt.plot(dis_losses, label="Discriminator_loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Loss plot")
        plt.show()



if __name__ == '__main__':
    """
    You can implement your training and testing loop here.
    You MUST use your class implementations to train the model and to get the results.
    DO NOT use pytorch or tensorflow get the results. The results generated using these
    libraries will be different as compared to your implementation.
    """
    #import ipdb;ipdb.set_trace()
    numbers = [2]

    data = mnist_reader(numbers)

    wgan = WGAN(Generator(), Discriminator(), numbers)

    wgan.train(data)


