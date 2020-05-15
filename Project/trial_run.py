import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

from log_helper import Logger

#Function loads MNIST_data
def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5,), (.5,))
        ]
    )
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

#Loads MNIST_data into data
data = mnist_data()

#This variable will allow us to iterate over the MNIST_data
loaded_data_iteratable = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

#Number of batches
num_batches = len(loaded_data_iteratable)
#print(num_batches) #Should show 600

# DISCCRIMINATOR NUERAL NET
# THIS WILL CHECK WHETHER OR NOT THE GENERATOR'S OUTPUT CAN PASS AS REAL OR SYNTHESIZED
# THE GOAL IS TO MINIMIZE THE PROBABILITY OF THE GENERATOR'S OUTPUT PASSING AS REAL
class DiscriminatorNetwork(torch.nn.Module):
    """
    We will convert the 784-sized feature vector (28x28 pixels) into 1 result vector
    using a neural network that contains 3 layers.
    Each layer contains a Linear Function that will perform a Linear transformation to the incoming feature vector: y=xA^T + b
    They will also contain a LeakyReLU non-linear activation function and random dropouts to prevent overfitting.
    A Sigmoid function is then used on the final feature vector to categorize the values into binary form (0,1).
    """
    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()
        n_features = 784
        n_out = 1

        self.hiddenLayer0 = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hiddenLayer1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.hiddenLayer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )

        self.outLayer = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )

    #Function used to run the feature vectors through the 3-layer network and return the result vector
    def forward(self, x):
        x = self.hiddenLayer0(x)
        x = self.hiddenLayer1(x)
        x = self.hiddenLayer2(x)
        x = self.outLayer(x)
        return x

discriminator = DiscriminatorNetwork()
print("Got to this point, created the Discriminator Network")

#Function to convert images into feature vectors
def images_to_vectors(images):
    return images.view(images.size(0), 784)

#Function to convert feature vectors into images
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

# GENERATOR NUERAL NET
# THIS WILL TRY CREATE IMAGES THAT CAN TRICK THE DISCRIMINATOR INTO THINKING THAT IMAGE IS REAL WHEN IT IS ACTUALLY SYNTHESIZED
# THE GOAL IS TO MAXIMIZE THE PROBABILITY OF THE GENERATOR'S OUTPUT PASSING AS REAL
# AND MINIMIZE THE PROBABILITY OF THE DISCRIMINATOR'S GUESS BEING FAKE
class GeneratorNetwork(torch.nn.Module):
    """
    We will convert a latent variable vector into a 784-sized feature vector (28x28 pixels)
    using a neural network that contains 3 layers.
    The main purpose of this network is to learn how to create realistic hand-written digits.
    Each layer contains a Linear Function that will perform a Linear transformation to the incoming feature vector: y=xA^T + b
    They will also contain a LeakyReLU non-linear activation function.
    A TanH (Hyperbolic Tangent) function is then used on the final feature vector to map the values into a range between (-1,1).
    That is the same range that the MNIST images are bounded on.
    """
    def __init__(self):
        super(GeneratorNetwork, self).__init__()
        n_features = 100
        n_out = 784

        self.hiddenLayer0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )

        self.hiddenLayer1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )

        self.hiddenLayer2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        self.outLayer = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )
    #Function used to run the latent variable vector through the 3-layer network and return the resulting feature vector
    def forward(self, x):
        x = self.hiddenLayer0(x)
        x = self.hiddenLayer1(x)
        x = self.hiddenLayer2(x)
        x = self.outLayer(x)
        return x

generator = GeneratorNetwork()
print("Got to this point, created the Generator Network")

# Function for creating random noise in the shape of a 1-D vector
def noise(size_of_noise):
    return Variable(torch.randn(size_of_noise, 100))

# Here we use the Adam Optimizer from PyTorch to optimize both networks
# The lr variable is the learning rate
discriminatorOptimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
generatorOptimizer = optim.Adam(generator.parameters(), lr=0.0002)

# The Loss Function we will be using is the Binary Cross Entropy Loss Function (BCEL)
# Will resemble the logarithmic loss for both the Generator and Discriminator.
# We will take the mean of the loss calculated for each minibatch.
# Also, since we want to maximize log(D(G(z))) and PyTorch and many other ML libraries only minimize,
# maximizing log(D(G(z))) is equivalent to minimizing the negative, and since BCEL has a negative sign,
# we don't have to worry about the sign.
loss = nn.BCELoss()

# Since we will always recognize real-images as ones and synthesized-images as zeros, we use the following functions 
# to return a target containing only ones and zeros
def ones(size):
    data = Variable(torch.ones(size, 1))
    return data

def zeros(size):
    data = Variable(torch.zeros(size, 1))
    return data

# We now create the method for training the Discriminator network. We will get the total mini-batch loss from summing the 
# error for identifying real-images and error for identifying synthesized-images together
def train_discriminator(optimizer, real_data, synthesized_data):
    N = real_data.size(0)
    # Zero out the gradients of the optimizer function
    optimizer.zero_grad()

    # Train on real data
    predict_real = discriminator(real_data)
    error_real = loss(predict_real, ones(N))
    # Backpropogate
    error_real.backward()

    # Train on synthesized data
    predict_synthesized = discriminator(synthesized_data)
    error_synthesized = loss(predict_synthesized, zeros(N))
    # Backpropogate
    error_synthesized.backward()

    # Update the weights in the optimizer with the gradients
    optimizer.step()

    # Return the error and predictions for real and synthesized
    return error_real + error_synthesized, predict_real, predict_synthesized

# We now create the method for training the Generator network. We will get the loss from checking the
# prediction from the discriminator and inputting that into the loss function to see if it predicted the images were real.
def train_generator(optimizer, synthesized_data):
    N = synthesized_data.size(0)
    # Zero out the gradients of the optimizer function
    optimizer.zero_grad()

    # Sample the noise and generate synthesized data
    prediction = discriminator(synthesized_data)

    error = loss(prediction, ones(N))
    # Backpropogate
    error.backward()

    # Update the weights in the optimizer with the gradients
    optimizer.step()

    #Return the error
    return error

# This test noise will be used to visualize the training process as the GAN learns
num_test_samples = 16
test_noise = noise(num_test_samples)

#Logger Instance
logger = Logger(model_name='VGAN', data_name='MNIST')


# Total number of epochs to train
num_epochs = 5

#This is the training process
for epoch in range(num_epochs):
        for n_batch, (real_batch,_) in enumerate(loaded_data_iteratable):
            N = real_batch.size(0)
            """
            Training Discriminator
            """
            real_data = Variable(images_to_vectors(real_batch))

            # Generating synthesized data and detach so they are not calculated for generator
            synthesized_data = generator(noise(N)).detach()

            discriminator_error, discriminator_pred_real, discriminator_pred_synthesized = train_discriminator(discriminatorOptimizer, real_data, synthesized_data)

            """
            Training Generator
            """
            
            #Generating synthesized data
            synthesized_data = generator(noise(N))

            generator_error = train_generator(generatorOptimizer, synthesized_data)

            # Log batch error
            logger.log(discriminator_error, generator_error, epoch, n_batch, num_batches)

            # Display progress every 10 batches
            if (n_batch % 10) == 0:
                test_images = vectors_to_images(generator(test_noise))
                test_images = test_images.data

                logger.log_images(test_images, num_test_samples, epoch, n_batch, num_batches)

                #Display status logs
                logger.display_status(epoch, num_epochs, n_batch, num_batches, discriminator_error, generator_error, discriminator_pred_real, discriminator_pred_synthesized)

