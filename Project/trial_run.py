import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets

# from utils import Logger

#Function loads MNIST_data
def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))
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

discrimator = DiscriminatorNetwork()
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