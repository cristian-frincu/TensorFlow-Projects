import NN
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = NN.Network([784, 5, 10])

net.SGD(training_data, 5, 10, 3.0, test_data=test_data)
