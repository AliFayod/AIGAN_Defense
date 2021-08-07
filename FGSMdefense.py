from FGSMmain import  x_test_adv
from FGSMmain import classifier
import torch
import torch.nn as nn
from advertorch.defenses import BitSqueezing
from advertorch.defenses import MedianSmoothing2D
from keras.models import load_model


from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.utils import load_dataset
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":

    # Read MNIST dataset
    (x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str("mnist"))


    #Load Minst Model
    classifier= load_model('/models/FGSMMODEL/model.h5')
    # Create Defence
    bits_squeezing = BitSqueezing(bit_depth=1)
    median_filter = MedianSmoothing2D(kernel_size=3)

    defense = nn.Sequential(

        bits_squeezing,
        median_filter,
    )

    torch_ex_float_tensor = torch.from_numpy(x_test_adv)
    torch_ex_float_tensor= torch.nn.functional.pad(torch_ex_float_tensor, (1, 1), mode='constant', value=0)
    Def_test=defense(torch_ex_float_tensor)

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(Def_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy on defenced sample: %.2f%%" % (acc * 100))