import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from advertorch.defenses import BitSqueezing
from advertorch.defenses import MedianSmoothing2D
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import _imshow
from advertorch_examples.utils import get_mnist_test_loader
import models
from models import MNIST_target_net
from sklearn.metrics import confusion_matrix

torch.manual_seed(0)

if __name__ == "__main__":

    use_cuda = True
    image_nc = 1
   # batch_size = 128
    gen_input_nc = image_nc

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # load the pretrained model
    pretrained_model = "./MNIST_target_model.pth"
    model = MNIST_target_net().to(device)
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()

    # load the generator of adversarial examples
    pretrained_generator_path = './models/netG.pth.tar'
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()

    # Load data
    batch_size = 5
    loader = get_mnist_test_loader(batch_size=batch_size)
    for cln_data, true_label in loader:
        break
    cln_data, true_label = cln_data.to(device), true_label.to(device)


    #Create Defence
    bits_squeezing = BitSqueezing(bit_depth=1)
    median_filter = MedianSmoothing2D(kernel_size=3)


    defense = nn.Sequential(


        bits_squeezing,
        median_filter,
    )





        #add defence.nn
    adv_untargeted = pretrained_G(cln_data)
    adv_untargeted = torch.clamp(adv_untargeted, -0.3, 0.3)
    adv = adv_untargeted+cln_data
    adv_img = torch.clamp(adv, 0, 1)
    adv_defended = defense(adv_img)
    pred_untargeted_adv = predict_from_logits(model(adv_untargeted))
    pred_cln = predict_from_logits(model(cln_data))

    pred_adv = predict_from_logits(model(adv))

    pred_adv_defended = predict_from_logits(model(adv_defended))


    plt.figure(figsize=(10, 8))
    for ii in range(batch_size):
        plt.subplot(3, batch_size, ii + 1)
        _imshow(cln_data[ii])
        plt.title("clean \n pred: {}".format(pred_cln[ii]))
        plt.subplot(3, batch_size, ii + 1 + batch_size)
        _imshow(adv_untargeted[ii])
        plt.title("untargeted \n AIGAN \n pred: {}".format(
            pred_untargeted_adv[ii]))
        plt.subplot(4, batch_size, ii + 1 + batch_size * 3)
        _imshow(adv_defended[ii])
        plt.title("defended AIGAN \n pred: {}".format(
            pred_adv_defended[ii]))


        plt.tight_layout()
        plt.show()

    loader = get_mnist_test_loader(batch_size=batch_size)

    num_correct = 0
    num_all = 0
    for i, data in enumerate(loader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        adv_untargeted = pretrained_G(test_img)
        adv_untargeted = torch.clamp(adv_untargeted, -0.3, 0.3)
        adv = adv_untargeted + test_img
        adv_img = torch.clamp(adv, 0, 1)
        adv_defended = defense(adv_img)
        pred_lab = torch.argmax((model(adv_defended)), 1)
        num_all += len(test_label)
        num_correct += torch.sum(pred_lab == test_label, 0)

    print('MNIST training dataset:')
    print('num_examples: ', num_all)
    print('num_correct: ', num_correct.item())
    print('accuracy of defense imgs in testing set: %f\n' % (num_correct.item() / num_all))
    loader = get_mnist_test_loader(batch_size=10000)
    for i, data in enumerate(loader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        adv_untargeted = pretrained_G(test_img)
        adv_untargeted = torch.clamp(adv_untargeted, -0.3, 0.3)
        adv = adv_untargeted + test_img
        adv_img = torch.clamp(adv, 0, 1)
        adv_defended = defense(adv_img)
        pred_lab = torch.argmax((model(adv_defended)), 1)
        cm = confusion_matrix(y_true = test_label, y_pred = pred_lab)
        print(cm)