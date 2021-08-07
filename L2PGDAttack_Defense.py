import torch
import torch.nn as nn
from models import MNIST_target_net
from advertorch.utils import predict_from_logits
from advertorch_examples.utils import get_mnist_test_loader
from advertorch_examples.utils import _imshow
from advertorch.attacks import L2PGDAttack
from advertorch.defenses import MedianSmoothing2D
from advertorch.defenses import BitSqueezing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from models import MNIST_target_net
import models

if __name__ == "__main__":
    image_nc = 1
    # batch_size = 128
    gen_input_nc = image_nc

    torch.manual_seed(0)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    pretrained_model = "./MNIST_target_model.pth"
    model = MNIST_target_net().to(device)
    model.load_state_dict(torch.load(pretrained_model))
    model.eval()

    batch_size = 5
    loader = get_mnist_test_loader(batch_size=batch_size)
    for cln_data, true_label in loader:
        break
    cln_data, true_label = cln_data.to(device), true_label.to(device)

    adversary = L2PGDAttack(
        model, eps=2., eps_iter=1. , nb_iter=40,

        rand_init=False, targeted=False)

    adv_ = adversary.perturb(cln_data, true_label)
    pred_cln = predict_from_logits(model(cln_data))
    pred_adv_ = predict_from_logits(model(adv_))

    plt.figure(figsize=(8, 6))
    for ii in range(batch_size):
        plt.subplot(3, batch_size, ii + 1)
        _imshow(cln_data[ii])
        plt.title("clean \n pred: {} ".format(pred_cln[ii]))
        plt.subplot(3, batch_size, ii + 1 + batch_size)

        _imshow(adv_[ii])

        plt.title("\n  \n adv \n pred: {}".format(
            pred_adv_[ii]))

    plt.tight_layout()
    plt.show()

    bits_squeezing = BitSqueezing(bit_depth=1)
    median_filter = MedianSmoothing2D(kernel_size=3)

    defense = nn.Sequential(

        bits_squeezing,
        median_filter,
    )

    adv = adv_
    adv_defended = defense(adv)
    cln_defended = defense(cln_data)

    pred_cln = predict_from_logits(model(cln_data))
    pred_cln_defended = predict_from_logits(model(cln_defended))
    pred_adv = predict_from_logits(model(adv))
    pred_adv_defended = predict_from_logits(model(adv_defended))

    plt.figure(figsize=(10, 10))
    for ii in range(batch_size):
        plt.subplot(6, batch_size, ii + 1 + batch_size * 2)
        _imshow(adv[ii])
        plt.title("adv \n pred: {}".format(
            pred_adv[ii]))
        plt.subplot(6, batch_size, ii + 1 + batch_size * 3)
        _imshow(adv_defended[ii])
        plt.title("defended adv \n pred: {}".format(
            pred_adv_defended[ii]))

    plt.tight_layout()
    plt.show()


    num_correct = 0
    num_all = 0
    for i, data in enumerate(loader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        adv_ = adversary.perturb(test_img, true_label)
        pred_cln = predict_from_logits(model(test_img))
        pred_adv_ = predict_from_logits(model(adv_))

        adv_defended = defense(adv_)
        pred_lab = predict_from_logits(model(adv_defended))
        num_all += len(test_label)
        num_correct += torch.sum(pred_adv_ == test_label, 0)

    print('MNIST training dataset:')
    print('num_examples: ', num_all)
    print('num_correct: ', num_correct.item())
    print('accuracy of defensed imgs in testing set: %f\n' % (num_correct.item() / num_all))

    loader = get_mnist_test_loader(batch_size=10000)
    for i, data in enumerate(loader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        adv_untargeted = adversary.perturb(test_img)
        adv_untargeted = torch.clamp(adv_untargeted, -0.3, 0.3)
        adv = adv_untargeted + test_img
        adv_img = torch.clamp(adv, 0, 1)
        adv_defended = defense(adv_img)
        pred_lab = torch.argmax((model(adv_img)), 1)
        cm = confusion_matrix(y_true = test_label, y_pred = pred_lab)
        print(cm)

        num_correct = 0
        num_all = 0
        for i, data in enumerate(loader, 0):
            test_img, test_label = data
            test_img, test_label = test_img.to(device), test_label.to(device)
            adv_ = adversary.perturb(test_img, true_label)
            pred_cln = predict_from_logits(model(test_img))
            pred_adv_ = predict_from_logits(model(adv_))
            num_all += len(test_label)
            num_correct += torch.sum(pred_adv_ == test_label, 0)

        print('MNIST training dataset:')
        print('num_examples: ', num_all)
        print('num_correct: ', num_correct.item())
        print('accuracy of adv imgs in testing set: %f\n' % (num_correct.item() / num_all))



        loader = get_mnist_test_loader(batch_size=10000)
        for i, data in enumerate(loader, 0):
            test_img, test_label = data
            test_img, test_label = test_img.to(device), test_label.to(device)
            adv_untargeted = adversary.perturb(test_img)
            adv_untargeted = torch.clamp(adv_untargeted, -0.3, 0.3)
            adv = adv_untargeted + test_img
            adv_img = torch.clamp(adv, 0, 1)
            pred_lab = torch.argmax((model(adv_img)), 1)
            cm = confusion_matrix(y_true=test_label, y_pred=pred_lab)
            print(cm)