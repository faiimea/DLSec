import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse

from fgsm import FGSM
from pgd import PGD
from difgsm import DIFGSM
from mifgsm import MIFGSM
from nifgsm import NIFGSM
from sinifgsm import SINIFGSM
from tifgsm import TIFGSM
from vmifgsm import VMIFGSM
from vnifgsm import VNIFGSM




# Hyberpara
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

'''
需要传递的参数：
data path
attack method
IF save image
Epsilon
Test Scale
'''

# def unnormalize(img, mean = np.array(norm_mean), std = np.array(norm_std)):
#     inverse_mean = - mean/std
#     inverse_std = 1/std
#     img = transforms.Normalize(mean=-mean/std, std=1/std)(img)
#     return img

# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))

# def ax_imshow(ax, img, label):
    # img = unnormalize(img)     # unnormalize
    # img = np.clip(img, 0., 1.)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # img = np.transpose(img, (1,2,0))
    # ax.imshow(img)
    # ax.set_title(classes[label])

def test_acc_new(model, testloader, test_images, n_image = 100, save_test_images = False):
        correct = 0
        total = 0
        p_labels = []
        saved_img = []
        i = 0
        for data in testloader:
            images, labels = data
            if total < n_image:
                outputs = model(test_images[i])
                _, predicted = torch.max(outputs.data, 1)
                # The torch.max return 2 value, which respectively represents the max value and the index
                # Because of the batch, the predicted here is a tensor with 4 elements

                # Judge the lael here
                for j in labels:
                    # if save_test_images and total % 4 == 0:
                    #     saved_img.append(test_images[total])
                    p_labels.append(predicted)
                    if predicted[total%4] == j:
                        correct += 1
                    total += 1
                i += 1
            else:
                break
            
        test_accuracy = (100.0 * correct / total)
        print('Accuracy of the network on the', total, "images is: ", test_accuracy, '%')
        print("Saving test images = ", save_test_images)
        if save_test_images == True:
            return test_accuracy, p_labels, saved_img
        else:
            return test_accuracy, p_labels

if __name__ == '__main__':

    parser=argparse.ArgumentParser(description='Adversarial Sample Test')
    parser.add_argument('--datapath',type=str,help='a',default='../data')
    parser.add_argument('--datatype',type=str,help='a',default='CIFAR10')
    parser.add_argument('--method',type=str,help='a',default='FGSM')
    parser.add_argument('--save',type=bool,help='a',default=False)
    parser.add_argument('--eps',type=str,help='a',default='?')
    parser.add_argument('--scale',type=str,help='a',default='?')

    args=parser.parse_args()

    # Todo - pass a pretrainded model
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    use_cuda = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if(args.datapath=='CIFAR10'):
        pass
    else:
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
        my_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

        trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                shuffle=False, num_workers=2)

    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    
    # Opt? Cache all image in Memory
    org_img = []
    org_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            print(images.shape)
            org_img.append(images)
            org_labels.append(labels)

    # definition of const

    # Small test
    st = 100
    # Midium test
    mt = 400
    # Large test
    lt = 1000
    # Full test
    ft = 2500
    
    if(args.scale=='st'):
        pass
    data_scale=mt

    test_accuracy, resnet56_labels, orig = test_acc_new(model, testloader, org_img, data_scale, True)
    # Here no need for another 0

    # TODO pass the epsilon

    epsilons=[0.005, 0.01,0.02,0.03,0.05]
    accuracies=[]
    atk_examples = []
    for eps in epsilons:
        visual_examples = 5
        attack_img = []
        if(args.method == 'FGSM'):
            attack = FGSM(model,eps=eps)
        elif(args.method == 'PGD'):
            attack = PGD(model, eps=eps)
        else:
            print('ERROR')
        count = 0
        for i in range(mt):
            attack_img.append(attack(org_img[i], org_labels[i]))

        # Here add guass noise

        # if () Save the images

        atk_test_accuracy, atk_labels, a_images = test_acc_new(model, testloader, attack_img, mt, True)
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        accuracies.append(atk_test_accuracy)
        atk_examples.append(a_images)


    # Visualize
    plt.figure(figsize=(5,5))
    plt.plot([0] + epsilons, [test_accuracy] + accuracies, "*-")
    plt.yticks(np.arange(0.0, test_accuracy, step=10))
    plt.xticks(np.arange(0.0, max(epsilons), step=max(epsilons)/5))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()