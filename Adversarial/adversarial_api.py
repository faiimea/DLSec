import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image,ImageFilter
import cv2
import io
import time
from tqdm import tqdm  # Import tqdm for creating progress bars

from Adversarial.fgsm import FGSM
from Adversarial.pgd import PGD
from Adversarial.difgsm import DIFGSM
from Adversarial.mifgsm import MIFGSM
from Adversarial.nifgsm import NIFGSM
from Adversarial.sinifgsm import SINIFGSM
from Adversarial.tifgsm import TIFGSM
from Adversarial.vmifgsm import VMIFGSM
from Adversarial.vnifgsm import VNIFGSM




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

def dataset_preprocess():
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
    
    return testloader

# 高斯噪声
def add_gaussian_noise(image, mean=0, std=1):
    """
    在图像上添加高斯噪声
    """
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return noisy_image

# 高斯模糊
def apply_gaussian_blur(image, radius=2):
    """
    对图像进行高斯模糊
    """
    blurred_images = []
    for i in range(image.shape[0]):  # 遍历 batch 中的每张图像
        image_tensor = image[i].squeeze(0)  # 去除单一的 batch 维度
        image_pil = transforms.ToPILImage()(image_tensor)
        blurred_image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred_image = transforms.ToTensor()(blurred_image_pil)
        blurred_images.append(blurred_image)

    # 将处理后的图像转换为张量，并在第一个维度上堆叠，形成 batch
    blurred_images = torch.stack(blurred_images, dim=0)
    return blurred_images

# 图像压缩
def compress_image(image, quality=85):
    """
    对图像进行压缩
    """
    compressed_images = []
    for i in range(image.shape[0]):  # 遍历 batch 中的每张图像
        image_tensor = image[i].squeeze(0)  # 去除单一的 batch 维度
        image_pil = transforms.ToPILImage()(image_tensor)
        
        # 压缩图像
        buffer = io.BytesIO()
        image_pil.save(buffer, format='JPEG', quality=quality)
        compressed_image_pil = Image.open(buffer)
        compressed_image = transforms.ToTensor()(compressed_image_pil)
        
        compressed_images.append(compressed_image)

    compressed_images = torch.stack(compressed_images, dim=0)
    
    return compressed_images

def test_acc_new(model, testloader, test_images, n_image = 448, save_test_images = False):
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

                    # Here change batchsize !!!
                    p_labels.append(predicted)
                    if predicted[total%64] == j:
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


def plot_ACC(epsilons,test_accuracy,accuracies):
    plt.figure(figsize=(5,5))
    plt.plot([0] + epsilons, [test_accuracy] + accuracies, "*-")
    plt.yticks(np.arange(0.0, test_accuracy, step=10))
    plt.xticks(np.arange(0.0, max(epsilons), step=max(epsilons)/5))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()


def adversarial_attack(model=None,method="fgsm", train_dataloader=None, params=None):
    print('atk_start')
    if model is None:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    
    #print(model)
    use_cuda = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if train_dataloader is None:
        testloader=dataset_preprocess()
    else:
        testloader=train_dataloader

    org_img = []
    org_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            org_img.append(images)
            org_labels.append(labels)

    '''
    TODO 图片高斯噪声 图片高斯模糊 图片压缩鲁棒
    '''
    print('Noise start')
    noisy_images = [add_gaussian_noise(img) for img in org_img]
    print('Blurred start')
    blurred_images = [apply_gaussian_blur(img) for img in org_img]
    print('Compressed start')
    compressed_images = [compress_image(img) for img in org_img]

    
    
    # Small test
    st = 100
    # Midium test
    mt = 400
    # Large test
    lt = 1000
    # Full test
    ft = 2500
    
    data_scale=mt

    test_accuracy, resnet56_labels, orig = test_acc_new(model, testloader, org_img, data_scale, True)

    # epsilons=[0.005,0.01,0.02,0.05,0.1]
    epsilons=[0.005]

    progress_bar = tqdm(total=4*len(epsilons), desc="Generating Adversarial Examples")
    accuracies=[]
    accuracies_noisy=[]
    accuracies_blurred=[]
    accuracies_compressed=[]
    atk_examples = []
    for eps in epsilons:
        #visual_examples = 5
        if method == 'fgsm':
            attack = FGSM(model, eps=eps)
        elif method == 'pgd':
            attack = PGD(model, eps=eps)
        elif method == 'difgsm':
            attack = DIFGSM(model, eps=eps)
        elif method == 'mifgsm':
            attack = MIFGSM(model, eps=eps)
        elif method == 'nifgsm':
            attack = NIFGSM(model, eps=eps)
        elif method == 'sinifgsm':
            attack = SINIFGSM(model, eps=eps)
        elif method == 'tifgsm':
            attack = TIFGSM(model, eps=eps)
        elif method == 'vmifgsm':
            attack = VMIFGSM(model, eps=eps)
        elif method == 'vnifgsm':
            attack = VNIFGSM(model, eps=eps)
        else:
            print('Unsupported attack method:', method)

        count = 0
        start_time = time.time()  # 记录开始时间
        attack_img = []
        for i in range(len(org_img)):
            attack_img.append(attack(org_img[i], org_labels[i]))
        elapsed_time = time.time() - start_time
        
        progress_bar.update(1)
        # Process noisy_images
        attack_img_noisy = []
        for i in range(len(noisy_images)):
            attack_img_noisy.append(attack(noisy_images[i], org_labels[i]))
        progress_bar.update(1)
        # Process blurred_images
        attack_img_blurred = []
        for i in range(len(blurred_images)):
            attack_img_blurred.append(attack(blurred_images[i], org_labels[i]))
        progress_bar.update(1)
        # Process compressed_images
        attack_img_compressed = []
        for i in range(len(compressed_images)):
            attack_img_compressed.append(attack(compressed_images[i], org_labels[i]))
        progress_bar.update(1)
        print(elapsed_time)
        atk_test_accuracy, atk_labels, a_images = test_acc_new(model, testloader, attack_img, mt, True)
        atk_test_accuracy_noisy, _, _ = test_acc_new(model, testloader, attack_img_noisy, mt, True)
        atk_test_accuracy_blurred, _, _ = test_acc_new(model, testloader, attack_img_blurred, mt, True)
        atk_test_accuracy_compressed, _, _ = test_acc_new(model, testloader, attack_img_compressed, mt, True)
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        accuracies.append(atk_test_accuracy)
        accuracies_noisy.append(atk_test_accuracy_noisy)
        accuracies_blurred.append(atk_test_accuracy_blurred)
        accuracies_compressed.append(atk_test_accuracy_compressed)
        atk_examples.append(a_images)


    # Visualize
    # plot_ACC(epsilons,test_accuracy,accuracies)
    # plot_ACC(epsilons,test_accuracy,accuracies_noisy)
    # plot_ACC(epsilons,test_accuracy,accuracies_blurred)
    # plot_ACC(epsilons,test_accuracy,accuracies_compressed)

if __name__ == '__main__':


    # Todo - pass a pretrainded model
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    use_cuda = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testloader=dataset_preprocess()

    org_img = []
    org_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
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
    
    data_scale=mt

    test_accuracy, resnet56_labels, orig = test_acc_new(model, testloader, org_img, data_scale, True)

    epsilons=[0.005,]
    accuracies=[]
    atk_examples = []
    for eps in epsilons:
        visual_examples = 5
        attack_img = []
        attack = FGSM(model,eps=eps)
        # if(args.method == 'FGSM'):
        #     attack = FGSM(model,eps=eps)
        # elif(args.method == 'PGD'):
        #     attack = PGD(model, eps=eps)
        # else:
        #     print('ERROR')
        count = 0
        for i in range(mt):
            attack_img.append(attack(org_img[i], org_labels[i]))

        atk_test_accuracy, atk_labels, a_images = test_acc_new(model, testloader, attack_img, mt, True)
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        accuracies.append(atk_test_accuracy)
        atk_examples.append(a_images)


    # Visualize
    plot_ACC(epsilons,test_accuracy,accuracies)