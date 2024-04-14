import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
from PIL import Image, ImageFilter
import cv2
import io
import time
from tqdm import tqdm  # Import tqdm for creating progress bars
from tabulate import tabulate

from Adversarial.fgsm import FGSM
from Adversarial.pgd import PGD
from Adversarial.difgsm import DIFGSM
from Adversarial.mifgsm import MIFGSM
from Adversarial.nifgsm import NIFGSM
from Adversarial.sinifgsm import SINIFGSM
from Adversarial.tifgsm import TIFGSM
from Adversarial.vmifgsm import VMIFGSM
from Adversarial.vnifgsm import VNIFGSM

from sklearn.metrics import accuracy_score

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


def test_CACC(model, data_loader,device="cuda"):
    model.to(device)
    model.eval()
    y_true = []
    y_predict = []
    for (batch_x, batch_y) in tqdm(data_loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        batch_y_predict = model(batch_x)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)
    return accuracy_score(y_true.cpu(), y_predict.cpu())


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
def add_gaussian_noise(image, mean=0, std=0.3):
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
        image_tensor = image[i].squeeze(0)
        image_pil = transforms.ToPILImage()(image_tensor)
        blurred_image_pil = image_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred_image = transforms.ToTensor()(blurred_image_pil)
        blurred_images.append(blurred_image)
    blurred_images = torch.stack(blurred_images, dim=0)
    return blurred_images


# 图像压缩
def compress_image(image, quality=98):
    """
    对图像进行压缩
    """
    compressed_images = []
    for i in range(image.shape[0]):  # 遍历 batch 中的每张图像
        image_tensor = image[i].squeeze(0)
        image_pil = transforms.ToPILImage()(image_tensor)
        buffer = io.BytesIO()
        image_pil.save(buffer, format='JPEG', quality=quality)
        compressed_image_pil = Image.open(buffer)
        compressed_image = transforms.ToTensor()(compressed_image_pil)
        compressed_images.append(compressed_image)
    compressed_images = torch.stack(compressed_images, dim=0)

    return compressed_images


def test_acc(model, testloader, test_images, n_image=448, save_test_images=False, batch_size=64):
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
                if predicted[total % len(labels)] == j:
                    correct += 1
                total += 1
            i += 1
        else:
            break

    test_accuracy = (100.0 * correct / total)
    # print('Accuracy of the network on the', total, "images is: ", test_accuracy, '%')
    # print("Saving test images = ", save_test_images)
    if save_test_images == True:
        return test_accuracy, p_labels, saved_img
    else:
        return test_accuracy, p_labels


def plot_ACC(epsilons, test_accuracy, accuracies):
    plt.figure(figsize=(5, 5))
    plt.plot([0] + epsilons, [test_accuracy] + accuracies, "*-")
    plt.yticks(np.arange(0.0, test_accuracy, step=10))
    plt.xticks(np.arange(0.0, max(epsilons), step=max(epsilons) / 5))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()


def adversarial_attack(model=None, method="fgsm", train_dataloader=None, params=None):
    if model is None:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

    # print(model)
    use_cuda = True
    device = params['device']
    model.to(device)

    if train_dataloader is None:
        testloader = dataset_preprocess()
    else:
        testloader = train_dataloader

    org_img = []
    org_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            org_img.append(images.to(device))
            org_labels.append(labels.to(device))


    print('Noise start')
    noisy_images = [add_gaussian_noise(img).to(device) for img in org_img]

    # Small test
    st = 100
    # Midium test
    mt = 400
    # Large test
    lt = 1000
    # Full test
    ft = 2500

    data_scale = mt

    # test_accuracy, resnet56_labels, orig = test_acc(model, testloader, org_img, data_scale, True)

    # epsilons=[0.005,0.01,0.02,0.05,0.1]
    epsilons = [0.005]
    table_data = []
    progress_bar = tqdm(total=4 * len(epsilons), desc="Generating Adversarial Examples")
    accuracies = []
    accuracies_noisy = []
    accuracies_blurred = []
    accuracies_compressed = []
    atk_examples = []
    for eps in epsilons:
        # visual_examples = 5
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
        atk_test_accuracy, atk_labels, a_images = test_acc(model, testloader, attack_img, mt, True)
        del attack_img
        progress_bar.update(1)
        # Process noisy_images

        attack_img_noisy = []
        for i in range(len(noisy_images)):
            attack_img_noisy.append(attack(noisy_images[i], org_labels[i]))
        atk_test_accuracy_noisy, _, _ = test_acc(model, testloader, attack_img_noisy, mt, True)
        del attack_img_noisy
        del noisy_images
        progress_bar.update(1)

        print('Blurred start')
        blurred_images = [apply_gaussian_blur(img).to(device) for img in org_img]

        # Process blurred_images
        attack_img_blurred = []

        for i in range(len(blurred_images)):
            attack_img_blurred.append(attack(blurred_images[i], org_labels[i]))
        atk_test_accuracy_blurred, _, _ = test_acc(model, testloader, attack_img_blurred, mt, True)

        del attack_img_blurred
        del blurred_images
        progress_bar.update(1)

        print('Compressed start')
        compressed_images = [compress_image(img).to(device) for img in org_img]
        # Process compressed_images
        attack_img_compressed = []
        for i in range(len(compressed_images)):
            attack_img_compressed.append(attack(compressed_images[i], org_labels[i]))
        progress_bar.update(1)
        print(elapsed_time)



        atk_test_accuracy_compressed, _, _ = test_acc(model, testloader, attack_img_compressed, mt, True)
        table_data.append([eps, atk_test_accuracy, atk_test_accuracy_noisy, atk_test_accuracy_blurred,
                           atk_test_accuracy_compressed, elapsed_time])
        dataiter = iter(testloader)
        images, labels = next(dataiter)
        accuracies.append(atk_test_accuracy)
        accuracies_noisy.append(atk_test_accuracy_noisy)
        accuracies_blurred.append(atk_test_accuracy_blurred)
        accuracies_compressed.append(atk_test_accuracy_compressed)
        atk_examples.append(a_images)
    headers = ["Epsilon", "ACC", "Noisy ACC", "Blurred ACC", "Compressed ACC", "Time (seconds)"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    return table_data

    # Visualize
    # plot_ACC(epsilons,test_accuracy,accuracies)
    # plot_ACC(epsilons,test_accuracy,accuracies_noisy)
    # plot_ACC(epsilons,test_accuracy,accuracies_blurred)
    # plot_ACC(epsilons,test_accuracy,accuracies_compressed)


def adversarial_mutiple_attack(model=None, train_dataloader=None, params=None):
    if model is None:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

    # print(model)
    use_cuda = True
    device = params['device']
    print(device)
    model.to(device)
    if train_dataloader is None:
        testloader = dataset_preprocess()
    else:
        testloader = train_dataloader

    org_img = []
    org_labels = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            org_img.append(images)
            org_labels.append(labels)

    # Small test
    st = 100
    # Midium test
    mt = 400
    # Large test
    lt = 1000
    # Full test
    ft = 2500

    data_scale = st

    test_accuracy, resnet56_labels, orig = test_acc(model, testloader, org_img, data_scale, True)

    # epsilons=[0.005,0.01,0.02,0.05,0.1]
    epsilons = [0.005]
    table_data = []
    accuracies = []
    # attack_methods = ['fgsm', 'pgd', 'difgsm', 'mifgsm', 'nifgsm', 'sinifgsm', 'tifgsm', 'vmifgsm', 'vnifgsm']
    attack_methods = ['fgsm', 'pgd', 'difgsm', 'mifgsm', 'nifgsm',  'tifgsm']
    progress_bar = tqdm(total=len(attack_methods) * len(epsilons), desc="Generating Adversarial Examples")
    for eps in epsilons:
        for method in attack_methods:
            if method == 'fgsm':
                attack = FGSM(model, eps=eps, device=params['device'])
            elif method == 'pgd':
                attack = PGD(model, eps=eps, device=params['device'])
            elif method == 'difgsm':
                attack = DIFGSM(model, eps=eps, device=params['device'])
            elif method == 'mifgsm':
                attack = MIFGSM(model, eps=eps, device=params['device'])
            elif method == 'nifgsm':
                attack = NIFGSM(model, eps=eps, device=params['device'])
            # 慢
            elif method == 'sinifgsm':
                attack = SINIFGSM(model, eps=eps, device=params['device'])
            elif method == 'tifgsm':
                attack = TIFGSM(model, eps=eps, device=params['device'])
            # 慢
            elif method == 'vmifgsm':
                attack = VMIFGSM(model, eps=eps, device=params['device'])
            # 慢
            elif method == 'vnifgsm':
                attack = VNIFGSM(model, eps=eps, device=params['device'])

            count = 0
            start_time = time.time()  # 记录开始时间
            attack_img = []
            for i in range(len(org_img)):
                attack_img.append(attack(org_img[i], org_labels[i]))
            elapsed_time = time.time() - start_time
            progress_bar.update(1)
            atk_test_accuracy, _, _ = test_acc(model, testloader, attack_img, data_scale, True)

            table_data.append([eps, method, atk_test_accuracy, elapsed_time])
            dataiter = iter(testloader)
            images, labels = next(dataiter)

    headers = ["Epsilon", "Attack Method", "ACC", "Time (seconds)"]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    return table_data


def robust_test():
    pass


if __name__ == '__main__':

    # Todo - pass a pretrainded model
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)
    use_cuda = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    testloader = dataset_preprocess()

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

    data_scale = mt

    test_accuracy, resnet56_labels, orig = test_acc_new(model, testloader, org_img, data_scale, True)

    epsilons = [0.005, ]
    accuracies = []
    atk_examples = []
    for eps in epsilons:
        visual_examples = 5
        attack_img = []
        attack = FGSM(model, eps=eps)
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
    plot_ACC(epsilons, test_accuracy, accuracies)
