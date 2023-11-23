if __name__ == '__main__':
    import torch
    import torchvision
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt
    import numpy as np

    # Todo - pass a pretrainded model
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet56", pretrained=True)

    use_cuda = True

    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    def unnormalize(img, mean = np.array(norm_mean), std = np.array(norm_std)):
        inverse_mean = - mean/std
        inverse_std = 1/std
        img = transforms.Normalize(mean=-mean/std, std=1/std)(img)
        return img

    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    def ax_imshow(ax, img, label):
        img = unnormalize(img)     # unnormalize
        img = np.clip(img, 0., 1.)
        ax.set_xticks([])
        ax.set_yticks([])
        img = np.transpose(img, (1,2,0))
        ax.imshow(img)
        ax.set_title(classes[label])

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

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        

    test_accuracy, resnet56_labels, orig = test_acc_new(model, testloader, org_img, mt, True)
    # Here no need for another 0

    # TODO pass the epsilon

    epsilons=[0.005, 0.01,0.02,0.03,0.05]
    accuracies=[]

    from pgd import PGD
    # from deepfool import DeepFool

    fgsm_img = []
    # attack=FGSM(model)
    # # attack = torchattacks.FGSM(model)
    # # batch?
    # for i in range(st):
    #         fgsm_img.append(attack(org_img[i], org_labels[i]))
    # # test_accuracy, labels= test_acc_new(model, testloader, org_img, mt, False)
    # # print(test_accuracy)
    # fgsm_test_accuracy, fgsm_labels= test_acc_new(model, testloader, fgsm_img, st, False)
    # print(fgsm_test_accuracy)

    eps_times = 0
    fin_image = []
    fgsm_examples = []
    for eps in epsilons:
        visual_examples = 5
        fgsm_img = []
        attack = PGD(model, eps=eps)
        count = 0
        for i in range(mt):
            fgsm_img.append(attack(org_img[i], org_labels[i]))

        # if () Save the images

        fgsm_test_accuracy, fgsm_labels, a_images = test_acc_new(model, testloader, fgsm_img, mt, True)
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        accuracies.append(fgsm_test_accuracy)
        fgsm_examples.append(a_images)



    plt.figure(figsize=(5,5))
    plt.plot([0] + epsilons, [test_accuracy] + accuracies, "*-")
    plt.yticks(np.arange(0.0, test_accuracy, step=10))
    plt.xticks(np.arange(0.0, max(epsilons), step=max(epsilons)/5))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.show()