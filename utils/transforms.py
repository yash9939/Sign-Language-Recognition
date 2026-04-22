from torchvision import transforms

train_transform = transforms.Compose([
<<<<<<< HEAD
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
=======
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(25),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.RandomAffine(0, translate=(0.1,0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
<<<<<<< HEAD
]) ##
=======
>>>>>>> e7f94120b8e68fc0d16059433aa44b447e4ec253
])
>>>>>>> 65ba29d3f8e1316bfc9b362047c694bf363a8ec2
