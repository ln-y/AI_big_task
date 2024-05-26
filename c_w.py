import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os, shutil
from tqdm import tqdm
from model import ViolenceClassifier  # Import from model.py
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-model',type=str,help="path of model file *.ckpt")
parser.add_argument('-eps',type=float,help="the value of `ep`")
parser.add_argument('-num',type=int,default=-1,help="the num of samples")
args = parser.parse_args()
print(args)


device="cuda" if torch.cuda.is_available() else "cpu"

# Load model and set to evaluation mode
model = ViolenceClassifier()
model_path = args.model
model.load_from_checkpoint(model_path)
model.eval()
model.to(device)

def load_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image.to(device)

def cw_attack(model, image, label, targeted=False, c=1e-4, kappa=0, max_iter=100, learning_rate=0.01):
    # Set device
    device = next(model.parameters()).device
    image = image.to(device)
    label = torch.tensor([label]).to(device)

    # Define box constraints [0, 1]
    box_min = torch.zeros_like(image).to(device)
    box_max = torch.ones_like(image).to(device)

    # Initialize perturbation
    w = torch.zeros_like(image).to(device)
    w.requires_grad = True

    optimizer = torch.optim.Adam([w], lr=learning_rate)

    for i in range(max_iter):
        perturbed_image = torch.tanh(w) * 0.5 + 0.5  # Scale to [0, 1]
        perturbed_image = image + perturbed_image * (box_max - box_min)
        output = model(perturbed_image)

        real = output.gather(1, label.unsqueeze(1)).squeeze(1)
        other = (output - 1e4 * torch.eye(output.shape[1])[label].to(device)).max(1)[0]

        if targeted:
            loss1 = F.relu(other - real + kappa)
        else:
            loss1 = F.relu(real - other + kappa)

        loss2 = torch.sum((perturbed_image - image) ** 2)
        loss = loss1 + c * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return perturbed_image.detach()

def save_perturbed_images(directory, cw_directory):
    if os.path.exists(cw_directory):
        shutil.rmtree(cw_directory)
    os.makedirs(cw_directory)

    files=os.listdir(directory)
    random.shuffle(files)
    if args.num>0:
        files=files[:args.num]
    # 遍历目录中的所有图片
    for filename in tqdm(files,desc="generating"):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)
            image = load_image(image_path)
            label = int(filename.split('_')[0])  # Extract label from filename
            perturbed_image = cw_attack(model, image, label, targeted=False, c=0.1, kappa=0, max_iter=20, learning_rate=0.001)

            # Save the adversarial image
            save_path = os.path.join(cw_directory, filename)
            save_image = transforms.ToPILImage()(perturbed_image.squeeze(0))
            save_image.save(save_path)
            # print(f'{filename} saved.')

# Set directories and parameters
test_directory = 'test'
cw_directory = f'c_w_eps={args.eps}'  # Directory for C&W adversarial images

# Generate and save adversarial images
save_perturbed_images(test_directory, cw_directory)
