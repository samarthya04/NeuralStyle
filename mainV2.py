import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from serpapi import GoogleSearch
import os
import time
import tracemalloc
import numpy as np
from sklearn.preprocessing import normalize
#from PyEMD import EMD  # Commented out initially - try different import methods
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# SerpAPI Key (replace with your actual key)
API_KEY = "26a42832f68a3fb2c572dd2ed728bc0cefcbab28791920814c40a443f4d95bd2"

def download_first_google_image(query, save_path="downloaded_image.jpg"):
    params = {
        "q": query,
        "tbm": "isch",
        "num": 1,
        "api_key": API_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    if "images_results" in results and results["images_results"]:
        image_url = results["images_results"][0]["original"]
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Image downloaded successfully: {save_path}")
        else:
            print("Failed to download image.")
            exit()
    else:
        print("No images found.")

def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_name).convert('RGB')
    return loader(image).unsqueeze(0).to(device, torch.float)

def imshow(tensor, title=None):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    return model, style_losses, content_losses

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=500,
                       style_weight=100000, content_weight=10):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses) * style_weight
            content_score = sum(cl.loss for cl in content_losses) * content_weight
            loss = style_score + content_score
            loss.backward()
            if run[0]%50 == 0:
                print(f'\nRun = {run[0]}: Total Loss = {loss}')
            run[0] += 1
            return loss
        optimizer.step(closure)
    with torch.no_grad():
        input_img.clamp_(0, 1)
    return input_img

# Attempt to import EMD and provide a fallback
emd_installed = False  # Initialize emd_installed *before* the try block
try:
    from PyEMD import EMD
    emd_installed = True
    print("EMD from PyEMD imported successfully.")
except ImportError:
    try:
        from EMD_signal import EMD  # Or the correct import for EMD-signal
        emd_installed = True
        print("EMD from EMD_signal imported successfully.")
    except ImportError:
        print("EMD library not found. STI will use histogram comparison.")
        emd_installed = False

def calculate_sti(style_img, output):
    global emd_installed # Access global variable

    if emd_installed:
        try:
            emd = EMD()
            style_imfs = emd(style_img.view(-1).detach().cpu().numpy())
            output_imfs = emd(output.view(-1).detach().cpu().numpy())
            return np.mean(np.abs(style_imfs - output_imfs))
        except Exception as e:
            print(f"Error calculating STI with EMD: {e}. Falling back to histogram comparison.")
            emd_installed = False  # Ensure we don't try EMD again if it fails once.
    if not emd_installed:  # Fallback if EMD is not available
        style_hist, _ = np.histogram(style_img.view(-1).detach().cpu().numpy(), bins=50, density=True)
        output_hist, _ = np.histogram(output.view(-1).detach().cpu().numpy(), bins=50, density=True)
        return np.mean(np.abs(style_hist - output_hist))

def calculate_content_preservation(content_img, styled_img):
    content_np = content_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    styled_np = styled_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    return ssim(content_np, styled_np, multichannel=True, win_size=3, data_range=1.0)

def calculate_style_similarity(style_img, styled_img):
    style_np = style_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    styled_np = styled_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    return ssim(style_np, styled_np, multichannel=True, win_size=3, data_range=1.0)


if __name__ == "__main__":
    imsize = 512 if torch.cuda.is_available() else 256
    content_img = image_loader("content.jpg", imsize)
    
    prompt = input("Enter the image style: ")
    download_first_google_image(prompt, "style.jpg")
    
    num = int(input("Enter the number of iterations you want (Default = 500): "))
    style_weight, content_weight = int(input("Enter the style weight (Default = 100000): ")), int(input("Enter the content weight (Default = 10): "))
    
    
    # Check if style image was downloaded successfully
    if not os.path.exists("style.jpg"):
        print("Failed to download style image. Exiting.")
        exit()
    
    style_img = image_loader("style.jpg", imsize)
    input_img = content_img.clone()

    # Start performance measurements
    start_time = time.time()
    tracemalloc.start()

    # Run style transfer
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, content_img, style_img, input_img, num_steps=num, style_weight=style_weight, content_weight=content_weight)

    # End performance measurements
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate performance metrics
    processing_time = end_time - start_time
    print(f"Style transfer took {processing_time:.2f} seconds")
    print(f"Current memory usage: {current / 10**6:.2f} MB")
    print(f"Peak memory usage: {peak / 10**6:.2f} MB")

    # Save the output image
    output_image = output.squeeze(0).cpu().clone()
    output_image = transforms.ToPILImage()(output_image)
    output_image.save("styled_image.jpg")

    # Calculate additional metrics
    sti_score = calculate_sti(style_img, output)
    print(f"Style Transfer Intensity (STI) score: {sti_score:.4f}")

    content_preservation_score = calculate_content_preservation(content_img, output)
    print(f"Content Preservation score: {content_preservation_score:.4f}")

    style_similarity_score = calculate_style_similarity(style_img, output)
    print(f"Style Similarity score: {style_similarity_score:.4f}")

    # Display the images
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(content_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    ax1.set_title(f"Content Image\nNumber of Iterations = {num}")
    ax1.axis('off')
    ax2.imshow(style_img.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    ax2.set_title("Style Image")
    ax2.axis('off')
    ax3.imshow(output.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
    ax3.set_title("Styled Image")
    ax3.axis('off')
    plt.tight_layout()
    plt.show()


    
