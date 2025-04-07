# CONTRIBUTION

This document outlines the individual contributions to the **Image Styling Using Neural Style Transfer** project, based on the standalone script provided. Each contributor’s work is detailed below, including the specific code sections they implemented.

---

## Samarthya Earnest Chattree (2205498)

**Role:** Implemented the style retrieval system and enforced image size limits. Also responsible for web deployment (not present in this standalone script).

**Contribution:**
- Developed the `download_first_google_image` function to fetch style images using SerpAPI (lines 35–52).
- Set image size limits to 512px (GPU) or 256px (CPU) to optimize performance (line 304).
- Deployed the NST model as a web application using Flask and Bootstrap (not reflected in this script).

**Code Snippet:**
```python
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

# Image size limit (line 304)
if __name__ == "__main__":
    imsize = 512 if torch.cuda.is_available() else 256
    content_img = image_loader("content.jpg", imsize)
    
    prompt = input("Enter the image style: ")
    download_first_google_image(prompt, "style.jpg")
```

---

## Harekrishna (2205470)

**Role:** Implemented the core Neural Style Transfer (NST) algorithm using VGG-19.

**Contribution:**
- Coded the `gram_matrix` function to compute style feature correlations (lines 85–90).
- Developed `get_style_model_and_losses` to extract content and style features from VGG-19 (lines 112–154).
- Implemented `run_style_transfer` to perform the style transfer optimization (lines 156–185).

**Code Snippet:**
```python
def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)

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

def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
```

---

## Satvik (2205503)

**Role:** Implemented performance evaluation and visualization functions.

**Contribution:**
- Developed `calculate_sti`, `calculate_content_preservation`, and `calculate_style_similarity` to assess style transfer quality (lines 198–235).
- Coded `plot_performance_metrics` to visualize processing time and memory usage (lines 280–295).
- Included EMD fallback with histogram comparison in `calculate_sti` (lines 213–218).

**Code Snippet:**
```python
# Attempt to import EMD and provide a fallback
emd_installed = False
try:
    from PyEMD import EMD
    emd_installed = True
    print("EMD from PyEMD imported successfully.")
except ImportError:
    try:
        from EMD_signal import EMD
        emd_installed = True
        print("EMD from EMD_signal imported successfully.")
    except ImportError:
        print("EMD library not found. STI will use histogram comparison.")
        emd_installed = False

def calculate_sti(style_img, output):
    global emd_installed

    if emd_installed:
        try:
            emd = EMD()
            style_imfs = emd(style_img.view(-1).detach().cpu().numpy())
            output_imfs = emd(output.view(-1).detach().cpu().numpy())
            return np.mean(np.abs(style_imfs - output_imfs))
        except Exception as e:
            print(f"Error calculating STI with EMD: {e}. Falling back to histogram comparison.")
            emd_installed = False
    if not emd_installed:
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

def plot_performance_metrics(processing_time, current_memory, peak_memory):
    metrics = ['Processing Time (s)', 'Current Memory (MB)', 'Peak Memory (MB)']
    values = [processing_time, current_memory / 10**6, peak_memory / 10**6]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['blue', 'green', 'red'])
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.title('Style Transfer Performance Metrics')
    plt.ylabel('Value')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    
    plt.savefig('performance_metrics.png')
    plt.show()
```

---

## Raunak (2205494)

**Role:** Enhanced user interaction and visualization.

**Contribution:**
- Implemented `plot_vgg19_diagram` to illustrate the VGG-19 architecture (lines 248–278).
- Added user input prompts for iterations and weights in the main script (lines 310–312).
- Developed image display logic to show content, style, and output images (lines 332–345).

**Code Snippet:**
```python
def plot_vgg19_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_title("VGG-19 Convolutional Architecture", fontsize=16, pad=20)
    ax.axis('off')

    blocks = [
        ("Block 1\nConv_1: 64 filters\nConv_2: 64 filters\nMaxPool", 0, "224x224x3", "112x112x64"),
        ("Block 2\nConv_3: 128 filters\nConv_4: 128 filters\nMaxPool", 1, "112x112x64", "56x56x128"),
        ("Block 3\nConv_5: 256 filters\nConv_6: 256 filters\nConv_7: 256 filters\nConv_8: 256 filters\nMaxPool", 2, "56x56x128", "28x28x256"),
        ("Block 4\nConv_9: 512 filters\nConv_10: 512 filters\nConv_11: 512 filters\nConv_12: 512 filters\nMaxPool", 3, "28x28x256", "14x14x512"),
        ("Block 5\nConv_13: 512 filters\nConv_14: 512 filters\nConv_15: 512 filters\nConv_16: 512 filters\nMaxPool", 4, "14x14x512", "7x7x512"),
    ]

    for i, (label, x, input_dim, output_dim) in enumerate(blocks):
        ax.text(x * 0.25 + 0.1, 0.5, label, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightblue"))
        ax.text(x * 0.25 + 0.1, 0.7, f"Input: {input_dim}", ha='center', va='center', fontsize=8, color='darkgreen')
        ax.text(x * 0.25 + 0.1, 0.3, f"Output: {output_dim}", ha='center', va='center', fontsize=8, color='darkred')
        if i < len(blocks) - 1:
            ax.arrow(x * 0.25 + 0.2, 0.5, 0.05, 0, head_width=0.02, head_length=0.02, fc='black', ec='black')

    ax.text(-0.1, 0.5, "Input\n224x224x3", ha='center', va='center', fontsize=10, color='blue')
    ax.text(1.35, 0.5, "Output\n7x7x512", ha='center', va='center', fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig("vgg19_architecture_detailed.png", dpi=300, bbox_inches='tight')
    plt.show()

# User input prompts and image display logic (lines 310–312, 332–345)
if __name__ == "__main__":
    # User input prompts
    num = int(input("Enter the number of iterations you want (Default = 500): "))
    style_weight, content_weight = int(input("Enter the style weight (Default = 100000): ")), int(input("Enter the content weight (Default = 10): "))
    
    # ... (other setup code) ...
    
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
```

---

## Soumyata Biswas (2205512)

**Role:** Implemented image preprocessing and normalization.

**Contribution:**
- Coded the `image_loader` function for image preprocessing (lines 54–60).
- Developed the `Normalization` class for VGG-19 compatibility (lines 98–104).
- Defined normalization parameters using ImageNet mean and std (lines 93–94).

**Code Snippet:**
```python
def image_loader(image_name, imsize):
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()
    ])
    image = Image.open(image_name).convert('RGB')
    return loader(image).unsqueeze(0).to(device, torch.float)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)
    def forward(self, img):
        return (img - self.mean) / self.std

# Normalization parameters (lines 93–94)
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
```

---

## Ujjwal Shrivastava (2205517)

**Role:** Focused on performance monitoring and quality assurance.

**Contribution:**
- Implemented `ContentLoss` and `StyleLoss` classes for loss computation (lines 67–90).
- Added memory tracking using `tracemalloc` to monitor resource usage (lines 314–320).

**Code Snippet:**
```python
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# Memory tracking and performance monitoring (lines 314–320)
if __name__ == "__main__":
    # Start performance measurements
    start_time = time.time()
    tracemalloc.start()

    # Run style transfer (assumed part of monitoring scope)
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
```

---

## Notes
- **Dependencies:** Each contributor’s code assumes the necessary imports (e.g., `torch`, `numpy`, `matplotlib`) and global variables (e.g., `device`, `cnn`) are available from the full script.
- **Scope:** This `CONTRIBUTION.md` reflects the standalone script provided. The Flask-based web deployment (mentioned by Samarthya) is not included here but is part of the project’s broader scope.
- **Line Numbers:** References to line numbers align with the structure of the provided standalone script.

