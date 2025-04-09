# 🖼️ Image Styling using Neural Style Transfer
This project performs **Neural Style Transfer (NST)** using PyTorch and VGG19 to blend the content of one image with the artistic style of another, fetched directly from Google Images via SerpAPI. It also computes key metrics to evaluate style intensity, content preservation, and style similarity.

---

## 🚀 Features

- Content-style image blending using **VGG19**.
- **Dynamic style fetching** from Google Images (via SerpAPI).
- Evaluation using:
  - **Style Transfer Intensity (STI)** using EMD or histogram comparison.
  - **Content Preservation** using SSIM.
  - **Style Similarity** using SSIM.
- **Performance profiling** (time + memory).
- Real-time display of content, style, and styled images.

---

## 🛠️ Installation

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/neural-style-transfer-metrics.git
cd neural-style-transfer-metrics
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

<details>
<summary>Requirements (if not using <code>requirements.txt</code>):</summary>

```bash
torch torchvision matplotlib requests beautifulsoup4 serpapi scikit-image numpy
```

> Optional (for STI via EMD):
```bash
pip install EMD-signal  # OR PyEMD
```
</details>

---

## ⚙️ Usage

1. **Place your content image** in the root directory as `content.jpg`.

2. **Run the script**:

```bash
python style_transfer.py
```

3. **Provide the following inputs when prompted**:
   - A **style prompt** (e.g., "Van Gogh painting", "cyberpunk art").
   - Number of **optimization iterations**.
   - Style and content **weights** (e.g., 100000 and 10 respectively).

The script will:
- Fetch the first Google Image result based on the prompt.
- Perform style transfer.
- Save the final result as `styled_image.jpg`.
- Print metrics and display visual comparisons.

---

## 📊 Metrics Explained

| Metric                     | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| **Style Transfer Intensity (STI)** | Measures how much style has been imposed over the content. Calculated via EMD or histogram difference. |
| **Content Preservation**   | SSIM-based measure of how well the original content is preserved.           |
| **Style Similarity**       | SSIM-based comparison between the styled image and style reference.         |
| **Time + Memory**          | Benchmarks runtime and memory usage (via `tracemalloc`).                    |

---

## 📁 Output Files

- `styled_image.jpg` — Final styled image.
- `style.jpg` — Downloaded image based on user’s style prompt.
- `content.jpg` — User-provided base image.

---

## 🔐 API Key Configuration

The script uses [SerpAPI](https://serpapi.com/) for image search. Replace the placeholder API key:

```python
API_KEY = "YOUR_SERPAPI_KEY"
```

Get a free API key from [serpapi.com/users/sign_up](https://serpapi.com/users/sign_up).

---

## 🧪 Example Prompts

- `"Van Gogh painting"`
- `"Cyberpunk cityscape"`
- `"Japanese ink art"`
- `"Picasso cubism"`

---

## 📌 Notes

- Works on **CPU and GPU**.
- EMD-based STI is optional but recommended for more robust evaluation.
- Tune `style_weight` and `content_weight` for desired output balance.

---

## 📸 Example Output

![Example Result](preview_sample.jpg) <!-- Add your example result here -->

---

## 🧑‍💻 Author

**Samarthya Earnest Chattree**

- B.Tech CSE @ KIIT
- ML/DL Researcher | IoT Enthusiast
- [LinkedIn](https://www.linkedin.com) | [GitHub](https://github.com/yourusername)

---

## 📜 License

MIT License — use freely with attribution.

