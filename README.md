# Embryo Image Merging with Deep Learning

## 💁 Collaborators

- Parth Thakre [https://github.com/parth-thakre]
- Shrishti Pandey [https://github.com/ShrishtiP0214]
- Shreya Dhanuka [https://github.com/shreyaadhanuka]
- Shivam Tadas [https://github.com/shivam-tadas]

## 📁 Project Overview

This project focuses on enhancing cell boundary visualization in embryo images by merging images captured at different focal lengths using deep learning techniques. The project uses a U-Net model as a baseline, with plans to experiment with SwinIR for improved performance.

## 🎯 Objective

- Merge multiple focal images to generate a clear and enhanced cell boundary image.
- Utilize deep learning models, starting with U-Net, and explore advanced architectures like SwinIR.
- Evaluate the effectiveness of merging methods using both qualitative and quantitative metrics.

---

## 🚦 Project Structure

```
Embryo/
│── Dataset/              # Raw dataset containing images at different focal lengths
│── Notebooks/
│    │── preprocessing_images.ipynb    # Jupyter Notebook for preprocessing and dataset initialization
│── Scripts/
│    │── preprocessing.py   # Dataset class and preprocessing pipeline
│    │── train_unet.py      # Script for training the U-Net model
│    │── test_unet.py       # Script for testing the trained model
│── embryo_env/             # Python virtual environment
│── processed_data/         # Directory for processed output images
│── preprocessed_data/      # Directory for preprocessed images
│── trained_models/         # Directory storing trained model weights
│── results/                # Directory for storing evaluation results and logs
│── README.md               # Project documentation
```

---

## 🧠 Methodology

### 1️⃣ Data Collection

- **Paired Images:** Each embryo has multiple images captured at different focal lengths (e.g., F0 to F5).
- **Ground Truth:** The best-focused image or a manually enhanced image is used as the target output.

### 2️⃣ Preprocessing

- **Resizing:** Images are resized to `256x256`.
- **Normalization:** Pixel intensity values are normalized for consistent model performance.
- **Augmentations:** Data augmentation using Albumentations (e.g., rotation, flipping, contrast adjustments) to increase dataset diversity.
- **Dataset Splitting:**
  - `80%` Training
  - `10%` Validation
  - `10%` Testing

### 3️⃣ Model Training

- **Baseline Model:** U-Net with 6 input channels (one for each focal plane) and 1 output channel (fused image).
- **Loss Function:** Mean Squared Error (MSE) for pixel-wise similarity.
- **Optimizer:** Adam with learning rate scheduling.
- **Hardware:** Training runs on GPU (if available) for better performance.
- **Checkpointing:** The best-performing model is saved to `trained_models/embryo_unet.pth`.

### 4️⃣ Testing & Inference

- Users can provide `6` images of an embryo, and the model will output a fused image.
- Run `test_unet.py` to evaluate a single embryo:

  ```sh
  python test_unet.py --image_paths path/to/img1 path/to/img2 path/to/img3 ...
  ```

- Output is saved to `results/fused_output.jpg`.

### 5️⃣ Evaluation

- **Quantitative Metrics:** PSNR, SSIM, and MAE between fused output and ground truth.
- **Qualitative Analysis:** Visual comparison between fused and ground truth images.
- **Error Logging:** Any truncated or corrupted images are logged in `results/error_log.txt`.

---

## 🚀 Training the Model

To train the U-Net model from scratch, run:

```sh
python unet.py
```

---

## 🧪 Testing the Model

To test the model with a given set of 6 images:

```sh
python test_unet.py --image_paths /path/to/F0.jpg /path/to/F1.jpg /path/to/F2.jpg /path/to/F3.jpg /path/to/F4.jpg /path/to/F5.jpg
```

This will generate and save the fused image as `results/fused_output.jpg`.

---

## 📊 Results

### **Evaluation Metrics**

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures image reconstruction quality.
- **SSIM (Structural Similarity Index Measure)**: Evaluates structural differences.
- **MAE (Mean Absolute Error)**: Computes the average pixel-wise difference.

Results are stored in `results/evaluation_metrics.txt`.

---

## 🛠 Future Work

- **Experiment with SwinIR** for improved high-frequency detail retention.
- **Refine preprocessing techniques** to handle truncated images more effectively.
- **Hyperparameter tuning** to improve training stability and convergence.
- **Explore additional loss functions** for better alignment with human perception.

---

## 🤝 Contribution

Feel free to submit issues or pull requests to contribute to this project! Contributions could include:

- Implementing new model architectures.
- Enhancing the dataset with more diverse embryo images.
- Improving preprocessing and augmentation techniques.

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
