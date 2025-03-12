# Embryo Image Merging with Deep Learning

## 📁 Project Overview

This project focuses on enhancing cell boundary visualization in embryo images by merging images captured at different focal lengths using deep learning techniques. The project uses a U-Net model as a baseline, with plans to experiment with SwinIR for improved performance.

## 🎯 Objective

- Merge multiple focal images to generate a clear and enhanced cell boundary image.
- Utilize deep learning models, starting with U-Net, and explore advanced architectures like SwinIR.

---

## 🚦 Project Structure

```
Embryo/
│── Dataset/              # Raw dataset containing images at different focal lengths
│── Notebooks/
│    │── preprocessing_images.ipynb    # Jupyter Notebook for preprocessing and dataset initialization
│── Scripts/
│    │── preprocessing.py   # Dataset class and preprocessing pipeline
│── embryo_env/             # Python virtual environment
│── processed_data/         # Directory for processed output images
│── preprocessed_data/      # Directory for preprocessed images
│── README.md               # Project documentation
```

---

## 🧠 Methodology

1. **Data Collection:** Paired images at different focal lengths, with corresponding ground truth images.
2. **Preprocessing:**
   - Resizing images to 256x256.
   - Applying augmentations using Albumentations.
   - Generating valid image pairs for training.
3. **Modeling:**
   - Baseline with U-Net.
   - Future work includes exploring SwinIR.
4. **Evaluation:**
   - Visual and quantitative metrics to assess merging quality.

---

<!-- ## ⚙️ Setup

### 1. Clone Repository

```sh
git clone <repository_url>
cd Embryo
```

### 2. Setup Python Environment

```sh
python -m venv embryo_env
source embryo_env/bin/activate
pip install -r requirements.txt
```

### 3. Dataset Preparation

- Place the dataset in the `Dataset/` directory with the following structure:

```
Dataset/
│── embryo_dataset_F0/
│── embryo_dataset_F1/
│── embryo_dataset_F2/
│── embryo_dataset_F3/
│── embryo_dataset_F4/
│── embryo_dataset_F5/
│── processed_data/
```

### 4. Run Preprocessing

```sh
jupyter notebook Notebooks/preprocessing_images.ipynb
```

---

## 🚀 Training the Model

To train the U-Net model, run:

```sh
python train_unet.py
```

---

## 📊 Results

- Evaluation metrics and qualitative results will be documented here.

--- -->

## 🛠 Future Work

- Experiment with SwinIR.
- Implement additional preprocessing and augmentation techniques.

---

## 🤝 Contribution

Feel free to submit issues or pull requests to contribute to this project!

---

## 📝 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
