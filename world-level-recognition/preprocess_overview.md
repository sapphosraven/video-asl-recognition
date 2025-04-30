This script is a **complete preprocessing pipeline** for converting the WLASL video dataset into a clean, labeled **image dataset** based on the best hand-detected frame from each video. Here's an overview of its full functionality:

---

### 🔍 **1. Load & Filter WLASL Dataset**
- Loads the JSON file (`WLASL_v0.3.json`) containing glosses and video metadata.
- Counts the number of videos per sign (gloss).
- **Selects the top N most frequent classes (default: 100).**

---

### 📽 **2. Sample & Process Videos**
- For each video:
  - **Extracts 5 frames** evenly spaced in the middle 70% of the video.
  - **Detects hands** in each sampled frame using **MediaPipe**.
  - Chooses the frame with the **highest hand detection confidence**.
  - Crops around the detected hands and resizes the image to **224×224** pixels.

---

### 🧪 **3. Train/Val/Test Split**
- Splits instances of each gloss into:
  - **Train (80%)**
  - **Validation (10%)**
  - **Test (10%)**
- Stores cropped hand images in:  
  ```
  wlasl_dataset/
    ├── train/
    ├── val/
    └── test/
  ```
  Each split has subfolders named after the gloss (class name), containing images. compatible with pytorch ImageFolder format.

---

### 🧹 **4. Dataset Cleanup**
- Removes classes that are:
  - Missing in any split, or
  - Empty in any split (e.g., failed hand detection).
- Ensures only consistent classes exist in all three splits.

---

### 📊 **5. Dataset Analysis & Imbalance Suggestions**
- Calculates:
  - Number of classes and images per split.
  - Min/max samples per class.
  - **Imbalance ratio** between largest and smallest class.
- Suggests:
  - `WeightedRandomSampler` for balanced training.
  - Data augmentation ideas.
  - Using class-weighted loss functions.

---

### ✅ **6. Final Output**
After running the script:
- You have a **clean image dataset** derived from WLASL.
- It’s ready for training image-based deep learning models (e.g., CNNs or ViTs).
- You also get a `label_mapping.txt` file with gloss-to-index mapping.

---

Would you like help modifying this script to include pose or face detection alongside hand crops?