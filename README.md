# 🩸 AI-Based Cuff-less Blood Pressure Waveform Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Status](https://img.shields.io/badge/Status-Completed-success) ![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This project focuses on **Continuous Blood Pressure Monitoring** without the use of invasive catheters or inflatable cuffs. By leveraging Deep Learning, specifically a **Hybrid CNN-BiLSTM (Seq2Seq)** architecture, the model reconstructs the full **Arterial Blood Pressure (ABP)** waveform using only **PPG** and **ECG** signals.

The final model achieved **Clinical-Grade Accuracy** with an MAE of **3.42 mmHg**, significantly outperforming traditional Machine Learning baselines.

---

## 📂 Dataset & Preprocessing
**Source:** [Kaggle Blood Pressure Dataset](https://www.kaggle.com/datasets/mkachuee/BloodPressureDataset) (Part 1)

The dataset consists of physiological signals from **1,000 subjects**. To prepare the data for the Seq2Seq model, we applied a windowing technique to multiply the available training samples:

* **Sampling Rate:** 125 Hz
* **Original Duration:** 5-6 minutes per patient (40-45k samples).
* **Window Size:** 5 Seconds (625 samples).
* **Slicing Logic:** `300 seconds / 5 seconds = 60 windows per patient`.
* **Total Training Samples:** ~60,000 (1,000 patients × 60 windows).

### Data Distribution
![Data Distribution](Images/Data%20Distribution.png)

---

## 🧠 Model Architecture: "The Dragon" 🐉
We utilized a sophisticated Hybrid Deep Learning architecture designed to capture both morphological features and temporal dependencies:

1.  **Input:** `(625, 2)` shape containing PPG and ECG signals.
2.  **Encoder (CNN Blocks):** Two layers of **1D Convolutions (128 filters)** to extract signal shape features (peaks, dicrotic notches).
3.  **Decoder (Bi-Directional LSTM):** Two layers of **Bi-LSTM (128 units)**. This allows the model to learn from both past and future context, ensuring smooth waveform generation.
4.  **Output:** A **TimeDistributed Dense** layer that outputs the predicted ABP value for every single time step.

---

## 📊 Comparative Analysis
We benchmarked the Deep Learning model against traditional ML approaches to justify the computational cost.

### 1. Baseline: Linear Regression
* **Result:** MAE ~13.11 mmHg.
* **Verdict:** Too simple. Failed to capture non-linearities in blood pressure dynamics.
![Linear Regression](Images/Linear%20Regression%20Evaluation%20Graph.png)

### 2. Baseline: Decision Tree Regressor
* **Result:** MAE ~16.50 mmHg.
* **Verdict:** Resulted in a "jagged" and low-resolution prediction. Unable to model smooth physiological curves.
![Decision Tree](Images/Decision%20Tree%20Evaluation%20Graph.png)

### 3. The Winner: Bi-Directional LSTM (Deep Learning)
* **Result:** **MAE 3.42 mmHg**.
* **Verdict:** Successfully reconstructed the full waveform, capturing Systolic peaks and Diastolic valleys with high precision.
![Final LSTM Result](Images/LSTM%20Evaluation%20Graph.png)

---

## 📉 Training Performance
The model was trained using **Mixed Precision (Float16)** on a T4 GPU. We utilized custom callbacks for monitoring and `EarlyStopping` to prevent overfitting.

* **Optimizer:** Adam (`lr=0.001`)
* **Loss Function:** Mean Squared Error (MSE)
* **Metric:** Mean Absolute Error (MAE)

![Training Graphs](Images/Training%20Graphs.png)

---

## 🏆 Final Results Table

| Model | MAE (Error) | RMSE | Status |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | 13.11 mmHg | N/A | ❌ Baseline |
| **Decision Tree** | 16.50 mmHg | N/A | ❌ Poor Resolution |
| **XGBoost (Subset)** | ~9.20 mmHg | N/A | ⚠️ Computationally Expensive |
| **Bi-Directional LSTM** | **3.42 mmHg** | **5.59 mmHg** | ✅ **SOTA Accuracy** |

---

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ahsanatiq05/AI-ABP-Waveform-Prediction.git](https://github.com/ahsanatiq05/AI-ABP-Waveform-Prediction.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy scikit-learn matplotlib opendatasets
    ```
3.  **Download Data:**
    The notebook automatically downloads the dataset from Kaggle using `opendatasets`. Ensure you have your `kaggle.json` key ready.
4.  **Run the Notebook:**
    Open `BloodPressure_Waveform_Prediction.ipynb` in Jupyter or Google Colab.

---

## 👨‍💻 Author
**Ahsan Atiq**
* **Institution:** COMSATS University Islamabad
* **Focus:** Data Science, Deep Learning, Medical Imaging
* **Contact:** [LinkedIn Profile](https://www.linkedin.com/in/ahsan-atiq-913219263/)

---
*Note: This project is for research and educational purposes. While the results are clinical-grade, it should not replace certified medical devices without clinical trials.*
