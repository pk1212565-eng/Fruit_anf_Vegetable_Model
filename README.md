# FreshScan — Fruits & Vegetables Streamlit UI

A professional Streamlit interface for the model in `Fruits_Veg_Dataset.ipynb`.

## Learning Hub
Use the **Learning Hub** option in the sidebar to explore:
- Perceptrons, forward passes, and backpropagation.
- Activation functions and classification loss.
- How CNNs extract image features and how this project's custom CNN is structured.
- Transfer learning with feature extraction and fine-tuning, including why it
  can help. This is educational content; the live scanner still uses its
  custom CNN trained from scratch.

The learning material describes concepts; it does not change the trained model
or claim that the scanner uses transfer learning.

## 1. Requirements
- Python 3.10 or 3.11 is recommended for TensorFlow compatibility.
- Your trained Keras model file: `Fruite_and_veg_model.keras`
- The class names in the same order used during training.

## 2. Put the model and labels in this folder
Copy `Fruite_and_veg_model.keras` into this folder.

If `class_names.json` is not present, the app uses sorted class-folder names
from `Fruits_Vegetables/train` when that dataset directory is available.
Otherwise, create `class_names.json` using the exact order from the notebook:
```python
import json
json.dump(data_labels, open("class_names.json", "w", encoding="utf-8"))
```
Run that code in the notebook after `data_labels = train_images.class_names`. Then copy the generated JSON file beside `app.py`.

## 3. Install and run (Windows)
Open Command Prompt or PowerShell in this folder:
```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Streamlit will show a local URL, usually `http://localhost:8501`.

## Notes
- The interface expects 180×180 input images to match the notebook.
- Your notebook uses `SparseCategoricalCrossentropy(from_logits=True)`, so the UI applies softmax to display prediction scores.
- If you retrain the model with a different preprocessing layer or class order, update the app accordingly.
