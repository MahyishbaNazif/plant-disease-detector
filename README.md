# Plant Disease Detector

Upload a leaf photo -> get the plant, the disease, and a suggested treatment.
Built with TensorFlow (MobileNetV2 transfer learning) and Streamlit.

## Setup
1. `pip install -r requirements.txt`
2. Download the PlantVillage dataset from Kaggle and put the class folders inside `data/`
   (e.g. `data/Tomato___Late_blight/`).

## Run
1. `python train.py`      -> creates `plant_model.keras` and `class_names.json`
2. `python evaluate.py`   -> accuracy report + `confusion_matrix.png`
3. `streamlit run app.py` -> opens the web app

## Tip
No GPU? Train on Google Colab or Kaggle notebooks (free GPU), then download
`plant_model.keras` and `class_names.json` next to `app.py`.
