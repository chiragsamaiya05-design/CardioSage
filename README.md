# CardioSage

Streamlit web app for a heart-disease risk estimate based on the included `heart.csv` dataset.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy with Streamlit Community Cloud

1. Push this project to your GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Select **Create app**, then choose your repository and branch.
4. Set the main file path to `Heart_Strock_Prediction/app.py`.
5. Click **Deploy**.

The model is trained from `heart.csv` when the app starts, so no separate `.pkl` model files are required.

> This app is an educational project only and does not provide a medical diagnosis.

Model: Logistic Regression

Accuracy : 87%

Precision : 88%

Recall : 85%

F1 Score : 86%

ROC-AUC : 92%
