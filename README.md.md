# 🎬 CineMatch AI — Streamlit Movie Recommendation System

## 1. Project files

Keep these files in the same folder:

```text
Movie-Recommendation/
│
├── app.py
├── movies_20000.xlsx
└── requirements.txt
```

## 2. Install dependencies

Open Command Prompt / PowerShell in the project folder:

```bash
pip install -r requirements.txt
```

## 3. Run locally

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

## 4. Deploy on Streamlit Community Cloud

### Step 1 — Create a GitHub repository

Create a new GitHub repository, for example:

`movie-recommendation-system`

Upload:

- `app.py`
- `movies_20000.xlsx`
- `requirements.txt`

### Step 2 — Open Streamlit Community Cloud

Go to Streamlit Community Cloud and sign in with GitHub.

### Step 3 — Create the app

Choose:

- Repository: your movie recommendation repository
- Branch: `main`
- Main file path: `app.py`

Then click **Deploy**.

### Step 4 — Wait for deployment

Streamlit installs the packages from `requirements.txt`, loads the Excel dataset, builds the TF-IDF model, and starts the application.

## ⚠️ Important

The Excel file is approximately a 20,000-movie dataset. If GitHub rejects the file because of its size, use Git LFS or move the dataset to a suitable hosted storage/data source and update `app.py` accordingly.

## 🎯 Model

The application uses:

`Movie Description → TF-IDF → Cosine Similarity → Top-N Recommendations`

The app calculates similarity between the selected movie and all movie descriptions without constructing a huge dense 20,000 × 20,000 similarity matrix.
