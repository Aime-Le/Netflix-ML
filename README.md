# Netflix Content Recommendation System
## This a ML Internship project in  Auspify Technologies No.1

### Project overview
This project implements a **content-based Netflix recommendation system**.
Given a selected Netflix title, the system recommends other titles with
similar content attributes.
### See Result in APP
https://netflix-recommender-aimele.streamlit.app/#netflix-content-recommendation-system
### Task 1 workflow
work follows the following rule

1. Prepare content-related features.
2. Convert text data into a machine-readable representation.
3. Calculate content similarity scores.
4. Generate recommendations for selected titles.
5. Evaluate recommendation quality.

### Features used
The recommendation model combines:

- `type`
- `director`
- `country`
- `rating`
- `listed_in` (genres/categories)

These fields are combined into a single `content` feature.

### Machine-learning method
**TF-IDF (Term Frequency–Inverse Document Frequency)** is used to transform
the combined content text into numerical feature vectors.

**Cosine similarity** is then used to compare the selected title with all
other Netflix titles.

A higher cosine similarity score means the content representation is more
similar to the selected title.

### Evaluation
Because this is a content-based recommendation system rather than a
supervised classification model, classification accuracy is not used.

The project evaluates recommendation quality using:

- Average similarity score
- Highest similarity score
- Lowest similarity score
- Genre/category overlap

### Files

```text
Netflix_Content_Recommendation_Task1_Submission/
│
├── Dataset.csv
├── netflix_recommendation.py
├── requirements.txt
├── netflix_prepared_dataset.csv
├── midnight_mass_recommendations.csv
├── evaluation_summary.txt
│
├── notebook/
│   ├── netflix_task1_recommendation.ipynb
│   └── netflix_task1_recommendation_executed.ipynb
│
└── screenshots/
    ├── 01_final_recommendations.png
    └── 02_dataset_loaded.png
```

### To run in Google Colab 

1. Upload `Dataset.csv`.
2. Upload or open the notebook in the `notebook` folder.
3. Run the cells from top to bottom.
4. Check the recommendation and evaluation outputs.

### To run in VS Code

Open a terminal in this project folder and install the requirements:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python netflix_recommendation.py
```
### Skills demonstrated

- Python
- Pandas
- Data preprocessing
- Feature engineering
- NLP fundamentals
- TF-IDF
- Cosine similarity
- Content-based recommendation systems
- Scikit-learn
- Model evaluation
