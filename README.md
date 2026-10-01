# Netflix Content Recommendation System
## Auspify Technologies — Machine Learning Internship — Task 1

### Project overview
This project implements a **content-based Netflix recommendation system**.
Given a selected Netflix title, the system recommends other titles with
similar content attributes.

### Task 1 workflow
The implementation follows the internship Task 1 workflow:

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

For the included example run, the evaluation summary is stored in
`evaluation_summary.txt`.

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

### How to run in Google Colab

1. Upload `Dataset.csv`.
2. Upload or open the notebook in the `notebook` folder.
3. Run the cells from top to bottom.
4. Check the recommendation and evaluation outputs.

### How to run in VS Code

Open a terminal in this project folder and install the requirements:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python netflix_recommendation.py
```

### Example
The included example uses:

```text
Midnight Mass
```

The system returns the top 10 similar Netflix titles together with their
type, genres/categories, and cosine similarity scores.

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

### Submission note
A separate Streamlit application is **not required for the core Task 1
machine-learning workflow**. This submission focuses on the required
recommendation-system implementation, results, evaluation, source code,
notebooks, and screenshots.
