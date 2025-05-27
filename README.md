# AI Task Management System Data Visualization

A Python-based data visualization and machine learning pipeline for classifying and analyzing issue priorities in an AI-powered task management system.

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Project Structure](#project-structure)
* [Usage](#usage)
* [Environment Variables](#environment-variables)
* [Dependencies](#dependencies)
* [Results and Outputs](#results-and-outputs)
* [Contributing](#contributing)
* [License](#license)

## Overview

This repository provides a complete workflow to:

1. Load and preprocess cleaned JIRA issue data.
2. Train and evaluate machine learning classifiers (Naive Bayes, SVM) for predicting issue priority levels (`High`, `Medium`, `Low`).
3. Balance classes using SMOTE to improve model performance.
4. Perform hyperparameter tuning with `GridSearchCV`.
5. Generate visualizations to explore data distributions and model performance.
6. Export trained models and vectorizers for deployment.

## Features

* **Data Loading**: Read cleaned CSV data from JIRA issue reports.
* **Text Preprocessing**: TF-IDF vectorization of issue descriptions.
* **Classification Models**: Naive Bayes and class-weighted Linear SVM.
* **Imbalanced Learning**: Oversample minority classes using SMOTE.
* **Hyperparameter Tuning**: Grid search over regularization strengths.
* **Model Evaluation**: Classification reports, confusion matrices.
* **Data Visualization**:

  * Class distribution before/after balancing
  * Text length histogram
  * Word cloud of frequent terms
  * Feature importance bar plots
* **Model Persistence**: Save final classifiers and TF-IDF vectorizer with `joblib`.

## Prerequisites

* Python 3.8 or above
* Jupyter Notebook (optional for interactive exploration)
* A Kaggle environment or local setup with the cleaned CSV file available at `/kaggle/input/cleaned-csv/jira_issues_cleaned.csv` (or modify path)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/AI-Task-Management-System-Data-Visualization.git
   cd AI-Task-Management-System-Data-Visualization
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate    # Linux/macOS
   venv\\Scripts\\activate   # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```text
zogratis17-ai-task-management-system-data-visualization/
├── README.md                                # Project overview and instructions
└── AI-Task-Management-System-Data-Visualization.ipynb  # Jupyter notebook with end-to-end workflow
```

## Usage

### Jupyter Notebook

1. Launch Jupyter:

   ```bash
   jupyter notebook
   ```
2. Open `AI-Task-Management-System-Data-Visualization.ipynb` and run cells sequentially.

### Python Scripts

1. Adjust file paths in `scripts/config.py` if applicable.
2. Run the training script:

   ```bash
   python scripts/train_model.py
   ```
3. Generate visualizations:

   ```bash
   python scripts/visualize.py
   ```

## Dependencies

Key Python packages:

* `requests`, `python-dotenv` for API and environment loading
* `pandas` for data manipulation
* `scikit-learn` for machine learning models and grid search
* `imbalanced-learn` for SMOTE oversampling
* `matplotlib`, `seaborn`, `wordcloud` for visualizations

See `requirements.txt` for full list.

## Results and Outputs

* **Models**: Saved under `models/`
* **Figures**: Displayed inline in notebook and can be saved via `plt.savefig()` calls

## Contributing

Contributions are welcome! Please open issues or submit pull requests for bug fixes, enhancements, or new features.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
