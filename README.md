# Predicting Credit Card Approvals

![Credit card being held in hand](./Image/credit_card.jpg)

## Project Overview

This project builds a machine learning model to predict whether a credit card application will be approved. Commercial banks receive numerous credit card applications, and manually reviewing them is time-consuming and prone to errors. By leveraging machine learning, we can automate the approval process, making it more efficient and cost-effective for banks.

This project uses the [Credit Card Approval dataset](https://archive.ics.uci.edu/ml/datasets/Credit+Approval) from the UCI Machine Learning Repository. The dataset is included in this repository as `Data/cc_approvals.data`.

For a more detailed exploratory data analysis and model development process, please open the Jupyter Notebook in the [Notebooks](./Notebooks/) directory.

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Predicting-Credit-Card-Approvals.git
   cd Predicting-Credit-Card-Approvals
   ```

2. **Create and activate a virtual environment:**

   * **Using `conda` (recommended):**
     ```bash
     conda create -n credit-approval-env python=3.9
     conda activate credit-approval-env
     ```

   * **Using `venv`:**
     ```bash
     python -m venv venv
     source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
     ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the credit card approval prediction model, execute the following command:

```bash
python src/main.py
```

The script will load the data, preprocess it, train a logistic regression model using GridSearchCV to find the best hyperparameters, and print the model's performance.

## Results

The logistic regression model achieves the following results on the test set:

*   **Accuracy:** Approximately **80%**
*   **Best Hyperparameters:** `{'max_iter': 100, 'tol': 0.01}`
