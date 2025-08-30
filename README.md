# Counterfeit Product Detection Model

This project builds a machine learning pipeline to detect counterfeit (fraudulent) product listings on an e-commerce platform.

Feature engineering on product, seller, logistics, and engagement data, has been used and then Logistic Regression and XGBoost Classifier models have been trained and evaluated.

# Features & Engineering

## Product-level

log_cost: Log-transformed price

low_price_flag: Price < 25% of category median

desc_short_flag: Very short descriptions

typo_density: Typos per character


## Logistics

loc_mismatch_flag: Vendor nation ≠ Dispatch location

## Engagement

conversion_rate = sales_vol / page_hits

wishlist_ratio = saved_items / page_hits

All categorical fields (product_type, manufacturer, vendor_nation, dispatch_loc, geo_inconsistency) are OneHotEncoded, while booleans are converted into ones and zeros.

# Models

## Two models were trained and compared:

### Logistic Regression

Baseline interpretable model

Good for understanding key fraud drivers

### XGBoost Classifier

Gradient boosting model for tabular fraud data

Handles non-linearities, interactions, and class imbalance well

# Evaluation Metrics

### Accuracy

### Precision / Recall / F1

### ROC-AUC Score


# Target Variable:

fraud_indicator = 1 → counterfeit product

fraud_indicator = 0 → legitimate product

### NOTE: Remember to install required dependencies from the ```requirements_txt``` file provided in the repository.
