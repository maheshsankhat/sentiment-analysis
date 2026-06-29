# AI-Driven Citizen Grievance & Sentiment Analysis System

## Overview

This project is an AI-powered civic grievance management system that automates complaint routing and urgency assessment. It predicts the responsible government department for a complaint and assigns a priority level to help authorities respond faster.

## Problem Statement

Manual grievance handling in public-sector systems is slow and inefficient. Complaints are often routed manually, which delays resolution. This project addresses that issue by building an automated pipeline for:

* Department / agency classification
* Priority / urgency scoring

## Features

* Predicts the responsible department from citizen complaint text
* Assigns complaint priority as **Low / Medium / High / Critical**
* Uses **machine learning** for department classification
* Uses a **rule-based priority engine** for urgency scoring
* Exposes the system through a **FastAPI REST API**

## Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* FastAPI
* Joblib

## Project Structure

```text
Citizen-Grievance-AI/
│
├── data/
├── notebooks/
├── models/
├── api/
├── requirements.txt
└── README.md
```

## Workflow

1. Load and clean grievance data
2. Perform EDA and text preprocessing
3. Train a department classification model
4. Build a rule-based priority engine
5. Deploy both through a FastAPI endpoint

## API Endpoint

### POST `/predict`

**Request**

```json
{
  "complaint": "There is no electricity in my building for 2 days"
}
```

**Response**

```json
{
  "complaint": "There is no electricity in my building for 2 days",
  "predicted_department": "HPD",
  "predicted_priority": "Critical"
}
```

## Future Improvements

* Use a real labeled urgency dataset for priority classification
* Fine-tune transformer-based NLP models such as BERT
* Add dashboard for complaint analytics and trend visualization
* Integrate multilingual complaint support
* Connect API with a web application for citizens and officials
