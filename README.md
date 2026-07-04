#  Bangalore House Price Predictor

A Machine Learning-powered web application that estimates house prices in Bangalore based on user inputs such as location, total square footage, number of bedrooms (BHK), and bathrooms. The application uses a trained regression model and provides instant price predictions through a modern, responsive Flask web interface.

##  Features

* Predicts Bangalore house prices using Machine Learning.
* Interactive and responsive user interface.
* Instant predictions without page reload (AJAX).
* Supports multiple Bangalore locations.
* Simple and intuitive form design.
* Flask backend integrated with a trained ML model.
* Deployed online using Render.

---

##  Tech Stack

### Frontend

* HTML
* Tailwind CSS
* JavaScript
* AJAX (XMLHttpRequest)

### Backend

* Python
* Flask

### Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Pickle

### Deployment

* Render

---

##  Project Structure

```text
bangalore-house-price-predictor/
│
├── app.py
├── house_price_model.pkl
├── columns.json
├── requirements.txt
├── templates/
│   └── index.html
├── static/
├── README.md
└── .gitignore
```

##  Machine Learning Workflow

* Data Collection
* Data Cleaning
* Feature Engineering
* One-Hot Encoding of Locations
* Model Training
* Model Evaluation
* Model Serialization
* Flask Integration
* Deployment on Render

---

##  Input Features

The model predicts the house price based on:

*  Location
*  BHK
*  Number of Bathrooms
*  Total Square Feet

---

##  Output

The application predicts the **estimated market price** of the property in **Lakhs (₹)**.

---

##  Future Improvements

* Allow users to enter custom square footage.
* Add more Bangalore localities.
* Display price confidence intervals.
* Add price trends based on locality.

---

##  Author

**Anay Bhattacharya**
