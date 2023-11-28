# AI-Driven Financial Advisory App for SMEs

![header](https://capsule-render.vercel.app/api?type=wave&color=gradient&height=300&section=header&text=FinAI%20SME%20Advisor&fontSize=50)

## 📍 Overview
FinAI SME Advisor is a sophisticated Flask application, leveraging OpenAI's GPT-3.5 Turbo, to offer bespoke financial and business advice, primarily for SMEs and loan-related queries. Featuring a machine learning model for predictive analytics, the app boasts a dynamic user interface and is conveniently deployed on [Render](https://finai-t4wc.onrender.com).

---

## 📍 Key Features
- *AI-Powered Advice*: Integrating OpenAI's GPT-3.5 Turbo for specialized financial and business consultation.
- *Predictive Analysis Model*: Incorporates a Random Forest algorithm for insightful financial decision-making.
- *Interactive User Interface*: Crafted with Flask, the application offers a responsive and engaging user experience.
- *Accessible Online*: Hosted on Render, ensuring easy access and widespread usability.

---

## 📍 Installation and Setup

### Prerequisites
- Python 3.x
- Pip (Python package manager)

### Installation Steps
1. Clone the GitHub repository:
   bash
   git clone https://github.com/Ajisco/finai.git
   
2. Navigate to the project directory:
   bash
   cd finai
   
3. Install the necessary Python libraries:
   bash
   pip install -r requirements.txt
   

### Running Locally
Execute the application on your local machine:
bash
python app.py

Access the app at `http://localhost:5000`.

---

## 📍 Application Structure
- `templates/`: HTML templates for the web interface.
- `static/assets/`: Static files such as CSS and JavaScript.
- `app.py`: Flask application script defining routes and functionalities.
- `Prediction.ipynb`: Jupyter notebook for machine learning model development.
- `random_forest_model.pkl`: Pre-trained Random Forest model file.
- `requirements.txt`: Lists all necessary Python packages.

---

## 📍 Application Functionality
- *Loan Prediction*: Employs machine learning to assess and predict loan approval outcomes.
- *Business Idea Generation*: AI-driven suggestions for viable business ideas, customized to user's financial capabilities.
- *Financial Advice*: Personalized, AI-powered financial guidance for specific business scenarios.

---

## 📍 Deployment 🚀
The application is available on [Render](https://finai-t4wc.onrender.com), offering easy and broad accessibility.

---

## 📍 Contributing
Interested in contributing? Here's how:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Submit a Pull Request.

---

## 📍 Acknowledgments
- Sincere thanks to OpenAI for providing the GPT-3.5 Turbo API.
- Gratitude to the Flask community for their exceptional web framework.
- Appreciation to Render for their reliable hosting and deployment services.

