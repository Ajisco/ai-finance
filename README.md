# AI-Driven Financial Advisory App for SMEs

![header](https://capsule-render.vercel.app/api?type=wave&color=gradient&height=300&section=header&text=FinAI%20SME%20Advisor&fontSize=50)

## 📍 Overview
FinAI SME Advisor is a sophisticated Flask application, leveraging OpenAI's GPT-3.5 Turbo, to offer bespoke financial and business advice, primarily for SMEs and loan-related queries. Featuring a machine learning model for predictive analytics, the app boasts a dynamic user interface and is conveniently deployed on [Render](https://finai-t4wc.onrender.com).

---

# 📍 Key Features Overview

| [![Watch the video](https://img.youtube.com/vi/y8iPCGCZJCo/maxresdefault.jpg)](https://youtu.be/y8iPCGCZJCo) |

## Services and UI Screenshots

### Home Page
- **Description**: Provides a comprehensive overview of the application, highlighting its main features and functionalities.
- **Screenshots**:
  | Home Page 1 | Home Page 2 |
  | :---: | :---: |
  | ![Home Page 1](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_home1.png) | ![Home Page 2](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_home2.png) |

### Sign In and Available Services
- **Description**: The Sign In page allows users to securely access their accounts, and the Available Services page showcases the variety of services offered.
- **Screenshots**:
  | Sign In Page | Available Services |
  | :---: | :---: |
  | ![Sign In](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_signin.png) | ![Services](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_services.png) |

### Loan Approval Service
- **Description**: Assesses the likelihood of loan approvals for user applications.
- **Functionality**:
  - Information on loan acquisition based on user details.
  - Interactive AI chat for additional queries.
- **Screenshots**:
  | Loan Evaluation Form | Loan Evaluation Form |
  | :---: | :---: |
  | ![Form 1](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_pred_form1.png) | ![Form 2](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_pred_form2.png) |
  | Loan Evaluation Chat | Loan Evaluation Chat |
  | ![Chat 1](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_pred_chat_1.png) | ![Chat 2](https://github.com/Ajisco/ai-finance/blob/master/fin_images/form_pred_chat_2.png) |

### Business Idea Service
- **Description**: Generates business ideas based on user parameters such as location, capital, and sector.
- **Functionality**:
  - Personalized ideas for various parameters.
  - Interactive AI chat for further exploration.
- **Screenshots**:
  | Business Idea Form | Business Idea Chat |
  | :---: | :---: |
  | ![Business Form](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_busin_form1.png) | ![Chat 1](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_busin_chat1.png) |

### Financial Advice Service
- **Description**: Provides personalized financial advice for SMEs.
- **Functionality**:
  - Custom advice based on financial parameters.
  - AI chat for detailed guidance.
- **Screenshots**:
  | Financial Advice Form | Financial Advice Chat |
  | :---: | :---: |
  | ![Advice Form](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_finan_form1.png) | ![Advice Chat](https://github.com/Ajisco/ai-finance/blob/master/fin_images/fin_finan_chat1.png) |

## Additional Features
- **Interactive User Interface**:
  - Backend: Flask.
  - Frontend: HTML, CSS, JavaScript.
- **Online Accessibility**:
  - Hosted on Render.

The visual presentation of the application's interfaces provides a clear insight into the user experience and functionality of each service. These neatly organized sections with screenshots create a comprehensive and visually appealing overview, enhancing the documentation's effectiveness and aesthetic appeal.

---

## 📍 Installation and Setup

### Prerequisites
- Python 3.x
- Pip (Python package manager)

### Installation Steps
1. Clone the GitHub repository:
   ```bash
   git clone https://github.com/Ajisco/finai.git
   ```
2. Change directory to the project folder:
   ```bash
   cd finai
   ```
3. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally
To run the Flask app locally:
```bash
python app.py
```
Visit `http://localhost:5000` in your web browser to interact with the application.

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

---

## 📍 Skills and Technologies
- *Programming:* Python, JavaScript
- *Large Languauge Model:* GPT 3.5
- *Backend Development:* Flask, Ajax.js
- *Prompt Enngineering*
- *Data Wrangling:* Pandas, Numpy
- *Machine Learning:* Scikit Learn, Random Forest Classifier
- *Frontend Development:* HTML, CSS, Bootstrap
- *Cloud Deployment:* Render
- *Data Analysis and Visualization:* Seaborn, Matplotlib

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

