# AI-Driven Financial Advisory App for SMEs

## Overview
This Flask application, utilizing OpenAI's GPT-3.5 Turbo, offers financial and business advice tailored for loans and SMEs. It includes a machine learning model for predictive analysis and features a dynamic user interface. The app is deployed on [Render](https://finai-t4wc.onrender.com).

## Key Features
- **AI-Powered Advice:** Uses OpenAI's GPT-3.5 Turbo for specialized financial guidance.
- **Machine Learning Model:** Incorporates a Random Forest model for predictive analysis in financial decisions.
- **Interactive User Interface:** Developed with Flask, rendering templates and managing static assets.
- **Deployment on Render:** Accessible online for wider reach and usability.

## Installation and Setup

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

## Application Structure
- `templates/`: HTML templates for the web interface.
- `static/assets/`: Static resources like CSS and JavaScript files.
- `app.py`: The main Flask application file detailing routes and logic.
- `Prediction.ipynb`: Jupyter notebook for developing the machine learning model.
- `random_forest_model.pkl`: The pre-trained Random Forest model.
- `requirements.txt`: Required Python packages for the application.

## Application Functionality
- **Loan Prediction:** Utilizes machine learning to predict loan approval based on user inputs.
- **Business Idea Generation:** AI-driven suggestions for business ideas based on user preferences and financial capacity.
- **Financial Advice:** Customized financial advice leveraging AI, tailored to user's specific business scenarios.

## Deployment
Hosted on Render, the application can be accessed at [finai-t4wc.onrender.com](https://finai-t4wc.onrender.com).

## Contributing
Contributions are welcome! Please follow these steps to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a new Pull Request.

## Acknowledgments
- OpenAI for the GPT-3.5 Turbo API.
- Flask community for the web framework.
- Render for hosting and deployment services.

## Contact
For questions or feedback, please reach out to [GitHub Profile](https://github.com/Ajisco).



