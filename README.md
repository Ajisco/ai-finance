
# AI-Driven Financial Advisory App for SMEs

## Overview
This AI-driven Flask application, leveraging OpenAI's GPT-3.5 Turbo, provides specialized financial and business advice, focusing on loans and SMEs. It integrates a machine learning model for predictive analysis, features a user-friendly interface, and is deployed on Render.

## Features
- **AI-Powered Financial Advice:** Utilizes OpenAI's GPT-3.5 Turbo for precise and custom financial guidance.
- **Machine Learning Integration:** Incorporates a Random Forest predictive model.
- **Intuitive User Interface:** Built with Flask, the frontend is organized under `templates` and `static/assets`.
- **Ready for Deployment:** Hosted on Render, available at [finai-t4wc.onrender.com](https://finai-t4wc.onrender.com).

## Getting Started

### Prerequisites
- Python 3.x
- Pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ajisco/finai.git
   ```
2. Navigate to the project directory:
   ```bash
   cd finai
   ```
3. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
Execute the Flask app:
```bash
python app.py
```
Access the application at [http://localhost:5000](http://localhost:5000).

## File Structure
- `templates/` - HTML files for the web interface.
- `static/assets/` - CSS, JS, and other static resources.
- `app.py` - Main Flask application script.
- `Prediction.ipynb` - Jupyter notebook for ML model development.
- `random_forest_model.pkl` - Serialized Random Forest model.
- `requirements.txt` - Python dependencies.

## Deployment
The app is deployed on Render and can be accessed at [finai-t4wc.onrender.com](https://finai-t4wc.onrender.com).

## Contributing
We welcome contributions. Here's how you can contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b new-feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin new-feature`).
5. Create a new Pull Request.

## Acknowledgments
- Thanks to OpenAI for providing GPT-3.5 Turbo.
- Gratitude to the Flask community for their excellent web framework.
- Render for hosting and deployment solutions.
```

This README is designed to provide a clear and professional overview of your project, along with instructions for installation, running the application, and contributing. You can adjust any part of it to better fit your project's specifics or personal preferences.
