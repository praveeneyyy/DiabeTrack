# DiabeTrack

DiabeTrack is an interactive diabetes risk assessment web application built with FastAPI. It guides users through a polished, question-driven experience and delivers a clear final report with a positive/negative diabetes prediction.

## Key Features

- Step-by-step questionnaire for user health data and symptoms
- BMI calculation based on height and weight
- Dark/light theme toggle with minimalist interface
- Final report page showing diabetes status as `Positive` or `Negative`
- Smart prediction using a pre-trained machine learning model

## Project Structure

- `flask/`
  - `app.py` — Flask application and prediction flow
  - `templates/` — HTML templates for home, questionnaire, and report pages
  - `static/css/app.css` — custom styling and dark mode support
  - `static/js/theme-toggle.js` — toggle theme logic
  - `static/js/interactive.js` — minimalist option selection interactions
  - `static/favicon.svg` — browser tab icon
  - `model.pkl` — saved diabetes prediction model
  - `diabetes.csv` — dataset used for modeling and default values
- `diabetes.ipynb` — exploratory notebook for data analysis and modeling
- `diabetes.py` — original script (if present) and analysis utilities
- `requirements.txt` — Python package dependencies

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Diabetes-Prediction-master.git
   cd Diabetes-Prediction-master
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the FastAPI app:
   ```bash
   uvicorn app:app --reload
   ```
5. Open the app in your browser:
   ```
   http://127.0.0.1:8000/
   ```

## Usage

- Start the assessment from the homepage
- Enter personal information such as age, gender, height, and weight
- Answer the symptom and lifestyle questions
- View the final report with diabetes status and health recommendations

## Notes

- The app currently uses a pre-trained model saved in `model.pkl`.
- If you want to retrain the model, use the dataset in `flask/diabetes.csv` and the notebook `diabetes.ipynb`.

## License

This project is for educational and demonstration purposes.
