from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, Form, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent / "flask"

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="your_random_secret_key_here")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

model = pickle.load(open(str(BASE_DIR / "model.pkl"), "rb"))
dataset = pd.read_csv(str(BASE_DIR / "diabetes.csv"))
dataset_X = dataset.iloc[:, [1, 4, 5, 7]].values
sc = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = sc.fit_transform(dataset_X)


def session_redirect_home(request: Request):
    return RedirectResponse(request.url_for("home"), status_code=303)


@app.get("/", name="home")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "session": request.session, "url_for": request.url_for},
    )


@app.get("/start", name="start")
async def start(request: Request):
    return templates.TemplateResponse(
        "personal.html",
        {"request": request, "session": request.session, "url_for": request.url_for},
    )


@app.post("/process_personal", name="process_personal")
async def process_personal(
    request: Request,
    age: int = Form(...),
    gender: str = Form(...),
    height: float = Form(...),
    weight: float = Form(...),
):
    request.session["age"] = age
    request.session["gender"] = gender
    request.session["height"] = height
    request.session["weight"] = weight
    height_m = height / 100
    request.session["bmi"] = round(weight / (height_m ** 2), 2)
    return RedirectResponse(request.url_for("question_1"), status_code=303)


@app.get("/question/1", name="question_1")
async def question_1(request: Request):
    return templates.TemplateResponse(
        "question1.html",
        {"request": request, "session": request.session, "url_for": request.url_for},
    )


@app.post("/process_question_1", name="process_question_1")
async def process_question_1(
    request: Request,
    family_history: str = Form(...),
    frequent_urination: str = Form(...),
    excessive_thirst: str = Form(...),
    frequent_hunger: str = Form(...),
    sudden_weight_loss: str = Form(...),
):
    request.session["family_history"] = family_history
    request.session["frequent_urination"] = frequent_urination
    request.session["excessive_thirst"] = excessive_thirst
    request.session["frequent_hunger"] = frequent_hunger
    request.session["sudden_weight_loss"] = sudden_weight_loss
    return RedirectResponse(request.url_for("question_2"), status_code=303)


@app.get("/question/2", name="question_2")
async def question_2(request: Request):
    return templates.TemplateResponse(
        "question2.html",
        {"request": request, "session": request.session, "url_for": request.url_for},
    )


@app.post("/process_question_2", name="process_question_2")
async def process_question_2(
    request: Request,
    tired_weak: str = Form(...),
    blurred_vision: str = Form(...),
    slow_healing: str = Form(...),
    numbness_tingling: str = Form(...),
    frequent_infections: str = Form(...),
):
    request.session["tired_weak"] = tired_weak
    request.session["blurred_vision"] = blurred_vision
    request.session["slow_healing"] = slow_healing
    request.session["numbness_tingling"] = numbness_tingling
    request.session["frequent_infections"] = frequent_infections
    return RedirectResponse(request.url_for("question_3"), status_code=303)


@app.get("/question/3", name="question_3")
async def question_3(request: Request):
    return templates.TemplateResponse(
        "question3.html",
        {"request": request, "session": request.session, "url_for": request.url_for},
    )


@app.post("/process_question_3", name="process_question_3")
async def process_question_3(
    request: Request,
    high_bp: str = Form(...),
    high_cholesterol: str = Form(...),
    blood_glucose: str = Form(""),
    hba1c: str = Form(""),
    exercise: str = Form(...),
    sugary_foods: str = Form(...),
    smoke: str = Form(...),
    alcohol: str = Form(...),
    sleep_hours: float = Form(...),
    stress: str = Form(...),
):
    request.session["high_bp"] = high_bp
    request.session["high_cholesterol"] = high_cholesterol
    request.session["blood_glucose"] = blood_glucose
    request.session["hba1c"] = hba1c
    request.session["exercise"] = exercise
    request.session["sugary_foods"] = sugary_foods
    request.session["smoke"] = smoke
    request.session["alcohol"] = alcohol
    request.session["sleep_hours"] = sleep_hours
    request.session["stress"] = stress
    if request.session.get("gender") == "female":
        return RedirectResponse(request.url_for("question_female"), status_code=303)
    request.session["prediabetes"] = "no"
    request.session["diabetes_medication"] = "no"
    request.session["gestational_diabetes"] = "n/a"
    request.session["pcos"] = "n/a"
    return RedirectResponse(request.url_for("predict_report"), status_code=303)


@app.get("/question/female", name="question_female")
async def question_female(request: Request):
    return templates.TemplateResponse(
        "question_female.html",
        {"request": request, "session": request.session, "url_for": request.url_for},
    )


@app.post("/process_female", name="process_female")
async def process_female(
    request: Request,
    prediabetes: str = Form(...),
    diabetes_medication: str = Form(...),
    gestational_diabetes: str = Form(...),
    pcos: str = Form(...),
):
    request.session["prediabetes"] = prediabetes
    request.session["diabetes_medication"] = diabetes_medication
    request.session["gestational_diabetes"] = gestational_diabetes
    request.session["pcos"] = pcos
    return RedirectResponse(request.url_for("predict_report"), status_code=303)


@app.post("/predict", name="predict")
async def predict(request: Request):
    form = await request.form()
    float_features = [float(value) for value in form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(sc.transform(final_features))[0]
    prediction_text = (
        "You have Diabetes, please consult a Doctor."
        if prediction == 1
        else "You don't have Diabetes."
    )
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "session": request.session,
            "url_for": request.url_for,
            "prediction_text": prediction_text,
        },
    )


@app.get("/predict_report", name="predict_report")
async def predict_report(request: Request):
    if not request.session:
        return session_redirect_home(request)

    blood_glucose = request.session.get("blood_glucose", "").strip()
    glucose = float(blood_glucose) if blood_glucose else 100.0
    insulin = 79.8
    bmi = float(request.session.get("bmi", 0.0))
    age = int(request.session.get("age", 0))

    features = np.array([[glucose, insulin, bmi, age]])
    scaled_features = sc.transform(features)
    prediction = int(model.predict(scaled_features)[0])

    request.session["prediction"] = prediction
    request.session["prediction_status"] = "Positive" if prediction == 1 else "Negative"
    request.session["prediction_text"] = (
        "Diabetes Positive. Please consult a doctor."
        if prediction == 1
        else "Diabetes Negative. Keep monitoring your health."
    )

    return templates.TemplateResponse(
        "report.html",
        {"request": request, "session": request.session, "url_for": request.url_for},
    )
