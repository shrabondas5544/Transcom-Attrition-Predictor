# 🚀 Transcom Attrition Predictor

> **An enterprise-grade, machine-learning-powered HR Intelligence Platform** that combines predictive analytics, explainable AI, generative retention planning, and interactive scenario simulation — built for Transcom field operations to identify, understand, and act on employee flight risk before it becomes turnover.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Feature Highlights](#-feature-highlights)
   - [Executive HR Dashboard](#1-executive-hr-dashboard)
   - [Interactive Chart Filtering](#2-interactive-chart-filtering--donut--bar-click-filter)
   - [Employee Directory & Predict Single Employee](#3-employee-directory--predict-single-employee)
   - [Clickable Row → Deep-Dive Modal](#4-clickable-row--deep-dive-modal)
   - [SHAP Top Attrition Risk Contributors](#5-top-attrition-risk-contributors--shap-explainer)
   - [What-If Retention Simulator](#6-what-if-retention-simulator)
   - [AI Retention Prescription Card](#7-ai-retention-prescription-card)
   - [AI Retention Advisor Chatbot](#8-ai-retention-advisor-chatbot)
   - [PDF Executive Report Export](#9-pdf-executive-report-export)
3. [Tech Stack](#-tech-stack)
4. [System Architecture](#-system-architecture)
5. [Project Structure](#-project-structure)
6. [API Reference](#-api-reference)
7. [How to Run on Any Device](#-how-to-run-on-any-device)
   - [Method A: Docker (Recommended)](#method-a-docker-recommended---works-on-windows-mac-linux)
   - [Method B: Local Python Virtual Environment](#method-b-local-python-virtual-environment)
8. [Environment Variables](#-environment-variables)
9. [Dataset & Model Training](#-dataset--model-training)
10. [ML Model Details](#-ml-model-details)

---

## 🌟 Project Overview

The **Transcom Attrition Predictor** is a full-stack web application that empowers HR managers to proactively manage employee turnover. Instead of reacting to resignations, this system scores every employee's attrition probability in real-time, explains *why* they are at risk using SHAP values, and generates personalized AI retention strategies using Google Gemini 2.5.

**The platform answers three critical HR questions:**

| Question | How the System Answers |
|---|---|
| **Who is at risk?** | Random Forest classifier scores every employee with an attrition probability (0–100%) |
| **Why are they at risk?** | SHAP explainability surfaces the top 3 positive drivers pushing each employee toward leaving |
| **What should we do?** | Google Gemini 2.5 generates a tailored, actionable retention prescription per employee |

---

## ✨ Feature Highlights

### 1. Executive HR Dashboard

The main landing page (`/`) presents a real-time command center for HR leadership, automatically loaded with live data from the SQLite database.

**What it displays:**

| KPI Card | Description |
|---|---|
| **Total Employees** | Live count of all employees in the system directory |
| **High Risk Count** | Employees with attrition probability ≥ 50% (classified as "High") |
| **Medium Risk Count** | Employees with attrition probability between 20–49% |
| **Current Attrition Rate** | `(High + Medium) / Total × 100%` — the composite flight risk rate across the roster |

**Charts rendered on the dashboard:**

- **Risk Distribution Donut Chart** — Visualizes the proportion of Low / Medium / High risk employees as a color-coded donut chart
- **Attrition by Location Bar Chart** — Horizontal bar chart showing the count of High+Medium risk employees across all 6 Bangladeshi field locations (Dhaka, Chittagong, Sylhet, Rajshahi, Khulna, Barisal)
- **Advanced Analytics Panel** — Contains a Radar Chart (workforce wellness KPIs), Bubble Chart (overtime × salary × distance correlation), and a 12-month Attrition Trend Line Chart

---

### 2. Interactive Chart Filtering — Donut & Bar Click-Filter

> **One of the most powerful UI features:** Every chart segment is clickable and functions as a live filter for the Employee Directory table below.

#### 🍩 Donut Chart (Risk Category Filter)

- **Click "High" segment** → The Employee Directory table instantly filters to show **only High-risk employees**
- **Click "Medium" segment** → Table filters to **Medium-risk employees only**
- **Click "Low" segment** → Table filters to **Low-risk employees only**
- **Click the same segment again** → Clears the filter and restores the full directory

**How it works technically:**
When a donut segment is clicked, a JavaScript event listener captures the risk category label. The Employee Directory fetches data from `GET /employees/` and applies a client-side `.filter()` against `risk_category`. The table re-renders instantly with filtered results and shows the active filter badge at the top.

#### 📊 Location Bar Chart Filter

- **Click any location bar** (e.g., "Dhaka") → The Employee Directory table filters to show **only employees from that location**
- **Click the same bar again** → Clears the location filter and restores all employees
- Works in **combination** with the donut chart filter — e.g., click "High" risk + click "Dhaka" to see only High-risk Dhaka employees

**Visual feedback:** When a filter is active, a highlighted filter badge appears above the table indicating which risk category and/or location is currently applied.

---

### 3. Employee Directory & Predict Single Employee

#### Employee Directory Table

The Employee Directory at the bottom of the dashboard displays all employees sorted by **attrition probability (descending)** — highest risk employees appear first for immediate visibility.

**Columns shown per employee:**

| Column | Description |
|---|---|
| **ID** | Auto-assigned employee identifier |
| **Location** | Field office city |
| **Age / Gender** | Demographics |
| **Tenure** | Years of service |
| **Monthly Salary** | In BDT |
| **Overtime Hours** | Hours per month |
| **Performance Rating** | 1–5 scale |
| **Risk Category** | Color-coded badge: 🔴 High / 🟡 Medium / 🟢 Low |
| **Attrition Probability** | Percentage score from the ML model |
| **Primary Driver** | Top SHAP feature identified as the main attrition risk factor |

#### 🔮 Predict Single Employee (Manual Input)

**Endpoint:** `POST /predict-single/`

The dashboard provides a dedicated form where an HR manager can input the raw attributes of **any employee** — including hypothetical or new hires not yet in the database — and receive an instant prediction.

**Input fields accepted:**

```
Age, Gender, Educational Qualification, Location, Tenure,
Monthly Salary, Incentive Earnings, Attendance %, Leave Utilization,
Distance from Workplace, Number of Transfers, Performance Rating,
Training Hours, Promotion History, Manager Effectiveness Score,
Employee Engagement Score, Overtime Hours
```

**Output returned:**

```json
{
  "probability": 0.7823,
  "risk_category": "High",
  "primary_driver": "Overtime Hours"
}
```

**Technical flow:**
1. Form data is POSTed as JSON to `PredictSingleView`
2. The view loads the saved `attrition_model.pkl` + `preprocessor.pkl` + `shap_explainer.pkl`
3. The input is run through `preprocessor.transform()` → `model.predict_proba()` for the probability score
4. SHAP values are computed on the preprocessed features; the feature with the **highest positive SHAP value** (most pushing toward attrition) becomes the Primary Driver
5. Risk category is assigned: ≥ 50% = High, 20–49% = Medium, < 20% = Low

---

### 4. Clickable Row → Deep-Dive Modal

Clicking **any row** in the Employee Directory table opens a beautiful, dark-themed pop-up modal that provides a comprehensive 360° view of that individual employee.

**The modal contains 3 sections:**

#### Section A — Full Employee Profile
All 17 raw metric fields are displayed in a clean grid layout:
- Demographics: Age, Gender, Education, Location
- Employment: Tenure, Salary, Incentive Earnings, Attendance %, Leave Utilization
- Performance: Rating, Training Hours, Promotion History
- Work Conditions: Overtime Hours, Distance from Workplace, Number of Transfers
- Management: Manager Effectiveness Score, Employee Engagement Score
- Prediction Results: Attrition Probability (displayed as a progress bar), Risk Category badge, Primary Driver label

#### Section B — SHAP Risk Driver Chart *(see Section 5 below)*

#### Section C — What-If Retention Simulator *(see Section 6 below)*

**Endpoint called:** `GET /employees/{employee_id}/insights/`

**Technical flow when modal opens:**
1. JavaScript captures the `data-employee-id` from the clicked row
2. An async AJAX `fetch()` call is made to `EmployeeDetailAPIView`
3. The backend fetches the Employee object, builds a single-row Pandas DataFrame, runs the SHAP explainer, and returns:
   - Full employee profile JSON
   - Top 3 positive SHAP contributors (feature name + SHAP value)
4. The modal renders instantly with employee data and draws the SHAP mini-bar chart

---

### 5. Top Attrition Risk Contributors — SHAP Explainer

> **Explainable AI (XAI)** is the core of what makes this system trustworthy for enterprise HR decisions. Instead of a black-box score, every prediction is accompanied by a precise, feature-level explanation.

**What is SHAP?**
SHAP (SHapley Additive exPlanations) is a game-theoretic approach that calculates each feature's contribution to pushing the model output higher or lower than the baseline prediction. A positive SHAP value means the feature is **increasing** attrition risk; a negative SHAP value means it is **decreasing** risk.

**How it's used in this system:**

1. **System-wide (Batch Inference):** When `Run Predictions` is triggered, every employee is scored and their **single top SHAP driver** is extracted and stored in the `primary_driver` database field. This drives the "Primary Driver" column in the table and the dominant driver stat in the PDF report.

2. **Individual (Modal Deep-Dive):** When a modal is opened, the **top 3 positive SHAP contributors** are computed fresh and returned to the frontend. These are displayed as a **horizontal mini-bar chart** inside the modal, visually showing which 3 features contribute the most to that specific person's risk score.

**Example output for a high-risk employee:**

| Rank | Feature | SHAP Value (Impact) |
|---|---|---|
| 1 | Overtime Hours | +0.312 |
| 2 | Distance from Workplace | +0.187 |
| 3 | Manager Effectiveness Score | +0.094 |

**Endpoint:** `GET /employees/{employee_id}/insights/`
Returns: `{ "top_drivers": [{"name": "Overtime Hours", "value": 0.312}, ...] }`

**Technical implementation:**
- The SHAP `TreeExplainer` is initialized once during model training and saved as `shap_explainer.pkl`
- On each inference, `explainer(X_proc_df)` computes SHAP values for the preprocessed feature vector
- The code handles both 2D and 3D SHAP output shapes (for binary classification with `predict_proba`)
- Positive SHAP indices are isolated, sorted by magnitude, and the top N are returned with clean feature names (removing sklearn pipeline prefixes like `cat__`, `num__`)

---

### 6. "What-If" Retention Simulator

> **The most interactive feature of the platform.** HR managers can dynamically test compensation and workload interventions and see how they would statistically change an employee's flight risk.

**Location:** Inside the Employee Deep-Dive Modal (Section C), below the SHAP chart.

**How to use:**

1. Open any employee's modal by clicking their row
2. Scroll to the **"Retention Simulator"** section at the bottom of the modal
3. Adjust the **Monthly Salary** input slider or text field (increase to simulate a raise, decrease to simulate a cut)
4. Adjust the **Overtime Hours** input slider or text field (increase to simulate heavier workload, decrease to simulate workload relief)
5. Click **"Simulate Adjustments"**
6. The updated **Flight Risk Score** and **Risk Category** appear immediately without reloading the page

**The business logic is HR-rational:**

| Change | Effect | Reasoning |
|---|---|---|
| **Salary ↑** | Risk Score **decreases** | Higher compensation reduces financial dissatisfaction |
| **Salary ↓** | Risk Score **increases** | Lower pay increases attrition likelihood |
| **Overtime ↑** | Risk Score **increases** | Excessive hours drive burnout and flight risk |
| **Overtime ↓** | Risk Score **decreases** | Workload relief reduces pressure to leave |

**Sensitivity calibration:**
- A 100% salary increase from baseline moves probability by approximately −0.50 (large raise, meaningful impact)
- A 100% overtime increase from baseline moves probability by approximately +0.35
- The simulator clamps the final probability to the `[0.0, 1.0]` range

**Endpoint:** `POST /predict-scenario/`

**Request body:**
```json
{
  "employee_id": 142,
  "monthly_salary": 85000,
  "overtime_hours": 15
}
```

**Response:**
```json
{
  "probability": 0.3241,
  "risk_category": "Medium"
}
```

**Technical flow:**
1. The current employee's full feature vector is loaded from the database
2. Only `monthly_salary` and `overtime_hours` are patched with the simulator values — all other features remain at their real values
3. The patched record is run through `preprocessor.transform()` → `model.predict_proba()` for the base model probability
4. A **delta-based directional adjustment** is applied: `prob -= salary_delta × 0.50` and `prob += overtime_delta × 0.35`
5. The adjusted probability and new risk category are returned to the frontend in real-time

---

### 7. AI Retention Prescription Card

> **Personalized, context-aware retention advice** generated by Google Gemini 2.5 Flash, tailored to each individual employee's specific risk profile and SHAP-identified drivers.

**Location:** Inside the Employee Deep-Dive Modal — triggered when you click the **"Generate AI Prescription"** button after the modal loads.

**How it works:**

1. When the prescription button is clicked, a `POST /employees/{employee_id}/prescription/` request is fired
2. The backend (`EmployeePrescriptionAPIView`) fetches the employee's full profile from the database
3. It loads the ML artifacts and re-computes the **top 3 SHAP drivers** for that employee in real-time
4. The employee profile (all 17 metrics + risk score) and the top SHAP drivers are formatted into a rich, structured **contextual prompt**
5. The prompt is sent to Google Gemini 2.5 Flash via LangChain's `ChatGoogleGenerativeAI` interface
6. Gemini returns a prescription in natural language — a specific, actionable plan for retaining that employee
7. The response is streamed and displayed in the modal as a formatted **"Prescription Card"**

**Example Prescription Output:**

```
Employee #142 is at HIGH flight risk (78.2%), primarily driven by:
  1. Excessive Overtime (40+ hrs/month) — main burnout signal
  2. Long Commute (27 km) — daily friction factor
  3. Low Manager Effectiveness (Score: 3/10) — leadership disconnect

Recommended Retention Actions:
  ✅ Reduce mandatory overtime to ≤ 20 hrs/month immediately
  ✅ Offer a travel/fuel stipend or remote-work 2 days/week
  ✅ Schedule a manager coaching session within 14 days
  ✅ Discuss a structured salary review aligned with performance rating
```

**Endpoint:** `POST /employees/{employee_id}/prescription/`

**Backend service:** `chatbot.services.generate_individual_prescription(employee_data, top_contributors)`

**Technology stack used:**
- `langchain-google-genai` (LangChain wrapper for Google Gemini)
- `ChatGoogleGenerativeAI(model="gemini-2.5-flash")`
- `GEMINI_API_KEY` environment variable (required)

---

### 8. AI Retention Advisor Chatbot

> **A conversational HR assistant** that allows managers to ask freeform questions about workforce attrition, HR policies, and retention strategies — and receive informed, Gemini-powered responses.

**Location:** Fixed floating widget (bottom-right corner of every page), accessible from the dashboard at all times.

**How to use:**

1. Click the **chat bubble icon** in the bottom-right corner to open the chat panel
2. Type any HR-related question in natural language, for example:
   - *"What are the most common reasons employees leave in Dhaka?"*
   - *"How should I handle an employee with a low manager effectiveness score?"*
   - *"What retention strategies work best for high-overtime employees?"*
   - *"Explain what SHAP values mean in the context of this dashboard"*
3. Press Enter or click Send
4. The Gemini 2.5 model returns a detailed, conversational response within seconds

**Technical Architecture (RAG-powered):**

The chatbot is powered by a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Document Indexing:** HR policy documents and retention guides are embedded using Google Generative AI Embeddings (`text-embedding-004`) and stored in a **FAISS** vector index
2. **Query Retrieval:** When a user submits a question, the query is embedded and FAISS searches for the most semantically relevant document chunks
3. **Augmented Generation:** The retrieved context is injected into the Gemini prompt alongside the user's question, so the model grounds its response in actual policy knowledge rather than hallucinating

**Endpoint:** `POST /chatbot/`

**Request body:**
```json
{
  "message": "How do I reduce overtime-related attrition?"
}
```

**Response:**
```json
{
  "success": true,
  "response": "To reduce overtime-related attrition, consider implementing..."
}
```

**Technology stack used:**
- `langchain-google-genai` for Gemini LLM + Embeddings
- `faiss-cpu` for vector similarity search
- `langchain` for the RAG chain orchestration
- `ChatGoogleGenerativeAI(model="gemini-2.5-flash")` as the LLM backbone

---

### 9. PDF Executive Report Export

**Endpoint:** `GET /export-report/`

Click the **"Export PDF Summary"** button on the dashboard to instantly download a professionally formatted PDF executive report.

**Report contents:**

| Section | Details |
|---|---|
| **Executive Summary KPIs** | Total employees, high risk count, composite flight risk rate, average attrition probability, dominant SHAP driver, model diagnostics state |
| **Location Risk Distribution Table** | Per-city breakdown of roster size, High risk count, Medium risk count, and the location's flight risk rate percentage |
| **Strategic Retention Guidelines** | Auto-generated policy recommendations based on the dominant SHAP drivers (Overtime Fatigue, Commute Mitigation, Manager Support) |

**Styling:** Corporate clean theme using ReportLab with `#0f172a` dark headers and `#4f46e5` accent color matching the dashboard design system.

---

## 🛠️ Tech Stack

| Layer | Technology | Version |
|---|---|---|
| **Backend Framework** | Django | 6.0.6 |
| **Machine Learning** | scikit-learn | 1.8.0 |
| **Class Imbalance** | imbalanced-learn (SMOTE) | 0.14.1 |
| **Explainable AI** | SHAP (TreeExplainer) | 0.52.0 |
| **Data Processing** | Pandas, NumPy | 3.0.3 / 2.4.6 |
| **Model Persistence** | joblib | 1.5.3 |
| **Vector Search** | FAISS (CPU) | 1.14.2 |
| **LLM Orchestration** | LangChain | 1.3.4 |
| **Generative AI** | Google Gemini 2.5 Flash | via `langchain-google-genai` 4.2.4 |
| **PDF Generation** | ReportLab | 4.5.1 |
| **Database** | SQLite3 | (Django built-in) |
| **Containerization** | Docker + Docker Compose | — |
| **Frontend** | Vanilla JS, Chart.js, HTML/CSS | — |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         BROWSER (HR Manager)                    │
│  Dashboard  ──── Click Chart ──── Open Modal ──── Chat Widget   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP Requests
┌──────────────────────────▼──────────────────────────────────────┐
│                      DJANGO WEB SERVER                          │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  dashboard/  │  │   chatbot/   │  │       api/          │   │
│  │   views.py   │  │  services.py │  │   (CSV Upload)      │   │
│  └──────┬───────┘  └──────┬───────┘  └─────────────────────┘   │
└─────────┼─────────────────┼───────────────────────────────────-─┘
          │                 │
    ┌─────▼─────┐    ┌──────▼──────────────────────────────┐
    │  SQLite3  │    │           ML Pipeline               │
    │ Database  │    │  preprocessor.pkl → model.pkl       │
    └───────────┘    │         ↓                           │
                     │  shap_explainer.pkl (SHAP values)   │
                     │         ↓                           │
                     │  FAISS Vector Index (RAG)           │
                     │         ↓                           │
                     │  Google Gemini 2.5 Flash API        │
                     └─────────────────────────────────────┘
```

---

## 📂 Project Structure

```
Transcom project/
├── generate_dataset.py          # Synthetic dataset generator (3,000 employees)
├── train_model.py               # Standalone model trainer (saves .pkl artifacts)
├── README.md                    # This documentation file
│
└── transcom_hr/                 # Django project root
    ├── manage.py
    ├── requirements.txt         # All Python dependencies (pinned versions)
    ├── Dockerfile               # Docker container specification
    ├── docker-compose.yml       # Multi-service orchestration
    ├── .env.template            # Environment variable template
    ├── .env                     # Your local secrets (NOT committed to git)
    ├── db.sqlite3               # SQLite database (persisted via Docker volume)
    │
    ├── transcom_hr/             # Django project settings package
    │   ├── settings.py          # Main Django settings (DB, installed apps, middleware)
    │   ├── urls.py              # Root URL router
    │   ├── middleware.py        # AutoAdminLoginMiddleware (password-less admin access)
    │   └── wsgi.py
    │
    ├── dashboard/               # Core HR analytics Django app
    │   ├── models.py            # Employee model (17 raw fields + 3 predictive fields)
    │   ├── views.py             # All API views and page renderers
    │   ├── urls.py              # Dashboard URL patterns
    │   ├── admin.py             # Django Admin registration
    │   │
    │   ├── ml/                  # Machine Learning sub-module
    │   │   ├── train_model.py   # Model training logic + get_top_contributors()
    │   │   └── saved_models/    # Pickled artifacts (not committed to git)
    │   │       ├── attrition_model.pkl
    │   │       ├── preprocessor.pkl
    │   │       ├── shap_explainer.pkl
    │   │       └── feature_names.pkl
    │   │
    │   ├── services/
    │   │   └── inference.py     # Batch inference runner + data drift checker
    │   │
    │   └── templates/
    │       └── dashboard/
    │           └── index.html   # Main dashboard template (JS, charts, modals)
    │
    ├── chatbot/                 # AI Chatbot + Prescription Django app
    │   └── services.py          # LangChain RAG pipeline + Gemini integration
    │
    └── data/                    # Training dataset CSV files
        └── transcom_field_officer_attrition.csv
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main dashboard page |
| `GET/POST` | `/run-predictions/` | Batch inference on all employees |
| `GET` | `/employees/` | List all employees (sorted by risk desc.) |
| `POST` | `/upload-csv/` | Upload CSV/XLSX and populate database |
| `POST` | `/predict-single/` | Predict risk for a new manual input |
| `POST` | `/chatbot/` | Send a message to the AI Advisor chatbot |
| `GET` | `/advanced-analytics/` | Aggregated radar/bubble/line chart data |
| `GET` | `/dashboard-stats/` | Live KPI stats and chart data |
| `GET` | `/export-report/` | Download PDF executive summary report |
| `GET` | `/employees/<id>/insights/` | Individual employee profile + SHAP top 3 |
| `POST` | `/employees/<id>/prescription/` | Generate AI retention prescription |
| `POST` | `/predict-scenario/` | What-If simulator (salary + overtime) |
| `GET` | `/admin/` | Django Admin panel (no login required) |

---

## 💻 How to Run on Any Device

### Method A: Docker (Recommended) — Works on Windows, Mac, Linux

Docker containerizes the entire application with all dependencies pre-installed. No Python setup needed on your machine.

**Prerequisites:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/shrabondas5544/Transcom-Attrition-Predictor.git
cd "Transcom-Attrition-Predictor"
```

**Step 2 — Navigate into the Django root:**
```bash
cd transcom_hr
```

**Step 3 — Create your environment file:**
```bash
# On Windows (PowerShell)
Copy-Item .env.template .env

# On Mac/Linux
cp .env.template .env
```

**Step 4 — Add your Gemini API Key:**
Open `.env` in any text editor and replace the placeholder:
```env
GEMINI_API_KEY=AIzaSy_YOUR_ACTUAL_KEY_HERE
```
> Get a free API key at https://aistudio.google.com/ → "Get API Key"

**Step 5 — Build and launch the container:**
```bash
docker-compose up --build
```
On first run this downloads the Python base image and installs all packages (takes ~2–5 minutes). Subsequent runs start in seconds.

**Step 6 — Access the application:**

| Page | URL |
|---|---|
| 🏠 Main Dashboard | http://localhost:8000/ |
| 🔧 Django Admin | http://localhost:8000/admin/ |
| 📊 Run Predictions | http://localhost:8000/run-predictions/ |

**Step 7 — Run initial predictions:**
After the server starts, either:
- Click the **"Run Predictions"** button on the dashboard, OR
- Visit http://localhost:8000/run-predictions/ in your browser

This scores all 3,000 employees and populates the risk columns.

**To stop the container:**
```bash
docker-compose down
```

---

### Method B: Local Python Virtual Environment

Use this method for development, debugging, or when Docker is not available.

**Prerequisites:**
- Python 3.11 or higher installed → https://python.org/downloads
- `pip` (comes with Python)
- Git (optional, for cloning)

---

**Step 1 — Get the project:**
```bash
git clone https://github.com/shrabondas5544/Transcom-Attrition-Predictor.git
cd "Transcom-Attrition-Predictor"
```

**Step 2 — Create a Python virtual environment:**
```bash
python -m venv venv
```

**Step 3 — Activate the virtual environment:**

| Operating System | Command |
|---|---|
| **Windows (PowerShell)** | `.\venv\Scripts\Activate.ps1` |
| **Windows (CMD)** | `.\venv\Scripts\activate.bat` |
| **macOS** | `source venv/bin/activate` |
| **Linux (Ubuntu/Debian)** | `source venv/bin/activate` |

You will see `(venv)` appear in your terminal prompt when activation succeeds.

**Step 4 — Install all Python dependencies:**
```bash
cd transcom_hr
pip install --upgrade pip
pip install -r requirements.txt
```
> ⚠️ This installs heavy scientific libraries (scikit-learn, SHAP, numpy, pandas, faiss). Allow 3–10 minutes depending on your internet connection.

**Step 5 — Configure environment variables:**
```bash
# Windows PowerShell
Copy-Item .env.template .env

# macOS/Linux
cp .env.template .env
```
Open `.env` and fill in your Gemini API key:
```env
GEMINI_API_KEY=AIzaSy_YOUR_ACTUAL_KEY_HERE
```

**Step 6 — *(Optional)* Generate a fresh dataset and retrain the model:**

If you want to regenerate the synthetic dataset and retrain the Random Forest from scratch:
```bash
# Go back to the project root first
cd ..

# Generate 3,000 synthetic employee records
python generate_dataset.py

# Train the model and save artifacts (.pkl files)
python train_model.py

# Return to Django directory
cd transcom_hr
```

> The pre-trained model artifacts (`.pkl` files) should already be present in `dashboard/ml/saved_models/`. Only run these if you want to retrain with different parameters.

**Step 7 — Apply database migrations:**
```bash
python manage.py migrate
```

**Step 8 — Start the development server:**
```bash
python manage.py runserver
```

**Step 9 — Open the application:**

| Page | URL |
|---|---|
| 🏠 Main Dashboard | http://127.0.0.1:8000/ |
| 🔧 Django Admin | http://127.0.0.1:8000/admin/ |

**Step 10 — Seed and score employees:**
Click **"Run Predictions"** on the dashboard or visit `http://127.0.0.1:8000/run-predictions/` to run batch inference on all employee records.

---

### Method C: Running on a Remote Server (VPS / Cloud VM)

**Ubuntu Server example:**

```bash
# Update and install Python
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip git

# Clone project
git clone https://github.com/shrabondas5544/Transcom-Attrition-Predictor.git
cd Transcom-Attrition-Predictor

# Setup venv
python3.11 -m venv venv
source venv/bin/activate

# Install deps
cd transcom_hr
pip install -r requirements.txt

# Configure .env
cp .env.template .env
nano .env   # Add GEMINI_API_KEY

# Run
python manage.py migrate
python manage.py runserver 0.0.0.0:8000
```

Access at `http://<your-server-ip>:8000/`

> For production deployment, use Gunicorn + Nginx instead of Django's development server.

---

## 🔐 Environment Variables

All secrets are stored in `transcom_hr/.env`:

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ Yes | Google Gemini API key for the AI chatbot and prescription card |
| `SECRET_KEY` | Auto | Django secret key (auto-generated in settings.py if missing) |

**Getting a Gemini API Key:**
1. Go to https://aistudio.google.com/
2. Sign in with your Google account
3. Click **"Get API Key"** → **"Create API key"**
4. Copy the key and paste it into your `.env` file

> ⚠️ **Never commit your `.env` file to git.** It is already listed in `.gitignore`.

---

## 📊 Dataset & Model Training

### Dataset Generation

The synthetic dataset is generated by `generate_dataset.py` (run from the project root). It creates 3,000 fictional Transcom field officers with statistically realistic attributes.

**Features generated:**

| Feature | Type | Range |
|---|---|---|
| Age | Integer | 22–60 |
| Gender | Categorical | Male / Female / Other |
| Educational Qualification | Categorical | High School / Bachelors / Masters / PhD |
| Location | Categorical | Dhaka / Chittagong / Sylhet / Rajshahi / Khulna / Barisal |
| Tenure | Integer | 1–15 years |
| Monthly Salary | Integer | 15,000–150,000 BDT |
| Incentive Earnings | Float | 5–20% of salary |
| Attendance % | Float | 75–100% |
| Leave Utilization | Integer | 5–25 days |
| Distance from Workplace | Integer | 1–30 km |
| Number of Transfers | Integer | 0–4 |
| Performance Rating | Integer | 1–5 |
| Training Hours | Integer | 10–100 hrs |
| Promotion History | Integer | 0–3 |
| Manager Effectiveness Score | Integer | 1–10 |
| Employee Engagement Score | Integer | 1–10 |
| Overtime Hours | Integer | 0–50 hrs/month |

**Attrition labeling logic:**
A logit-weighted scoring system assigns "Yes" to the top 32% of employees with the highest attrition probability based on: overtime > 40 hrs, distance > 25 km, manager score < 2.5, salary < 22,000 BDT, or performance rating ≤ 2.

### Model Training

Run `train_model.py` to train and save all artifacts:

```bash
python train_model.py
```

**Pipeline:**
1. `StandardScaler` on numerical features
2. `OneHotEncoder` on categorical features (Gender, Qualification, Location)
3. `SMOTE` oversampling to address class imbalance (32% attrition rate)
4. `RandomForestClassifier` with `n_estimators=100`, `class_weight='balanced'`
5. `TreeExplainer` SHAP explainer fitted on transformed training data

**Artifacts saved to `transcom_hr/dashboard/ml/saved_models/`:**
- `attrition_model.pkl` — The trained Random Forest classifier
- `preprocessor.pkl` — The fitted ColumnTransformer (scaler + encoder)
- `shap_explainer.pkl` — The TreeExplainer instance
- `feature_names.pkl` — List of post-encoding feature name strings

---

## 🤖 ML Model Details

| Parameter | Value |
|---|---|
| **Algorithm** | Random Forest Classifier |
| **Estimators** | 100 trees |
| **Class Weight** | Balanced (handles imbalance) |
| **Oversampling** | SMOTE (synthetic minority oversampling) |
| **Train/Test Split** | 80% / 20% (stratified) |
| **Optimization Target** | Recall (catch as many true attrition cases as possible) |
| **Risk Thresholds** | High ≥ 50% \| Medium 20–49% \| Low < 20% |
| **Explainability** | SHAP TreeExplainer (additive feature attribution) |

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Built with ❤️ by Shrabon Das · scikit-learn + Django*