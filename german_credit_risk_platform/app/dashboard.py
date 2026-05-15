from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT_FOR_IMPORTS / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from credit_risk.config import ARTIFACTS_DIR, METRICS_PATH, PROJECT_ROOT
from credit_risk.data import load_raw_data
from credit_risk.predict import (
    load_model,
    load_threshold,
    model_artifacts_ready,
    score_single_application,
)

st.set_page_config(
    page_title="German Credit Risk Command Center",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

DASHBOARD_CSS = """
<style>
:root {
    --navy: #1f2a3a;
    --navy-2: #26364b;
    --blue: #1368ce;
    --text: #232733;
    --muted: #6b7280;
    --panel: #ffffff;
    --line: #dde3ec;
    --canvas: #f7f9fc;
    --green: #18a058;
    --red: #d64545;
    --amber: #cc7a00;
}
[data-testid="stAppViewContainer"] {
    background: var(--canvas);
}
[data-testid="stHeader"] {
    background: rgba(247, 249, 252, 0.96);
}
[data-testid="stSidebar"] {
    background: var(--navy);
    border-right: 0;
}
[data-testid="stSidebar"] * {
    color: #edf3fb !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {
    color: #edf3fb !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: #33465f !important;
    border-color: #475b73 !important;
    border-radius: 7px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span,
[data-testid="stSidebar"] input {
    color: #ffffff !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] [role="slider"] {
    background: var(--blue) !important;
    border-color: var(--blue) !important;
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1680px;
}
.report-hero {
    background: linear-gradient(90deg, #1f2a3a 0%, #23405f 50%, #1368ce 100%);
    color: white;
    padding: 24px 28px;
    border-radius: 8px;
    margin-bottom: 16px;
    box-shadow: 0 8px 22px rgba(31, 42, 58, 0.16);
}
.report-hero h1 {
    color: #ffffff;
    font-size: 31px;
    line-height: 1.15;
    margin: 0;
    letter-spacing: 0;
}
.report-hero p {
    color: #d8e7ff;
    margin: 8px 0 0 0;
    font-size: 14px;
}
.report-status {
    display: inline-block;
    margin-top: 12px;
    padding: 6px 10px;
    border-radius: 5px;
    background: rgba(255, 255, 255, 0.14);
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
}
.kpi-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-top: 4px solid var(--blue);
    border-radius: 8px;
    padding: 16px 17px;
    min-height: 118px;
    box-shadow: 0 3px 10px rgba(31, 42, 58, 0.07);
}
.kpi-card.warning { border-top-color: var(--amber); }
.kpi-card.success { border-top-color: var(--green); }
.kpi-card.risk { border-top-color: var(--red); }
.kpi-label {
    color: #5e6c83;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.kpi-value {
    color: #111827;
    font-size: 31px;
    font-weight: 850;
    line-height: 1;
}
.kpi-subtitle {
    color: #69778c;
    font-size: 12px;
    margin-top: 9px;
}
.visual-title {
    color: var(--text);
    font-size: 16px;
    font-weight: 800;
    margin: 2px 0 8px 0;
}
.visual-note {
    color: var(--muted);
    font-size: 12px;
}
.badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 5px;
    background: #e7f0ff;
    color: #155cb8;
    font-size: 12px;
    font-weight: 800;
}
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 13px 14px;
    box-shadow: 0 2px 8px rgba(31, 42, 58, 0.06);
}
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"] {
    color: #5e6c83 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #111827 !important;
}
[data-testid="stTabs"] [role="tablist"] {
    gap: 5px;
    border-bottom: 1px solid var(--line);
}
[data-testid="stTabs"] button[role="tab"] {
    background: #ffffff;
    border: 1px solid var(--line);
    border-bottom: 0;
    border-radius: 6px 6px 0 0;
    color: #435067;
    padding: 9px 14px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    background: var(--navy);
    color: #ffffff;
    border-color: var(--navy);
}
[data-testid="stTabs"] button[aria-selected="true"] p {
    color: #ffffff !important;
}
[data-testid="stDataFrame"] {
    border-radius: 8px;
}
</style>
"""


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return load_raw_data()


@st.cache_data
def load_json_artifact(path: str) -> dict:
    artifact_path = Path(path)
    if not artifact_path.exists():
        return {}
    return json.loads(artifact_path.read_text(encoding="utf-8"))


@st.cache_resource
def load_runtime_model():
    if not model_artifacts_ready():
        return None, None
    return load_model(), load_threshold()


def risk_label(value: str) -> str:
    return "Bad risk" if value.lower() == "bad" else "Good risk"


def all_options(values: list[str]) -> list[str]:
    return ["All"] + sorted([str(value) for value in values])


def render_kpi(label: str, value: str, subtitle: str, tone: str = "") -> None:
    tone_class = f" {tone}" if tone else ""
    st.markdown(
        f"""
        <div class="kpi-card{tone_class}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def filter_equals(frame: pd.DataFrame, column: str, value: str) -> pd.DataFrame:
    if value == "All":
        return frame
    return frame[frame[column].astype(str) == value]


def show_confusion_matrix(matrix: list[list[int]]) -> None:
    if not matrix:
        st.info("Confusion matrix artifact is not available yet.")
        return
    frame = pd.DataFrame(
        matrix,
        index=["Actual good", "Actual bad"],
        columns=["Predicted good", "Predicted bad"],
    )
    st.dataframe(frame, use_container_width=True, height=145)


def show_metric_bars(metrics: dict[str, float]) -> None:
    labels = {
        "roc_auc": "ROC-AUC",
        "average_precision": "Avg precision",
        "accuracy": "Accuracy",
        "bad_precision": "Bad precision",
        "bad_recall": "Bad recall",
        "bad_f2": "Bad F2",
    }
    frame = pd.DataFrame(
        {"score": [metrics.get(name, 0.0) for name in labels]},
        index=list(labels.values()),
    )
    st.bar_chart(frame, y="score", height=330)


def show_fairness_attribute(attribute: str, payload: dict) -> None:
    st.markdown(f"<div class='visual-title'>{attribute} fairness monitor</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Parity ratio", f"{payload.get('demographic_parity_ratio', 0):.3f}")
    c2.metric("Parity difference", f"{payload.get('demographic_parity_difference', 0):.3f}")
    c3.metric("Equalized odds", f"{payload.get('equalized_odds_difference', 0):.3f}")

    groups = pd.DataFrame(payload.get("groups", []))
    if groups.empty:
        st.info("No group-level fairness rows available.")
        return
    chart_frame = groups.set_index("group")[["approval_rate", "true_positive_rate", "false_positive_rate"]]
    st.bar_chart(chart_frame, height=320)
    st.dataframe(groups, use_container_width=True, height=210)


def build_sidebar_filters(raw: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## Showcase")
    st.sidebar.caption("German Credit Risk Platform")
    st.sidebar.divider()
    st.sidebar.markdown("### Filters")

    sex = st.sidebar.selectbox("Sex", all_options(raw["Sex"].dropna().unique().tolist()))
    housing = st.sidebar.selectbox("Housing", all_options(raw["Housing"].dropna().unique().tolist()))
    purpose = st.sidebar.selectbox("Purpose", all_options(raw["Purpose"].dropna().unique().tolist()))
    risk = st.sidebar.selectbox("Risk", all_options(raw["Risk"].dropna().map(risk_label).unique().tolist()))

    credit_range = st.sidebar.slider(
        "Credit amount",
        int(raw["Credit amount"].min()),
        int(raw["Credit amount"].max()),
        (int(raw["Credit amount"].min()), int(raw["Credit amount"].max())),
    )
    duration_range = st.sidebar.slider(
        "Duration months",
        int(raw["Duration"].min()),
        int(raw["Duration"].max()),
        (int(raw["Duration"].min()), int(raw["Duration"].max())),
    )

    filtered = raw.copy()
    filtered = filter_equals(filtered, "Sex", sex)
    filtered = filter_equals(filtered, "Housing", housing)
    filtered = filter_equals(filtered, "Purpose", purpose)
    if risk != "All":
        filtered = filtered[filtered["Risk"].map(risk_label) == risk]
    return filtered[
        filtered["Credit amount"].between(*credit_range)
        & filtered["Duration"].between(*duration_range)
    ]


def applicant_form(defaults: pd.DataFrame) -> pd.DataFrame | None:
    model, threshold = load_runtime_model()
    if model is None or threshold is None:
        st.warning("Model artifacts are missing. Run `python -m credit_risk.train` first.")
        return None

    with st.form("applicant_scoring_form"):
        cols = st.columns(3)
        with cols[0]:
            age = st.slider("Age", 18, 100, 35)
            sex = st.selectbox("Applicant sex", sorted(defaults["Sex"].dropna().unique().tolist()))
            job = st.selectbox("Job", sorted(defaults["Job"].dropna().unique().tolist()), index=2)
        with cols[1]:
            housing = st.selectbox("Applicant housing", sorted(defaults["Housing"].dropna().unique().tolist()))
            saving = st.selectbox("Saving accounts", sorted(defaults["Saving accounts"].fillna("No Account").unique().tolist()))
            checking = st.selectbox("Checking account", sorted(defaults["Checking account"].fillna("No Account").unique().tolist()))
        with cols[2]:
            credit_amount = st.number_input("Credit amount", min_value=1, max_value=50000, value=5000, step=250)
            duration = st.slider("Duration", 1, 84, 24)
            purpose = st.selectbox("Applicant purpose", sorted(defaults["Purpose"].dropna().unique().tolist()))

        submitted = st.form_submit_button("Score applicant")

    if not submitted:
        return None

    applicant = pd.DataFrame(
        [
            {
                "Age": age,
                "Sex": sex,
                "Job": int(job),
                "Housing": housing,
                "Saving accounts": None if saving == "No Account" else saving,
                "Checking account": None if checking == "No Account" else checking,
                "Credit amount": int(credit_amount),
                "Duration": int(duration),
                "Purpose": purpose,
            }
        ]
    )
    scored = score_single_application(applicant, model=model, threshold=threshold)
    return pd.DataFrame([scored])


def main() -> None:
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    raw = load_dataset()
    metrics = load_json_artifact(str(METRICS_PATH))
    fairness = load_json_artifact(str(ARTIFACTS_DIR / "fairness_report.json"))
    threshold = load_json_artifact(str(ARTIFACTS_DIR / "decision_threshold.json"))
    filtered = build_sidebar_filters(raw)
    test_metrics = metrics.get("test_metrics", {})

    st.markdown(
        f"""
        <div class="report-hero">
            <h1>German Credit Risk Command Center</h1>
            <p>Interactive portfolio intelligence, XGBoost scoring, SHAP drivers, and fairness controls for retail credit decisions.</p>
            <span class="report-status">Model: {metrics.get('selected_model', 'not trained')} | Threshold: {threshold.get('threshold', 0):.3f} | Filtered records: {len(filtered):,}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        render_kpi("Applicants", f"{len(filtered):,}", f"of {len(raw):,} records", "success")
    with k2:
        bad_rate = (filtered["Risk"].str.lower() == "bad").mean() if len(filtered) else 0
        render_kpi("Bad-risk rate", f"{bad_rate:.1%}", "filtered portfolio", "risk")
    with k3:
        render_kpi("Bad recall", f"{test_metrics.get('bad_recall', 0):.3f}", "test set focus metric", "warning")
    with k4:
        render_kpi("ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}", "test set discrimination")
    with k5:
        render_kpi("F2 score", f"{test_metrics.get('bad_f2', 0):.3f}", "recall-weighted score")

    overview, performance, fairness_tab, scoring, data, deployment = st.tabs(
        ["Executive View", "Model Performance", "Fairness Audit", "Applicant Scoring", "Data Explorer", "Deployment"]
    )

    with overview:
        left, middle = st.columns([0.95, 1.05])
        with left:
            with st.container(border=True):
                st.markdown("<div class='visual-title'>Risk distribution</div>", unsafe_allow_html=True)
                risk_counts = filtered["Risk"].map(risk_label).value_counts().to_frame("count")
                st.bar_chart(risk_counts, y="count", height=350)
        with middle:
            with st.container(border=True):
                st.markdown("<div class='visual-title'>Average credit by purpose</div>", unsafe_allow_html=True)
                purpose_frame = filtered.groupby("Purpose", dropna=False)["Credit amount"].mean().sort_values().to_frame("avg_credit_amount")
                st.bar_chart(purpose_frame, y="avg_credit_amount", height=350)
        with st.container(border=True):
            st.markdown("<div class='visual-title'>Portfolio snapshot</div>", unsafe_allow_html=True)
            snapshot = filtered[["Age", "Sex", "Housing", "Credit amount", "Duration", "Purpose", "Risk"]].head(15)
            st.dataframe(snapshot, use_container_width=True, height=360)

    with performance:
        left, right = st.columns([1.2, 0.8])
        with left:
            with st.container(border=True):
                st.markdown("<div class='visual-title'>Model scorecard</div>", unsafe_allow_html=True)
                show_metric_bars(test_metrics)
        with right:
            with st.container(border=True):
                st.markdown("<div class='visual-title'>Confusion matrix</div>", unsafe_allow_html=True)
                show_confusion_matrix(test_metrics.get("confusion_matrix", []))
                st.markdown("<span class='visual-note'>Rows are actual outcomes. Columns are predictions.</span>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<div class='visual-title'>Validation metrics payload</div>", unsafe_allow_html=True)
            st.json(metrics.get("validation_metrics", {}))

    with fairness_tab:
        st.markdown("<span class='badge'>Protected attributes: Age and Sex</span>", unsafe_allow_html=True)
        st.write("Fairness visuals compare approval rates, true positive rates, false positive rates, demographic parity, and equalized odds.")
        for attribute, payload in fairness.items():
            with st.container(border=True):
                show_fairness_attribute(attribute, payload)

    with scoring:
        left, right = st.columns([0.9, 1.1])
        with left:
            with st.container(border=True):
                st.markdown("<div class='visual-title'>Applicant scoring form</div>", unsafe_allow_html=True)
                scored_frame = applicant_form(raw)
        with right:
            with st.container(border=True):
                st.markdown("<div class='visual-title'>Decision and explainability output</div>", unsafe_allow_html=True)
                if scored_frame is None:
                    st.info("Score an applicant to see probability, recommendation, and SHAP drivers.")
                else:
                    result = scored_frame.iloc[0].to_dict()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Bad-loan probability", f"{result['bad_loan_probability']:.1%}")
                    c2.metric("Predicted bad", int(result["predicted_bad"]))
                    c3.metric("Action", result["recommended_action"])
                    drivers = pd.DataFrame(result.get("explanation", {}).get("top_features", []))
                    st.markdown("<div class='visual-title'>Top SHAP drivers</div>", unsafe_allow_html=True)
                    if not drivers.empty:
                        st.bar_chart(drivers.set_index("feature")[["shap_value"]], y="shap_value", height=310)
                        st.dataframe(drivers, use_container_width=True, height=230)
                    with st.expander("Raw scoring response"):
                        st.json(result)

    with data:
        with st.container(border=True):
            st.markdown("<div class='visual-title'>Filtered applicant table</div>", unsafe_allow_html=True)
            st.dataframe(filtered, use_container_width=True, height=520)

    with deployment:
        left, right = st.columns(2)
        with left:
            with st.container(border=True):
                st.markdown("<div class='visual-title'>Local services</div>", unsafe_allow_html=True)
                st.code("uvicorn api.main:app --host 127.0.0.1 --port 8000", language="powershell")
                st.code("streamlit run app/dashboard.py --server.port 8501", language="powershell")
        with right:
            with st.container(border=True):
                st.markdown("<div class='visual-title'>GCP deployment path</div>", unsafe_allow_html=True)
                st.write("Use `deployment/gcp/README.md` for Artifact Registry and Cloud Run deployment.")
                st.write(f"Project root: `{PROJECT_ROOT}`")


if __name__ == "__main__":
    main()

