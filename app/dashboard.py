from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from credit_risk.config import ARTIFACTS_DIR, METRICS_PATH, PROJECT_ROOT
from credit_risk.data import load_raw_data, prepare_modeling_table
from credit_risk.predict import load_model, load_threshold, model_artifacts_ready, score_single_application

st.set_page_config(
    page_title="German Credit Risk Dashboard",
    page_icon="",
    layout="wide",
)


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


def metric_card(label: str, value: float | int | str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def show_confusion_matrix(matrix: list[list[int]]) -> None:
    if not matrix:
        st.info("Confusion matrix artifact is not available yet.")
        return
    frame = pd.DataFrame(
        matrix,
        index=["Actual good", "Actual bad"],
        columns=["Predicted good", "Predicted bad"],
    )
    st.dataframe(frame, use_container_width=True)


def show_metric_bars(metrics: dict[str, float]) -> None:
    names = ["roc_auc", "average_precision", "accuracy", "bad_precision", "bad_recall", "bad_f2"]
    values = {name: metrics.get(name, 0.0) for name in names}
    frame = pd.DataFrame.from_dict(values, orient="index", columns=["score"])
    st.bar_chart(frame, y="score", height=320)


def show_fairness_attribute(attribute: str, payload: dict) -> None:
    st.subheader(attribute)
    c1, c2, c3 = st.columns(3)
    c1.metric("Demographic parity ratio", f"{payload.get('demographic_parity_ratio', 0):.3f}")
    c2.metric("Demographic parity difference", f"{payload.get('demographic_parity_difference', 0):.3f}")
    c3.metric("Equalized odds difference", f"{payload.get('equalized_odds_difference', 0):.3f}")

    groups = pd.DataFrame(payload.get("groups", []))
    if groups.empty:
        st.info("No group-level fairness rows available.")
        return
    st.dataframe(groups, use_container_width=True)
    chart_frame = groups.set_index("group")[["approval_rate", "true_positive_rate", "false_positive_rate"]]
    st.bar_chart(chart_frame, height=320)


def applicant_form(defaults: pd.DataFrame) -> pd.DataFrame | None:
    model, threshold = load_runtime_model()
    if model is None or threshold is None:
        st.warning("Model artifacts are missing. Run `python -m credit_risk.train` first.")
        return None

    with st.form("applicant_scoring_form"):
        cols = st.columns(3)
        with cols[0]:
            age = st.slider("Age", 18, 100, 35)
            sex = st.selectbox("Sex", sorted(defaults["Sex"].dropna().unique().tolist()))
            job = st.selectbox("Job", sorted(defaults["Job"].dropna().unique().tolist()), index=2)
        with cols[1]:
            housing = st.selectbox("Housing", sorted(defaults["Housing"].dropna().unique().tolist()))
            saving = st.selectbox("Saving accounts", sorted(defaults["Saving accounts"].fillna("No Account").unique().tolist()))
            checking = st.selectbox("Checking account", sorted(defaults["Checking account"].fillna("No Account").unique().tolist()))
        with cols[2]:
            credit_amount = st.number_input("Credit amount", min_value=1, max_value=50000, value=5000, step=250)
            duration = st.slider("Duration", 1, 84, 24)
            purpose = st.selectbox("Purpose", sorted(defaults["Purpose"].dropna().unique().tolist()))

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
    st.title("German Credit Risk Dashboard")
    st.caption("Interactive model monitoring, fairness audit, data exploration, and XGBoost SHAP scoring workspace.")

    raw = load_dataset()
    metrics = load_json_artifact(str(METRICS_PATH))
    fairness = load_json_artifact(str(ARTIFACTS_DIR / "fairness_report.json"))
    threshold = load_json_artifact(str(ARTIFACTS_DIR / "decision_threshold.json"))

    overview, data, performance, fairness_tab, scoring, deployment = st.tabs(
        ["Overview", "Data Explorer", "Model Performance", "Fairness Audit", "Applicant Scoring", "Deployment"]
    )

    with overview:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Applicants", f"{len(raw):,}")
        c2.metric("Bad-risk rate", f"{(raw['Risk'].str.lower() == 'bad').mean():.1%}")
        c3.metric("Selected model", metrics.get("selected_model", "not trained"))
        c4.metric("Decision threshold", f"{threshold.get('threshold', 0):.3f}")

        st.subheader("Risk distribution")
        risk_counts = raw["Risk"].map(risk_label).value_counts().to_frame("count")
        st.bar_chart(risk_counts, y="count", height=300)

        st.subheader("Credit portfolio snapshot")
        st.dataframe(
            raw[["Age", "Sex", "Job", "Housing", "Credit amount", "Duration", "Purpose", "Risk"]].head(20),
            use_container_width=True,
        )

    with data:
        st.subheader("Filter applicant population")
        left, right = st.columns(2)
        with left:
            selected_sex = st.multiselect("Sex", sorted(raw["Sex"].dropna().unique()), default=sorted(raw["Sex"].dropna().unique()))
            selected_housing = st.multiselect("Housing", sorted(raw["Housing"].dropna().unique()), default=sorted(raw["Housing"].dropna().unique()))
        with right:
            credit_range = st.slider(
                "Credit amount range",
                int(raw["Credit amount"].min()),
                int(raw["Credit amount"].max()),
                (int(raw["Credit amount"].min()), int(raw["Credit amount"].max())),
            )
            duration_range = st.slider(
                "Duration range",
                int(raw["Duration"].min()),
                int(raw["Duration"].max()),
                (int(raw["Duration"].min()), int(raw["Duration"].max())),
            )
        filtered = raw[
            raw["Sex"].isin(selected_sex)
            & raw["Housing"].isin(selected_housing)
            & raw["Credit amount"].between(*credit_range)
            & raw["Duration"].between(*duration_range)
        ]
        st.metric("Filtered applicants", f"{len(filtered):,}")
        st.dataframe(filtered, use_container_width=True)
        st.subheader("Average credit amount by purpose")
        purpose_frame = filtered.groupby("Purpose", dropna=False)["Credit amount"].mean().sort_values().to_frame("avg_credit_amount")
        st.bar_chart(purpose_frame, y="avg_credit_amount", height=360)

    with performance:
        test_metrics = metrics.get("test_metrics", {})
        validation_metrics = metrics.get("validation_metrics", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test ROC-AUC", f"{test_metrics.get('roc_auc', 0):.3f}")
        c2.metric("Bad recall", f"{test_metrics.get('bad_recall', 0):.3f}")
        c3.metric("Bad F2", f"{test_metrics.get('bad_f2', 0):.3f}")
        c4.metric("Accuracy", f"{test_metrics.get('accuracy', 0):.3f}")

        left, right = st.columns(2)
        with left:
            st.subheader("Test metrics")
            show_metric_bars(test_metrics)
        with right:
            st.subheader("Confusion matrix")
            show_confusion_matrix(test_metrics.get("confusion_matrix", []))

        with st.expander("Validation metrics"):
            st.json(validation_metrics)

    with fairness_tab:
        st.write("Protected attributes are audited with approval rates, true positive rates, false positive rates, demographic parity, and equalized odds.")
        for attribute, payload in fairness.items():
            show_fairness_attribute(attribute, payload)

    with scoring:
        st.subheader("Score a single applicant")
        scored_frame = applicant_form(raw)
        if scored_frame is not None:
            result = scored_frame.iloc[0].to_dict()
            c1, c2, c3 = st.columns(3)
            c1.metric("Bad-loan probability", f"{result['bad_loan_probability']:.1%}")
            c2.metric("Predicted bad", int(result["predicted_bad"]))
            c3.metric("Recommended action", result["recommended_action"])

            explanation = result.get("explanation", {})
            st.subheader("Top SHAP drivers")
            drivers = pd.DataFrame(explanation.get("top_features", []))
            if not drivers.empty:
                st.dataframe(drivers, use_container_width=True)
                chart = drivers.set_index("feature")[["shap_value"]]
                st.bar_chart(chart, y="shap_value", height=360)
            with st.expander("Raw scoring response"):
                st.json(result)

    with deployment:
        st.subheader("Local services")
        st.code("uvicorn api.main:app --host 127.0.0.1 --port 8000", language="powershell")
        st.code("streamlit run app/dashboard.py --server.port 8501", language="powershell")
        st.subheader("GCP path")
        st.write("Use the Dockerfile and deployment notes under `deployment/gcp/README.md` for Artifact Registry and Cloud Run deployment.")
        st.write(f"Project root: `{PROJECT_ROOT}`")


if __name__ == "__main__":
    main()
