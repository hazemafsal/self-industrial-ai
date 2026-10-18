
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from rag_engine import create_rag_components, answer_with_rag

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "dataset.csv"
MODEL_DIR = BASE_DIR / "models"

NUMERIC_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

FEATURES = NUMERIC_FEATURES + ["Type"]

st.set_page_config(
    page_title="Self-Aware Industrial AI",
    page_icon="🏭",
    layout="wide",
)


@st.cache_resource
def load_models():
    return (
        joblib.load(MODEL_DIR / "failure_model.joblib"),
        joblib.load(MODEL_DIR / "novelty_detector.joblib"),
    )


@st.cache_resource
def load_rag():
    return create_rag_components()


@st.cache_data
def load_metrics():
    import json

    path = MODEL_DIR / "metrics.json"
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset():
    if not DATA_FILE.exists():
        return None
    return pd.read_csv(DATA_FILE)


st.title("🏭 Self-Aware Industrial AI")
st.subheader("Failure prediction • Anomaly detection • RAG • Gemini LLM")

st.info(
    "Research prototype. Predictions and generated recommendations are "
    "decision-support information, not proof that equipment is safe."
)

# -----------------------------
# Load machine-learning models
# -----------------------------
try:
    failure_model, novelty_detector = load_models()
except Exception as error:
    st.error(
        f"Cannot load the trained models: {error}. "
        "Run python download_data.py and python train_model.py first."
    )
    st.stop()


# -----------------------------
# Sensor inputs
# -----------------------------
st.sidebar.header("Machine sensor readings")

air_temp = st.sidebar.number_input(
    "Air temperature",
    min_value=200.0,
    max_value=400.0,
    value=300.0,
    step=0.1,
)

process_temp = st.sidebar.number_input(
    "Process temperature",
    min_value=200.0,
    max_value=450.0,
    value=310.0,
    step=0.1,
)

rpm = st.sidebar.number_input(
    "Rotational speed",
    min_value=0.0,
    max_value=10000.0,
    value=1500.0,
    step=10.0,
)

torque = st.sidebar.number_input(
    "Torque",
    min_value=0.0,
    max_value=500.0,
    value=40.0,
    step=0.5,
)

tool_wear = st.sidebar.number_input(
    "Tool wear",
    min_value=0.0,
    max_value=1000.0,
    value=100.0,
    step=1.0,
)

product_type = st.sidebar.selectbox(
    "Product type",
    ["L", "M", "H"],
)

risk_threshold = st.sidebar.slider(
    "Failure-risk alert threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

sensor_data = pd.DataFrame(
    [{
        "Air temperature": air_temp,
        "Process temperature": process_temp,
        "Rotational speed": rpm,
        "Torque": torque,
        "Tool wear": tool_wear,
        "Type": product_type,
    }],
    columns=FEATURES,
)

st.header("1. Machine assessment")
st.dataframe(sensor_data, use_container_width=True, hide_index=True)

# -----------------------------
# Predict failure, detect novelty,
# and evaluate sensor quality
# -----------------------------
probabilities = failure_model.predict_proba(sensor_data)[0]

classes = list(
    failure_model.named_steps["classifier"].classes_
)

if 1 not in classes:
    st.error("The trained model does not contain failure class 1.")
    st.stop()

failure_risk = float(
    probabilities[classes.index(1)]
)


# Transform inputs using the same preprocessing as training
preprocessor = failure_model.named_steps["preprocessor"]
transformed = preprocessor.transform(sensor_data)

# Novelty detection
novelty_prediction = int(
    novelty_detector.predict(transformed)[0]
)

novelty_score = float(
    novelty_detector.decision_function(transformed)[0]
)

is_unusual = novelty_prediction == -1

# Quality checks
warnings = []

if process_temp < air_temp:
    warnings.append(
        "Process temperature is below air temperature. "
        "Verify sensor readings and operating conditions."
    )

if rpm <= 0:
    warnings.append(
        "Rotational speed is zero. Confirm whether the machine is stopped."
    )

if torque > 300:
    warnings.append(
        "Torque exceeds the prototype warning threshold of 300 Nm. "
        "Verify the manufacturer's operating limits."
    )

if tool_wear > 250:
    warnings.append(
        "Tool wear exceeds the prototype warning threshold of 250 minutes. "
        "Verify the maintenance specification."
    )

# Additional configurable sensor-pattern rules
if rpm > 0 and torque > 0:
    if rpm < 500 and torque > 70:
        warnings.append(
            "Low rotational speed combined with high torque. "
            "Verify whether this combination is expected."
        )

# Display metrics
col1, col2, col3 = st.columns(3)

col1.metric(
    "Predicted failure-risk score",
    f"{failure_risk:.1%}"
)

col2.metric(
    "Unusual sensor pattern",
    "Flagged" if is_unusual else "Not flagged"
)

col3.metric(
    "Quality warnings",
    len(warnings)
)

# Failure-risk alert
if failure_risk >= risk_threshold:
    st.error(
        "Failure-risk alert: the model score meets or exceeds "
        "your selected threshold."
    )
else:
    st.info(
        "The model score is below your selected threshold. "
        "This does not guarantee that the machine is safe."
    )

# Explain anomaly result
with st.expander("Understand the novelty detector"):
    st.write(f"Novelty prediction: {novelty_prediction}")
    st.write(f"Decision score: {novelty_score:.6f}")

    if is_unusual:
        st.warning(
            "The novelty detector classified this sensor combination "
            "as unusual compared with its learned reference data."
        )
    else:
        st.write(
            "The novelty detector did not flag this input. "
            "This does not prove that the input is normal or safe."
        )

# Explain quality warnings
if warnings:
    st.subheader("Sensor and operating warnings")

    for warning in warnings:
        st.warning(warning)
else:
    st.success(
        "No configured quality rules were triggered. "
        "This is not a guarantee of safe operation."
    )

if warnings or is_unusual:
    st.error(
        "Reliability warning: verify sensor readings and operating "
        "conditions before relying on the prediction."
    )

machine_context = f"""
Sensor readings: {sensor_data.to_dict(orient="records")[0]}
Failure-risk score: {failure_risk:.4f}
User-selected risk threshold: {risk_threshold:.2f}
Unusual sensor pattern flagged: {is_unusual}
Novelty detector decision score: {novelty_score:.6f}
Quality warning count: {len(warnings)}
Quality warnings: {warnings}
"""



# -----------------------------
# RAG + Gemini LLM assistant
# -----------------------------
st.divider()
st.header("2. RAG + Gemini maintenance assistant")

st.write(
    "Ask about the current prediction or maintenance guidance. "
    "The assistant retrieves relevant passages from your local knowledge "
    "base and uses them to ground its response."
)

question = st.text_area(
    "Ask a question",
    placeholder=(
        "Why might this prediction be unreliable? "
        "What should an engineer check?"
    ),
    height=100,
)

if st.button("Ask Gemini", type="primary"):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Retrieving knowledge and generating an answer..."):
                vectorstore, llm = load_rag()

                result = answer_with_rag(
                    question=question.strip(),
                    machine_context=machine_context,
                    vectorstore=vectorstore,
                    llm=llm,
                )

            st.subheader("LLM response")
            st.markdown(result["answer"])

            st.subheader("Retrieved RAG sources")
            if result["documents"]:
                for index, document in enumerate(
                    result["documents"], start=1
                ):
                    source = document.metadata.get("source", "Unknown")
                    with st.expander(f"Source {index}: {source}"):
                        st.write(document.page_content)
            else:
                st.warning("No relevant knowledge passages were retrieved.")

        except Exception as error:
            st.error(
                f"RAG/LLM could not run: {error}. Check your API key, "
                "network connection, model availability, and ChromaDB setup."
            )

# -----------------------------
# Model evaluation
# -----------------------------
st.divider()
st.header("3. Model evaluation")

metrics = load_metrics()

if metrics:
    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Balanced accuracy", f"{metrics['balanced_accuracy']:.3f}")

    roc_auc = metrics.get("roc_auc")
    col3.metric(
        "ROC AUC",
        f"{roc_auc:.3f}" if roc_auc is not None else "N/A",
    )

    st.write("Confusion matrix:")
    st.dataframe(
        pd.DataFrame(
            metrics["confusion_matrix"],
            index=["Actual: no failure", "Actual: failure"],
            columns=["Predicted: no failure", "Predicted: failure"],
        ),
        use_container_width=True,
    )

    with st.expander("Full classification report"):
        st.json(metrics["classification_report"])
else:
    st.warning("Metrics not found. Run python train_model.py first.")

# -----------------------------
# Raw dataset preview
# -----------------------------
st.divider()
st.header("4. Original dataset")

raw_df = load_dataset()

if raw_df is not None:
    st.write(f"{len(raw_df):,} rows • {len(raw_df.columns)} columns")
    st.dataframe(raw_df.head(20), use_container_width=True)

    st.download_button(
        "Download original CSV",
        data=raw_df.to_csv(index=False).encode("utf-8"),
        file_name="ai4i2020_raw.csv",
        mime="text/csv",
    )
else:
    st.warning("Dataset not found. Run python download_data.py.")

st.caption(
    "The failure-risk score may not be calibrated. An anomaly score is "
    "not the probability that a prediction is wrong. Validate this "
    "prototype using real equipment data before operational use."
)
