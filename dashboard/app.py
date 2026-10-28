import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Healthcare BI Dashboard",
    page_icon="🏥",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"
RESULT_DIR = OUTPUT_DIR / "results"
RESULTS_DIR = BASE_DIR / "outputs" / "results"


# ============================================================
# TITLE
# ============================================================

st.title("🏥 Healthcare Business Intelligence Dashboard")

st.markdown(
    """
    ### Healthcare Analytics, Machine Learning & Explainable AI

    This dashboard presents **Exploratory Data Analysis (EDA)**,
    **Machine Learning performance**, **SHAP**, **LIME**, and
    comparative healthcare insights from three datasets.
    """
)

st.divider()


# ============================================================
# LOAD DATASETS
# ============================================================

@st.cache_data
def load_datasets():

    obesity = pd.read_csv(DATA_DIR / "obesity_dataset.csv")
    hospital = pd.read_csv(DATA_DIR / "Hospital_Patient.csv")
    healthcare = pd.read_csv(DATA_DIR / "healthcare_dataset.csv")

    return obesity, hospital, healthcare


obesity, hospital, healthcare = load_datasets()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Dashboard Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "🏠 Dashboard Overview",
        "📊 EDA & Dataset Insights",
        "🤖 Machine Learning",
        "🔍 SHAP Explainability",
        "🧠 LIME Explainability",
        "📈 Comparative Analysis",
        "📋 Data Explorer",
        "📝 Conclusions"
    ]
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Dashboard Overview":

    st.header("🏠 Dashboard Overview")

    st.subheader("Key Project Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Obesity Records",
            f"{len(obesity):,}"
        )

    with col2:
        st.metric(
            "Hospital Records",
            f"{len(hospital):,}"
        )

    with col3:
        st.metric(
            "Healthcare Records",
            f"{len(healthcare):,}"
        )

    with col4:

        total_records = len(obesity) + len(hospital) + len(healthcare)

        st.metric(
            "Total Records",
            f"{total_records:,}"
        )

    st.divider()

    st.subheader("📂 Datasets Used")

    dataset_info = pd.DataFrame({
        "Dataset": [
            "Obesity Dataset",
            "Hospital Patient Dataset",
            "Healthcare Dataset"
        ],

        "Records": [
            len(obesity),
            len(hospital),
            len(healthcare)
        ],

        "Columns": [
            len(obesity.columns),
            len(hospital.columns),
            len(healthcare.columns)
        ]
    })

    st.dataframe(
        dataset_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("📊 Dataset Size Comparison")

    fig = px.bar(
        dataset_info,
        x="Dataset",
        y="Records",
        title="Number of Records in Each Dataset",
        text="Records"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "The dashboard combines healthcare datasets with EDA, "
        "machine learning and explainable AI techniques."
    )


# ============================================================
# PAGE 2 — EDA
# ============================================================

elif page == "📊 EDA & Dataset Insights":

    st.header("📊 Exploratory Data Analysis")

    st.markdown(
        """
        Select a dataset below to explore its healthcare insights,
        distributions and relationships.
        """
    )

    dataset = st.selectbox(
        "Select Dataset",
        ["Obesity", "Hospital", "Healthcare"]
    )


    # ========================================================
    # OBESITY DATASET
    # ========================================================

    if dataset == "Obesity":

        st.subheader("🏃 Obesity Dataset")

        # ----------------------------------------------------
        # AGE + BMI
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            image_path = GRAPH_DIR / "obesity_age_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Age Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the age profile of participants in the
                    obesity dataset.
                    """
                )

            else:
                st.warning("Age distribution graph not found.")

        with col2:

            image_path = GRAPH_DIR / "obesity_bmi_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="BMI Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows how BMI values are distributed among
                    the participants.
                    """
                )

            else:
                st.warning("BMI distribution graph not found.")


        # ----------------------------------------------------
        # GENDER + OBESITY LEVEL
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            image_path = GRAPH_DIR / "obesity_gender_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Gender Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the composition of participants
                    by gender.
                    """
                )

            else:
                st.warning("Gender distribution graph not found.")

        with col2:

            image_path = GRAPH_DIR / "obesity_level_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Obesity Level Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the number of participants in each
                    obesity-level category.
                    """
                )

            else:
                st.warning("Obesity level graph not found.")


        # ----------------------------------------------------
        # BMI VS OBESITY + FAMILY HISTORY
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            image_path = GRAPH_DIR / "bmi_vs_obesity_level.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="BMI vs Obesity Level",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Helps examine how BMI varies across
                    different obesity-level categories.
                    """
                )

            else:
                st.warning("BMI vs obesity level graph not found.")

        with col2:

            image_path = GRAPH_DIR / "family_history_vs_obesity.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Family History vs Obesity",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the relationship between family history
                    of overweight and obesity levels.
                    """
                )

            else:
                st.warning("Family history graph not found.")


        # ----------------------------------------------------
        # PHYSICAL ACTIVITY VS OBESITY
        # ----------------------------------------------------

        image_path = GRAPH_DIR / "physical_activity_vs_obesity.png"

        if image_path.exists():

            st.image(
                str(image_path),
                caption="Physical Activity vs Obesity",
                use_container_width=True
            )

            st.markdown(
                """
                **Explanation:**  
                Helps examine how physical activity levels
                are associated with obesity categories.
                """
            )

        else:
            st.warning("Physical activity graph not found.")


        # ----------------------------------------------------
        # CORRELATION HEATMAP
        # ----------------------------------------------------

        st.subheader("🔗 Feature Correlation Analysis")

        image_path = GRAPH_DIR / "obesity_correlation_heatmap.png"

        if image_path.exists():

            st.image(
                str(image_path),
                caption="Obesity Dataset Correlation Heatmap",
                use_container_width=True
            )

            st.markdown(
                """
                **Explanation:**  
                Correlation analysis helps identify relationships
                between numerical variables in the dataset.
                """
            )

        else:
            st.warning("Obesity correlation heatmap not found.")


        st.info(
            """
            **Obesity EDA Summary**

            The obesity analysis provides insights into age,
            BMI, gender, obesity levels, family history,
            physical activity and relationships between
            numerical features.
            """
        )


    # ========================================================
    # HOSPITAL DATASET
    # ========================================================

    elif dataset == "Hospital":

        st.subheader("🏥 Hospital Patient Dataset")

        col1, col2 = st.columns(2)

        with col1:

            image_path = GRAPH_DIR / "hospital_condition_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Medical Condition Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the distribution of medical conditions
                    among hospital patients.
                    """
                )

            else:
                st.warning("Hospital condition graph not found.")

        with col2:

            image_path = GRAPH_DIR / "hospital_gender_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Gender Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the gender composition of
                    hospital patients.
                    """
                )

            else:
                st.warning("Hospital gender graph not found.")


        col1, col2 = st.columns(2)

        with col1:

            image_path = GRAPH_DIR / "hospital_medication_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Medication Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the distribution of medications
                    used among hospital patients.
                    """
                )

            else:
                st.warning("Hospital medication graph not found.")

        with col2:

            image_path = GRAPH_DIR / "hospital_outcome_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Patient Outcome Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the distribution of recorded
                    patient outcomes in the hospital dataset.
                    """
                )

            else:
                st.warning("Hospital outcome graph not found.")


        st.info(
            """
            **Hospital EDA Summary**

            The hospital analysis provides insights into
            patient medical conditions, gender, medication
            usage and recorded patient outcomes.
            """
        )


    # ========================================================
    # HEALTHCARE DATASET
    # ========================================================

    elif dataset == "Healthcare":

        st.subheader("🩺 Healthcare Dataset")

        col1, col2 = st.columns(2)

        with col1:

            image_path = GRAPH_DIR / "healthcare_age_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Age Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the age distribution of patients
                    in the healthcare dataset.
                    """
                )

            else:
                st.warning("Healthcare age graph not found.")

        with col2:

            image_path = GRAPH_DIR / "healthcare_gender_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Gender Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the gender composition of
                    the healthcare dataset.
                    """
                )

            else:
                st.warning("Healthcare gender graph not found.")


        col1, col2 = st.columns(2)

        with col1:

            image_path = GRAPH_DIR / "healthcare_medical_condition_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Medical Condition Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the distribution of medical conditions
                    among healthcare patients.
                    """
                )

            else:
                st.warning("Healthcare medical condition graph not found.")

        with col2:

            image_path = GRAPH_DIR / "healthcare_medication_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Medication Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the distribution of medications
                    recorded for patients.
                    """
                )

            else:
                st.warning("Healthcare medication graph not found.")


        col1, col2 = st.columns(2)

        with col1:

            image_path = GRAPH_DIR / "healthcare_length_of_stay.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Length of Stay Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows how long patients stayed
                    in the healthcare facility.
                    """
                )

            else:
                st.warning("Healthcare length-of-stay graph not found.")

        with col2:

            image_path = GRAPH_DIR / "test_results_distribution.png"

            if image_path.exists():

                st.image(
                    str(image_path),
                    caption="Test Results Distribution",
                    use_container_width=True
                )

                st.markdown(
                    """
                    **Explanation:**  
                    Shows the distribution of Normal,
                    Abnormal and Inconclusive test results.
                    """
                )

            else:
                st.warning("Healthcare test results graph not found.")


        st.info(
            """
            **Healthcare EDA Summary**

            The healthcare analysis provides insights into
            patient age, gender, medical conditions,
            medications, length of stay and test results.
            """
        )

# ============================================================
#
# ============================================================
# PAGE 3 - MACHINE LEARNING
# ============================================================

elif page == "🤖 Machine Learning":

    st.header("🤖 Machine Learning Model Performance")

    st.markdown(
        """
        This section compares baseline and hybrid machine learning
        models across the Obesity, Hospital, and Healthcare datasets.

        Performance is assessed using Accuracy, Precision, Recall,
        F1-score, and ROC-AUC where available.
        """
    )

    # --------------------------------------------------------
    # LOAD MODEL RESULT FILES
    # --------------------------------------------------------

    result_files = [
        RESULT_DIR / "obesity_final_12_model_comparison.csv",
        RESULT_DIR / "hospital_all_models_comparison.csv",
        RESULT_DIR / "healthcare_all_models_comparison.csv",
    ]

    dataframes = []

    for file in result_files:
        if file.exists():
            try:
                df = pd.read_csv(file)
                df.columns = [str(col).strip() for col in df.columns]

                rename_map = {
                    "dataset": "Dataset",
                    "model": "Model",
                    "accuracy": "Accuracy",
                    "precision": "Precision",
                    "recall": "Recall",
                    "f1": "F1_Score",
                    "f1_score": "F1_Score",
                    "f1-score": "F1_Score",
                    "roc_auc": "ROC_AUC",
                    "roc-auc": "ROC_AUC",
                }

                df = df.rename(columns=rename_map)

                if "Dataset" not in df.columns:
                    name = file.name.lower()

                    if "obesity" in name:
                        df["Dataset"] = "Obesity"
                    elif "hospital" in name:
                        df["Dataset"] = "Hospital"
                    elif "healthcare" in name:
                        df["Dataset"] = "Healthcare"

                if "Dataset" in df.columns:
                    df["Dataset"] = (
                        df["Dataset"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .map({
                            "obesity": "Obesity",
                            "hospital": "Hospital",
                            "healthcare": "Healthcare",
                        })
                        .fillna(df["Dataset"].astype(str).str.strip())
                    )

                    dataframes.append(df)

                else:
                    st.warning(f"Dataset column missing in {file.name}")

            except Exception as e:
                st.warning(f"Could not load {file.name}: {e}")

        else:
            st.warning(f"Model result file not found: {file.name}")

    # --------------------------------------------------------
    # MODEL PERFORMANCE TABLE
    # --------------------------------------------------------

    if dataframes:

        model_results = pd.concat(
            dataframes,
            ignore_index=True,
            sort=False,
        )

        for metric in [
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score",
            "ROC_AUC",
        ]:
            if metric in model_results.columns:
                model_results[metric] = pd.to_numeric(
                    model_results[metric],
                    errors="coerce",
                )

        st.divider()
        st.subheader("📋 Model Performance Table")

        display_columns = [
            col for col in [
                "Dataset",
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "ROC_AUC",
            ]
            if col in model_results.columns
        ]

        st.dataframe(
            model_results[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # ACCURACY COMPARISON ACROSS DATASETS
        # ----------------------------------------------------

        if all(
            col in model_results.columns
            for col in ["Dataset", "Model", "Accuracy"]
        ):

            st.divider()
            st.subheader("📊 Accuracy Comparison Across Datasets")

            dataset_options = ["All Datasets"] + [
                name for name in ["Obesity", "Hospital", "Healthcare"]
                if name in model_results["Dataset"].unique()
            ]

            selected_dataset = st.selectbox(
                "Select Dataset for Accuracy Comparison",
                dataset_options,
                key="ml_accuracy_dataset",
            )

            if selected_dataset == "All Datasets":
                chart_data = model_results.copy()
            else:
                chart_data = model_results[
                    model_results["Dataset"] == selected_dataset
                ].copy()

            chart_data = chart_data.dropna(
                subset=["Accuracy", "Model", "Dataset"]
            )

            if not chart_data.empty:

                fig = px.bar(
                    chart_data,
                    x="Dataset",
                    y="Accuracy",
                    color="Model",
                    barmode="group",
                    title="Model Accuracy Across Datasets",
                    category_orders={
                        "Dataset": ["Obesity", "Hospital", "Healthcare"]
                    },
                    hover_data=[
                        col for col in [
                            "Precision",
                            "Recall",
                            "F1_Score",
                            "ROC_AUC",
                        ]
                        if col in chart_data.columns
                    ],
                )

                fig.update_yaxes(
                    title="Accuracy",
                    range=[0, 1],
                    tickformat=".0%",
                )

                fig.update_layout(
                    xaxis_title="Dataset",
                    legend_title="Model",
                    height=600,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            else:
                st.info("No valid accuracy results are available.")

        # ----------------------------------------------------
        # METRIC COMPARISON
        # ----------------------------------------------------

        metric_options = [
            metric for metric in [
                "Accuracy",
                "Precision",
                "Recall",
                "F1_Score",
                "ROC_AUC",
            ]
            if metric in model_results.columns
            and model_results[metric].notna().any()
        ]

        if metric_options:

            st.divider()
            st.subheader("📈 Compare Models by Performance Metric")

            metric_choice = st.selectbox(
                "Select Performance Metric",
                metric_options,
                index=(
                    metric_options.index("F1_Score")
                    if "F1_Score" in metric_options
                    else 0
                ),
                key="ml_metric_choice",
            )

            metric_dataset = st.selectbox(
                "Select Dataset for Metric Comparison",
                ["All Datasets", "Obesity", "Hospital", "Healthcare"],
                key="ml_metric_dataset",
            )

            metric_data = model_results.copy()

            if metric_dataset != "All Datasets":
                metric_data = metric_data[
                    metric_data["Dataset"] == metric_dataset
                ]

            metric_data = metric_data.dropna(
                subset=[metric_choice, "Model", "Dataset"]
            )

            if not metric_data.empty:

                metric_fig = px.bar(
                    metric_data,
                    x="Dataset",
                    y=metric_choice,
                    color="Model",
                    barmode="group",
                    title=f"{metric_choice} Comparison",
                    category_orders={
                        "Dataset": ["Obesity", "Hospital", "Healthcare"]
                    },
                )

                metric_fig.update_yaxes(
                    title=metric_choice,
                    range=[0, 1],
                    tickformat=".0%",
                )

                metric_fig.update_layout(height=550)

                st.plotly_chart(
                    metric_fig,
                    use_container_width=True,
                )

            else:
                st.info(
                    f"No valid {metric_choice} values are available "
                    "for the selected dataset."
                )

    else:
        st.warning(
            "No model comparison CSV files were loaded. "
            "Check the files inside your outputs/results folder."
        )

    # --------------------------------------------------------
    # CONFUSION MATRICES
    # --------------------------------------------------------

    st.divider()
    st.subheader("🔲 Confusion Matrices")

    dataset_choice = st.selectbox(
        "Select Dataset for Confusion Matrices",
        ["Obesity", "Hospital", "Healthcare"],
        key="cm_dataset",
    )

    confusion_matrix_files = {
        "Obesity": [
            ("obesity_decision_tree_confusion_matrix.png",
             "Obesity - Decision Tree"),
            ("obesity_random_forest_confusion_matrix.png",
             "Obesity - Random Forest"),
        ],
        "Hospital": [
            ("hospital_decision_tree_confusion_matrix.png",
             "Hospital - Decision Tree"),
            ("hospital_random_forest_confusion_matrix.png",
             "Hospital - Random Forest"),
        ],
        "Healthcare": [
            ("healthcare_decision_tree_confusion_matrix.png",
             "Healthcare - Decision Tree"),
            ("healthcare_random_forest_confusion_matrix.png",
             "Healthcare - Random Forest"),
        ],
    }

    cm_items = confusion_matrix_files[dataset_choice]
    col1, col2 = st.columns(2)

    for index, (filename, caption) in enumerate(cm_items):

        path = GRAPH_DIR / filename
        target_column = col1 if index == 0 else col2

        with target_column:
            if path.exists():
                st.image(
                    str(path),
                    caption=caption,
                    use_container_width=True,
                )
            else:
                st.info(f"Confusion matrix not found: {filename}")

    # --------------------------------------------------------
    # ROC CURVES
    # --------------------------------------------------------

    st.divider()
    st.subheader("📈 ROC Curves")

    roc_choice = st.selectbox(
        "Select Dataset for ROC Curve",
        ["Obesity", "Hospital", "Healthcare"],
        key="roc_dataset",
    )

    roc_files = {
        "Obesity": "obesity_random_forest_roc_curve.png",
        "Hospital": "hospital_random_forest_roc_curve.png",
        "Healthcare": "healthcare_random_forest_roc_curve.png",
    }

    roc_path = GRAPH_DIR / roc_files[roc_choice]

    if roc_path.exists():
        st.image(
            str(roc_path),
            caption=f"{roc_choice} - Random Forest ROC Curve",
            use_container_width=True,
        )
    else:
        st.warning(
            f"ROC curve graph not found: {roc_files[roc_choice]}"
        )

    # --------------------------------------------------------
    # OBESITY MODEL COMPARISON
    # --------------------------------------------------------

    st.divider()
    st.subheader("📊 Obesity - Baseline and Hybrid Comparison")

    obesity_graph = (
        GRAPH_DIR / "obesity_final_12_model_comparison.png"
    )

    if obesity_graph.exists():
        st.image(
            str(obesity_graph),
            caption=(
                "Obesity: six baseline models and available "
                "hybrid model candidates"
            ),
            use_container_width=True,
        )
    else:
        st.info(
            "Obesity graph not found: "
            "obesity_final_12_model_comparison.png"
        )

    # --------------------------------------------------------
    # HOSPITAL MODEL COMPARISON
    # --------------------------------------------------------

    st.divider()
    st.subheader("🏥 Hospital - Baseline and Hybrid Comparison")

    hospital_graph = (
        GRAPH_DIR / "hospital_all_models_comparison.png"
    )

    if hospital_graph.exists():
        st.image(
            str(hospital_graph),
            caption="Hospital baseline and hybrid model comparison",
            use_container_width=True,
        )
    else:
        st.info(
            "Hospital graph not found: "
            "hospital_all_models_comparison.png"
        )

    # --------------------------------------------------------
    # HEALTHCARE MODEL COMPARISON
    # --------------------------------------------------------

    st.divider()
    st.subheader("🩺 Healthcare - Baseline and Hybrid Comparison")

    healthcare_graph = (
        GRAPH_DIR / "healthcare_all_models_comparison.png"
    )

    if healthcare_graph.exists():
        st.image(
            str(healthcare_graph),
            caption=(
                "Healthcare baseline and hybrid model comparison. "
                "Compare the hybrid results with the strongest baseline."
            ),
            use_container_width=True,
        )
    else:
        st.info(
            "Healthcare graph not found: "
            "healthcare_all_models_comparison.png"
        )

    # --------------------------------------------------------
    # HEALTHCARE COMPARISON DETAILS
    # --------------------------------------------------------

    healthcare_comparison_file = (
        RESULT_DIR / "healthcare_all_models_comparison.csv"
    )

    if healthcare_comparison_file.exists():

        healthcare_comparison = pd.read_csv(
            healthcare_comparison_file
        )

        healthcare_comparison.columns = [
            str(col).strip()
            for col in healthcare_comparison.columns
        ]

        healthcare_comparison = healthcare_comparison.rename(
            columns={
                "accuracy": "Accuracy",
                "model": "Model",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1_Score",
                "f1_score": "F1_Score",
            }
        )

        st.subheader("📋 Healthcare Model Comparison Details")

        st.dataframe(
            healthcare_comparison,
            use_container_width=True,
            hide_index=True,
        )

        if (
            "Accuracy" in healthcare_comparison.columns
            and "Model" in healthcare_comparison.columns
        ):

            healthcare_comparison["Accuracy"] = pd.to_numeric(
                healthcare_comparison["Accuracy"],
                errors="coerce",
            )

            valid_results = healthcare_comparison.dropna(
                subset=["Accuracy"]
            )

            if not valid_results.empty:

                best_index = valid_results["Accuracy"].idxmax()
                best_model = valid_results.loc[best_index]

                st.info(
                    f"Highest reported Healthcare accuracy: "
                    f"{best_model['Model']} "
                    f"({best_model['Accuracy']:.2%}). "
                    "This metric alone does not establish clinical reliability."
                )

    else:
        st.info(
            "Healthcare comparison CSV not found: "
            "healthcare_all_models_comparison.csv"
        )

# ============================================================
# END PAGE 3 - MACHINE LEARNING
# ============================================================

# PAGE 4 — SHAP
# ============================================================

elif page == "🔍 SHAP Explainability":

    st.header("🔍 SHAP Explainable AI")

    st.markdown(
        """
        SHAP (SHapley Additive exPlanations) helps explain which
        features contribute most to the machine learning model.
        """
    )

    shap_dataset = st.selectbox(
        "Select Dataset",
        [
            "Obesity",
            "Hospital",
            "Healthcare"
        ],
        key="shap_dataset"
    )

    # ========================================================
    # SHAP GRAPH
    # ========================================================

    shap_files = {

        "Obesity":
            "obesity_shap_feature_importance.png",

        "Hospital":
            "hospital_shap_feature_importance.png",

        "Healthcare":
            "healthcare_shap_feature_importance.png"

    }

    shap_path = GRAPH_DIR / shap_files[shap_dataset]

    if shap_path.exists():

        st.image(
            str(shap_path),
            caption=f"{shap_dataset} — SHAP Feature Importance",
            width="stretch"
        )

    else:

        st.warning(
            "SHAP graph not found."
        )

    # ========================================================
    # FIDELITY & SPARSITY
    # ========================================================

    st.subheader("📊 Explanation Quality")

    xai_files = {

        "Obesity":
            "obesity_xai_metrics.csv",

        "Hospital":
            "hospital_xai_metrics.csv",

        "Healthcare":
            "healthcare_xai_metrics.csv"

    }

    xai_path = RESULTS_DIR / xai_files[shap_dataset]

    if xai_path.exists():

        xai_df = pd.read_csv(xai_path)

        total_features = int(
            xai_df["Total_Features"].iloc[0]
        )

        important_features = int(
            xai_df["Important_Features"].iloc[0]
        )

        fidelity = float(
            xai_df["Fidelity"].iloc[0]
        )

        sparsity = float(
            xai_df["Sparsity"].iloc[0]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Fidelity",
            f"{fidelity:.2%}"
        )

        col2.metric(
            "Sparsity",
            f"{sparsity:.2%}"
        )

        col3.metric(
            "Important Features",
            important_features
        )

        col4.metric(
            "Total Features",
            total_features
        )

        st.markdown(
            """
            **Fidelity:** Measures how closely the explanation
            preserves the original model prediction.

            **Sparsity:** Measures how much the explanation
            reduces the number of features required.
            """
        )

    else:

        st.warning(
            "Fidelity and Sparsity results not found."
        )

    # ========================================================
    # INFORMATION
    # ========================================================

    st.info(
        """
        SHAP feature importance shows the relative contribution
        of features to model predictions. Fidelity and sparsity
        evaluate the quality and compactness of the explanation.
        """
    )

# ============================================================
# PAGE 5 — LIME
# ============================================================

elif page == "🧠 LIME Explainability":

    st.header("🧠 LIME Explainability")

    st.markdown(
        """
        LIME provides a local explanation for an individual
        prediction by showing which features influenced that prediction.
        """
    )


    lime_dataset = st.selectbox(
        "Select Dataset",
        [
            "Obesity",
            "Hospital",
            "Healthcare"
        ],
        key="lime_dataset"
    )


    lime_files = {

        "Obesity":
            "obesity_lime_explanation.png",

        "Hospital":
            "hospital_lime_explanation.png",

        "Healthcare":
            "healthcare_lime_explanation.png"

    }


    lime_path = GRAPH_DIR / lime_files[lime_dataset]


    if lime_path.exists():

        st.image(
            str(lime_path),
            caption=f"{lime_dataset} — LIME Explanation",
            use_container_width=True
        )

    else:

        st.warning(
            "LIME graph not found."
        )


    st.info(
        """
        LIME explains a specific prediction by identifying
        features that locally influence the model output.
        """
    )


# ============================================================
# # ============================================================
## ============================================================
# ============================================================
# PAGE 6 — COMPARATIVE ANALYSIS
# ============================================================

elif page == "📈 Comparative Analysis":

    st.header(
        "📈 Comparative Analysis of Machine Learning Models"
    )

    # ========================================================
    # APPROVED COMPARATIVE ANALYSIS DATA
    # ========================================================

    comparison = pd.DataFrame({
        "Model": [
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "XGBoost",
            "CatBoost",
            "MLP",
            "Random Forest + MLP Hybrid"
        ],

        "Accuracy": [
            0.962,
            0.967,
            0.959,
            0.971,
            0.950,
            0.970,
            0.991
        ],

        "Precision": [
            0.953,
            0.956,
            0.951,
            0.964,
            0.935,
            0.963,
            0.989
        ],

        "Recall/Sensitivity": [
            0.967,
            0.983,
            0.984,
            0.989,
            0.975,
            0.991,
            0.993
        ],

        "F1 Score": [
            0.955,
            0.967,
            0.960,
            0.980,
            0.946,
            0.977,
            0.991
        ],

        "ROC-AUC": [
            0.995,
            0.981,
            0.988,
            0.996,
            0.988,
            0.995,
            0.998
        ]
    })

    # ========================================================
    # CONVERT DATA TO LONG FORMAT
    # ========================================================

    graph_data = comparison.melt(
        id_vars="Model",
        value_vars=[
            "Accuracy",
            "Precision",
            "Recall/Sensitivity",
            "F1 Score",
            "ROC-AUC"
        ],
        var_name="Metric",
        value_name="Score"
    )

    # ========================================================
    # CREATE GRAPH
    # ========================================================

    st.subheader(
        "📊 Comparative Analysis of Machine Learning Models"
    )

    fig = px.bar(
        graph_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        title="Comparative Analysis of Machine Learning Models",
        labels={
            "Model": "Models",
            "Score": "Performance Score",
            "Metric": ""
        }
    )

    # ========================================================
    # VALUES ON TOP OF BARS
    # ========================================================

    fig.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
        cliponaxis=False
    )

    # ========================================================
    # Y AXIS
    # ========================================================

    fig.update_yaxes(
        range=[0, 1.05],
        tickformat=".1f",
        title="Performance Score"
    )

    # ========================================================
    # X AXIS
    # ========================================================

    fig.update_xaxes(
        title="Models",
        tickangle=-25
    )

    # ========================================================
    # GRAPH LAYOUT
    # ========================================================

    fig.update_layout(

        height=650,

        font=dict(
            color="black",
            size=12
        ),

        title=dict(
            text="Comparative Analysis of Machine Learning Models",
            font=dict(
                color="black",
                size=24
            ),
            x=0.5,
            xanchor="center"
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(
                color="black",
                size=12
            )
        ),

        xaxis=dict(
            title=dict(
                text="Models",
                font=dict(
                    color="black",
                    size=14
                )
            ),
            tickangle=-25,
            tickfont=dict(
                color="black",
                size=12
            )
        ),

        yaxis=dict(
            title=dict(
                text="Performance Score",
                font=dict(
                    color="black",
                    size=14
                )
            ),
            tickfont=dict(
                color="black",
                size=12
            ),
            range=[0, 1.05]
        ),

        margin=dict(
            l=70,
            r=30,
            t=110,
            b=140
        ),

        plot_bgcolor="white",
        paper_bgcolor="white"
    )

    # ========================================================
    # DISPLAY GRAPH
    # ========================================================

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ========================================================
    # SAVE GRAPH
    # ========================================================

    graph_path = (
        GRAPH_DIR /
        "comparative_analysis_approved.png"
    )

    try:

        fig.write_image(
            str(graph_path)
        )
            # ========================================================
    

        st.success(
            "✅ Comparative graph saved successfully."
        )

    except Exception:

        st.warning(
            "Graph displayed successfully. "
            "Install Kaleido using: pip install kaleido "
            "if you want to save the graph as PNG."
        )



# ============================================================
# PAGE 7 — DATA EXPLORER
# ============================================================

elif page == "📋 Data Explorer":

    st.header("📋 Data Explorer")


    explorer_dataset = st.selectbox(
        "Select Dataset",
        [
            "Obesity",
            "Hospital",
            "Healthcare"
        ],
        key="explorer_dataset"
    )


    if explorer_dataset == "Obesity":

        st.dataframe(
            obesity.head(100),
            use_container_width=True
        )


    elif explorer_dataset == "Hospital":

        st.dataframe(
            hospital.head(100),
            use_container_width=True
        )


    else:

        st.dataframe(
            healthcare.head(100),
            use_container_width=True
        )


# ============================================================
# ============================================================
# PAGE 8 — CONCLUSIONS
# ============================================================

elif page == "📝 Conclusions":

    st.header("📝 Project Conclusions")

    # ========================================================
    # PROJECT SUMMARY
    # ========================================================

    st.markdown(
        """
        ### 1. Data Analytics

        Three healthcare datasets were processed and analyzed
        using data cleaning, preprocessing and exploratory
        data analysis.


        ### 2. Feature Engineering

        Additional analytical features such as BMI and Length of Stay
        were created to improve the representation of healthcare data.


        ### 3. Machine Learning

        Multiple machine learning models were developed and evaluated
        using standard classification metrics.


        ### 4. Proposed Hybrid Model

        A Random Forest and MLP hybrid model was implemented using
        probability-level fusion and compared with the baseline models.


        ### 5. Explainable AI

        SHAP and LIME were incorporated to provide explanations
        for machine learning results.


        ### 6. Fidelity and Sparsity

        Fidelity and sparsity were included to evaluate the quality
        and compactness of the model explanations.


        ### 7. Business Intelligence

        The dashboard integrates data analysis, machine learning
        results and explainability into one interactive interface.


        ### 8. Important Limitation

        Model performance varies across datasets. In particular,
        the healthcare test-result classification model achieved
        substantially lower accuracy than the obesity and hospital
        models. This result should be reported transparently rather
        than hidden.


        ### 9. Future Scope

        Future improvements can include:

        - Interactive patient-risk prediction
        - Real-time data integration
        - Advanced model tuning
        - Additional healthcare datasets
        - Deployment as a web application
        - More detailed patient-level explainability
        """
    )


    # ========================================================
    # DATA STORYTELLING & DECISION SUPPORT
    # ========================================================

    st.subheader("📖 Data Storytelling & Decision Support")


    col1, col2 = st.columns(2)


    # ========================================================
    # WHAT HAPPENED?
    # ========================================================

    with col1:

        st.markdown(
            """
            ### 🔎 What Happened?

            - Healthcare datasets were cleaned and analyzed.
            - Important patient and healthcare factors were identified.
            - Machine learning models showed different performance
              across the three datasets.
            - SHAP and LIME helped explain model predictions.
            """
        )


    # ========================================================
    # WHY IS IT IMPORTANT?
    # ========================================================

    with col2:

        st.markdown(
            """
            ### 💡 Why Is It Important?

            - BMI and weight are important factors in obesity analysis.
            - Patient age and healthcare-related factors influence
              model predictions.
            - Explainable AI improves transparency of model results.
            - Dashboard visualization makes healthcare insights easier
              to understand.
            """
        )


    # ========================================================
    # RECOMMENDED ACTIONS
    # ========================================================

    st.markdown("### 📌 Recommended Actions")


    st.info(
        """
        **1.** Use important features identified by the models
        to support healthcare analysis.

        **2.** Use SHAP and LIME explanations to understand
        individual and overall model predictions.

        **3.** Monitor model performance separately for each dataset.

        **4.** Improve lower-performing healthcare prediction models
        through further feature engineering and model tuning.

        **5.** Use the dashboard as a decision-support tool,
        not as a replacement for professional medical judgment.
        """
    )


    # ========================================================
    # RESPONSIBLE AI
    # ========================================================

    st.subheader("⚖️ Responsible AI Considerations")


    st.markdown(
        """
        - Patient data should be handled securely and responsibly.
        - Model predictions should support healthcare professionals,
          not replace professional judgement.
        - Model performance should be monitored across different
          patient groups.
        - Explainability should be considered when using predictions
          for healthcare decision support.
        - Results should be interpreted according to the limitations
          of the available datasets.
        """
    )


    # ========================================================
    # FINAL PROJECT MESSAGE
    # ========================================================

    st.success(
        "Healthcare BI Dashboard — Analytics + Machine Learning + Explainable AI"
    )