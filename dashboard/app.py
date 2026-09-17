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

    # Dataset selector
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

        # ----------------------------------------------------
        # CONDITION + GENDER
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # MEDICATION + OUTCOME
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # AGE + GENDER
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # MEDICAL CONDITION + MEDICATION
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # LENGTH OF STAY + TEST RESULTS
        # ----------------------------------------------------

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
# PAGE 3 — MACHINE LEARNING
# ============================================================

elif page == "🤖 Machine Learning":

    st.header("🤖 Machine Learning Model Performance")

    st.markdown(
        """
        Two classification algorithms were evaluated:

        - Decision Tree
        - Random Forest

        The evaluation uses Accuracy, Precision, Recall and F1-score.
        """
    )


    # ========================================================
    # LOAD MODEL RESULTS
    # ========================================================

    result_files = [
        RESULT_DIR / "obesity_model_results.csv",
        RESULT_DIR / "hospital_model_results.csv",
        RESULT_DIR / "healthcare_model_results.csv"
    ]

    dataframes = []

    for file in result_files:

        if file.exists():

            df = pd.read_csv(file)

            dataframes.append(df)


    if dataframes:

        model_results = pd.concat(
            dataframes,
            ignore_index=True
        )

        st.subheader("📋 Model Performance Table")

        st.dataframe(
            model_results,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # ACCURACY COMPARISON
        # ====================================================

        accuracy_column = None

        for column in model_results.columns:

            if column.lower() == "accuracy":

                accuracy_column = column

                break


        if accuracy_column:

            st.subheader("📊 Accuracy Comparison")

            fig = px.bar(
                model_results,
                x="Dataset",
                y=accuracy_column,
                color="Model",
                barmode="group",
                title="Model Accuracy Across Datasets"
            )

            fig.update_yaxes(
                range=[0, 1]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "Model result CSV files were not found."
        )


    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    st.divider()

    st.subheader("🔲 Confusion Matrices")

    dataset_choice = st.selectbox(
        "Select Dataset",
        [
            "Obesity",
            "Hospital",
            "Healthcare"
        ],
        key="cm_dataset"
    )


    if dataset_choice == "Obesity":

        col1, col2 = st.columns(2)

        with col1:

            path = GRAPH_DIR / "obesity_decision_tree_confusion_matrix.png"

            if path.exists():

                st.image(
                    str(path),
                    caption="Obesity — Decision Tree",
                    use_container_width=True
                )

        with col2:

            path = GRAPH_DIR / "obesity_random_forest_confusion_matrix.png"

            if path.exists():

                st.image(
                    str(path),
                    caption="Obesity — Random Forest",
                    use_container_width=True
                )


    elif dataset_choice == "Hospital":

        col1, col2 = st.columns(2)

        with col1:

            path = GRAPH_DIR / "hospital_decision_tree_confusion_matrix.png"

            if path.exists():

                st.image(
                    str(path),
                    caption="Hospital — Decision Tree",
                    use_container_width=True
                )

        with col2:

            path = GRAPH_DIR / "hospital_random_forest_confusion_matrix.png"

            if path.exists():

                st.image(
                    str(path),
                    caption="Hospital — Random Forest",
                    use_container_width=True
                )


    elif dataset_choice == "Healthcare":

        col1, col2 = st.columns(2)

        with col1:

            path = GRAPH_DIR / "healthcare_decision_tree_confusion_matrix.png"

            if path.exists():

                st.image(
                    str(path),
                    caption="Healthcare — Decision Tree",
                    use_container_width=True
                )

        with col2:

            path = GRAPH_DIR / "healthcare_random_forest_confusion_matrix.png"

            if path.exists():

                st.image(
                    str(path),
                    caption="Healthcare — Random Forest",
                    use_container_width=True
                )


    # ========================================================
    # ROC CURVES
    # ========================================================

    st.divider()

    st.subheader("📈 ROC Curves")

    roc_choice = st.selectbox(
        "Select Dataset for ROC Curve",
        [
            "Obesity",
            "Hospital",
            "Healthcare"
        ],
        key="roc_dataset"
    )


    roc_files = {

        "Obesity":
            "obesity_random_forest_roc_curve.png",

        "Hospital":
            "hospital_random_forest_roc_curve.png",

        "Healthcare":
            "healthcare_random_forest_roc_curve.png"

    }


    roc_path = GRAPH_DIR / roc_files[roc_choice]


    if roc_path.exists():

        st.image(
            str(roc_path),
            caption=f"{roc_choice} — Random Forest ROC Curve",
            use_container_width=True
        )

    else:

        st.warning(
            "ROC curve graph not found."
        )


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
            use_container_width=True
        )

    else:

        st.warning(
            "SHAP graph not found."
        )


    st.info(
        """
        SHAP feature importance shows the relative contribution
        of features to model predictions.
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
# PAGE 6 — COMPARATIVE ANALYSIS
# ============================================================

elif page == "📈 Comparative Analysis":

    st.header("📈 Comparative Analysis")

    st.subheader("Dataset Comparison")


    comparison = pd.DataFrame({

        "Dataset": [
            "Obesity",
            "Hospital",
            "Healthcare"
        ],

        "Records": [
            len(obesity),
            len(hospital),
            len(healthcare)
        ],

        "Features": [
            len(obesity.columns),
            len(hospital.columns),
            len(healthcare.columns)
        ]

    })


    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )


    fig = px.bar(
        comparison,
        x="Dataset",
        y="Records",
        title="Dataset Record Comparison",
        text="Records"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.divider()


    st.subheader("🔬 Explainability Comparison")


    explainability = pd.DataFrame({

        "Dataset": [
            "Obesity",
            "Hospital",
            "Healthcare"
        ],

        "Explainability": [
            "SHAP + LIME",
            "SHAP + LIME",
            "SHAP + LIME"
        ]

    })


    st.dataframe(
        explainability,
        use_container_width=True,
        hide_index=True
    )


    st.markdown(
        """
        ### Key Comparative Insight

        All three datasets were processed through the same overall
        analytics pipeline:

        **Data Preprocessing → EDA → Feature Engineering →
        Feature Selection → Machine Learning → Model Evaluation →
        SHAP → LIME → Dashboard**

        The model performance differs across datasets because the
        underlying features, target variables and class distributions
        are different.
        """
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
# PAGE 8 — CONCLUSIONS
# ============================================================

elif page == "📝 Conclusions":

    st.header("📝 Project Conclusions")


    st.markdown(
        """
        ### 1. Data Analytics

        Three healthcare datasets were processed and analyzed
        using data cleaning, preprocessing and exploratory data analysis.


        ### 2. Feature Engineering

        Additional analytical features such as BMI and Length of Stay
        were created to improve the representation of healthcare data.


        ### 3. Machine Learning

        Decision Tree and Random Forest classification models were
        developed and evaluated using standard classification metrics.


        ### 4. Explainable AI

        SHAP and LIME were incorporated to provide explanations
        for machine learning results.


        ### 5. Business Intelligence

        The dashboard integrates data analysis, machine learning
        results and explainability into one interactive interface.


        ### 6. Important Limitation

        Model performance varies across datasets. In particular,
        the healthcare test-result classification model achieved
        substantially lower accuracy than the obesity and hospital
        models. This result should be reported transparently rather
        than hidden.


        ### 7. Future Scope

        Future improvements can include:

        - Interactive patient-risk prediction
        - Real-time data integration
        - Advanced model tuning
        - Additional healthcare datasets
        - Deployment as a web application
        - More detailed patient-level explainability
        """
    )


    st.success(
        "Healthcare BI Dashboard — Analytics + Machine Learning + Explainable AI"
    )