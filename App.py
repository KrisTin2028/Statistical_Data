import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import norm, shapiro, ks_2samp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


# Define a function to load data
def load_data(file):
    data = pd.read_csv(file)
    return data

# Sidebar options
st.sidebar.title("MLtRS")
option = st.sidebar.selectbox("Choose Option", ["Home", "Data Sets", "Statistical Analysis", "Supervised Learning", "Unsupervised Learning", "Contact"])

# Home tab
if option == "Home":
    st.title("Hi Welcome to MLtRS")
    st.write("This is a web application for Machine Learning and AI! Right now application supports a few techniques related to Supervised Learning.")

# Data Sets tab
elif option == "Data Sets":
    st.title("Upload a CSV file")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Data preview:")
        st.write(data.head())
        st.write("Data summary:")
        st.write(data.describe())
        data_columns = data.columns.tolist()
        st.multiselect("Choose columns to display", options=data_columns, default=data_columns)
        if st.button("Download data"):
            st.write("Downloading data is not yet implemented")

# Statistical Analysis tab
elif option == "Statistical Analysis":
    st.title("Statistical Analysis")
    st.subheader("Summary Statistics")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Data preview:")
        st.write(data.head())
        columns = st.multiselect("Choose columns to summarize", options=data.columns.tolist(), default=data.columns.tolist())
        st.write(data[columns].describe())
    st.subheader("Frequency Tables")
    # Add implementation for frequency tables here

# Supervised Learning tab
elif option == "Supervised Learning":
    st.title("Supervised Learning")
    algorithm = st.selectbox("Choose Algorithm", ["Logistic Regression", "kNN", "SVM", "Decision Trees", "Random Forests", "Naive Bayes Classifier", "Neural Networks"])
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Data preview:")
        st.write(data.head())
        target = st.selectbox("Choose target variable", options=data.columns.tolist())
        features = st.multiselect("Choose feature variables", options=[col for col in data.columns if col != target])
        if st.button("Train and Test"):
            X = data[features]
            y = data[target]
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            if algorithm == "Logistic Regression":
                model = LogisticRegression()
            elif algorithm == "kNN":
                model = KNeighborsClassifier()
            elif algorithm == "SVM":
                model = SVC()
            elif algorithm == "Decision Trees":
                model = DecisionTreeClassifier()
            elif algorithm == "Random Forests":
                model = RandomForestClassifier()
            elif algorithm == "Naive Bayes Classifier":
                model = GaussianNB()
            elif algorithm == "Neural Networks":
                model = MLPClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))
            st.write("Classification Report:")
            st.write(classification_report(y_test, y_pred))

# Unsupervised Learning tab
elif option == "Unsupervised Learning":
    st.title("Unsupervised Learning")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], key="unsupervised_file")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Data preview:")
        st.write(data.head())
        st.sidebar.header("Principal Component Analysis (PCA)")
        pca_var = st.sidebar.multiselect("Select Columns for PCA", data.columns.tolist())
        pca_n_components = st.sidebar.slider("Number of PCA Components", 1, min(len(data.columns), 10), 2)
        if pca_var:
            X = data[pca_var]
            X = StandardScaler().fit_transform(X)
            pca = PCA(n_components=pca_n_components)
            principalComponents = pca.fit_transform(X)
            principalDf = pd.DataFrame(data=principalComponents,
                                    columns=[f'Principal Component {i + 1}' for i in range(pca_n_components)])
            st.write("PCA Result")
            st.write(principalDf)
            fig, ax = plt.subplots()
            sns.scatterplot(x=principalDf.iloc[:, 0], y=principalDf.iloc[:, 1], ax=ax)
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            st.pyplot(fig)

# Contact tab
elif option == "Contact":
    st.title("Contact Information")
    st.write("Information to contact")

# Upload dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview")
    st.dataframe(data.head())

    # Data processing options
    st.sidebar.header("Data Processing Options")
    columns = data.columns.tolist()
    selected_columns = st.sidebar.multiselect("Select Columns", columns)

    transformation = st.sidebar.selectbox(
        "Select Transformation",
        ["None", "Log", "Inverse Log", "Exponential", "Log Normal", "Standardize"]
    )

    if transformation == "Log":
        data[selected_columns] = np.log(data[selected_columns])
    elif transformation == "Inverse Log":
        data[selected_columns] = 1 / np.log(data[selected_columns])
    elif transformation == "Exponential":
        data[selected_columns] = np.exp(data[selected_columns])
    elif transformation == "Log Normal":
        data[selected_columns] = np.log1p(data[selected_columns])
    elif transformation == "Standardize":
        scaler = StandardScaler()
        data[selected_columns] = scaler.fit_transform(data[selected_columns])

    st.write("Transformed Data")
    st.dataframe(data.head())

    # Download processed dataset
    @st.cache_data
    def convert_df(df):
        return df.to_csv().encode('utf-8')

    csv = convert_df(data)

    st.sidebar.download_button(
        label="Download Processed Data",
        data=csv,
        file_name='processed_data.csv',
        mime='text/csv',
    )

    # Summary statistics
    st.sidebar.header("Summary Statistics")
    summary_option = st.sidebar.selectbox(
        "Select Summary Option",
        ["Summary", "Length", "Dimensions", "Type", "Class"]
    )

    if summary_option == "Summary":
        st.write(data[selected_columns].describe())
    elif summary_option == "Length":
        st.write(len(data[selected_columns]))
    elif summary_option == "Dimensions":
        st.write(data[selected_columns].shape)
    elif summary_option == "Type":
        st.write(data[selected_columns].dtypes)
    elif summary_option == "Class":
        st.write(data[selected_columns].apply(lambda x: x.__class__.__name__))

    # Frequency table
    st.sidebar.header("Frequency Table")
    freq_col1 = st.sidebar.selectbox("Select First Column", columns, key="freq1")
    freq_col2 = st.sidebar.selectbox("Select Second Column", columns, key="freq2")

    if freq_col1 and freq_col2:
        freq_table = pd.crosstab(data[freq_col1], data[freq_col2])
        st.write("Frequency Table")
        st.write(freq_table)

    # Cross Tabulation
    st.sidebar.header("Cross Tabulation")
    cross_col1 = st.sidebar.selectbox("Select First Column", columns, key="cross1")
    cross_col2 = st.sidebar.selectbox("Select Second Column", columns, key="cross2")

    if cross_col1 and cross_col2:
        cross_tab = pd.crosstab(data[cross_col1], data[cross_col2])
        st.write("Cross Tabulation")
        st.write(cross_tab)
        chi2_test = sm.stats.Table(cross_tab).test_nominal_association()
        st.write("Chi-Square Test")
        st.write(chi2_test)

    # Plotting
    st.sidebar.header("Plotting")
    plot_option = st.sidebar.selectbox(
        "Select Plot Type",
        ["Histogram", "Bar Plot", "Scatter", "Pie"]
    )
    plot_column = st.sidebar.selectbox("Select Column for Plot", columns, key="plot")

    if plot_option == "Histogram":
        st.write("Histogram")
        fig, ax = plt.subplots()
        sns.histplot(data[plot_column], kde=True, ax=ax)
        st.pyplot(fig)
    elif plot_option == "Bar Plot":
        st.write("Bar Plot")
        fig, ax = plt.subplots()
        data[plot_column].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    elif plot_option == "Scatter":
        scatter_x = st.sidebar.selectbox("Select X-axis Column", columns, key="scatter_x")
        scatter_y = st.sidebar.selectbox("Select Y-axis Column", columns, key="scatter_y")
        st.write("Scatter Plot")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x=scatter_x, y=scatter_y, ax=ax)
        st.pyplot(fig)
    elif plot_option == "Pie":
        st.write("Pie Chart")
        fig, ax = plt.subplots()
        data[plot_column].value_counts().plot(kind='pie', ax=ax)
        st.pyplot(fig)

    # Statistical Tests
    st.sidebar.header("Statistical Tests")
    normaltest_col = st.sidebar.selectbox("Select Column for Normality Test", columns)
    normaltest_option = st.sidebar.selectbox(
        "Select Normality Test",
        ["Anderson-Darling", "Shapiro-Wilk", "Kolmogorov-Smirnov"]
    )

    if normaltest_option == "Anderson-Darling":
        result = sm.stats.anderson(data[normaltest_col])
    elif normaltest_option == "Shapiro-Wilk":
        result = shapiro(data[normaltest_col])
    elif normaltest_option == "Kolmogorov-Smirnov":
        result = ks_2samp(data[normaltest_col], np.random.normal(size=len(data)))

    st.write("Normality Test Result")
    st.write(result)

    # Correlation & Regression
    st.sidebar.header("Correlation & Regression")
    cor_col1 = st.sidebar.selectbox("Select First Column for Correlation", columns, key="cor1")
    cor_col2 = st.sidebar.selectbox("Select Second Column for Correlation", columns, key="cor2")
    cor_method = st.sidebar.selectbox(
        "Select Correlation Method",
        ["Covariance", "Pearson", "Spearman", "Kendall"]
    )

    if cor_method == "Covariance":
        correlation = np.cov(data[cor_col1], data[cor_col2])
    else:
        correlation = data[cor_col1].corr(data[cor_col2], method=cor_method.lower())

    st.write("Correlation Result")
    st.write(correlation)

    # Linear Regression
    st.sidebar.header("Linear Regression")
    reg_col1 = st.sidebar.selectbox("Select X Column for Regression", columns, key="reg1")
    reg_col2 = st.sidebar.selectbox("Select Y Column for Regression", columns, key="reg2")
    reg_method = st.sidebar.selectbox(
        "Select Regression Method",
        ["Fit", "Summary", "ANOVA"]
    )

    X = sm.add_constant(data[reg_col1])
    y = data[reg_col2]
    model = sm.OLS(y, X).fit()

    if reg_method == "Fit":
        st.write("Regression Fit")
        st.write(model.params)
    elif reg_method == "Summary":
        st.write("Regression Summary")
        st.write(model.summary())
    elif reg_method == "ANOVA":
        st.write("ANOVA Result")
        anova_results = sm.stats.anova_lm(model)
        st.write(anova_results)

    # Logistic Regression
    st.sidebar.header("Logistic Regression")
    logr_var = st.sidebar.selectbox("Select Dependent Variable", columns, key="logr")
    logr_test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)
    logr_options = st.sidebar.selectbox(
        "Select Logistic Regression Option",
        ["Show Proportion", "Fit", "Coefficients", "Prediction Accuracy"]
    )

    X = data.drop(columns=[logr_var])
    y = data[logr_var]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=logr_test_size, random_state=0)
    logr_model = LogisticRegression(max_iter=1000).fit(X_train, y_train)

    if logr_options == "Show Proportion":
        st.write("Training Set Proportion")
        st.write(len(y_train) / len(y))
    elif logr_options == "Fit":
        st.write("Logistic Regression Model")
        st.write(logr_model)
    elif logr_options == "Coefficients":
        st.write("Model Coefficients")
        st.write(np.exp(logr_model.coef_))
    elif logr_options == "Prediction Accuracy":
        y_pred = logr_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        st.write("Prediction Accuracy")
        st.write(f"Accuracy: {accuracy}")
        st.write("Confusion Matrix")
        st.write(cm)
    
    # K-Nearest Neighbors (KNN)
    st.sidebar.header("K-Nearest Neighbors")
    knn_var = st.sidebar.selectbox("Select Dependent Variable", columns, key="knn_var")
    knn_test_size = st.sidebar.slider("Test Size for KNN", 0.1, 0.5, 0.3)
    knn_options = st.sidebar.selectbox(
        "Select KNN Option",
        ["Fit", "Prediction Accuracy"]
    )

    if knn_var:
        X = data.drop(columns=[knn_var])
        y = data[knn_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=knn_test_size, random_state=0)
        knn_model = KNeighborsClassifier().fit(X_train, y_train)

        if knn_options == "Fit":
            st.write("KNN Model")
            st.write(knn_model)
        elif knn_options == "Prediction Accuracy":
            y_pred = knn_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            st.write("Prediction Accuracy")
            st.write(f"Accuracy: {accuracy}")
            st.write("Confusion Matrix")
            st.write(cm)

    # Support Vector Machine (SVM)
    st.sidebar.header("Support Vector Machine")
    svm_var = st.sidebar.selectbox("Select Dependent Variable", columns, key="svm_var")
    svm_test_size = st.sidebar.slider("Test Size for SVM", 0.1, 0.5, 0.3)
    svm_options = st.sidebar.selectbox(
        "Select SVM Option",
        ["Fit", "Prediction Accuracy"]
    )

    if svm_var:
        X = data.drop(columns=[svm_var])
        y = data[svm_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=svm_test_size, random_state=0)
        svm_model = SVC().fit(X_train, y_train)

        if svm_options == "Fit":
            st.write("SVM Model")
            st.write(svm_model)
        elif svm_options == "Prediction Accuracy":
            y_pred = svm_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            st.write("Prediction Accuracy")
            st.write(f"Accuracy: {accuracy}")
            st.write("Confusion Matrix")
            st.write(cm)

    # Decision Tree
    st.sidebar.header("Decision Tree")
    tree_var = st.sidebar.selectbox("Select Dependent Variable", columns, key="tree_var")
    tree_test_size = st.sidebar.slider("Test Size for Decision Tree", 0.1, 0.5, 0.3)
    tree_options = st.sidebar.selectbox(
        "Select Decision Tree Option",
        ["Fit", "Prediction Accuracy"]
    )

    if tree_var:
        X = data.drop(columns=[tree_var])
        y = data[tree_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tree_test_size, random_state=0)
        tree_model = DecisionTreeClassifier().fit(X_train, y_train)

        if tree_options == "Fit":
            st.write("Decision Tree Model")
            st.write(tree_model)
        elif tree_options == "Prediction Accuracy":
            y_pred = tree_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            st.write("Prediction Accuracy")
            st.write(f"Accuracy: {accuracy}")
            st.write("Confusion Matrix")
            st.write(cm)

    # Random Forest
    st.sidebar.header("Random Forest")
    rf_var = st.sidebar.selectbox("Select Dependent Variable", columns, key="rf_var")
    rf_test_size = st.sidebar.slider("Test Size for Random Forest", 0.1, 0.5, 0.3)
    rf_options = st.sidebar.selectbox(
        "Select Random Forest Option",
        ["Fit", "Prediction Accuracy"]
    )

    if rf_var:
        X = data.drop(columns=[rf_var])
        y = data[rf_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=rf_test_size, random_state=0)
        rf_model = RandomForestClassifier().fit(X_train, y_train)

        if rf_options == "Fit":
            st.write("Random Forest Model")
            st.write(rf_model)
        elif rf_options == "Prediction Accuracy":
            y_pred = rf_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            st.write("Prediction Accuracy")
            st.write(f"Accuracy: {accuracy}")
            st.write("Confusion Matrix")
            st.write(cm)

    # Principal Component Analysis (PCA)
    st.sidebar.header("Principal Component Analysis (PCA)")
    pca_var = st.sidebar.multiselect("Select Columns for PCA", columns)
    pca_n_components = st.sidebar.slider("Number of PCA Components", 1, min(len(columns), 10), 2)

    if pca_var:
        X = data[pca_var]
        X = StandardScaler().fit_transform(X)
        pca = PCA(n_components=pca_n_components)
        principalComponents = pca.fit_transform(X)
        principalDf = pd.DataFrame(data=principalComponents,
                                columns=[f'Principal Component {i + 1}' for i in range(pca_n_components)])
        st.write("PCA Result")
        st.write(principalDf)

        fig, ax = plt.subplots()
        sns.scatterplot(x=principalDf.iloc[:, 0], y=principalDf.iloc[:, 1], ax=ax)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        st.pyplot(fig)
