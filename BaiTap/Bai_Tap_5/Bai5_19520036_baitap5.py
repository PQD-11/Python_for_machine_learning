import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

st.set_page_config(
    page_title="Bài tập 5: Linear Regression với Streamlit", initial_sidebar_state="expanded"
)

st.write(
    """
# Linear Regression
#### Sinh viên: Phạm Quốc Đăng
#### MSSV: 19520036
"""
)

uploaded_file = st.file_uploader("Choose a CSV file", type=".csv")  
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("### 1. Data preview ")
    st.dataframe(df)

    st.markdown("### 2. Setup")
    st.write("&ensp; **`Input Features`**")
    input_feature = st.multiselect(
        " ",
        label_visibility='collapsed',
        options = df.columns[: -1]
    )
    st.write("&ensp; **`Output Features (Default): `**", df.columns[-1])
    
    st.write("&ensp; **`Type of Splitting Data`**")
    cols = st.columns(3)
    with cols[1]: 
        method_test = st.radio(
                " ",
                label_visibility='collapsed',
                options=["Train/Test split", "K-fold cross validation"],
                index=0,
                key="method",
            )
        
    if method_test ==  "Train/Test split":
        st.write("&ensp; **`Train Ratio`**")
        value_slider = st.slider(
        " ",
        label_visibility='collapsed',
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.1,
        key="alpha",
        )
        coefficient = ", Train Ratio: "
    else:
        st.write("&ensp; **`Numbers of Fold`**")
        value_slider = st.slider(
        " ",
        label_visibility='collapsed',
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        key="alpha",
        )
        coefficient = ", Numbers of Fold: "

    if not input_feature:
        st.warning("Please select input feature")
        st.stop()

    name = (
        "Website_Results.csv" if isinstance(uploaded_file, str) else uploaded_file.name
    )
    
if uploaded_file:      
    st.markdown("### 3. Train Model")  
    cols = st.columns(11)
    with cols[5]: 
        button = st.button("Run")
    if button:
        st.markdown("### 4. Results")
        st.write("&ensp; `Column chart (`", method_test, coefficient, value_slider, "`)`")
        x_data = df[input_feature]
        y_data = df[df.columns[-1]]
        if method_test == "Train/Test split":
            x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = value_slider, random_state=0)
            model = LinearRegression().fit(x_train, y_train)
            y_pred = model.predict(x_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            st.write("mae: ", mae)
            st.write("mse: ", mse)
            
            
            plt.figure(figsize=(8,4))
            ax1 = plt.subplot()
            ax1.bar(np.arange(1) - 0.21, [mae], 0.4, label='MAE', color='blue')
            plt.xticks(np.arange(1), [str(value_slider)])
            plt.xlabel("Ratio", color='blue')
            plt.ylabel("Mean Absolute Error", color='maroon')
            ax2 = ax1.twinx()
            ax2.bar(np.arange(1) + 0.21, [mse], 0.4, label='MSE', color='green')
            plt.ylabel('Mean Squared Error', color='green')
            plt.title("EVALUATION METRIC")
            plt.savefig('chart.png')
            
        else: 
            kf = KFold(n_splits = value_slider)
            model = LinearRegression()
            
            score_mae = cross_val_score(model, x_data, y_data, cv=kf, scoring='neg_mean_absolute_error')
            mae = np.absolute(score_mae)
            
            score_mse = cross_val_score(model, x_data, y_data, cv=kf, scoring='neg_mean_squared_error')
            mse = np.absolute(score_mse)
            
            st.write("mae: ", mae)
            st.write("score_mae: ", score_mae)
            
            folds = [str(fold) for fold in range(1, value_slider+1)]
            plt.figure(figsize=(8, 4))
            ax1 = plt.subplot()
            ax1.bar(np.arange(len(folds)) - 0.21, mae, 0.4, label='MAE', color='maroon')
            plt.xticks(np.arange(len(folds)), folds)
            plt.xlabel("Folds", color='blue')
            plt.ylabel("Mean Absolute Error", color='maroon')
            ax2 = ax1.twinx()
            ax2.bar(np.arange(len(folds)) + 0.21, mse, 0.4, label='MSE', color='green')
            plt.ylabel('Mean Squared Error', color='green')
            plt.title("EVALUATION METRIC")
            plt.savefig('chart.png')
            
        img = cv2.imread('chart.png')
        if img is not None: 
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                