import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss

st.set_page_config(
    page_title="Bài tập 6: Logistic Regression với Streamlit", initial_sidebar_state="expanded"
)

st.write(
    """
# Logistic Regression
#### Sinh viên: Phạm Quốc Đăng
#### MSSV: 19520036
"""
)

uploaded_file = st.file_uploader("Choose a CSV file", type=".csv")  
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    custom_cols = str(df.columns[-1]).split(';')
    df[custom_cols] = df[df.columns[-1]].str.split(pat=";", expand=True)
    df = df.drop(df.columns[0], axis=1)
    df = df.astype(int) 
    
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
        x_data = df[input_feature]
        X = x_data.to_numpy()
        y_data = df[df.columns[-1]]
        Y = y_data.to_numpy()
        if method_test == "Train/Test split":
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = value_slider, random_state=None, stratify = Y)
            model = LogisticRegression().fit(x_train, y_train)
            y_pred = model.predict(x_test)

            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            y_pred_loss = model.predict_proba(x_test)
            logloss = log_loss(y_test, y_pred_loss)
            print(precision)
            print(recall)
            print(f1)
            print(logloss)
                 
            plt.figure(figsize=(8,4))
            ax1 = plt.bar([0], [precision], 0.2, label = str("{:.5f}".format(precision)), color='b')
            ax2 = plt.bar([0.5], [recall], 0.2, label = str("{:.5f}".format(recall)), color='c')
            ax3 = plt.bar([1], [f1], 0.2, label = str("{:.5f}".format(f1)), color='g')
            ax4 = plt.bar([1.5], [logloss], 0.2, label = str("{:.5f}".format(logloss)), color='m')
            
            plt.xlabel("Model", color='black')
            plt.ylabel("Precision score", color='black')
            plt.title("Train/Test Split",fontweight="bold", color= 'blue')
            plt.xticks(np.arange(0, 2, 0.5) , ['precision', 'recall', 'f1', 'logloss'])
            plt.legend(bbox_to_anchor =(1, 1))
            plt.tight_layout()
            plt.savefig('chart.png')          
        else: 
            kf = KFold(n_splits=value_slider, random_state=None)
            folds = [str(fold) for fold in range(1, value_slider+1)]
            precision = []
            recall = []
            f1 = []
            logloss = []
            for train_index, test_index in kf.split(X):
                print("train_index: ",train_index)
                print("test_index: ", test_index)
                X_train, X_test = X[train_index], X[test_index]
                Y_train, Y_test = Y[train_index], Y[test_index]
                # print("X_train: ", X_train)
                # print("Y_train: ", Y_train)
                model = LogisticRegression().fit(X_train, Y_train)
                Y_pred = model.predict(X_test)
                
                precision.append(precision_score(Y_test, Y_pred, average='macro'))
                recall.append(recall_score(Y_test, Y_pred, average='macro'))
                f1.append(f1_score(Y_test, Y_pred, average='macro'))
                y_pred_loss = model.predict_proba(X_test)
                logloss.append(log_loss(Y_test, y_pred_loss))
                print(precision)
                print(recall)
                print(f1)
                print(logloss)
            
            precision.append(sum(precision) / len(precision))
            recall.append(sum(recall) / len(recall))
            f1.append(sum(f1) / len(f1))
            logloss.append(sum(logloss) / len(logloss))
             
            plt.figure(figsize=(8,4))
            ax1 = plt.bar([0], [precision[-1]], 0.2, label = str("{:.5f}".format(precision[-1])), color='b')
            ax2 = plt.bar([0.5], [recall[-1]], 0.2, label = str("{:.5f}".format(recall[-1])), color='c')
            ax3 = plt.bar([1], [f1[-1]], 0.2, label = str("{:.5f}".format(f1[-1])), color='g')
            ax4 = plt.bar([1.5], [logloss[-1]], 0.2, label = str("{:.5f}".format(logloss[-1])), color='m')
            
            plt.xlabel("Model", color='black')
            plt.ylabel("Precision score", color='black')
            plt.title("K-Fold ",fontweight="bold", color= 'blue')
            plt.xticks(np.arange(0, 2, 0.5) , ['precision', 'recall', 'f1', 'logloss'])
            plt.legend(bbox_to_anchor =(1, 1))
            plt.tight_layout()
            plt.savefig('chart.png')      
            
        img = cv2.imread('chart.png')
        if img is not None: 
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                
