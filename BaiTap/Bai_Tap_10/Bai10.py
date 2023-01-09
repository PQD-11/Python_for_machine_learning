import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier
import xgboost


st.set_page_config(
    page_title="Bài tập 10: XGBoost", initial_sidebar_state="expanded"
)

st.write(
    """
# XGBoost
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
                options=["K-fold cross validation", "Train/Test split"],
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
        value=0.1,
        step=0.1,
        key="alpha",
        )
    else:
        st.write("&ensp; **`Numbers of Fold`**")
        value_slider = st.slider(
        " ",
        label_visibility='collapsed',
        min_value=4,
        max_value=20,
        value=4,
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
    cols = st.columns(9)
    with cols[4]: 
        button = st.button("Run")
    if button:
        st.markdown("### 4. Results")
        x_data = df[input_feature]
        X = x_data.to_numpy()
        y_data = df[df.columns[-1]]
        Y = y_data.to_numpy()
        
        print("X: ", X)
        print("Y: ", Y)
        
        if method_test == "Train/Test split":
            le = LabelEncoder()
            Y = le.fit_transform(Y)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = value_slider, stratify = Y)
            
            XGB_model = xgboost.XGBClassifier().fit(x_train, y_train)
            XGB_y_pred = XGB_model.predict(x_test)
            print("pred: ", XGB_y_pred)
            print("test: ",y_test)
            XGB_precision = precision_score(y_test, XGB_y_pred, average='macro') 
            print(XGB_precision)
            
            LR_model = LogisticRegression().fit(x_train, y_train)
            LR_y_pred = LR_model.predict(x_test)
            LR_precision = precision_score(y_test, LR_y_pred, average='macro')
            print(LR_precision)
            
            SVM_model = SVC(kernel = 'linear').fit(x_train, y_train)
            SVM_y_pred = SVM_model.predict(x_test)
            SVM_precision = precision_score(y_test, SVM_y_pred, average='macro')
            print(SVM_precision)
            
            DTC_model = DecisionTreeClassifier().fit(x_train, y_train)
            DTC_y_pred = DTC_model.predict(x_test)
            DTC_precision = precision_score(y_test, DTC_y_pred, average='macro')
            print(DTC_precision)
                        
            plt.figure(figsize=(8,4))
            ax1 = plt.bar([0], [XGB_precision], 0.2, label = str("{:.5f}".format(XGB_precision)), color='b')
            ax2 = plt.bar([0.5], [LR_precision], 0.2, label = str("{:.5f}".format(LR_precision)), color='c')
            ax3 = plt.bar([1], [SVM_precision], 0.2, label = str("{:.5f}".format(SVM_precision)), color='g')
            ax4 = plt.bar([1.5], [DTC_precision], 0.2, label = str("{:.5f}".format(DTC_precision)), color='m')
            
            plt.xlabel("Model", color='black')
            plt.ylabel("Precision score", color='black')
            plt.title("Train/Test Split",fontweight="bold", color= 'blue')
            plt.xticks(np.arange(0, 2, 0.5) , ['XGBoost', 'Logistic', 'SVM', 'Decision'])
            plt.legend(bbox_to_anchor =(1, 1))
            plt.tight_layout()
            plt.savefig('chart.png')            
        else: 
            le = LabelEncoder()
            Y = le.fit_transform(Y)
            kf = KFold(n_splits=value_slider, random_state=None)
            folds = [str(fold) for fold in range(1, value_slider+1)]
            XGB_precision = []
            LR_precision = []
            SVM_precision = []
            DTC_precision = []
            
            for train_index, test_index in kf.split(X):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                
                XGB_model = xgboost.XGBClassifier().fit(x_train, y_train)
                XGB_y_pred = XGB_model.predict(x_test)
                print("pred: ", XGB_y_pred)
                print("test: ",y_test)
                XGB_precision.append(precision_score(y_test, XGB_y_pred, average='macro'))
                
                LR_model = LogisticRegression().fit(x_train, y_train)
                LR_y_pred = LR_model.predict(x_test)
                LR_precision.append(precision_score(y_test, LR_y_pred, average='macro'))
                
                SVM_model = SVC(kernel = 'linear').fit(x_train, y_train)
                SVM_y_pred = SVM_model.predict(x_test)
                SVM_precision.append(precision_score(y_test, SVM_y_pred, average='macro'))
                
                DTC_model = DecisionTreeClassifier().fit(x_train, y_train)
                DTC_y_pred = DTC_model.predict(x_test)
                DTC_precision.append(precision_score(y_test, DTC_y_pred, average='macro'))
            
            XGB_precision.append(sum(XGB_precision) / len(XGB_precision))
            LR_precision.append(sum(LR_precision) / len(LR_precision))
            SVM_precision.append(sum(SVM_precision) / len(SVM_precision))
            DTC_precision.append(sum(DTC_precision) / len(DTC_precision))
            
            plt.figure(figsize=(8,4))
            ax1 = plt.bar([0], [XGB_precision[-1]], 0.2, label = str("{:.5f}".format(XGB_precision[-1])), color='b')
            ax2 = plt.bar([0.5], [LR_precision[-1]], 0.2, label = str("{:.5f}".format(LR_precision[-1])), color='c')
            ax3 = plt.bar([1], [SVM_precision[-1]], 0.2, label = str("{:.5f}".format(SVM_precision[-1])), color='g')
            ax4 = plt.bar([1.5], [DTC_precision[-1]], 0.2, label = str("{:.5f}".format(DTC_precision[-1])), color='m')
            
            plt.xlabel("Model", color='black')
            plt.ylabel("Precision score", color='black')
            plt.title("K-Fold ",fontweight="bold", color= 'blue')
            plt.xticks(np.arange(0, 2, 0.5) , ['XGBoost', 'Logistic', 'SVM', 'Decision'])
            plt.legend(bbox_to_anchor =(1, 1))
            plt.tight_layout()
            plt.savefig('chart.png')   
            
        img = cv2.imread('chart.png')
        if img is not None: 
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                