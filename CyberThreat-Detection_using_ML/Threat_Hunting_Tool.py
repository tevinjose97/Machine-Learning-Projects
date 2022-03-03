from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.base import TransformerMixin

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import IsolationForest
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np

from math import ceil

import itertools
import datetime
import random
import pickle
import bisect
import time
import os
import re

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns) #np.dtype('O')

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

class tandem_model():
    
    def __init__(self,model_name, normal_label):

        self.model_name=model_name
        self.normal_label=normal_label

        # Use LabelEncoder to encode string labels  to numericals
        self.le = LabelEncoder()
        # Use Custom Imputer to handle both categorical and numerical data
        self.df_imputer=DataFrameImputer()
        
        # Model
        # Initialize CatBoostClassifier
        # If system does not have GPU, remove parameter setting task_type="GPU" to use CPU by default
        self.sp_model = CatBoostClassifier(loss_function='MultiClass',task_type="GPU",random_state=101)
        # Initialize Isolation Forest
        self.usp_model= IsolationForest(random_state=42)
        
        # Initialize empty set to store columns to be dropped
        self.drop_cols= set()

        # Set default scoring metric for supervised grid search model (training done after resampling)
        self.grid_sp_score= 'f1_macro'

        # Flag to check if preprocess train has been called
        self.prp_train_flag= False
    
    def preprocess_train(self,anomaly_trn_ds, option="split",label_col="class"):
        
        # Set Label Column
        self.label_col= label_col        
        # Store Dataset columns
        self.columns= list(anomaly_trn_ds.columns)

        if option=="split":
            # Split Dataset
            anomaly_trn_X, anomaly_tst_X, anomaly_trn_y, anomaly_tst_y = train_test_split(anomaly_trn_ds.drop([self.label_col],axis=1),anomaly_trn_ds[self.label_col], test_size=0.2, random_state=42)

        elif option=="complete":
            # Should be called after initial model training
            # Separate Features and Label Column
            anomaly_trn_X,anomaly_trn_y= anomaly_trn_ds.drop([self.label_col],axis=1),anomaly_trn_ds[self.label_col]

        # 'id' column is set, if any 
        if 'id' in anomaly_trn_X.columns:
            anomaly_trn_X=anomaly_trn_X.set_index('id')

        # Get string (object) column names
        object_cols=list(anomaly_trn_X.select_dtypes(include='object').columns)

        # Initialize object and non object columns lists
        self.obj_cols=[]
        self.non_obj_cols=[]

        reg_ex= "^[-+]?((\d*\.?\d+)|(\d+\.?\d*))$"

        # Identify misinterpreted numerical columns
        for col in object_cols:
            # Check misinterpreted numerical columns using regex 
            # (temporarily interpret it as string column)
            if all(bool(re.fullmatch(reg_ex,val)) for val in anomaly_trn_X[col].\
                astype(str).dropna()):
                anomaly_trn_X[col] = pd.to_numeric(anomaly_trn_X[col])
                self.non_obj_cols.append(col)
            else:
                # Store as string columns
                anomaly_trn_X[col] = anomaly_trn_X[col].astype(str)
                self.obj_cols.append(col)
        
        # Finding columns with high percent of missing values (>30%)
        for col in anomaly_trn_X.columns:
            pct_missing = np.mean(anomaly_trn_X[col].isnull())
            if (pct_missing)>0.30:
                self.drop_cols.update([col])

        # Finding columns with low variance, excluding the label col
        var_df=anomaly_trn_X.var()
        self.drop_cols.update(list(var_df[var_df<0.01].index))
        
        # Partial data preprocessing - Removing Columns with low variance, Dropping columns with high missing values,
        # Dropping unimportant features if any
        if len(self.drop_cols)>=1:  
            anomaly_trn_X=anomaly_trn_X.drop(self.drop_cols, axis=1)

        # Update object and non object columns lists after dropping columns 
        self.obj_cols= [col for col in self.obj_cols if col not in self.drop_cols]
        self.non_obj_cols= [col for col in self.non_obj_cols if col not in self.drop_cols]

        # Impute Train Dataset Values
        anomaly_trn_X=self.df_imputer.fit_transform(anomaly_trn_X)
        
        # Obtain string and numerical column names
        self.numerical_cols= list(anomaly_trn_X.select_dtypes(exclude='O').columns)
        self.string_cols= list(anomaly_trn_X.select_dtypes(include='O').columns)
                
        # Fit Label Encoder to y_data
        self.le.fit(anomaly_trn_y)

        # Insert 'Unknown Anomaly' into Label Encoder
        le_classes = self.le.classes_.tolist()
        bisect.insort_left(le_classes, 'Unknown Anomaly')

        self.le.classes_= le_classes
        # Mapping between labels and numbers
        self.label_num_dict= dict(zip(list(self.le.classes_),[i for i in range (len(self.le.classes_))]))
        #self.label_num_dict['Unkown Anomaly']= -1 
        self.num_label_dict= {value:key for key, value in self.label_num_dict.items()}

        # Use LabelEncoder to encode string labels  to numericals
        anomaly_trn_y= self.le.transform(anomaly_trn_y)

        # Store X and y datasets without resampling
        unsampled_X, unsampled_y= anomaly_trn_X, anomaly_trn_y

        try:

            # Undersampling Strategy
            u_smp_dic= under_sampling_strat(anomaly_trn_y,self.label_num_dict,self.normal_label)
            
            undersample= RandomUnderSampler(sampling_strategy=u_smp_dic, random_state=42,)
            resampled_trn_X, resampled_trn_y= undersample.fit_resample(anomaly_trn_X,anomaly_trn_y)

            string_indexes=[]
            for ind,col in enumerate(resampled_trn_X.columns):
                if col in self.string_cols:
                    string_indexes.append(ind)
            
            # Oversampling Strategy
            sm = SMOTENC(categorical_features=string_indexes, random_state=42)
            #st.write("Start of Oversampling")
            resampled_trn_X, resampled_trn_y = sm.fit_resample(resampled_trn_X, resampled_trn_y)

            # Replace dataset with resampled values
            anomaly_trn_X, anomaly_trn_y= resampled_trn_X, resampled_trn_y

        except Exception as e:
            # Set suspervised grid search scoring parameter to 'f1_micro' due to imbalanced training data
            self.grid_sp_score= 'f1_micro'

            # To display the error
            # st.warning(f'Error: {e}')

            st.warning('Dataset resampling failed due to high imbalance. Training to be continued without balancing.')
        
        self.prp_train_flag= True

        if option=="split":
            return(anomaly_trn_X, unsampled_X, anomaly_tst_X, anomaly_trn_y, unsampled_y, anomaly_tst_y)
        elif option=="complete":
            return(anomaly_trn_X, unsampled_X, anomaly_trn_y)

    def preprocess_test(self, anomaly_tst_X):

        if (self.prp_train_flag):
            
            # 'id' column is set, if any 
            if 'id' in anomaly_tst_X.columns:
                anomaly_tst_X=anomaly_tst_X.set_index('id')

            if len(self.drop_cols)>=1:
                anomaly_tst_X=anomaly_tst_X.drop(self.drop_cols,axis=1)

            for col in self.obj_cols:
                anomaly_tst_X[col] = anomaly_tst_X[col].astype(str)

            for col in self.non_obj_cols:
                anomaly_tst_X[col] = pd.to_numeric(anomaly_tst_X[col])

            # Impute Test Dataset Values
            anomaly_tst_X=self.df_imputer.transform(anomaly_tst_X)
                        
            return(anomaly_tst_X)
        
        error_message='Preprocess train method [preprocess_train()] needs to be called before calling Preprocess test method [preprocess_test()] !'
        raise RuntimeError(error_message)
    
    def get_feature_imp(self, anomaly_trn_X):

        # Find positive important features of supervised model
        self.featureImp = pd.DataFrame(
              list(zip(anomaly_trn_X.columns, self.sp_model.feature_importances_)),
              columns=["Feature", "Importance"]).sort_values(by="Importance", ascending=False)

        self.featureImp['Cumulative_Importance'] = np.cumsum(self.featureImp['Importance'])
        self.featureImp['Index']=list(range(1,len(self.featureImp)+1))
        self.featureImp=self.featureImp.set_index('Index')

    def model_train(self, anomaly_trn_X, unsampled_X, anomaly_trn_y, unsampled_y):

        # Supervised GridSearch Model
        sp_param_grid = {
            'learning_rate': [0.03, 0.1],
            'depth':[4, 6, 10],
            'l2_leaf_reg': [3, 7, 11]
        } 

        grid_sp_model=GridSearchCV(self.sp_model, param_grid=sp_param_grid, cv=4, scoring= self.grid_sp_score, refit=True) #n_jobs=-1
        
        grid_sp_model.fit(anomaly_trn_X, anomaly_trn_y, cat_features=self.string_cols,verbose=False)
        self.sp_params= grid_sp_model.best_params_
        self.sp_model= grid_sp_model.best_estimator_

        # Obtain encoded value for normal label
        self.normal_num= self.label_num_dict[self.normal_label]
        # Creating normal and anomaly labels for dataset
        labelled_anomalies= [1 if threat==self.normal_num else -1 for threat in unsampled_y ]

        # Unsupervised
        usp_param_grid = {'contamination': [0.01, 0.03, 0.06, 0.09, 'auto']}

        grid_usp_model= GridSearchCV(self.usp_model, param_grid=usp_param_grid, cv=4, scoring='f1_micro')
        grid_usp_model.fit(unsampled_X[self.numerical_cols],labelled_anomalies)
        self.usp_params= grid_usp_model.best_params_
        
        self.usp_model= grid_usp_model.best_estimator_

        self.get_feature_imp(anomaly_trn_X)

        st.write('Supervised Model - Important Features (Sorted)')
        st.dataframe(self.featureImp[self.featureImp['Importance']>0].drop(['Cumulative_Importance'], axis=1))

    def model_predict(self, anomaly_tst_X):
        sp_preds= self.sp_model.predict(anomaly_tst_X)
        # st.write(self.numerical_cols)
        usp_preds= self.usp_model.predict(anomaly_tst_X[self.numerical_cols])
        
        sp_anomaly_pred= np.array([1 if pred==self.normal_num else -1 for pred in sp_preds])
        # Find positions where the predictions values disagree between the models
        pred_diff=sp_anomaly_pred-usp_preds
        
        # Threats categorised as anomalies by sp model and combined with usp model predictions
        # Any missing usp anomaly preds are added to sp model preds
        tdm_anomaly_preds=sp_anomaly_pred.copy()
        tdm_anomaly_preds[pred_diff>0]=-1

        # Combined predictions
        tdm_preds=sp_preds.copy()
        tdm_preds[pred_diff>0]= self.label_num_dict["Unknown Anomaly"]
        
        return (sp_preds,usp_preds,tdm_preds,tdm_anomaly_preds)
    
    def prep_train_score(self,anomaly_trn_ds,label_col="class"):
        
        # Split Dataset and Preprocess Train Data
        anomaly_trn_X, unsampled_X, anomaly_tst_X, anomaly_trn_y, unsampled_y, anomaly_tst_y= self.\
        preprocess_train(anomaly_trn_ds, option="split",label_col= label_col)
        
        self.sp_model.fit(anomaly_trn_X, anomaly_trn_y, cat_features=self.string_cols,verbose=False)
        # Get important features to distinguish unimportant features and remove them
        self.get_feature_imp(anomaly_trn_X)

        # Finding all unimportant features to be dropped (Only keeping important features that 
        # contribute 90% to prediction)
        unimp_feats= list(self.featureImp[self.featureImp['Cumulative_Importance']>90]['Feature'])

        # Dropping unimportant features if any
        if len(unimp_feats)>=1:
            anomaly_trn_X=anomaly_trn_X.drop(unimp_feats, axis=1)
            anomaly_tst_X=anomaly_tst_X.drop(unimp_feats, axis=1)
            unsampled_X= unsampled_X.drop(unimp_feats, axis=1)

        # Update object and non object columns lists after dropping unimportant features 
        self.obj_cols= [col for col in self.obj_cols if col not in unimp_feats]
        self.non_obj_cols= [col for col in self.non_obj_cols if col not in unimp_feats]
        
        # Update string and numerical column names
        self.numerical_cols= list(anomaly_trn_X.select_dtypes(exclude='O').columns)
        self.string_cols= list(anomaly_trn_X.select_dtypes(include='O').columns)

        # Tune Hyperparameters and Train model
        self.model_train(anomaly_trn_X, unsampled_X, anomaly_trn_y, unsampled_y)
        
        # Preprocess Test
        anomaly_tst_X= self.preprocess_test(anomaly_tst_X)

        # Replace unknown test labels with 'Unknown Anomaly'
        anomaly_tst_y.map(lambda threat: 'Unknown Anomaly' if threat not in self.le.classes_ else threat)
        # Use LabelEncoder to encode string labels  to numericals
        anomaly_tst_y= self.le.transform(anomaly_tst_y)
        
        # Obtain predictions for Test Set
        sp_preds,usp_preds,tdm_preds,tdm_anomaly_preds= self.model_predict(anomaly_tst_X)

        # Compute Performance Metrics and store them
        self.train_preformance =performance_metrics(anomaly_tst_y, sp_preds, usp_preds, tdm_preds, tdm_anomaly_preds, self.normal_num)

        # Retrain model on complete dataset after obtaining accuracy on test split

        # Preprocess Complete Train Dataset
        anomaly_trn_X, unsampled_X, anomaly_trn_y= self.preprocess_train(anomaly_trn_ds, option="complete", label_col=label_col)
        # Retrain both models
        self.sp_model.fit(anomaly_trn_X, anomaly_trn_y, cat_features=self.string_cols,verbose=False)
        self.usp_model.fit(unsampled_X[self.numerical_cols])

def performance_metrics(anomaly_tst_y, sp_preds, usp_preds, tdm_preds, tdm_anomaly_preds, normal_num):

    # Initialize dataframe to store performance metrics
    performance_df={'Model Type':[], 'Classification':[], 'Accuracy':[], 'Precision':[], 'Recall':[], 'F1-Score':[]}
    performance_df['Index']= list(range(1,5))

    # Create anomaly labels using threat labels
    labelled_anomalies= [1 if threat==normal_num else -1 for threat in anomaly_tst_y]
    # Create anomaly labels using supervised prediction labels
    sp_anomaly_preds= [1 if threat==normal_num else -1 for threat in sp_preds]

    performance_df['Model Type'].append('CatBoost Classifier')
    performance_df['Classification'].append('Threat (MC)')
    performance_df['Accuracy'].append(accuracy_score(anomaly_tst_y, sp_preds))
    performance_df['Precision'].append(precision_score(anomaly_tst_y, sp_preds,average='micro'))
    performance_df['Recall'].append(recall_score(anomaly_tst_y, sp_preds,average='micro'))
    performance_df['F1-Score'].append(f1_score(anomaly_tst_y, sp_preds,average='micro'))

    performance_df['Model Type'].append('Isolation Forest')
    performance_df['Classification'].append('Anomaly (BC)')
    performance_df['Accuracy'].append(accuracy_score(labelled_anomalies, usp_preds))
    performance_df['Precision'].append(precision_score(labelled_anomalies, usp_preds))
    performance_df['Recall'].append(recall_score(labelled_anomalies, usp_preds))
    performance_df['F1-Score'].append(f1_score(labelled_anomalies, usp_preds))

    performance_df['Model Type'].append('CatBoost Classifier')
    performance_df['Classification'].append('Anomaly (BC)')
    performance_df['Accuracy'].append(accuracy_score(labelled_anomalies, sp_anomaly_preds))
    performance_df['Precision'].append(precision_score(labelled_anomalies, sp_anomaly_preds))
    performance_df['Recall'].append(recall_score(labelled_anomalies, sp_anomaly_preds))
    performance_df['F1-Score'].append(f1_score(labelled_anomalies, sp_anomaly_preds))

    performance_df['Model Type'].append('Tandem Model')
    performance_df['Classification'].append('Anomaly (BC)')
    performance_df['Accuracy'].append(accuracy_score(labelled_anomalies, tdm_anomaly_preds))
    performance_df['Precision'].append(precision_score(labelled_anomalies, tdm_anomaly_preds))
    performance_df['Recall'].append(recall_score(labelled_anomalies, tdm_anomaly_preds))
    performance_df['F1-Score'].append(f1_score(labelled_anomalies, tdm_anomaly_preds))

    # Converting dictionary to Dataframe
    performance_df= pd.DataFrame(performance_df)
    # Setting Index
    performance_df= performance_df.set_index('Index')

    st.dataframe(performance_df)

    st.text('MC - Multiclass Classification | BC - Binary Classification')
    st.markdown('The results of Supervised Anomaly Classification are also shown in order\
     to compare them to the results of Tandem Anomaly Classification')

    return (performance_df)

def under_sampling_strat(y_data,label_num_dict,normal_label):

    # Obtain unique values and its counts from numpy array
    unique_labels, label_counts = np.unique(y_data, return_counts=True)

    # Calculate approx mean of label counts 
    # [Normal label count is capped at max of anomalous label]
    temp_np=np.delete(label_counts,label_num_dict[normal_label]-1)
    label_mean= int((sum(temp_np)+max(temp_np))/len(label_counts))

    # Create undersampling dict using key and value lists
    u_smp_keys=list(unique_labels[label_counts>label_mean])
    u_smp_vals=[label_mean]*len(u_smp_keys)
    u_smp_dic=dict(zip(u_smp_keys,u_smp_vals))
    return (u_smp_dic)

# Function to plot Confusion Matrix   
def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    
    # Setting plot attributes
    plt.clf()
    plt.cla()
    plt.figure(figsize=(8,9))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
        
    # Plotting confusion matrix 'cm' elements 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    
    # Setting x-axis and y-axis labels 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    fig= plt.gcf()

    st.pyplot(fig)
    #plt.figure(figsize=(8,8))

@st.cache
def load_data(f_name, header_option,col_names,na_val):

    if header_option=='manual':
        loaded_ds=pd.read_csv(f_name,names=col_names,na_values=na_val)
    elif header_option=='auto':
        loaded_ds=pd.read_csv(f_name,na_values=na_val)
    return (loaded_ds)

# Phase 1
def initial_user_input():

    # Set default values
    selected_ds,na_val,label_col,col_names,md_name=('—','?','—','—','—')
    normal_label='normal'
    header_option='manual'
    submit_button=False

    # Options to specify dataset type, w/ column headers
    df_options=['Dataset with separate Column Headers','Dataset with integrated Column Headers']
    selected_df_option = st.selectbox('Choose Dataset Option', df_options)

    # List current directory file list and create select box 
    folder_path='.'
    filenames = ['—']
    filenames.extend(os.listdir(folder_path))
    selected_ds = st.selectbox('Select a Dataset to train the model', filenames)

    # Verify if DS selected is a csv file
    if selected_ds!='—':

        # Option with Separate Column Headers
        if selected_df_option==df_options[0]:

            header_option= 'manual'

            col_names=[]

            selected_column_txt = st.selectbox('Select the Column Text file', filenames)
            # Verify if Column file selected is a txt file
            if selected_column_txt!='—':
                with open(selected_column_txt) as cols_fp:
                    for line_num, name in enumerate(cols_fp):
                        col_names.append(name.rstrip())

                # Default na value
                na_val=['']
                # Input and store na_val
                na_inp = st.text_input("Enter missing value characters of DS without separation, if any")
                na_val.extend(list(na_inp))

                # Wait for time(s) to input missing vals
                label_col_opts=['—']
                label_col_opts.extend(col_names[::-1])
                label_col = st.selectbox('Select the Label Column name', label_col_opts)
                normal_label= st.text_input('Enter normal label','normal')
                md_name= st.text_input('Enter custom name to save model','—')

        # Option with separate Column Headers
        else:
            # Store header option
            header_option= 'auto'

            # Default na value
            na_val=['']
            # Input and store na_val
            na_inp = st.text_input("Enter missing value characters of DS without separation, if any")
            na_val.extend(list(na_inp))

            anomaly_trn_ds= pd.read_csv(selected_ds,nrows=0) #na_values=na_val

            label_col_opts=['—']
            label_col_opts.extend(list(anomaly_trn_ds.columns)[::-1])
            label_col = st.selectbox('Select the Label Column name', label_col_opts)
            normal_label= st.text_input('Enter normal label','normal')
            md_name= st.text_input('Enter custom folder name to save model after training','—')

    if md_name!='—':
        if md_name in os.listdir(md_save_folder):
            st.error('Please enter a new folder name')
        else:
            submit_button = st.button('Submit Configuration Options')

    return (selected_ds,na_val,label_col,header_option,col_names,normal_label,md_name,submit_button)

def train_ensemble_model(selected_ds,na_val,label_col,header_option,col_names,normal_label,md_name,submit_button):

    with st.spinner('Loading Dataset...'):

        # Load training data
        anomaly_trn_ds= load_data(selected_ds,header_option,col_names,na_val)  

        # Sample a small portion of data for testing the program
        # anomaly_trn_ds=anomaly_trn_ds.sample(100000)

    with st.spinner('Training in progress...'):
        td_mod=tandem_model(model_name= md_name,normal_label= normal_label)
        td_mod.prep_train_score(anomaly_trn_ds, label_col)

    with st.spinner('Saving Model...'):
        if md_name not in os.listdir(md_save_folder):
            os.mkdir(f'{md_save_folder}\\{md_name}')
            pickle.dump(td_mod, open(f'{md_save_folder}\\{md_name}\\tdm_model.sav', 'wb'))

def reset_session_states():

    # Removing session state variables if they exist to reset them
    if 'saved_model' in st.session_state:
        del st.session_state['saved_model']

    if 'anomaly_tst_ds' in st.session_state:
        del st.session_state['anomaly_tst_ds']

    if 'tdm_preds' in st.session_state:
        del st.session_state['tdm_preds']

    if 'label_color_dict' in st.session_state:
        del st.session_state['label_color_dict']

    if 'def_choice' in st.session_state:
        del st.session_state['def_choice']


def test_predict(md_save_folder,option='test'):

    # option 'test' is for model testing
    # option 'predict' is for model prediction

    # Set default values
    submit_button=False

    # Obtain Saved model names
    model_names = ['—']
    model_names.extend(os.listdir(f'.\\{md_save_folder}')) 

    if len(model_names)>1:
        # User Input for saved models
        selected_model = st.selectbox('Select the saved model to be used', model_names)

        # List current directory file list and create select box 
        filenames = ['—']
        filenames.extend(os.listdir('.'))
        selected_ds = st.selectbox('Select a Dataset to test the model', filenames)

        # Options to specify dataset type, w/ column headers
        df_options=['Dataset with Column Headers','Dataset without Column Headers']
        selected_df_option = st.selectbox('Choose Dataset Option', df_options)

        # Default na value
        na_val=['']
        # Input and store na_val
        na_inp = st.text_input("Enter missing value characters of DS without separation, if any")
        na_val.extend(list(na_inp))

        # Separate submit text for test and predict
        if option=='test':
            button_text = 'Submit Configuration and Test'
        else:
            button_text= 'Submit Configuration and Predict'

        submit_button = st.button(button_text)

        # Submit button to start testing
        if (submit_button):
            st_time= time.time()
            if (selected_ds!='—' and selected_model!='—'):
                with st.spinner('Loading Model...'):
                    # Load saved model
                    saved_model= pickle.load(open(f'{md_save_folder}\\{selected_model}\\tdm_model.sav', 'rb'))
                    st.session_state["saved_model"]=saved_model

                with st.spinner('Loading Dataset...'):

                    if option=='test':
                        # Testing ds has label column
                        ds_cols= saved_model.columns
                    else:
                        # Prediction ds does not have label column
                        ds_cols= list(saved_model.columns)
                        ds_cols.remove(saved_model.label_col)

                    # Load test dataset
                    if selected_df_option==df_options[0]:
                        anomaly_tst_ds=load_data(selected_ds,'auto',ds_cols,na_val)
                    else:
                        anomaly_tst_ds=load_data(selected_ds,'manual',ds_cols,na_val)

                if option=='test':
                    # Separate Features and Label Column
                    anomaly_tst_X,anomaly_tst_y= anomaly_tst_ds.drop([saved_model.label_col],axis=1),anomaly_tst_ds[saved_model.label_col]
                else:
                    # Use ds as is
                    anomaly_tst_X= anomaly_tst_ds

                with st.spinner('Test Data Preprocessing...'):

                    # Preprocess Test Dataset
                    anomaly_tst_X= saved_model.preprocess_test(anomaly_tst_X)

                    if option=='test':

                        # Replace unknown test labels with 'Unknown Anomaly'
                        anomaly_tst_y.map(lambda threat: 'Unknown Anomaly' if threat not in saved_model.le.classes_ else threat)

                        # Use LabelEncoder to encode string labels  to numericals
                        anomaly_tst_y= saved_model.le.transform(anomaly_tst_y)

                with st.spinner('Prediction in progress...'):
                    # Obtain predictions for Test Set
                    sp_preds, usp_preds, tdm_preds, tdm_anomaly_preds= saved_model.model_predict(anomaly_tst_X)
                
                with st.spinner('Performance Evaluation...'):

                    if option=='test':

                        _= performance_metrics(anomaly_tst_y, sp_preds, usp_preds, tdm_preds, tdm_anomaly_preds, saved_model.normal_num)

                    else:

                        # Obtain prediction labels
                        tdm_preds=np.vectorize(saved_model.num_label_dict.get)(tdm_preds)
                        # Store tandem predictions in session state
                        st.session_state["tdm_preds"]=tdm_preds
                        # Store test dataset in session state
                        st.session_state['anomaly_tst_ds']= anomaly_tst_ds

                        # Ensure reset of var if menu options haven't changed
                        if 'label_color_dict' in st.session_state:
                            del st.session_state['label_color_dict']

                        if 'def_choice' in st.session_state:
                            del st.session_state['def_choice']

                        exec_time= time.time() - st_time
                        formatted_time= datetime.timedelta(seconds=exec_time) 

                        st.success(f"Predictions obtained in {str(formatted_time)} (h : m : s)!")
                        # View predictions under 'Results' or explore other visualization options"
                        st.markdown("Predictions can be viewed using the 'Results' option")
                        st.markdown("Feature-Threat frequency plots can be viewed by selecting the\
                         'Feature-Threat Visualization' option")

            else:
                st.error('Please ensure the Model and Dataset are selected')

    else:
        st.error('No models found ! Train and save models prior to testing/ prediction.')

def highlight_row(s, label_color_dict):

    color_val= label_color_dict[s.Predictions]
    return [f'background-color: {color_val}']*len(s)

def prediction_visualization(viz_option='results'):

    # Check if tdm_preds and saved_model session states variables exist
    if all (key in st.session_state for key in ("tdm_preds","saved_model","anomaly_tst_ds")):

        saved_model= st.session_state['saved_model']
        tdm_preds= st.session_state['tdm_preds'].copy()
        preds_ds= st.session_state['anomaly_tst_ds'].copy()

        # Prediction Dataset
        preds_ds['Predictions']= tdm_preds

        if viz_option=='results':

            st.header('Predictions Pie-Chart')
            # Display prediction-threat histogram counts
            threat_plots(preds_ds, option='Predictions', normal_label= saved_model.normal_label)

            if 'label_color_dict' not in st.session_state:
                # Obtain list of threat labels
                labels= list(saved_model.label_num_dict.keys())

                # Hex Value of Colors
                # The list of colors is limited to the below collection,
                # program needs to be updated to handle the color selection
                # if count of labels is greater than collection count
                color_hex= ['#D9CED5', '#C9E3C3', '#D4D8D2', '#CAE1Df', 
                '#C7C1C6', '#D1E4DD', '#C3CAE5', '#EDA2A2', '#FFAD99',
                '#FF9872', '#D7DF01', '#FFBF00', '#01DFD7', '#F7FE2E']

                # Shuffle Colors
                random.shuffle(color_hex)

                # Label Color Dictionary mapping
                label_color_dict= dict(zip(labels,color_hex[:len(labels)]))
                label_color_dict[saved_model.normal_label]= 'white'

                # Store label_color_dict in session state
                st.session_state['label_color_dict']= label_color_dict

            else:
                label_color_dict= st.session_state['label_color_dict']

            st.header('Predictions')

            # Dataframe Columns
            columns= saved_model.columns.copy()
            columns.remove(saved_model.label_col)

            # Important Features
            imp_feats= list(saved_model.featureImp.iloc[:10,0]) #['Feature']

            # Obtain predicted labels
            pred_labels= list(preds_ds['Predictions'].unique())
            selected_labels= st.multiselect(f'Select prediction labels to be displayed (All selected by default)',options= pred_labels, default= pred_labels)

            # Filtering dataset on 'selected_labels'
            filtered_preds= preds_ds[preds_ds.apply(lambda row: row['Predictions'] in selected_labels, axis=1)]

            # Input to select columns to display
            selected_cols= st.multiselect(f'Select columns to be displayed (Top {len(imp_feats)} important columns selected by default)',options= columns, default= imp_feats)

            # Page navigator
            page_size = 100
            page_number = st.number_input(
                label="Page Number",
                min_value=1,
                max_value=ceil(len(filtered_preds)/page_size),
                step=1,
            )
            current_start = (page_number-1)*page_size
            current_end = page_number*page_size

            if selected_cols!=[] and selected_labels!=[]:
                # Filter rows and columns of Dataframe to be displayed
                df_to_display= filtered_preds.iloc[current_start:current_end].loc[:,selected_cols+['Predictions']]

                # Display dataframe with rows highlighted according to the threats
                st.dataframe(df_to_display.style.apply(highlight_row, label_color_dict= label_color_dict, axis=1))

            else:
                st.error('No columns selected !')

        elif viz_option=='feat-threat':

            st.header('Feature-Threat Visualization')
            # Display column-threat histogram counts
            threat_plots(preds_ds, option= 'Manual', normal_label= saved_model.normal_label)

        else:
            st.error('Invalid Choice')

    else:
        st.error("Visualization can only be called after 'Configure Dataset Options and Predict'")

def threat_plots(dataset, option= 'Manual', normal_label= 'normal'):
    
    # Initialize vals
    selected_column= '—'
    selected_threats=[]

    if option=='Predictions':

        selected_column= 'Predictions'
        pred_val_counts= dataset['Predictions'][dataset.Predictions!=normal_label].value_counts()

        # Find threats and threat_counts
        threats= list(pred_val_counts.index)
        threat_counts= list(pred_val_counts)

        plt.figure()
        patches = plt.pie(threat_counts,radius=.9,counterclock=False)

        # Obtain percentage labels for piechart legend
        threat_counts=np.array(threat_counts)
        threat_counts_percent=(threat_counts/sum(threat_counts))*100
        labels=[(str(name)+ f' ({req:4.2f}%)') for (name,req) in zip(threats,threat_counts_percent)]

        # Create plot legend
        plt.legend(labels, title ='Threat Labels:', fontsize=10, bbox_to_anchor=(.85,.81))

        plt.tight_layout()

        fig= plt.gcf()

        st.pyplot(fig)

        pred_val_counts= pd.DataFrame({'Index':range(1,len(pred_val_counts)+1), selected_column: threats,
                                            'Counts':threat_counts}).set_index('Index')
        st.subheader('Prediction Frequencies')
        st.dataframe(pred_val_counts)

    elif option=='Manual':

        col_opts= ['—']
        col_opts.extend(list(dataset.columns))
        selected_column= st.selectbox('Select the column to be inspected',col_opts)

        threat_opts= list(dataset.Predictions.unique())
        default_opts= list(threat_opts)
        default_opts.remove(normal_label)

        if 'def_choice' not in st.session_state:
            st.session_state['def_choice']= random.choice(default_opts)

        selected_threats= st.multiselect('Select the predicted threats to be filtered on',options= threat_opts, default= st.session_state['def_choice'])
        
        if selected_column!= '—' and selected_threats!=[]:
            # Filtering Columns using 'selected_columns'
            # Filtering Rows using 'selected_threats'
            pred_val_counts= dataset[selected_column][dataset.apply(lambda row: row['Predictions'] in selected_threats, axis=1)].value_counts()

            # Find threats and threat_counts
            threats= np.array(pred_val_counts.index) #.astype('str')
            threat_counts= np.array(pred_val_counts)

            # Sort the threat counts based on decreasing order
            sorted_indexes= threat_counts.argsort()[::-1]

            # Reorder values based on 'sorted_indexes'
            threats= threats[sorted_indexes]
            threat_counts= threat_counts[sorted_indexes]

            # Page navigator
            threat_size = 20
            threat_pg_number = st.number_input(
                label="Bar Chart Number",
                min_value=1,
                max_value=ceil(len(threats)/threat_size),
                step=1,
            )
            current_start = (threat_pg_number-1)*threat_size
            current_end = threat_pg_number*threat_size

            chart_data = pd.DataFrame({
                'Label': threats,
                'Counts': threat_counts
            }).iloc[current_start:current_end]

            bar_chart= alt.Chart(chart_data).mark_bar(tooltip=True).encode(
                x= 'Counts:Q',
                y= alt.Y('Label', type='nominal', sort=None)
            ).interactive()

            st.altair_chart(bar_chart, use_container_width=True)

            pred_val_counts= pd.DataFrame({'Index':range(1,len(pred_val_counts)+1), 
                                                selected_column: threats,
                                                'Counts':threat_counts}).\
                                                set_index('Index')#.sort_values(by="Counts", ascending=False)
            
            st.subheader('Feature-Threat Frequencies')                                 
            st.dataframe(pred_val_counts)

def train_test_performance():

    # Obtain Saved model names
    model_names = ['—']
    model_names.extend(os.listdir(f'.\\{md_save_folder}')) 

    if len(model_names)>1:
        # User Input for saved models
        selected_model = st.selectbox('Select the saved model to be used', model_names)

        if selected_model!= '—':

            with st.spinner('Loading Model...'):
                saved_model= pickle.load(open(f'{md_save_folder}\\{selected_model}\\tdm_model.sav', 'rb'))

            performance_df= saved_model.train_preformance

            st.dataframe(performance_df)

            st.text('MC - Multiclass Classification | BC - Binary Classification')
            st.markdown('The results of Supervised Anomaly Classification are also shown in order\
             to compare them to the results of Tandem Anomaly Classification')

    else:
        st.error('No models found ! Train and save models prior to viewing performance metrics.')

def about_us():

    st.header('Information Guide')

    st.subheader('1) Train Option')
    st.markdown('**Function:** To train and save the model on a specified dataset.')
    st.markdown('**Info:** The program only supports labelled datasets with threats and normal instances \
         to train the machine learning models. The model assumes the training dataset is highly imbalanced \
         where the normal class count >>> anomaly/ threat class count. The dataset must be in csv \
         format and could have integrated column headers or separate\
         column headers (text file). Certain additional info regarding the dataset such as the\
         label column and the missing value (na) character should also be specified to correctly load\
         and infer data before training the model.')

    st.subheader('2) Train-Test Performance')
    st.markdown('**Function:** To view the performance of the model on the Testing split (20%)\
        of the Training set')
    st.markdown('**Info:** The model is selected from the list of saved models to view the performance\
        metrics')

    st.subheader('3) Test Option')
    st.markdown('**Function:** To test the saved models on a specified dataset and obtain\
        performance metrics.')
    st.markdown('**Info:** The program requires labelled datasets for testing the models.\
         It must also have the same features as the training dataset.\
         The dataset must be in csv format and could have integrated column headers or separate\
         column headers (text file). Certain additional info regarding the dataset such as the\
         missing value (na) character should also be specified to correctly load\
         and infer data before testing the model.')

    st.subheader('4) Threat/ Anomaly Prediction Option')
    st.markdown('**Function:** To obtain predictions and its visualization using the saved models on\
        a specified dataset.')
    st.markdown('**Info:** The dataset provided should have the same features as the training \
         dataset and does not require the label column (to be predicted).\
         The dataset must be in csv format and could have integrated column headers or separate\
         column headers (text file). Certain additional info regarding the dataset such as the\
         missing value (na) character should also be specified to correctly load and infer \
         data before making predictions. Visualization can only be called after configuring\
         the input dataset and obtaining the predictions.')
    
def main():


    global md_save_folder

    md_save_folder= 'Saved_Models'

    if md_save_folder not in os.listdir():
        os.mkdir(md_save_folder)

    st.title('CyberThreat Hunting using AI')

    nav_options= ["Information Guide", "Train Model", "Train-Test Performance", "Test Model", "Threat/ Anomaly Prediction"]

    sidebar_box= st.sidebar.selectbox("Navigation Bar", nav_options)

    if sidebar_box==nav_options[0]:
        about_us()

    if sidebar_box==nav_options[1]:

        st.write('Model Train')

        selected_ds,na_val,label_col,header_option,col_names,normal_label,md_name,submit_button= initial_user_input()

        if submit_button:
            st_time= time.time()
            train_ensemble_model(selected_ds,na_val,label_col,header_option,col_names,normal_label,md_name,submit_button)
            exec_time= time.time() - st_time
            formatted_time= datetime.timedelta(seconds=exec_time)    
            st.success(f"Model training completed in {str(formatted_time)} (h : m : s). Model has been saved under directory \'.\\{md_save_folder}\\{md_name}\'")

    if sidebar_box==nav_options[2]:

        st.write('Performance metrics on Test split (20%) of Training dataset')
        train_test_performance()

    if sidebar_box==nav_options[3]:

        st.write('Model Test')
        test_predict(md_save_folder,option='test')

    if sidebar_box==nav_options[4]:
        radio_opts= ['Configure Dataset Options and Predict', 'Results (Predictions)', 'Feature-Threat Visualization']
        selected_radio= st.sidebar.radio('Follow the sequence', radio_opts)

        if selected_radio==radio_opts[0]:
            st.write('Dataset Configuration and Model Prediction')
            test_predict(md_save_folder,option='predict')

        elif selected_radio==radio_opts[1]:
            prediction_visualization(viz_option='results')

        elif selected_radio==radio_opts[2]:
            prediction_visualization(viz_option='feat-threat')

        else:
            st.error('Invalid Choice')

    else:
        # Reset session state variables
        reset_session_states()

if __name__ == "__main__":
    main()