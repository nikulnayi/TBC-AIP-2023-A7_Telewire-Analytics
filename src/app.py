import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
from pathlib import Path
import plotly.express as px
import seaborn as sns

# from preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import dill
from xgboost import XGBClassifier
from sklearn import set_config
from predict import predict

project_dir = Path(__file__).resolve().parents[1]
with open(project_dir.joinpath('notebooks/pipelines/PreprocessingPipeline2.0.pkl'), 'rb') as f:
    pipeline, categories = dill.load(f)
with open(project_dir.joinpath('models/RandomForestClassifier.pkl'), 'rb') as f:
        model = dill.load(f)
# top_n_features = []
st.set_page_config(
    page_title="Telewire Dashboard",
    page_icon="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTeqOxnyNX_UCarYAKSkIsY7CWQZlBzULPyMg&usqp=CAU",
)
with st.sidebar:
    
    st.header("Upload Data file here for prediction")

    file = st.file_uploader("", type=["csv"])
    options = ["Company Management", "Data Science", "Uploaded Data"]
    selected_option = st.radio("Who is viewing?", options)
col1,col2 = st.columns([0.75,6])

# with col2:
st.title("Telewire Analytics")

if file is not None:

    df = pd.read_csv(file,encoding= 'unicode_escape')
    df = df.drop(["Unusual"], axis=1)
    temp_df = df.copy()
    prediction = predict(temp_df)
    df['Unusual'] = prediction  
        

    if selected_option == 'Company Management':
        with st.spinner('Processing...'):
            
            # Once the computation is done, remove the spinner
                # st.success('Processing done!')
                
                count_normal = df[df['Unusual']==0]['Unusual'].count()
                count_abnormal = df[df['Unusual']==1]['Unusual'].count()

                # new_data = pd.DataFrame({'No of Normal/Abnormal':[count_normal,count_abnormal]},index=['Count of Nomal Tower','Count of Abnormal Tower'])
                # st.dataframe(new_data)

                # create the columns 
                kpi1,kpi2 = st.columns(2)
                kpi1.metric(
                    label="Count of Nomal Cell Tower",
                    value=count_normal
                )
                kpi2.metric(
                    label="Count of Abnormal Cell Tower",
                    value=count_abnormal
                )
                col1,col2=st.columns(2)
                with col1:
                    # create a pie chart with the data
                    st.markdown("Percentage share of Usual and Unusual")
                    sizes = [count_normal,count_abnormal]
                    explode = (0, 0.1)
                    fig1, ax1 = plt.subplots()
                    labels=df['Unusual'].replace({0:"Usual",1:"Unusual"}).unique()

                    ax1.pie(sizes, labels=labels[::-1],explode=explode, autopct='%1.1f%%', startangle=90)
                    ax1.axis('equal')

                    st.pyplot(fig1)
                with col2:
                    # scatter plot CellName vs Time
                    Unusual = df[df['Unusual']==1]
                    Unusual = Unusual[['Time','CellName','Unusual']]
                    st.write('Below is the list of unusual acitivties predicted. \
                        Kindly take a look and consider re-configuration of the Cell Towers')
                    
                    Unusual = (Unusual.drop('Unusual',axis=1)).sample(100)
                    c = alt.Chart(Unusual).mark_circle(size=40).encode(
                            x='CellName', y='Time').properties(
                            width=700,
                            height=300
                            )
                    st.altair_chart(c, use_container_width=True)




                # Bar graph
                    
                st.markdown("Count of Cell Tower with respect to its behavior")
                    # group by 'Cell Name' and count the number of occurrences of 0 and 1

                temp_df['Unusual']=df['Unusual'].replace({0:"Usual",1:"Unusual"})
                counts = temp_df.groupby(['CellName', 'Unusual'])['Unusual'].count()
                    
                    # convert counts to a DataFrame and unstack the 'Status' index level
                counts_df = counts.unstack(level='Unusual', fill_value=0)
                    
                st.bar_chart(counts_df)

                # feature importance graph
                # def slider_callback():
                     
                    
                # n = st.slider('Parameters', 1, 13,3,key="myslider")
                
                x = df.drop(['Time','CellName','maxUE_UL+DL','Unusual'],axis=1)
                y = df['Unusual']
                dict_parameter = {
                     "meanUE_UL":"Average number of user equipment (UE) devices that were simultaneously active during the last 15 minutes. UL means communication from User to CellTower.",
                     "meanUE_DL":"Average number of user equipment (UE) devices that were simultaneously active during the last 15 minutes. DL means communication from CellTower to User.",
                     "PRBUsageDL":"Level of resource utilization in that cell measured as the portion of Physical Radio Blocks (PRB) that were in use (%) in the previous 15 minutes",
                     "PRBUsageUL":"Level of resource utilization in that cell measured as the portion of Physical Radio Blocks (PRB) that were in use (%) in the previous 15 minutes",
                     "maxThr_UL":"Maximum carried traffic (in Mbps) measured in the last 15 minutes.",
                     "maxThr_DL":"Maximum carried traffic (in Mbps) measured in the last 15 minutes.",
                     "meanThr_UL":"Average carried traffic (in Mbps) during the past 15 minutes.",
                     "meanThr_DL":"Average carried traffic (in Mbps) during the past 15 minutes",
                     "maxUE_DL":"Maximum number of user equipment (UE) devices that were simultaneously active during the last 15 minutes.",
                     "maxUE_UL":"Maximum number of user equipment (UE) devices that were simultaneously active during the last 15 minutes."
                }
                model.fit(x,y)
                if "myslider" not in st.session_state:
                     st.session_state.myslider = 3
                # Create a slider widget and specify the callback function
                n = st.slider("Select number of top parameters to display", min_value=1,value=st.session_state.myslider, max_value=len(x.columns),step=1,)
                feat_importances = pd.Series(model.named_steps['RandomForestClassifier'].feature_importances_, index=x.columns)
                top_n_features = feat_importances.nlargest(n)
                figure = px.bar(top_n_features,
                x=top_n_features.values,
                y=top_n_features.keys(), labels = {'x':'Ratio of Importance Value', 'index':'Parameter(s)'},
                text=np.round(top_n_features.values, 2),
                title= f'Top {len(top_n_features)} Parameter(s)',
                    )
                figure.update_layout({
                        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                })
                st.plotly_chart(figure)
                for i in top_n_features.keys():
                     st.write(f"**{i}:**")
                     st.write(f"{dict_parameter[i]}")



    if selected_option == 'Data Science':
        with st.spinner('Processing...'):

            st.markdown("This is a web app that uses the Cell Tower Log data to predict if it perfroms abnormal or not. The source code of this app is available on [Github](https://github.com/Anupriya-Sri/TBC-AIP-2023-A7_Telewire-Analytics)")   

            col1,col2 = st.columns(2)
            with col1:
                # Stastics Summary
                st.write("## Stastical Summary")
                st.write(pd.DataFrame(df).describe())
            with col2:
                # missing values
                st.write("## Columns with missing values")
                if (df.isna().sum()>0).any():
                    missing_value_count = df.isnull().sum()
                    columns_missing_values = missing_value_count[missing_value_count > 0].index
                    # Subset the original DataFrame with the selected columns
                    df_subset = df[columns_missing_values].isnull().sum()
                    st.dataframe(df_subset.transpose())
                else:
                    st.markdown("### There are no missing values in uploaded data")

            # heat map to see the relationship of Parameters

            st.markdown("### Heatmap to show relationship between Parameters")
        
            st.markdown("* If the value is postive, that means when one variable increases, the other variable also increases.")
            st.markdown("* If the value is negative, that means when one variable increases, the other variable also decreases.")
            st.markdown("* If the value is 0,  there is no correlation between the two variables. This means that the variables changes in a random manner with respect to each other")
            fig, ax = plt.subplots(figsize=(18,15))
            
            corr = df.corr()
            sns.heatmap(corr, ax=ax,annot=True,cmap="coolwarm",mask=np.triu(np.ones_like(corr)))

            st.pyplot(fig)
            
            
            st.markdown("### Box plot")
            st.markdown("* Identify outliers or anomalous data points")
            st.markdown("* To understand the spread/range of the data")
            # boxplot
            fig, ax = plt.subplots(figsize=(12,8))
            sns.boxplot(data=df.drop('Unusual',axis=1))
            plt.ylim(0, 80)
            # Set the axis labels and title
            ax.set_xlabel('Parameters')
            ax.set_ylabel('Distribution')
            ax.set_title('Box Plot')
            # Display the box plot in Streamlit
            st.pyplot(fig)

    if selected_option == 'Uploaded Data':
        with st.spinner('Processing...'):
            datatable = df.drop('Unusual',axis=1)
            st.markdown("### Overview of uploaded data")
            st.markdown("The following table gives you a quick view of the uploaded data.")
            st.dataframe(datatable)# will display the table
