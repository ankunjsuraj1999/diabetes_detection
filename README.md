# Diabetes Detection

* Diabetes is a horrible chronic disease that leads to a relatively lower quality of life, and can be life threatning if not diagnosed early. In this project, we work with clinical data, and train a model to predict diabetes status in a patient.
* We perform all the required analysis in three parts :
    * We will acquire and collect clinical data from UCI Machnine Learning Repository and then perform the cleaning action on it.
    * We perform statical analysis and visulise the data using data visualization methods.
    * We train a machine learning model, using the data we collected and cleaned. 


## Collect and clean the data
* In this, we collected the data related to diabetes patient and their symptoms from UCI machine learning repository.
* We then perform the data processing and cleaning action on the data collected.
* Then we exported the clean dataframe to an new csv file i.e., 'diabetes_data_clean.csv'.
* Now we can use this cleaned dataframe in the next two part of our analysis.


## Statistical Analysis of the data and visualising the data using visualization mathods
* In this, we used python libraries like, matplotlib, seaborn for data visualization and scipy.stats for statical analysis of data. Using seaborn we can plot count-plot and box-plot.
* We also analysed that whether obesity is related to diabetes or not. Similarly we also analysed whether age is related to diabetes or not. This can be achieved by using cross-tab.
* We have looked at single columns i.e., **Univariate Analysis** and also looked at the relationship between two columns i.e., **Bivariate Analysis**.
* We also contucted a statical test of difference between ages of non-diabetic patients and diabetic patients and plotted a correlation **Heatmap**.


##  Training a Machine Learning Model to predict Diabeties Patient based on symptoms
* In this part, we train machine learning models to predict diabetes. For this we use python libraries **sklearn**.
* For training the model we use **train_test_split** methods of sklearn model.
* In this part, we basically train three machine learning models:
    * Logistics Regression
    * Decission Tree Classifier
    * Random Forest Classifier
* After performing all these model, we found that the best performing model among these are **Random Forest** model with high accuracy of prediction.
