# Email/SMS Spam Classifier

### Software & Tools Requiremntts

1. [Github Accounts](https://github.com)
2. [GitCLI](https://git-scm.com/)
3. [Anaconda](https://www.anaconda.com/)
4. [VS Code IDE](https://code.visualstuido.com/)
5. [Streamlit Cloud](https://streamlit.io/cloud)


### Introduction (Problem Statement)

There are myriads of promotional e-mails/SMSs that people receive everyday. This project is about the building a machine learning model for classification of e-mail/SMSs as 'spam' or 'ham'. An end-to-end website is developed where the user can punch-in the content of the e-mail/SMS the user received to identify whether the e-mail/SMS is spam or not.

### Dataset Information

The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam. The dataset was acquired from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

##### Columns provided in dataset

- v1 - Target
- v2 - Text
- Unnamed:2
- Unnamed:3
- Unnamed:4

### Data Cleaning

1. Dataset was downloaded & loaded using pd.read_csv. Dataset was stored in a dataframe 'df'
2. Dataset information was analyzed to understand the datatype of the features/variables (columns).
3. Columns ('Unnamed:2', 'Unnamed:3', 'Unnamed:4') contained mostly null/nan values. So, these columns were dropped from the dataframe.
4. Column v1 was renamed to 'target' & column v2 was renamed to 'text'
5. Null values of the dataframe was analyzed. The dataframe contained no null values.
6. Data frame contained 403 duplicates. Duplicate values were dropped while retaining the original data points.

### Exploratory Data Analysis (EDA)

1. The 'target' variable's class distribution as analyzed. It was discovered that the dataset was imbalanced as the target column comprised of 87.37% 'ham' & 12.63% spam.
2. A pie chart was plotted. 
