# Email/SMS Spam Classifier

### Software & Tools Requiremntts

1. [Github Accounts](https://github.com)
2. [GitCLI](https://git-scm.com/)
3. [Anaconda](https://www.anaconda.com/)
4. [VS Code IDE](https://code.visualstuido.com/)
5. [Streamlit Cloud](https://streamlit.io/cloud)


### Introduction (Problem Statement)

There are myriads of promotional e-mails/SMSs that people receive every day. This project is about building a machine learning model for the classification of e-mail/SMSs as 'spam' or 'ham'. An end-to-end website is developed where the user can feed the content of the e-mail/SMS the user received to identify whether the e-mail/SMS is spam or not.

### Dataset Information

The SMS Spam Collection is a set of SMS-tagged messages collected for SMS Spam research. It contains one set of SMS messages in English of 5,574 messages, tagged as being ham (legitimate) or spam. The dataset was acquired from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

##### Columns provided in the dataset

- v1 - Target
- v2 - Text
- Unnamed:2
- Unnamed:3
- Unnamed:4

### Data Cleaning

1. Dataset was downloaded & loaded using pd.read_csv. The dataset was stored in a dataframe 'df'
2. Dataset information was analyzed to understand the datatype of the features/variables (columns).
3. Columns ('Unnamed:2', 'Unnamed:3', 'Unnamed:4') contained mostly null/nan values. So, these columns were dropped from the dataframe.
4. Column v1 was renamed to 'target' & column v2 was renamed to 'text'
5. Null values of the dataframe were analyzed. The dataframe contained no null values.
6. The dataframe contained 403 duplicates. Duplicate values were dropped while retaining the original data points.

### Exploratory Data Analysis (EDA)

1. The 'target' variable's class distribution as analyzed. It was discovered that the dataset was imbalanced as the target column comprised of 87.37% 'ham' & 12.63% spam.
2. A pie chart was plotted for analysis of ham/spam distribution.

![Plot 1](https://github.com/AnupamKNN/EmailSMSClassifier/blob/main/Plots/Ham_V_s_Spam_Percentage_Distribution.png)

3. A column 'num_characters' was added to the dataframe which has the count of characters per text message.
4. A column 'num_words' was added to the dataframe & 'word_tokenizer' from 'nltk' was utilized to tokenize the words in the 'text' for counting the number of words per text message.
5. A column 'num_sentences' was added to the dataframe & 'sent_tokenizer' from 'nltk' was used to tokenize the sentences for counting the number of sentences per text message.
6. Description of all the newly-added columns was deduced to analyze the statistical information of the data points of these columns.
7. In 'num_characters' column, it was found that there were outliers that contained approximately 910 characters where df[df['target']] = 0 & 224 characters where df[df'target] == 1.
8. In 'num_words' column, the outlier count was 220 words where df[df['target']] = 0 & 46 words where df[df['target]] = 1.
9. In 'num_sentences' column, the outlier count was '28' where df[df['target']] = 0 and 8 sentences where df[df['target]] = 1.
10. A histogram was plotted to analyze the distribution of 'num_characters'. It was found that spam messages contained a high count of characters in the majority of spam messages while ham messages contained fewer characters for the majority of messages.

![Plot 2](https://github.com/AnupamKNN/EmailSMSClassifier/blob/main/Plots/num_characters_hist_plot.png)

11. A histogram was plotted to analyze the distribution of 'num_words'. It was observed that spam messages had a high count of words for the majority of messages & ham messages contained fewer words for the majority of messages.

![Plot 3](https://github.com/AnupamKNN/EmailSMSClassifier/blob/main/Plots/num_words_histplot.png)

13. A pairplot was plotted for all the newly-added features keeping 'target' as the parameter.

![Plot 4](https://github.com/AnupamKNN/EmailSMSClassifier/blob/main/Plots/Pairplot.png)

14. A correlation matrix was established & plotted using heatmap.

![Plot 5](https://github.com/AnupamKNN/EmailSMSClassifier/blob/main/Plots/Correlation_heatmap.png)


### Data Preprocessing

1. A function 'transform_text' was defined for carrying-out following operations:-
   - Lowercase
   - Tokenization
   - Removing special characters
   - Removing stop words and punctuation
   - Stemming

2. After applying 'transform_text' for carrying out the above operations, all the transformed texts were stored in 'transformed_text' column in the data frame.
3. A word cloud was plotted to analyze the most frequent words used in spam messages based on the size of the words & their color in the word cloud plot.

![Plot 6](https://github.com/AnupamKNN/EmailSMSClassifier/blob/main/Plots/spam_wc.png)

4. A word cloud was plotted to analyze the most frequent words used in ham messages based on the size of the words & their color in the word cloud plot.

![Plot 6](https://github.com/AnupamKNN/EmailSMSClassifier/blob/main/Plots/ham_wc.png)

5. A bar plot was plotted to analyze the top 30 most frequent words used in spam messages.

![Plot 7](https://github.com/AnupamKNN/EmailSMSClassifier/blob/main/Plots/most_common_spam_words.png)

6. A bar plot was plotted to analyze the top 30 most frequent words used in ham messages.

![Plot 8](https://github.com/AnupamKNN/EmailSMSClassifier/blob/main/Plots/most_common_ham_words.png)

### Model Building
Note:- For model building, testing & selection, the metric 'precision_score' was considered since the dataset is imbalanced.

1. Feature 'transformed' was selected as the independent variable & stored in 'X'. The textual data was vectorized using 'TfIdfVectorizer' and stored as array in X.
2. Feature 'target' was selected as the dependent/target variable.
3. Both X and Y were split for creating training & testing datasets with 80% data reserved for training & 20% data for testing.
4. Metrics 'accuracy_score', 'confusion_matrix' & 'precision_score' were used for initial model building.
5. Three types of Naive Bayes model algorithms were utilized for preliminary training to select the best among the 3 based on the precision score. The algorithms were:
   - Guasian Naive Bayes:- Accuracy Score:- 86.94 %, Precision Score:- 50.68 &
   - Multinomial Naive Bayes:- Accuracy Score:- 97.09 %, Precision Score:- 100.00 %
   - Binomial Naive Bayes:- Accuracy Score:- 98.35 %, Precision Score:- 99.10 %
Among the three, based on the precision score, Multinomial Naive Bayes was selected.

6. Now, the for building final model, along with multinomial naive bayes, the algorithms that were selected were:-
  - LogisticRegression
  - SVC
  - DecisionTreeClassifier
  - KNeighborsClassifier
  - RandomForestClassifier
  - AdaBoostClassifier
  - BaggingClassifier
  - ExtraTreesClassifier
  - GradientBoostingClassifier
  - XGBClassifier

7. Model training was conducted & accuracy scores & precision scores were stored in a dataframe for comparison of the models' performance on the basis of which final model will be chosen. Model training was conducted in 4 stages.

8. For the first stage, the performance metrics are as follows:-

SNo.   Algorithm	                      Accuracy	  Precision
01.   KNeighborsClassifier	           90.0387 %	  100.0000 %
02.   MultinomialNB	                   95.9381 %	  100.0000 %
03.   RandomForestClassifier	         97.3888 %	  100.0000 %
04.   ExtraTreesClassifier	           97.5822 %	   98.2906 %
05.   SVC	                             97.2921 %	   97.4138 %
06.   AdaBoostClassifier	             96.1315 %	   94.5455 %
07.   LogisticRegression	             95.1644 %	   94.0000 %
08.   XGBClassifier	                   96.9052 %	   93.4426 %
09.   GradientBoostingClassifier	     95.2611 %	   92.3810 %
10.   BaggingClassifier	               95.8414 %	   86.2595 %
11.   DeccisionTreeClassifier	         93.5203 %	   83.8095 %

A categorical plot was plotted for comparison of performance metrics of all the above algorithms.

![Plot 9](https://github.com/AnupamKNN/EmailSMSClassifier/blob/main/Plots/Classifier%20Accuracy%20%26%20Precision%20Comparison.png)

9. For the 2nd stage, the 'transfrormed_text' was vectorized using 'TfIdfVectorizer' with the 'max_feature' parameter set to 3000. This means that only 3000 features (i.e. unique words) will be selected. The performance metrics are as follows:-

SNo. Algorithm	                   Accuracy_max_ft_3000	  Precision_max_ft_3000
1.	KNeighborsClassifier	            90.5222 %	            100.0000 %	
2.  MultinomialNB                       97.0986 %	            100.0000 %
3.  RandomForestClassifier          	97.4855	%                98.2759 %
4.	ExtraTreesClassifier            	97.4855 %	             97.4576 %
5.	SVC                              	97.5822 %	             97.4790 %
6.  AdaBoostClassifier              	96.0348 %	             92.9204 %
7.	LogisticRegression            	    95.8414 %	             97.0297 %
8.	XGBClassifier	                    97.1954 %	             94.3089 %
9.  GradientBosstingClassifier      	94.7776 %	             92.0000 %
10.	BaggingClassifier	                95.7447 %	             86.7188 %
10	DecissionTreeClassifier          	92.9400 %	             82.8283 %

10. For the 3rd stage, the vectorized text was scaled using MinMaxScaler, which scales the features & values to the range of (0, 1). The performance metrics are as follows:-

SNo.  Algorithm	            	 Accuracy_scaling	    Precision_scaling
1.   KNeighborsClassifier		    90.5222 %	            97.6190 %
2.   MultinomialNB                  97.8723 %	            94.6154 %
3.   RandomForestClassifier	    	97.4855 %            	98.2759 %
4.   ExtraTreesCassifier	        97.4855 %	            97.4576 %
5.   SVC	                        96.6151 %            	92.5620 %
6.	 AdaBoostClassifier	            96.0348	%               92.9204 %
7.   LogisticRegression          	96.7118 %	            96.4286 %
8.	 XGBClassifier              	97.1954 %            	94.3089 %
9.	 GradientBoostingClassifier  	94.7776 %	            92.0000 %
10.  BaggingClassifier          	95.7447 %	            86.7188 %
11.	 DecisionTreeClassifier	        92.7466 %	            81.1881 %

11. For the 4th stage, along with the 'transformed_text', 'num_characters' feature was also included for training the model. The performance metrics are as follows:-  

SNo.  Algorithm	                Accuracy_num_chars	Precision_num_chars
1.	KNeighborsClassifier	        93.4236 %	           82.4074 %
2.	MultinomialNB                	94.1006 %	          100.0000 %
3.	RandomForestClassifier	        96.8085 %	           98.1651 %
4. 	ExtraTreesClassifier	        98.0658 %	           97.5806 %
5.  SVC	0.972921	                86.6538 %	            0.0000 %
6.	AdaBoostClassifier	            96.4217 %	           93.1624 %
7.  LogisticRegression	            96.1315 %	           96.2264 %
8. 	XGBClassifier                	97.0986 %	           94.2623 %
9.	GradientBosstingClassifier	    95.1644 %	           93.1373 %
10.	BaggingClassifier	            96.6151	%              89.9225 %
11.	DecisionTreeClassifier	        94.3907	%              87.7358 %

Considering all the stage of model training & testing & analyzing the results, the algorithm that was selected for model building was Multinomial Naive Bayes based on the precision score of the model.
