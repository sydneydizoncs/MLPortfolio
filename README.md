# Binary Classification: Neural Network - Sentiment Analysis via Vectorization

The output I will be predicting will be the overall opinion of a certain Amazon book based on its review; whether the content of the book review contains positive connotations or negative connotations (true/false value). This is a classification problem because the target values of Good Review/Bad Review are different categories, and cannot be measured in a numeric way (i.e. no average can be taken). By taking the Positive Review column and running code that counts the values of True and the values of False, I got a distribution of 993 False and 980 True; because the values are balanced, there is no class imbalance, and therefore, no need to tamper or clean the dataset before use. Predicting whether a book review is positive or not can help companies optimize marketing efforts; they can market books with more positive reviews in order to focus selling more well-liked books, and therefore increase sales. Along with that, they can use this to filter and personalize content recommendations to users, which can increase user engagement and satisfaction.

-  Features:
The main feature would be the text content of the book reviews. Technique-wise, I can use TF-IDF or word embeddings to convert the text into numerical vectors that can be fed into the model.

-  Model Selection:
My goal is to start a with simpler model like Logistic Regression, and possibly experiment with more complex models like Random Forest or Gradient Boosting.

-  Evaluation Metric:
Since this is a classification problem, F1-score is a suitable evaluation metric, as I can use it to access the precision of the model.

-  Model Building and Validation Plan:
I will divide the dataset into training, validation, and test sets in order to train different models using the training set and evaluate their performance on the validation set; I can then select and use the best-performing model based on F1-score on the validation set.

-  Analyze and Improve Model:
In order to analyze the model and improve it, I can look at misclassified instances to identify patterns or challenges that the model struggles with. In order to fix this, I can add additional features, such as sentiment scores, or try different vectorization methods to improve model performance. 
