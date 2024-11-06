# LLMRAG


This code uses a RAG LLM to analyse the data sets on consumer choice in the UK and Ireland. It relies on the LEMUR methodology for text pre-processing and on the word2vec embeddings trained on Google news data sets. The word embeddings represent three million words and phrases as 300-dimensional vectors. The cosine similarity is used as a distance metric. 

The similarity search is done both across the rows of the tables and across the questions separately. The chosen rows are then matched to the questions and the whole part of the tables associated with these questions is deemed to contain information relevant for answering the query. In the data sets, some questions contain similar answer options. These questions are tabulated across the columns and not across the rows. The parts of the tables which correspond to these questions are isolated into separate data sets and then transposed before the analysis. Furthermore, some questions have sub-questions: see Question 15, Question 16 and Question 17 in the Dataset 1. Therefore, the corresponding data chunk which is supplied into the LLM includes the parts of the table corresponding to the question and to the sub-questions. 

The Mistral model from the mistralai is used. The API key can be obtained by registering in the webpage. 

In order to obtain an answer to the query, a user should use the function makeQueries(query: str, contextWindow: int, dumyData: int). In this function, the contextWindow defines the number of the most relevant rows and questions to choose and the dumyData allows to use one data set only or both data sets for the analysis: dummyData = 1 leads to the analysis using the Dataset 1 only, dummyData = 2 leads to the analysis using the Dataset 2 only while dummyData = 3 to the use of the both datasets. The program also automatically prints out the parts of the data chosen.

THE QUERY ANSWER IS PRINTED IN THE END.


