
# LLM-Based Book Search System

This is the source code for the final project of the "Foundation Models in NLP" course taught by Dr. Yadollah Yaghoobzadeh (Ex-Microsoft) and Dr. Mohammad Javad Dousti (Ex-Meta, Ex-Nvidia). The project implements an LLM-based search system using Langchain, ChromaDB, and OpenAI Embeddings.

> **Note**: This Repository is a work in progress. The code is being cleaned and organized for better accessibility.


## Project Overview

The main goal of this project is to develop a conversational book search system that can suggest suitable books to users based on their queries and preferences. The system leverages large language models (LLMs) in combination with information retrieval techniques to provide personalized book recommendations.

## Key Features

1. **Conversational Search**: The system engages in a conversational interaction with the user to understand their book preferences and requirements.
2. **Retrieval-Augmented Generation**: The system combines retrieval-based methods and LLM-based generation to provide relevant and coherent book recommendations.
3. **Persian Book Dataset**: The project utilizes a curated dataset of Persian books crawled from the "Ketabrah" website, allowing the system to cater to Persian-speaking users.
4. **Query-Document Relevance Dataset**: The team created a dataset of AI-generated queries and their relevance to the books in the dataset, which is used to evaluate the retrieval module.
5. **Hybrid Retrieval Approach**: The system employs a hybrid retrieval approach, utilizing both BM25 and semantic similarity (Self-Query Retriever) to achieve optimal performance.
6. **Human Evaluation**: The system's performance is evaluated through a human annotation process, where a human annotator compares the system's responses to those of the vanilla ChatGPT.

## Dataset

The project utilizes two main datasets:

1. **Persian Books Dataset**:
   - Crawled from the "Ketabrah" website, the dataset contains information about 4,719 Persian books.
   - The dataset includes details such as book title, author, translator, publisher, publication year, rating, number of pages, and more.
   - The dataset was preprocessed and stored in a format suitable for the retrieval module.

2. **AI-Generated Query-Document Relevance Dataset**:
   - To evaluate the retrieval module, the team created a dataset of AI-generated queries and their relevance to the books in the Persian Books Dataset.
   - The process involved:
     - Extracting keywords from the book dataset
     - Filtering and selecting a diverse set of keywords using GPT-4 and GPT-3.5
     - Generating 10 queries for each selected keyword using GPT-3.5
     - Establishing the relevance of the generated queries to the books based on the extracted keywords.

## Approach

The project's approach consists of two main components:

1. **Retrieval Module**:
   - The retrieval module employs a hybrid approach, combining BM25 and semantic similarity (Self-Query Retriever).
   - BM25 focuses on exact lexical matching between the user's input and the stored book data.
   - The Self-Query Retriever utilizes a language model to extract relevant metadata from the user's input and incorporate it into the database query.
   - An ensemble of the two retrievers, with more weight assigned to the Self-Query Retriever, achieved the best performance on the query-document relevance dataset.

2. **Chatbot Module**:
   - The chatbot module is responsible for suggesting books based on the user's input and the top results from the retrieval module.
   - The module combines the retrieved book information with its own knowledge to provide personalized book recommendations.
   - An interesting observation was that the chatbot module often provided better responses than the retrieval-only approach, even when the requested book was not present in the retrieved bookset.

## Experiments and Evaluation

The project's experiments and evaluation involved two main components:

1. **Systematic Evaluation of Retrieval Module**:
   - The retrieval module was evaluated using the AI-generated query-document relevance dataset.
   - Metrics such as Mean NDCG, Mean Precision at K, and Mean Average Precision were used to assess the performance of different retrieval configurations.
   - The hybrid retriever, with more weight on the Self-Query Retriever, achieved the highest scores across all metrics.

2. **Human Annotator Evaluation**:
   - A human annotator actively collaborated with the system, providing queries and evaluating the responses.
   - The annotator compared the system's responses to those of the vanilla ChatGPT, assessing which one was better, whether the system's response was available in the retrieved bookset, and the rank of the chosen book by ChatGPT within the retrieved bookset.
   - Out of 20 comprehensive evaluations, the system's model was preferred over ChatGPT in 13 instances, ChatGPT was chosen twice, and both models performed equally well or poorly in the remaining 5 instances.

## Error Analysis

The error analysis identified the following issues:

1. **Limitations of Embeddings**: The embeddings used, including the LaBSE model, did not adequately capture the semantic meaning or provide suitable reflections, resulting in suboptimal retrieval performance.
2. **Recommendation of Less-Known Books**: The final model tended to recommend less well-known books, which could be improved by incorporating factors like the number of user ratings and the book's popularity.
3. **Metadata Extraction Errors**: In some cases, the language model extracted features that did not belong to the metadata, leading to errors when executing the database query. Careful verification of the extracted metadata is necessary.
4. **Limitations in Handling Similar Book Requests**: The model's output was not suitable for handling requests for books similar to another book. Exploring the use of a language model trained on book text could help generate similar queries or expand the user's query.

## Future Work

The report outlines several avenues for future research and improvement:

1. **Exploring Additional Retrieval Algorithms**: Investigating the use of other retrieval algorithms, such as FAISS, to further enhance the retrieval performance.
2. **Expanding and Diversifying the Book Dataset**: Increasing the dataset size and coverage to include a broader range of book categories and genres.
3. **Testing Alternative Embeddings**: Evaluating the performance of other embedding models, such as OpenAI embeddings, to address the limitations of the current embeddings.
4. **Prompt Engineering and Prompt Refinement**: Continuing the process of prompt engineering and refining the prompts used for the language models to improve their performance.
5. **Incorporating Book Text-Based Language Models**: Exploring the use of language models trained on book text to better handle requests for similar books and generate more relevant queries.

## Repository Structure

The project repository contains the following main components:

```
├── data
│   ├── books_dataset
│   └── query_relevancy_dataset
├── notebooks
│   ├── data_cleaning.ipynb
│   ├── retriever_evaluation.ipynb
│   └── chatbot_integration.ipynb
├── src
│   ├── retriever.py
│   ├── chatbot.py
│   ├── utils.py
│   └── main.py
├── final_report.pdf
├── proposal.md
└── presentation.pptx
```

1. `data`: Contains the Persian Books Dataset and the AI-Generated Query-Document Relevance Dataset.
2. `notebooks`: Includes Jupyter Notebooks for data cleaning, retriever evaluation, and chatbot integration.
3. `src`: Houses the source code for the retriever, chatbot, and utility functions.
4. `final_report.pdf`: The comprehensive final report for the project.
5. `proposal.md`: The initial project proposal.
6. `presentation.pptx`: The presentation slides for the project.

## Usage

To run the LLM-based Book Search System, follow these steps:

1. Set up the required dependencies, including Langchain, ChromaDB, and OpenAI Embeddings.
2. Prepare the dataset by running the data cleaning and preprocessing steps in the `data_cleaning.ipynb` notebook.
3. Train and evaluate the retrieval module by running the `retriever_evaluation.ipynb` notebook.
4. Integrate the retrieval module and the chatbot module by running the `chatbot_integration.ipynb` notebook.
5. Utilize the `main.py` script to interact with the complete LLM-based Book Search System.

For detailed instructions and further information, please refer to the `final_report.pdf` and the source code in the repository.