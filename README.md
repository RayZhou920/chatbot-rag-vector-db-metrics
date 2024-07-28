# info7375-chatbot-rag-vector-db-metrics

### Report and Video

[Report](https://drive.google.com/file/d/1XBynK3bGEpFS5wBlikBiRZ9s8FsVcuh_/view?usp=drive_link)
[Video](https://www.youtube.com/watch?v=ENwYoVSQIkU)

### Installation of how to use this script to calculate the RAG metrics
1. Clone the repository - https://github.com/RayZhou920/Adaptive-Recommendation-Chatbot-with-RAG-and-Vector-Database.
2. Clone this repository, and add the evaluate_rag.py file to the Adaptive-Recommendation-Chatbot-with-RAG-and-Vector-Database repository.
3. Navigate to your repository directory containing the whole project: ‘cd your-repository’.
4. Create a virtual environment: 'pipenv shell'.
5. Install the required packages: 'pipenv install'.
6. Set up environment variables:
   Create a .env file in the root directory of your project and add your Pinecone API key, OpenAI API key
7. Fetch data from the MySQL website for the example cases:
   mkdir mysql-docs
   wget -r -P mysql-docs -E https://www.mysql.com/docs/manual
8. Pre-process the data by running the process_data.py script. You should see the following message if successful:
   Going to add xxx to Pinecone
   Loading to vectorstore done
9. Calculate the metrics for RAG:
   Run the evaluate_rag.py script\
   python evaluate_rag.py
