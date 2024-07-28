from typing import List, Dict, Any, Set
from rag import run_llm
from fuzzywuzzy import fuzz
import time

# calculate context precision
def calculate_answer_precision(generated_answers: List[str], ground_truths: List[str]) -> float:
    correct_answers = 0
    for gen, truth in zip(generated_answers, ground_truths):
        similarity = fuzz.partial_ratio(gen, truth)
        #  similarity = fuzz.(gen, truth)
        print(f"Generated: {gen}")
        print(f"Ground Truth: {truth}")
        print(f"Similarity: {similarity}")
        if similarity > 60:  # Consider it correct if similarity is above 80%
            correct_answers += 1
    print(f"CorrectAnswers: {correct_answers}")
    print(f"LengthOfGeneratedAnswers: {len(generated_answers)}")
    precision = len(generated_answers) / len(generated_answers) if generated_answers else 0
    print(f"Precision: {precision}")
    return precision


# calculate the recall of the retrieved contexts
# def calculate_context_recall(retrieved_contexts: List[str], relevant_contexts: List[str]) -> float:
#     relevant_retrieved = [ctx for ctx in retrieved_contexts if any(rel in ctx for rel in relevant_contexts)]
#     recall = len(relevant_retrieved) / len(relevant_contexts) if relevant_contexts else 0
#     print(f"LengthOfRelevantRetrieved: {len(relevant_retrieved)}")
#     print(f"LengthOfRelevantContexts: {len(relevant_contexts)}")
#     print(f"Recall: {recall}")
#     return recall

def calculate_context_recall(retrieved_contexts: List[str], relevant_contexts: List[str]) -> float:
    relevant_retrieved = []
    for rel_ctx in relevant_contexts:
        for ret_ctx in retrieved_contexts:
            if fuzz.partial_ratio(rel_ctx, ret_ctx) > 70:  # Adjust threshold as needed
                relevant_retrieved.append(ret_ctx)
                break
    recall = len(relevant_retrieved) / len(relevant_contexts) if relevant_contexts else 0
    print(f"LengthOfRelevantRetrieved: {len(relevant_retrieved)}")
    print(f"LengthOfRelevantContexts: {len(relevant_contexts)}")
    print(f"Recall: {recall}")
    return recall


# measures the faithfulness of the generated answers against the ground truth
def calculate_faithfulness(generated_answers: List[str], ground_truths: List[str]) -> float:
    faithful_answers = [ans for ans, truth in zip(generated_answers, ground_truths) if ans == truth]
    faithfulness = len(faithful_answers) / len(generated_answers)
    return faithfulness

# measures the response time of the system
def calculate_latency(query: str, chat_history: List[Dict[str, Any]]) -> float:
    start_time = time.time()
    run_llm(query=query, chat_history=chat_history)
    end_time = time.time()
    return end_time - start_time

# main function 
def evaluate_rag_pipeline(test_cases: List[Dict[str, Any]], ground_truths: List[str]) -> Dict[str, float]:
    metrics = {
        "context_precision": [],
        "context_recall": [],
        "faithfulness": [],
        "latency": [],
    }

    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        relevant_contexts = test_case["relevant_contexts"]
        chat_history = test_case["chat_history"]

        result = run_llm(query=query, chat_history=chat_history)
        retrieved_contexts = [doc.page_content for doc in result["context"]]
        generated_answer = result["answer"]

        print(f"Query: {query}")
        print(f"Relevant Contexts: {relevant_contexts}")
        print(f"Retrieved Contexts: {retrieved_contexts}")
        print(f"Generated Answer: {generated_answer}")

        metrics["context_precision"].append(calculate_answer_precision(retrieved_contexts, relevant_contexts))
        metrics["context_recall"].append(calculate_context_recall(retrieved_contexts, relevant_contexts))
        metrics["faithfulness"].append(calculate_faithfulness([generated_answer], [ground_truths[i]]))
        metrics["latency"].append(calculate_latency(query, chat_history))

    averaged_metrics = {metric: sum(values) / len(values) for metric, values in metrics.items()}
    return averaged_metrics

if __name__ == "__main__":
    # Example test cases
    test_cases = [
        {
            "query": "What is MySQL?",
            "relevant_contexts": ["MySQL is a relational database management system."],
            "chat_history": []
        },
        {
            "query": "What is MySQL Shell?",
            "relevant_contexts": ["MySQL Shell is an advanced client and code editor for MySQL Server that provides SQL functionality, scripting capabilities for JavaScript and Python, and includes APIs for working with MySQL. It is a component that can be installed separately from MySQL Server."],
            "chat_history": []
        },
        {
            "query": "What is InnoDB?",
            "relevant_contexts": ["InnoDB is a storage engine for MySQL."],
            "chat_history": []
        }
        # {
        #     "query": "What is MySQL Workbench?",
        #     "relevant_contexts": ["MySQL Workbench is a unified visual tool for database architects, developers, and DBAs."],
        #     "chat_history": []
        # },
        # {
        #     "query": "What is MySQL Cluster?",
        #     "relevant_contexts": ["MySQL Cluster is a technology that provides high availability and scalability."],
        #     "chat_history": []
        # },
        # {
        #     "query": "What is MySQL Replication?",
        #     "relevant_contexts": ["MySQL Replication allows data from one MySQL database server to be replicated to another."],
        #     "chat_history": []
        # },
        # {
        #     "query": "What is MySQL Enterprise?",
        #     "relevant_contexts": ["MySQL Enterprise is a comprehensive suite of solutions for enterprise needs."],
        #     "chat_history": []
        # }
    ]
    
    ground_truths = [
        "MySQL is a relational database management system.",
        "MySQL Shell is an advanced client and code editor for MySQL Server that provides SQL functionality, scripting capabilities for JavaScript and Python, and includes APIs for working with MySQL. It is a component that can be installed separately from MySQL Server.",
        "InnoDB is a storage engine for MySQL databases that provides features such as support for transactions, row-level locking, and foreign keys. It is known for its reliability, performance, and adherence to the ACID (Atomicity, Consistency, Isolation, Durability) model."
        # "MySQL Workbench is a GUI tool provided by Oracle Corporation for administering MySQL servers and databases. It is used to create, execute, and evaluate queries, as well as to migrate schemas and data from other relational database management systems for use with MySQL.",
        # "MySQL Cluster is a high-availability, high-redundancy version of MySQL that is adapted for the distributed computing environment.",
        # "MySQL Replication is a feature that allows data from one MySQL database server (source) to be copied to one or more MySQL database servers (replicas).",
        # "MySQL Enterprise is a commercial product that offers a comprehensive package of MySQL database software, services, and support."
    ]

    results = evaluate_rag_pipeline(test_cases, ground_truths)
    print(f"Evaluation Results: {results}")
