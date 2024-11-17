"""
run with  python -m pytest test/test_rag.py 
"""

import logging
from src import rag 

logging.basicConfig(level=logging.INFO)

# doc_id, question, expected_keywords
testcases = [
    (0, "What is the benefit of the mobile app for Move 4 Life?", ["Move 4 Life", "greater value", "efficient"]),
    (1, "What are the trade-offs of a web-first approach?", ["reach", "customer", "launch"]),
    (2, "What is the goal of John Lewis & Partners?", ["scale", "accelerate", "improve"]),
    (14, "What are the benefits of an event-driven architecture?", ["immutable", "subsequent actions"]),
    (30, "What is the main priority of O2?", ["tickets", "offering", "platform"]),
]

def test_rag():
    for doc_id, question, expected_keywords in testcases:
        output = rag.query_rag(question)
        texts = ' '.join([doc['entity']['text'] for doc in output[0]])

        assert doc_id in [doc['entity']['doc_id'] for doc in output[0]], f"Expected doc_id '{doc_id}' not found in retrieved docs"

        for keyword in expected_keywords:
            assert keyword in texts, f"Expected keyword '{keyword}' not found in retrieved texts"

