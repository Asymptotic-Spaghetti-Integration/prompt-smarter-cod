import json
from data_models.qa import QAModel

def load_qa_data(file_path="datasets/truthfulqa/processed_qa.json"):
    """
    Load processed QA data from a JSON file and convert to QAModel objects.
    
    Args:
        file_path: Path to the processed QA data JSON file
        
    Returns:
        List of QAModel objects
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    
    # Convert each dictionary back to a QAModel object
    qa_models = [QAModel(**item) for item in data]
    
    print(f"Loaded {len(qa_models)} QA models from {file_path}")
    return qa_models
