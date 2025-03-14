import json
import random
from string import ascii_uppercase
from data_models.qa import QAModel

DATA_PATH = "datasets/truthfulqa/mc_task.json"
OUTPUT_PATH = "datasets/truthfulqa/processed_qa.json"

def main():
    with open(DATA_PATH, "r") as f:
        qa_data = json.load(f)
    
    qa_models = []
    
    for item in qa_data:
        question = item["question"]
        
        # mc1_targets is a dictionary where keys are option texts and values are 0/1
        options_data = []
        for option_text, is_correct in item["mc1_targets"].items():
            options_data.append((option_text, is_correct))
        
        # Randomly shuffle the options to eliminate bias in letter assignment
        random.shuffle(options_data)
        
        # Get enough letters for all options
        option_count = len(options_data)
        option_letters = list(ascii_uppercase[:option_count])
        
        # Pair options with letters and identify correct answer
        final_options = []
        answer_letter = None
        
        for i, (option_text, is_correct) in enumerate(options_data):
            letter = option_letters[i]
            final_options.append((letter, option_text))
            if is_correct == 1:
                answer_letter = letter
        
        # Create QAModel
        qa_model = QAModel(
            question=question,
            options=[f"{letter}. {text}" for letter, text in final_options],
            answer=answer_letter,
            source="TruthfulQA MC1"
        )
        
        qa_models.append(qa_model)
    
    print(f"Parsed {len(qa_models)} questions")
    # Example: Print the first model to verify
    if qa_models:
        print(f"First question: {qa_models[0].question}")
        print(f"Options: {qa_models[0].options}")
        print(f"Answer: {qa_models[0].answer}")
    
    # Save the parsed data to a JSON file
    save_data = [model.dict() for model in qa_models]
    with open(OUTPUT_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"Saved {len(qa_models)} QA models to {OUTPUT_PATH}")
    
if __name__ == "__main__":
    main()
