from utils.load_data import load_qa_data
from ollama import chat
from ollama import ChatResponse
from data_models.answer import Answer, AnswerChoice
from prompts.prompt_manager import PromptManager
import time
import json
import csv
import os
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.layout import Layout
from rich import box
import statistics
import asyncio
from ollama import AsyncClient

# Example usage
MODEL_NAMES = [
    'phi4:latest',
    # 'gemma3:12b',
    'llama3.2:latest',
    'qwq:latest',
    'deepseek-r1:14b',
    'qwen2.5:14b',
    'llama3.1:8b'
]

COMPARISON_TABLE_COLUMNS = [
    'model_name',
    'model_type',
    'parameters',
    'dataset',
    'accuracy',
    'reasoning_length',
    'inference_time',
    'instruction_type',
    'completion_rate',
    'error_rate'
]

INSTRUCTIONS = [
    'cod_instruction',
    'standard_instruction',
    'cot_instruction'
]

# Model metadata - manually defined since it's hard to extract programmatically
MODEL_METADATA = {
    'phi4:latest': {'model_type': 'Phi', 'parameters': '4.2B'},
    'gemma3:12b': {'model_type': 'Gemma', 'parameters': '12B'},
    'llama3.2:latest': {'model_type': 'Llama', 'parameters': '8B'},
    'qwq:latest': {'model_type': 'QWQ', 'parameters': 'Unknown'},
    'deepseek-r1:14b': {'model_type': 'DeepSeek', 'parameters': '14B'},
    'qwen2.5:14b': {'model_type': 'Qwen', 'parameters': '14B'},
    'llama3.1:8b': {'model_type': 'Llama', 'parameters': '8B'}
}

async def evaluate_model_async(model_name, instruction_type, qa_models, sample_size=None, console=None, timeout=120):
    """
    Async version of evaluate_model that uses streaming responses
    """
    if console is None:
        console = Console()
        
    try:
        instruction = PromptManager.get_prompt(
            instruction_type,
            num_words=5
        )
    except Exception as e:
        console.print(f"[red]Error getting instruction prompt for {instruction_type}: {str(e)}[/red]")
        raise
    
    if sample_size and sample_size < len(qa_models):
        import random
        evaluation_set = random.sample(qa_models, sample_size)
    else:
        evaluation_set = qa_models
    
    total_questions = len(evaluation_set)
    correct_answers = 0
    errors = 0
    timeouts = 0
    results = []
    total_reasoning_length = 0
    total_inference_time = 0
    total_tokens = 0
    
    client = AsyncClient()
    
    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]Testing {model_name} with {instruction_type}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Processing", total=total_questions)
        
        for i, qa_model in enumerate(evaluation_set):
            try:
                prompt = PromptManager.get_prompt(
                    'truthfulqa_mc1',
                    instruction=instruction,
                    question=qa_model.question,
                    options=qa_model.options
                )
                
                try:
                    # Start the stream but don't wait for it to complete
                    stream = client.chat(
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                        }],
                        model=model_name,
                        options={"temperature": 0},
                        format=Answer.model_json_schema(),
                        stream=True
                    )
                    
                    # Get the initial response (this starts the stream)
                    stream_iterator = await asyncio.wait_for(stream, timeout=timeout)
                    
                    start_time = time.time()
                    full_response = ""
                    token_count = 0
                    last_valid_json = None
                    
                    # Set a timeout for the entire streaming process
                    stream_timeout = start_time + timeout
                    
                    async for part in stream_iterator:
                        # Check if we've exceeded our timeout
                        if time.time() > stream_timeout:
                            raise asyncio.TimeoutError(f"Streaming process exceeded timeout of {timeout}s")
                            
                        chunk = part['message']['content']
                        full_response += chunk
                        token_count += 1
                        
                        # Try to parse as JSON at each step
                        try:
                            parsed = json.loads(full_response)
                            if isinstance(parsed, dict) and 'answer' in parsed:
                                last_valid_json = parsed
                        except json.JSONDecodeError:
                            pass
                        
                        progress.update(
                            task,
                            description=f"[cyan]Q{i+1} - Tokens: {token_count}"
                        )
                    
                    end_time = time.time()
                    inference_time = end_time - start_time
                    tokens_per_second = token_count / inference_time if inference_time > 0 else 0
                    
                    # Use the last valid JSON if available, otherwise try parsing the full response
                    try:
                        if last_valid_json:
                            model_answer = Answer.model_validate(last_valid_json)
                        else:
                            # Clean up response if needed
                            cleaned_response = full_response.strip()
                            if cleaned_response.endswith(','):
                                cleaned_response = cleaned_response[:-1]
                            if not cleaned_response.startswith('{'):
                                cleaned_response = '{' + cleaned_response
                            if not cleaned_response.endswith('}'):
                                cleaned_response = cleaned_response + '}'
                            
                            model_answer = Answer.model_validate_json(cleaned_response)
                    except Exception as e:
                        raise ValueError(f"Failed to parse response: {str(e)}\nResponse: {full_response[:100]}...")
                    
                    # Validate answer format
                    if not hasattr(model_answer, 'answer') or not hasattr(model_answer.answer, 'value'):
                        raise ValueError(f"Invalid answer format: {model_answer}")
                    
                    is_correct = model_answer.answer.value == qa_model.answer
                    if is_correct:
                        correct_answers += 1
                        console.print(f"[green]✓ Q{i+1} - Correct[/green]")
                    else:
                        console.print(f"[red]✗ Q{i+1} - Wrong (Model: {model_answer.answer.value}, Correct: {qa_model.answer})[/red]")
                    
                    reasoning_length = 0
                    if model_answer.reasoning:
                        reasoning_length = len(model_answer.reasoning.split())
                        total_reasoning_length += reasoning_length
                    
                    total_inference_time += inference_time
                    total_tokens += token_count
                    
                    results.append({
                        "question": qa_model.question,
                        "model_answer": model_answer.answer.value,
                        "correct_answer": qa_model.answer,
                        "is_correct": is_correct,
                        "reasoning": model_answer.reasoning,
                        "reasoning_length": reasoning_length,
                        "inference_time": inference_time,
                        "tokens": token_count,
                        "tokens_per_second": tokens_per_second,
                        "error": None,
                        "raw_response": full_response  # Store raw response for debugging
                    })
                    
                    # Show detailed stats
                    console.print(
                        f"[dim]Q{i+1} - {token_count} tokens at {tokens_per_second:.1f} tokens/sec - Time: {inference_time:.2f}s[/dim]"
                    )
                    
                except asyncio.TimeoutError:
                    timeouts += 1
                    errors += 1
                    console.print(f"[yellow]Timeout for {model_name} on question {i+1} (>{timeout}s)[/yellow]")
                    results.append({
                        "question": qa_model.question,
                        "error": f"Timeout after {timeout}s",
                        "inference_time": timeout,
                        "tokens": token_count
                    })
                    
                except Exception as e:
                    errors += 1
                    error_msg = f"Error with {model_name} on question {i+1}: {str(e)}"
                    console.print(f"[red]{error_msg}[/red]")
                    results.append({
                        "question": qa_model.question,
                        "error": error_msg,
                        "inference_time": time.time() - start_time,
                        "tokens": token_count
                    })
                
            except Exception as e:
                errors += 1
                error_msg = f"Error preparing question {i+1}: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                results.append({
                    "question": qa_model.question,
                    "error": error_msg,
                    "inference_time": 0,
                    "tokens": 0
                })
            
            finally:
                progress.update(task, advance=1)
    
    accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    avg_reasoning_length = total_reasoning_length / (total_questions - errors) if (total_questions - errors) > 0 else 0
    avg_inference_time = total_inference_time / (total_questions - errors) if (total_questions - errors) > 0 else 0
    avg_tokens = total_tokens / (total_questions - errors) if (total_questions - errors) > 0 else 0
    completion_rate = ((total_questions - errors) / total_questions) * 100 if total_questions > 0 else 0
    error_rate = (errors / total_questions) * 100 if total_questions > 0 else 0
    timeout_rate = (timeouts / total_questions) * 100 if total_questions > 0 else 0
    
    if errors > 0:
        console.print(f"\n[yellow]Error Summary for {model_name}:[/yellow]")
        console.print(f"• Total errors: {errors}")
        console.print(f"• Timeouts: {timeouts}")
        console.print(f"• Error rate: {error_rate:.1f}%")
        console.print(f"• Timeout rate: {timeout_rate:.1f}%")
    
    evaluation_results = {
        "model_name": model_name,
        "model_type": MODEL_METADATA.get(model_name, {}).get('model_type', 'Unknown'),
        "parameters": MODEL_METADATA.get(model_name, {}).get('parameters', 'Unknown'),
        "dataset": "TruthfulQA MC1",
        "accuracy": accuracy,
        "reasoning_length": avg_reasoning_length,
        "inference_time": avg_inference_time,
        "tokens_per_response": avg_tokens,
        "instruction_type": instruction_type,
        "completion_rate": completion_rate,
        "error_rate": error_rate,
        "timeout_rate": timeout_rate,
        "total_questions": total_questions,
        "correct_answers": correct_answers,
        "errors": errors,
        "timeouts": timeouts,
        "total_tokens": total_tokens,
        "detailed_results": results
    }
    
    return evaluation_results

def evaluate_model(model_name, instruction_type, qa_models, sample_size=None, console=None, timeout=120):
    """
    Synchronous wrapper for evaluate_model_async
    """
    return asyncio.run(evaluate_model_async(
        model_name=model_name,
        instruction_type=instruction_type,
        qa_models=qa_models,
        sample_size=sample_size,
        console=console,
        timeout=timeout
    ))

def save_results(all_results, output_dir="results"):
    """Save evaluation results to files"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save full results as JSON
    json_path = os.path.join(output_dir, f"model_comparison_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save summary as CSV
    csv_path = os.path.join(output_dir, f"model_comparison_summary_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COMPARISON_TABLE_COLUMNS)
        writer.writeheader()
        
        for result in all_results:
            # Extract just the summary columns
            summary = {col: result.get(col) for col in COMPARISON_TABLE_COLUMNS}
            writer.writerow(summary)
    
    # Save detailed question-by-question results
    question_results = {}
    
    # Collect all questions and answers across models
    for result in all_results:
        model_name = result["model_name"]
        instruction_type = result["instruction_type"]
        model_key = f"{model_name}_{instruction_type}"
        
        for detail in result.get("detailed_results", []):
            question = detail.get("question")
            if not question:
                continue
                
            if question not in question_results:
                question_results[question] = {
                    "question": question,
                    "correct_answer": detail.get("correct_answer"),
                    "model_answers": {},
                    "correct_count": 0,
                    "total_attempts": 0
                }
            
            # Add this model's answer
            question_results[question]["model_answers"][model_key] = {
                "answer": detail.get("model_answer"),
                "is_correct": detail.get("is_correct", False),
                "reasoning": detail.get("reasoning", ""),
                "inference_time": detail.get("inference_time", 0),
                "error": detail.get("error")
            }
            
            # Update counts
            if detail.get("is_correct", False):
                question_results[question]["correct_count"] += 1
            if not detail.get("error"):
                question_results[question]["total_attempts"] += 1
    
    # Calculate accuracy per question
    for question_data in question_results.values():
        attempts = question_data["total_attempts"]
        if attempts > 0:
            question_data["accuracy"] = (question_data["correct_count"] / attempts) * 100
        else:
            question_data["accuracy"] = 0
    
    # Sort questions by accuracy (ascending to show most difficult first)
    sorted_questions = sorted(question_results.values(), key=lambda x: x["accuracy"])
    
    # Save detailed question results
    questions_path = os.path.join(output_dir, f"question_analysis_{timestamp}.json")
    with open(questions_path, 'w') as f:
        json.dump(sorted_questions, f, indent=2)
    
    # Save a CSV with the most difficult questions
    difficult_questions_path = os.path.join(output_dir, f"difficult_questions_{timestamp}.csv")
    with open(difficult_questions_path, 'w', newline='') as f:
        fieldnames = ["question", "correct_answer", "accuracy", "correct_count", "total_attempts"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for question_data in sorted_questions:
            row = {field: question_data[field] for field in fieldnames}
            writer.writerow(row)
    
    return json_path, csv_path, questions_path, difficult_questions_path

def display_summary_table(all_results, console):
    """Display a summary table of all results"""
    # Create the summary table
    table = Table(title="Model Comparison Summary", box=box.ROUNDED)
    
    # Add columns
    table.add_column("Model", style="cyan")
    table.add_column("Instruction", style="magenta")
    table.add_column("Accuracy", style="green")
    table.add_column("Avg Time (s)", style="yellow")
    table.add_column("Avg Reasoning", style="blue")
    table.add_column("Completion %", style="green")
    
    # Add rows for each result
    for result in all_results:
        table.add_row(
            result["model_name"],
            result["instruction_type"],
            f"{result['accuracy']:.2f}%",
            f"{result['inference_time']:.2f}",
            f"{result['reasoning_length']:.1f} words",
            f"{result['completion_rate']:.1f}%"
        )
    
    console.print(table)

def display_best_per_model(all_results, console):
    """Display the best instruction type for each model"""
    # Group results by model
    model_results = {}
    for result in all_results:
        model_name = result["model_name"]
        if model_name not in model_results:
            model_results[model_name] = []
        model_results[model_name].append(result)
    
    # Create the best results table
    table = Table(title="Best Instruction Type Per Model", box=box.ROUNDED)
    
    # Add columns
    table.add_column("Model", style="cyan")
    table.add_column("Best Instruction", style="magenta")
    table.add_column("Accuracy", style="green")
    table.add_column("Avg Time (s)", style="yellow")
    
    # Find the best instruction for each model
    for model_name, results in model_results.items():
        # Sort by accuracy (descending)
        best_result = sorted(results, key=lambda x: x["accuracy"], reverse=True)[0]
        
        table.add_row(
            model_name,
            best_result["instruction_type"],
            f"{best_result['accuracy']:.2f}%",
            f"{best_result['inference_time']:.2f}"
        )
    
    console.print(table)

def display_instruction_comparison(all_results, console):
    """Display comparison tables for each instruction type"""
    # Group results by instruction type
    instruction_results = {}
    for result in all_results:
        instruction = result["instruction_type"]
        if instruction not in instruction_results:
            instruction_results[instruction] = []
        instruction_results[instruction].append(result)
    
    # Create a table for each instruction type
    for instruction, results in instruction_results.items():
        # Sort by accuracy (descending)
        sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
        
        table = Table(title=f"Model Comparison with {instruction}", box=box.ROUNDED)
        
        # Add columns
        table.add_column("Rank", style="dim")
        table.add_column("Model", style="cyan")
        table.add_column("Parameters", style="magenta")
        table.add_column("Accuracy", style="green")
        table.add_column("Avg Time (s)", style="yellow")
        table.add_column("Avg Reasoning", style="blue")
        
        # Add rows
        for i, result in enumerate(sorted_results):
            table.add_row(
                str(i+1),
                result["model_name"],
                result["parameters"],
                f"{result['accuracy']:.2f}%",
                f"{result['inference_time']:.2f}",
                f"{result['reasoning_length']:.1f} words"
            )
        
        console.print(table)
        console.print()  # Add some space between tables

def display_difficult_questions(all_results, console, top_n=5):
    """Display the most difficult questions across all models"""
    # Collect question results
    question_results = {}
    
    # Process all detailed results
    for result in all_results:
        for detail in result.get("detailed_results", []):
            question = detail.get("question")
            if not question or detail.get("error"):
                continue
                
            if question not in question_results:
                question_results[question] = {
                    "question": question,
                    "correct_answer": detail.get("correct_answer"),
                    "correct_count": 0,
                    "total_attempts": 0
                }
            
            # Update counts
            if detail.get("is_correct", False):
                question_results[question]["correct_count"] += 1
            question_results[question]["total_attempts"] += 1
    
    # Calculate accuracy per question
    for question_data in question_results.values():
        attempts = question_data["total_attempts"]
        if attempts > 0:
            question_data["accuracy"] = (question_data["correct_count"] / attempts) * 100
        else:
            question_data["accuracy"] = 0
    
    # Sort questions by accuracy (ascending)
    sorted_questions = sorted(
        question_results.values(), 
        key=lambda x: x["accuracy"]
    )
    
    # Display the most difficult questions
    if sorted_questions:
        console.print("\n[bold]Most Difficult Questions:[/bold]")
        
        table = Table(box=box.ROUNDED)
        table.add_column("Question", style="cyan", no_wrap=False)
        table.add_column("Correct Answer", style="green")
        table.add_column("Accuracy", style="yellow")
        table.add_column("Correct/Total", style="magenta")
        
        for i, q in enumerate(sorted_questions[:top_n]):
            table.add_row(
                q["question"],
                q["correct_answer"],
                f"{q['accuracy']:.1f}%",
                f"{q['correct_count']}/{q['total_attempts']}"
            )
        
        console.print(table)

def main():
    # Initialize rich console for nice formatting
    console = Console()
    
    # Start timing the full run
    start_time = time.time()
    
    # Load the processed data
    qa_models = load_qa_data()
    
    # Ask user if they want to run a quick test or full evaluation
    console.print(Panel("[bold yellow]Model Comparison Configuration[/bold yellow]"))
    console.print("This will evaluate multiple models with different instruction types.")
    console.print("The full evaluation may take a long time.")
    
    # Get sample size (0 for full dataset)
    sample_size = 0
    try:
        sample_input = input("Enter sample size (0 for full dataset, recommended 10-20 for testing): ")
        sample_size = int(sample_input) if sample_input.strip() else 0
    except ValueError:
        console.print("[red]Invalid input, using full dataset[/red]")
    
    # Get models to evaluate
    console.print("\n[bold]Available Models:[/bold]")
    for i, model in enumerate(MODEL_NAMES):
        console.print(f"  {i+1}. {model}")
    
    selected_models = MODEL_NAMES
    try:
        models_input = input("Enter model numbers to evaluate (comma-separated, empty for all): ")
        if models_input.strip():
            model_indices = [int(idx.strip()) - 1 for idx in models_input.split(",")]
            selected_models = [MODEL_NAMES[idx] for idx in model_indices if 0 <= idx < len(MODEL_NAMES)]
    except (ValueError, IndexError):
        console.print("[red]Invalid input, using all models[/red]")
    
    # Get instruction types to evaluate
    console.print("\n[bold]Available Instruction Types:[/bold]")
    for i, instruction in enumerate(INSTRUCTIONS):
        console.print(f"  {i+1}. {instruction}")
    
    selected_instructions = INSTRUCTIONS
    try:
        instructions_input = input("Enter instruction numbers to evaluate (comma-separated, empty for all): ")
        if instructions_input.strip():
            instruction_indices = [int(idx.strip()) - 1 for idx in instructions_input.split(",")]
            selected_instructions = [INSTRUCTIONS[idx] for idx in instruction_indices if 0 <= idx < len(INSTRUCTIONS)]
    except (ValueError, IndexError):
        console.print("[red]Invalid input, using all instruction types[/red]")
    
    # Take the first N questions instead of random sampling
    if sample_size and sample_size < len(qa_models):
        evaluation_subset = qa_models[:sample_size]
    else:
        evaluation_subset = qa_models
    
    # Update dataset_size to reflect the actual subset size
    dataset_size = len(evaluation_subset)
    
    # Confirm evaluation plan
    total_evaluations = len(selected_models) * len(selected_instructions)
    
    console.print(Panel(f"""
[bold green]Evaluation Plan:[/bold green]
• Models to evaluate: {len(selected_models)}
• Instruction types: {len(selected_instructions)}
• Total evaluations: {total_evaluations}
• Questions per evaluation: {dataset_size}
• Total questions to process: {total_evaluations * dataset_size}
    """))
    
    confirm = input("Proceed with evaluation? (y/n): ").lower().strip()
    if confirm != 'y':
        console.print("[yellow]Evaluation cancelled[/yellow]")
        return
    
    # Run evaluations
    all_results = []
    evaluation_start_time = time.time()
    
    for model_name in selected_models:
        for instruction_type in selected_instructions:
            model_start_time = time.time()
            console.print(f"\n[bold]Evaluating {model_name} with {instruction_type}...[/bold]")
            
            result = evaluate_model(
                model_name=model_name,
                instruction_type=instruction_type,
                qa_models=evaluation_subset,  # Use the same subset for all evaluations
                sample_size=None,  # Don't sample again, we're passing the subset directly
                console=console
            )
            
            all_results.append(result)
            
            model_elapsed_time = time.time() - model_start_time
            # Show quick result
            console.print(f"[green]Accuracy: {result['accuracy']:.2f}%[/green]")
            console.print(f"[blue]Time taken: {model_elapsed_time:.2f} seconds[/blue]")
    
    evaluation_elapsed_time = time.time() - evaluation_start_time
    
    # Display results
    console.print("\n[bold green]Evaluation Complete![/bold green]\n")
    
    # Display summary tables
    display_summary_table(all_results, console)
    console.print()
    
    display_best_per_model(all_results, console)
    console.print()
    
    display_instruction_comparison(all_results, console)
    console.print()
    
    # Display most difficult questions
    display_difficult_questions(all_results, console)
    
    # Save results to files
    json_path, csv_path, questions_path, difficult_questions_path = save_results(all_results)
    console.print(f"\n[bold]Results saved to:[/bold]")
    console.print(f"• Summary JSON: {json_path}")
    console.print(f"• Summary CSV: {csv_path}")
    console.print(f"• Question Analysis: {questions_path}")
    console.print(f"• Difficult Questions: {difficult_questions_path}")
    
    # Display total time taken
    total_elapsed_time = time.time() - start_time
    console.print(Panel(f"""
[bold cyan]Time Summary:[/bold cyan]
• Evaluation time: {evaluation_elapsed_time:.2f} seconds ({evaluation_elapsed_time/60:.2f} minutes)
• Total run time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)
• Average time per evaluation: {evaluation_elapsed_time/total_evaluations:.2f} seconds
    """))

def test_run():
    """Run a quick test evaluation using 1% of the dataset for all models and instructions"""
    console = Console()
    
    # Start timing the test run
    start_time = time.time()
    
    # Load the processed data
    qa_models = load_qa_data()
    
    # Calculate sample size
    sample_size = max(int(len(qa_models) * 0.05), 1)  # At least 1 question
    
    # Take the first N questions instead of random sampling
    evaluation_subset = qa_models[:sample_size]
    
    console.print(Panel(f"""
[bold yellow]Test Run Configuration[/bold yellow]
• Using {len(evaluation_subset)} questions ({(len(evaluation_subset)/len(qa_models)*100):.1f}% of dataset)
• Testing all {len(MODEL_NAMES)} models
• Testing all {len(INSTRUCTIONS)} instruction types
• Total evaluations: {len(MODEL_NAMES) * len(INSTRUCTIONS)}
    """))
    
    # Run the evaluation
    console.print(f"\n[bold]Starting test evaluation...[/bold]")
    
    all_results = []
    evaluation_start_time = time.time()
    
    for model_name in MODEL_NAMES:
        for instruction_type in INSTRUCTIONS:
            model_start_time = time.time()
            console.print(f"\n[bold]Evaluating {model_name} with {instruction_type}...[/bold]")
            
            result = evaluate_model(
                model_name=model_name,
                instruction_type=instruction_type,
                qa_models=evaluation_subset,  # Use the same subset for all evaluations
                sample_size=None,  # Don't sample again, we're passing the subset directly
                console=console
            )
            
            all_results.append(result)
            
            model_elapsed_time = time.time() - model_start_time
            # Show quick result
            console.print(f"[green]Accuracy: {result['accuracy']:.2f}%[/green]")
            console.print(f"[blue]Time taken: {model_elapsed_time:.2f} seconds[/blue]")
    
    evaluation_elapsed_time = time.time() - evaluation_start_time
    
    # Display results
    console.print("\n[bold green]Test Run Complete![/bold green]\n")
    
    # Display all visualization tables
    display_summary_table(all_results, console)
    console.print()
    
    display_best_per_model(all_results, console)
    console.print()
    
    display_instruction_comparison(all_results, console)
    console.print()
    
    # Display most difficult questions
    display_difficult_questions(all_results, console)
    
    # Save test results
    json_path, csv_path, questions_path, difficult_questions_path = save_results(all_results, output_dir="test_results")
    console.print(f"\n[bold]Test results saved to:[/bold]")
    console.print(f"• Summary JSON: {json_path}")
    console.print(f"• Summary CSV: {csv_path}")
    console.print(f"• Question Analysis: {questions_path}")
    console.print(f"• Difficult Questions: {difficult_questions_path}")
    
    # Display total time taken
    total_elapsed_time = time.time() - start_time
    console.print(Panel(f"""
[bold cyan]Time Summary:[/bold cyan]
• Evaluation time: {evaluation_elapsed_time:.2f} seconds ({evaluation_elapsed_time/60:.2f} minutes)
• Total run time: {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} minutes)
• Average time per evaluation: {evaluation_elapsed_time/(len(MODEL_NAMES) * len(INSTRUCTIONS)):.2f} seconds
• Estimated full dataset time: {evaluation_elapsed_time * (len(qa_models)/sample_size):.2f} seconds ({evaluation_elapsed_time * (len(qa_models)/sample_size)/60:.2f} minutes)
    """))

if __name__ == "__main__":
    # Run the test by default
    test_run()
    
    # Ask if user wants to run full evaluation
    if input("\nRun full evaluation with complete dataset? (y/n): ").lower().strip() == 'y':
        main()