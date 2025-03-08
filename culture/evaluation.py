import json
import os
import argparse
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

def extract_from_json(task: str, data: dict) -> dict:
    """
    Extract test data from the JSON structure for evaluation.
    
    Args:
        task: The task name (e.g., 'train', 'test')
        data: The loaded JSON data dictionary
        
    Returns:
        A dictionary containing the extracted data for the specified task
    """
    if task in data:
        return data[task]
    else:
        raise KeyError(f"Task '{task}' not found in the data. Available tasks: {list(data.keys())}")

def evaluate_model(model: str, task: str, data: dict) -> dict:
    return extract_from_json(task, data)

def generate_response(base_model_path: str, adapter_path: str, task: str, data: dict) -> dict:
    llm = LLM(
        model=base_model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=64
    )
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512
    )
    prompts = [item['instruction'] for item in data.values()]
    outputs = llm.generate(
        prompts, 
        sampling_params,
        lora_request=LoRARequest(
            "adapter",
            1,
            adapter_path
        )
    )
    return outputs

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a given task")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the adapter model")
    parser.add_argument("--task", type=str, default="test", help="Task name (train/test)")
    parser.add_argument("--data_dir", type=str, default="sparta_alignment/data/culture", help="Path to the data directory")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()
    
    # Specific JSON files to evaluate
    target_files = [
        "country_dataset.json",
        "country_value_dataset.json", 
        "rule_of_thumb_dataset.json"
    ]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = {}
    overall_correct = 0
    overall_total = 0
    
    # Initialize the model once to avoid reloading for each dataset
    llm = LLM(
        model=args.base_model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        enable_lora=True,
        max_lora_rank=64
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512
    )
    
    # Print header for dataset metrics
    print("\n" + "=" * 80)
    print(f"{'Dataset':<30} {'Accuracy':<10} {'Correct':<10} {'Total':<10}")
    print("-" * 80)
    
    for data_file in target_files:
        data_path = os.path.join(args.data_dir, data_file)
        print(f"Processing {data_path}...")
        
        try:
            with open(data_path, 'r') as file:
                data = json.load(file)
            
            # Extract data for the specified task
            task_data = extract_from_json(args.task, data)
            
            # Prepare prompts
            prompts = [item['instruction'] for item in task_data.values()]
            example_ids = list(task_data.keys())
            
            # Generate model responses
            outputs = llm.generate(
                prompts, 
                sampling_params,
                lora_request=LoRARequest(
                    "adapter",
                    1,
                    args.adapter_path
                )
            )
            
            # Process results
            correct = 0
            total = len(outputs)
            
            for i, output in enumerate(outputs):
                example_id = example_ids[i]
                ground_truth = task_data[example_id]["label"]
                generated_text = output.outputs[0].text.strip().lower()
                
                # Check for yes/no/neutral in the generated text
                if "neutral" in generated_text:
                    prediction = "neutral"
                elif "yes" in generated_text:
                    prediction = "yes"
                elif "no" in generated_text:
                    prediction = "no"
                else:
                    # Default case if none of the expected answers are found
                    prediction = "neutral"  # You might want to adjust this default
                
                if prediction == ground_truth:
                    correct += 1
            
            accuracy = correct / total if total > 0 else 0
            
            # Update overall statistics
            overall_correct += correct
            overall_total += total
            
            # Save dataset results (only accuracy)
            dataset_name = os.path.splitext(data_file)[0]
            all_results[dataset_name] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total
            }
            
            # Print dataset metrics in table format
            print(f"{dataset_name:<30} {accuracy:.4f}    {correct:<10} {total:<10}")
            
        except Exception as e:
            print(f"Error processing {data_path}: {str(e)}")
    
    # Calculate overall accuracy
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0
    all_results["overall"] = {
        "accuracy": overall_accuracy,
        "correct": overall_correct,
        "total": overall_total
    }
    
    # Print overall metrics
    print("-" * 80)
    print(f"{'Overall':<30} {overall_accuracy:.4f}    {overall_correct:<10} {overall_total:<10}")
    print("=" * 80 + "\n")
    
    # Save all results to a JSON file
    results_path = os.path.join(args.output_dir, f"evaluation_results_{args.task}.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"All results saved to {results_path}")

if __name__ == "__main__":
    main()