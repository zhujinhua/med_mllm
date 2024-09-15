"""
Author: jhzhu
Date: 2024/8/24
Description: 
"""
import glob
import json
import csv
import os
import random


def convert_to_ceval():
    answer_convert_dict = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
    input_file_path = '/Users/jhzhu/Downloads/ARC-Easy-Dev.jsonl'
    output_file_path = '/Users/jhzhu/Downloads/ARC-Easy-Dev.csv'
    # Open the JSONL file and CSV file
    with open(input_file_path, 'r') as jsonl_file, open(output_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)

        # Write CSV header
        csv_writer.writerow(['id', 'question', 'A', 'B', 'C', 'D', 'answer', 'explanation'])

        # Process each line in the JSONL file
        for line in jsonl_file:
            # Parse the JSON object
            data = json.loads(line)

            # Extract the required fields
            q_id = data.get('id')
            question = data['question']['stem']
            choices = {choice['label']: choice['text'] for choice in data['question']['choices']}
            answer = data.get('answerKey')
            if answer in answer_convert_dict:
                answer = answer_convert_dict.get(answer)
            if len(choices) > 4:
                continue
            # Extract choices text based on labels 'A', 'B', 'C', 'D'
            A = choices.get('A') if 'A' in choices else choices.get('1', '')
            B = choices.get('B') if 'B' in choices else choices.get('2', '')
            C = choices.get('C') if 'C' in choices else choices.get('3', '')
            D = choices.get('D') if 'D' in choices else choices.get('4', '')

            # Write the row to the CSV file
            csv_writer.writerow([q_id, question, A, B, C, D, answer, ''])


def convert_eval_format():
    input_file_path = '/Users/jhzhu/Downloads/software/medical/test_zh_0.json'
    output_file_path = '/Users/jhzhu/Downloads/test_zh_0.jsonl'
    json_objects = []

    # Open the file in read mode with UTF-8 encoding
    with open(input_file_path, 'r', encoding='utf-8') as file:
        # Iterate over each line in the file
        for line in file:
            # Strip any extra whitespace and ensure the line is not empty
            stripped_line = line.strip()
            if stripped_line:  # Only process non-empty lines
                try:
                    json_object = json.loads(stripped_line)
                    formatted_entry = {
                        "query": json_object['instruction'],
                        "response": json_object['output'],
                        "history": []
                    }
                    json_objects.append(formatted_entry)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    print(f"Line content causing error: {line}")
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for obj in json_objects:
            json_string = json.dumps(obj, ensure_ascii=False)
            file.write(json_string + '\n')


def convert_mlec_qa():
    input_file_path = '/Users/jhzhu/Downloads/mlec-qa/TCM_dev.json'
    output_file_path = '/Users/jhzhu/Downloads/mlec-qa/TCM_dev.csv'

    formatted_data = []
    with open(input_file_path, 'r', encoding='utf-8') as f, open(output_file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['id', 'question', 'A', 'B', 'C', 'D', 'answer', 'explanation'])
        conversations = json.load(f)

        # id,question,A,B,C,D,answer,explanation
        for conversation in conversations:
            id = conversation.get('qid')
            question = conversation.get('qtext').replace('（　　）。', '')
            answer = conversation.get('answer')
            options = conversation.get('options')
            if len(options) > 5:
                print('Error')
            if answer == 'E':
                answer = random.choice(['A', 'B', 'C', 'D'])
                options[answer] = options.get('E')
            csv_writer.writerow([id, question, options.get('A'), options.get('B'), options.get('C'), options.get('D'), answer, ''])


def convert_history():
    input_file_path = '/Users/jhzhu/Downloads/software/medical/valid_zh_0.json'
    output_file_path = '/Users/jhzhu/Downloads/validate_zh_0.json'

    formatted_data = []

    try:
        # Open the input file in read mode
        with open(input_file_path, 'r', encoding='utf-8') as f:
            # Load the large JSON file as a list of conversations
            conversations = json.load(f)

            # Iterate over each conversation in the list
            for conversation in conversations:
                # Extract patient's question and doctor's response
                if len(conversation) < 2:
                    print(f"Skipping conversation with insufficient data: {conversation}")
                    continue

                    # Initialize a history list
                history = []

                # If the conversation length is greater than 2, populate history
                if len(conversation) > 2:
                    for i in range(0, len(conversation) - 2, 2):
                        human_instruction = conversation[i][3:]
                        model_response = conversation[i + 1][3:]
                        history.append([human_instruction, model_response])

                # The last two elements are the current instruction and output
                instruction = conversation[-2][3:]  # Second last element as the instruction
                output = conversation[-1][3:]  # Last element as the output

                # Create a formatted entry
                if history:
                    formatted_entry = {
                        "instruction": instruction,
                        "output": output,
                        "history": history
                    }
                else:
                    formatted_entry = {
                        "instruction": instruction,
                        "output": output
                    }

                # Add the formatted entry to the list
                formatted_data.append(formatted_entry)

        # Step 3: Write the formatted data to a new JSON file
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, ensure_ascii=False, indent=2)

        print(f"Data has been successfully written to {output_file_path}")

    except json.JSONDecodeError as e:
        print(f"Error reading JSON file: {e}")
    except FileNotFoundError:
        print(f"Input file {input_file_path} not found.")
    except MemoryError:
        print("MemoryError: The file is too large to fit into memory. Consider processing the file in smaller chunks.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    convert_mlec_qa()
