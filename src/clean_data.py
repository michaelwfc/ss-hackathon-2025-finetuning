import os
import re
import json
import random

def clean_unicode_characters(input_file  ,data_dir="data"):
    input_path =  os.path.join(data_dir, input_file)
    output_file = os.path.splitext(input_file)[0] + '_validated.jsonl'
    output_path = os.path.join(data_dir, output_file)
    
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            with open(output_path, 'w', encoding='utf-8') as f_out:
                for line_num, line in enumerate(f_in, 1):
                    try:
                        # Parse as JSON to ensure validity
                        data = json.loads(line.strip())
                        
                        # Convert back to JSON string with ASCII-only characters
                        cleaned_line = json.dumps(data, ensure_ascii=True)
                        f_out.write(cleaned_line + '\n')
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON on line {line_num}, skipping...")
                        continue
        
        print(f"Successfully cleaned {input_file} -> {output_file}")
        return output_file
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
        return False
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False


# shuffle data
def shuffle_data( input_file ,data_dir="data"):
    
    input_path =  os.path.join(data_dir, input_file)
    output_file = os.path.splitext(input_file)[0] + '_shuffled.jsonl'
    output_path = os.path.join(data_dir, output_file)
    
    try:
        # Read all lines from the input file
        
        with open(input_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
        
        # Shuffle the lines randomly
        random.shuffle(lines)
        
        # Write shuffled lines to the output file
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(lines)
        
        print(f"Successfully shuffled {len(lines)} lines from {input_file} -> {output_file}")
        return output_file
        
    except FileNotFoundError:
        print(f"Error: Could not find input file {input_file}")
        return False
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False


def postprocess_data(input_file ,  data_dir="data",):

    
    output_file = clean_unicode_characters(input_file=input_file)
    output_file = shuffle_data(input_file=output_file)
    print(output_file)


if __name__ == "__main__":

    input_file="state_data_gen_cleaned2.jsonl"
    
    # clean_unicode_characters(input_file="v3_dataset.jsonl")
    # shuffle_data()
    
    postprocess_data(input_file)