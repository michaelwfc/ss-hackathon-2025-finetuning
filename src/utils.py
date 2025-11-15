

def extract_json_objects(text: str):
    """Extract valid JSON objects from model output using regex."""
    pattern = r"\{.*?\}"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    json_objects = []
    for m in matches:
        try:
            obj = json.loads(m)
            if all(k in obj for k in ["instruction", "context", "response"]):
                    # Ensure context is empty string
                    obj["context"] = ""
                    # Validate response length (should be substantial)
                    if len(obj["response"]) > 100:
                        json_objects.append(obj)
                        
        except json.JSONDecodeError:
            continue
    return json_objects


def add_json_objects(examples, output_path):
    """Add JSON objects to model output."""
    
    with open(output_path, "a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=True) + "\n")

    print(f"✅ Saved {len(examples)} valid items to {output_path}")
    return output_path
  
  
def load_jsonl(file_path):
    """Load JSONL file with error handling."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            try:
                obj = json.loads(line.strip())
                data.append(obj)
  
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON on line {line_num}: {e}")
                continue    
    return data
  
  
# Example usage
if __name__ == "__main__":
    # Load environment variables
    load_env()
  
    dashscope_api_key = get_env("DASHSCOPE_API_KEY")
    print(dashscope_api_key)
