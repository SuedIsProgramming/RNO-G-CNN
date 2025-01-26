import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Loop through all files in the directory
for filename in os.listdir(script_dir):
    # Check if the file ends with .err or .out
    if filename.endswith('.err') or filename.endswith('.out'):
        # Construct the full file path
        file_path = os.path.join(script_dir, filename)
        # Delete the file
        os.remove(file_path)
        print(f"Deleted {file_path}")
