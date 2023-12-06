import csv
import os

# Define the path to your CSV file
csv_file_path = "/afs/crc.nd.edu/user/r/rshi2/mimik/xlsr-53-english/audios/metadata.csv"

# Check if the CSV file exists
if not os.path.exists(csv_file_path):
    print(f"CSV file '{csv_file_path}' does not exist.")
    exit(1)

# Define the column name containing file names
file_name_column = "file_name"

# Create a dictionary to store the content of each text file
modified_text = []

# Read the CSV file and populate the dictionary with file content
with open(csv_file_path, 'r') as csv_file:
    csv_reader = csv.DictReader(csv_file)

    script_file_path = "/afs/crc.nd.edu/user/r/rshi2/mimik/audios/transcriptions/"
    for row in csv_reader:
        file_name = row["transcription"]

        # Check if the file name ends with ".txt"
        if file_name.endswith(".txt"):
            file_path = os.path.join(os.path.dirname(script_file_path), file_name)

            # Check if the file exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as text_file:
                    file_content = text_file.read()
            
                # Store the content in the dictionary
                    print(file_content)
                    row["transcription"] = file_content
                    modified_text.append(row)
output_csv_path = 'output.csv'  # Replace with your desired output CSV file path
with open(output_csv_path, 'w', newline='') as output_file:
    fieldnames = csv_reader.fieldnames  # Get the field names from the input CSV
    csv_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
    csv_writer.writeheader()  # Write the header row

    # Write the modified rows to the output CSV file
    csv_writer.writerows(modified_text)
print(f"CSV file '{csv_file_path}' has been updated with file content.")

