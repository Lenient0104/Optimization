# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        text = file.read()
    return text


# Function to split text by spaces and store in a list (array)
def split_by_spaces(text):
    sentences = text.split()  # Split by spaces
    return sentences


# Main script
file_path = 'formatted_edges.txt'  # Replace with your file path
text = read_text_file(file_path)
sentences_array = split_by_spaces(text)

# Display the result as a list
print(sentences_array)
