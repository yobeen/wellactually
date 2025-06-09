import csv

def process_repo_comparisons(input_csv_file, output_txt_file):
    """
    Process CSV file with repo comparisons and output formatted statements.
    """
    output_lines = []
    
    with open(input_csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            repo_a = row['repo_a'].strip()
            repo_b = row['repo_b'].strip()
            choice = float(row['choice'])
            multiplier = float(row['multiplier'])
            
            # Extract just the repo name from the GitHub URL
            repo_a_name = repo_a.split('/')[-1] if repo_a.startswith('https://github.com/') else repo_a
            repo_b_name = repo_b.split('/')[-1] if repo_b.startswith('https://github.com/') else repo_b
            
            # Determine which repo is more important based on choice
            if choice == 1.0:
                # repo_a was chosen as more important
                statement = f"{repo_a} is [multiplier] times [compare] important than {repo_b}"
            elif choice == 2.0:
                # repo_b was chosen as more important
                statement = f"{repo_a} is [multiplier] times [compare] important than {repo_b}"
            else:
                # Handle unexpected choice values
                statement = f"Unexpected choice value {choice} for {repo_a_name} vs {repo_b_name}"
            
            output_lines.append(statement)
    
    # Write to output file
    with open(output_txt_file, 'w', encoding='utf-8') as txtfile:
        for line in output_lines:
            txtfile.write(line + '\n')
    
    print(f"Processed {len(output_lines)} comparisons.")
    print(f"Output saved to {output_txt_file}")
    
    # Also print to console for verification
    print("\nGenerated statements:")
    for line in output_lines:
        print(line)

# Usage
if __name__ == "__main__":
    input_file = "data/raw/train.csv"  # Replace with your CSV file name
    output_file = "comparison_results.txt"
    
    process_repo_comparisons(input_file, output_file)