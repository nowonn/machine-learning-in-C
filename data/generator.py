import csv
import random
import numpy as np

def generate_csv(n, m):
    # Initialize random parameters as integers
    parameters = [random.randint(-10, 10) for _ in range(n+1)]
    print(f"Random parameters: {parameters}")

    # Open the CSV file in write mode
    with open('input.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        
        writer.writerow([f"{m} {n}"])

        # Generate m rows
        for _ in range(m):
            # Generate n random features for each row
            features = [random.randint(0, 12) for _ in range(n)]
            
            # Calculate the expected output (y) using dot product of features and parameters
            y = np.dot(features, parameters[:-1]) + parameters[-1]
            
            # Write the features and expected output to the CSV file
            writer.writerow(features + [y])

# Read n and m from standard input
n = int(input("Enter the value for n: "))
m = int(input("Enter the value for m: "))

# Call the function with your desired values of n and m
generate_csv(n, m)
