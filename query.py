import pandas as pd
import os


cities = [
    "Bangkok", "Barcelona", "Boston", "Brussels", "BuenosAires", 
    "Chicago", "Lisbon", "London", "LosAngeles", "Madrid", 
    "Medellin", "Melbourne", "MexicoCity", "Miami", "Minneapolis", 
    "OSL", "Osaka", "PRG", "PRS", "Phoenix", "Rome", "TRT", "WashingtonDC"
]

print("Available Cities:")
for index, city in enumerate(cities):
    print(f"{index + 1}. {city}")

#Asking the user which city they want
try:
    city_choice = int(input("\nEnter the number of the city you want to view: ")) - 1
    selected_city = cities[city_choice]
except (ValueError, IndexError):
    print("Invalid selection. Please run the script again and choose a valid number.")
    exit() 
#Asking the user how many rows they want to see
try:
    row_count = int(input(f"How many rows of data for {selected_city} would you like to see? "))
except ValueError:
    print("Invalid number entered. Defaulting to 5 rows.")
    row_count = 5

file_path = f"archive/Dataframes/{selected_city}.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print(f"\n--- Displaying the first {row_count} rows for {selected_city} ---")
    print(df.head(row_count))
else:
    print(f"\nError: Could not find the file at {file_path}.")