from dataPrep import split_all_cities_to_folders

if __name__ == "__main__":
    summary = split_all_cities_to_folders(
        raw_root="Resized_Images",
        output_root="resized_images_split",
        seed=42
    )

    print("Done! Split summary:")
    for city, counts in summary.items():
        print(city, counts)