import os
from PIL import Image

input_folder = "archive/Images" 
output_folder = "Resized_Images" 

#target size required by most neural networks
target_size = (224, 224) 

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for city_name in os.listdir(input_folder):
    city_path = os.path.join(input_folder, city_name)
    
    if os.path.isdir(city_path):
        print(f"Processing city: {city_name}...")
        
        out_city_path = os.path.join(output_folder, city_name)
        if not os.path.exists(out_city_path):
            os.makedirs(out_city_path)
            
        for file_name in os.listdir(city_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(city_path, file_name)
                out_img_path = os.path.join(out_city_path, file_name)
                
                try:
                    with Image.open(img_path) as img:
                        #Converting to RGB just in case some images are grayscale or have transparent backgrounds
                        img = img.convert('RGB') 
                        img_resized = img.resize(target_size)
                        img_resized.save(out_img_path)
                except Exception as e:
                    print(f"Failed to process {file_name}: {e}")

print("\nAll images have been successfully resized to 224x224!")