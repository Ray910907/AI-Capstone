import requests
import os
import time
from PIL import Image
#Use Unsplash API to download pictures
ACCESS_KEY = "e_C8HzfEE_DUrCbB0QhH01FRSjoh2BQVRxSmHnwQMAw"
QUERY_LIST = ["Nature"]

IMG_COUNT = 1000 
SAVE_DIR = "dataset"
IMG_SIZE = (512,512)

os.makedirs(SAVE_DIR, exist_ok=True)
#Download pictures
def download_images(query, total_count):
    folder_path = SAVE_DIR
    os.makedirs(folder_path, exist_ok=True)
    
    remaining = total_count
    batch_size = 30  
    img_index = 0
    page = 1
    #Download pictures until tome run out, also we turn to nest page to download different pictures after the batch
    while remaining > 0:
        count = min(batch_size, remaining)
        url = f"https://api.unsplash.com/search/photos?query={query}&per_page={count}&page={page}&client_id={ACCESS_KEY}"

        try:
            response = requests.get(url)
            if response.status_code == 200:
                images = response.json()['results']
                for img in images:
                    img_url = img['urls']['regular']
                    img_data = requests.get(img_url).content
                    img_path = os.path.join(folder_path, f"{query}_{img_index}.jpg")

                    with open(img_path, 'wb') as f:
                        f.write(img_data)

                    resize_image(img_path)
                    print(f"Downloaded: {img_path}")
                    img_index += 1

                remaining -= count
                page += 1
                time.sleep(1)

            elif response.status_code == 429:
                print("Hit rate limit. Waiting for 30 seconds before retrying...")
                time.sleep(30)

            else:
                print(f"Error downloading {query} images, status code: {response.status_code}")
                break  

        except Exception as e:
            print(f"Error downloading {query} images: {e}")
            time.sleep(5)

#Maintaining the pictures consistent format
def resize_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.resize(IMG_SIZE, Image.LANCZOS)
            img.save(image_path)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        os.remove(image_path)



for category in QUERY_LIST:
    success = download_images(category,IMG_COUNT)

print("Done！")
