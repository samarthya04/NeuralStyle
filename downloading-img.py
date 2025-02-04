import requests
import os
from serpapi import GoogleSearch

# Replace with your SerpAPI key
API_KEY = "26a42832f68a3fb2c572dd2ed728bc0cefcbab28791920814c40a443f4d95bd2"

def download_first_google_image(query, save_path="downloaded_image.jpg"):
    params = {
        "q": query,
        "tbm": "isch",  # Image search
        "num": 1,  # Get only 1 result
        "api_key": API_KEY
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    if "images_results" in results and results["images_results"]:
        image_url = results["images_results"][0]["original"]
        
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            print(f"Image downloaded successfully: {save_path}")
        else:
            print("Failed to download image.")
    else:
        print("No images found.")

# Example usage
download_first_google_image("starry night")
