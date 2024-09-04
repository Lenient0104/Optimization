import json
import requests

url = "https://api.openchargemap.io/v3/poi/?output=json&countrycode=IE&maxresults=10?key=b865d350-4204-453a-ae91-a8bb29e1b645"


response = requests.get(url)
# Check if the request was successful
if response.status_code == 200:
    data = response.json()

    # Save the data to a JSON file
    with open('map_data.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)  # 'indent=4' is used for pretty printing the JSON

    print("Data saved to gtfs_data.json")
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")
