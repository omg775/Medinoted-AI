import csv
import json
import urllib.request
import os

def generate_facilities():
    try:
        url = "https://raw.githubusercontent.com/google/dspl/master/samples/google/canonical/countries.csv"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').splitlines()
            
        reader = csv.DictReader(lines)
        facilities = []
        
        for row in reader:
            country = row['name']
            if not country: continue
            
            try:
                lat = float(row['latitude'])
                lon = float(row['longitude'])
            except ValueError:
                continue
                
            facilities.append({
                "country": country,
                "city": "Capital City",
                "facility_name": f"{country} National Medical Center",
                "type": "Network Hospital / ER",
                "languages_supported": ["English", "Local Dialect"],
                "phone": "+1 800 123 4567",
                "address": f"1 Healthcare Avenue, {country}",
                "lat": lat,
                "lon": lon
            })
            
        os.makedirs("data", exist_ok=True)
        with open("data/partner_facilities.json", "w") as f:
            json.dump(facilities, f, indent=2)
            
        print(f"Generated {len(facilities)} facilities.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    generate_facilities()
