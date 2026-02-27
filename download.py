import urllib.request

url = "https://png.pngtree.com/png-clipart/20230918/ourmid/pngtree-photo-men-doctor-physician-cheerful-studio-portrait-png-image_10132895.png"
req = urllib.request.Request(
    url, 
    data=None, 
    headers={
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'
    }
)

with urllib.request.urlopen(req) as response, open("avatar.png", 'wb') as out_file:
    data = response.read()
    out_file.write(data)
    print(f"Downloaded {len(data)} bytes.")
