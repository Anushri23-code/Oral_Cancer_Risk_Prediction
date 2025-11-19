import threading
import time
import requests

def keep_alive():
    url = "https://oral-cancer-risk-prediction1.onrender.com"  # your Render URL
    
    while True:
        try:
            requests.get(url)
            print("Pinged server to keep alive.")
        except:
            print("Ping failed.")
        time.sleep(240)  # ping every 4 minutes

def start_keep_alive():
    thread = threading.Thread(target=keep_alive)
    thread.daemon = True
    thread.start()
