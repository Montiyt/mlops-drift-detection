import requests
import random
import time

url = "http://localhost:8000/predict"

print("Sending 50 test predictions...")
for i in range(50):
    payload = {
        "trip_distance": round(random.uniform(0.5, 15.0), 2),
        "fare_amount":   round(random.uniform(3.0, 60.0), 2),
        "PULocationID":  random.randint(1, 263),
        "DOLocationID":  random.randint(1, 263),
        "passenger_count": random.randint(1, 4)
    }
    r = requests.post(url, json=payload)
    print(f"  [{i+1}] {r.json()['predicted_duration_minutes']} mins")
    time.sleep(0.2)

print("\nDone! Check Grafana dashboard now.")