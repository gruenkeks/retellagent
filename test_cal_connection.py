import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_cal_api():
    api_key = os.environ.get("CAL_API_KEY")
    event_type_id = os.environ.get("CAL_EVENT_TYPE_ID")
    
    if not api_key:
        print("❌ CAL_API_KEY is missing in environment.")
        return
    if not event_type_id:
        print("❌ CAL_EVENT_TYPE_ID is missing in environment.")
        return

    print(f"✅ Found API Key: {api_key[:4]}...{api_key[-4:]}")
    print(f"✅ Found Event Type ID: {event_type_id}")

    # Test v2 slots endpoint
    url = "https://api.cal.com/v2/slots"
    
    # Use a dummy time range (e.g. tomorrow)
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    start = (now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (now + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    params = {
        "eventTypeId": int(event_type_id),
        "start": start,
        "end": end,
        "timeZone": "Europe/Berlin",
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "cal-api-version": "2024-08-13",
    }
    
    print(f"\nTesting Connection to {url}...")
    print(f"Params: {params}")
    
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.text[:500]}...") # Print first 500 chars
        
        if r.status_code == 200:
            print("\n✅ API Call Successful!")
            try:
                data = r.json()
                if "data" in data and "slots" in data["data"]:
                     print(f"Found {len(data['data']['slots'])} slots.")
                else:
                     print("Response JSON structure seems unexpected (v2 format check).")
            except:
                pass
        else:
            print("\n❌ API Call Failed.")
            
    except Exception as e:
        print(f"\n❌ Exception during request: {e}")

if __name__ == "__main__":
    test_cal_api()

