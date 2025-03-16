import requests
import pandas as pd
from geopy.geocoders import Nominatim
from time import sleep


CITIES = [
    'Quito', 'Santo Domingo', 'Cayambe', 'Latacunga', 'Riobamba',
    'Ibarra', 'Guaranda', 'Puyo', 'Ambato', 'Guayaquil', 'Salinas',
    'Daule', 'Babahoyo', 'Quevedo', 'Playas', 'Libertad', 'Cuenca',
    'Loja', 'Machala', 'Esmeraldas', 'Manta', 'El Carmen'
]
START_DATE = "2013-01-01"
END_DATE = "2017-08-15"


def get_coordinates(city):
    geolocator = Nominatim(user_agent="weather_app")
    location = geolocator.geocode(f"{city}, Ecuador")
    return (location.latitude, location.longitude) if location else (None, None)


def get_weather_data(lat, lon):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        f"&daily=temperature_2m_mean,precipitation_sum"
        f"&timezone=auto"
    )
    response = requests.get(url)
    return response.json()


all_data = []
for city in CITIES:
    lat, lon = get_coordinates(city)
    if not lat or not lon:
        print(f"Coordinates not found for {city}")
        continue
    
    try:
        data = get_weather_data(lat, lon)
        df = pd.DataFrame({
            'date': data['daily']['time'],
            'temperature': data['daily']['temperature_2m_mean'],
            'precipitation': data['daily']['precipitation_sum'],
            'city': city
        })
        all_data.append(df)
        print(f"Success: {city}")
        sleep(1)
    except Exception as e:
        print(f"Failed {city}: {str(e)}")


if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv("ecuador_weather_2013-2017.csv", index=False)
    print("Data saved to ecuador_weather_2013-2017.csv")
else:
    print("No data retrieved")