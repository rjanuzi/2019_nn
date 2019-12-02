import csv
from datetime import datetime

def get_data():
    features = []
    targets = []
    weather_mains = []
    weather_descriptions = []
    with open(r'dataset\Metro_Interstate_Traffic_Volume_Data_set.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            temp_temp = float(row.get('temp'))
            if temp_temp < 230.0 or temp_temp > 320.0:
                continue # skip

            temp_rain = float(row.get('rain_1h'))
            if temp_rain > 100:
                continue

            temp_weather_main = row.get('weather_main').strip().replace(' ', '_')
            if temp_weather_main not in weather_mains:
                weather_mains.append(temp_weather_main)

            temp_weather_description = row.get('weather_description').strip().replace(' ', '_')
            if temp_weather_description not in weather_descriptions:
                weather_descriptions.append(temp_weather_description)

            temp_datetime = datetime.strptime(row.get('date_time'), '%Y-%m-%d %H:%M:%S')

            features.append([
                                temp_temp,
                                temp_rain,
                                float(row.get('snow_1h')),
                                float(row.get('clouds_all')),
                                weather_mains.index(temp_weather_main),
                                weather_descriptions.index(temp_weather_description),
                                temp_datetime.year,
                                temp_datetime.month,
                                temp_datetime.day,
                                temp_datetime.hour
            ])

            targets.append(float(row.get('traffic_volume')))

    return features, targets
