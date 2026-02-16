import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)


@dataclass
class Sample:
    sid: str
    text: str          
    schema: str         
    reference: str    


# toy dataset as fallback
TOY_DATASET = [
    {
        "id": "person-1",
        "text": "John Smith is a 35-year-old software engineer from San Francisco. He has been working at TechCorp for 8 years and specializes in machine learning.",
        "schema": "name, age, occupation, city, company, years_experience, specialty",
        "reference": {
            "name": "John Smith",
            "age": 35,
            "occupation": "software engineer",
            "city": "San Francisco",
            "company": "TechCorp",
            "years_experience": 8,
            "specialty": "machine learning"
        }
    },
    {
        "id": "person-2",
        "text": "Dr. Maria Garcia, aged 42, is a cardiologist at Boston General Hospital. She graduated from Harvard Medical School and has published over 50 research papers.",
        "schema": "name, age, occupation, workplace, education, publications",
        "reference": {
            "name": "Maria Garcia",
            "age": 42,
            "occupation": "cardiologist",
            "workplace": "Boston General Hospital",
            "education": "Harvard Medical School",
            "publications": 50
        }
    },
    {
        "id": "place-1",
        "text": "The Eiffel Tower is located in Paris, France. It was built in 1889 and stands 330 meters tall. It attracts approximately 7 million visitors annually.",
        "schema": "name, city, country, year_built, height_meters, annual_visitors",
        "reference": {
            "name": "Eiffel Tower",
            "city": "Paris",
            "country": "France",
            "year_built": 1889,
            "height_meters": 330,
            "annual_visitors": 7000000
        }
    },
    {
        "id": "place-2",
        "text": "Central Park spans 843 acres in Manhattan, New York City. It was designed by Frederick Law Olmsted and opened in 1858. The park features 21 playgrounds and 36 bridges.",
        "schema": "name, size_acres, location, designer, year_opened, playgrounds, bridges",
        "reference": {
            "name": "Central Park",
            "size_acres": 843,
            "location": "Manhattan, New York City",
            "designer": "Frederick Law Olmsted",
            "year_opened": 1858,
            "playgrounds": 21,
            "bridges": 36
        }
    },
    {
        "id": "product-1",
        "text": "The iPhone 15 Pro is manufactured by Apple and retails for $999. It features a 6.1-inch display, 256GB storage, and an A17 Pro chip. Available in titanium finish.",
        "schema": "name, manufacturer, price_usd, display_inches, storage_gb, processor, finish",
        "reference": {
            "name": "iPhone 15 Pro",
            "manufacturer": "Apple",
            "price_usd": 999,
            "display_inches": 6.1,
            "storage_gb": 256,
            "processor": "A17 Pro",
            "finish": "titanium"
        }
    },
    {
        "id": "product-2",
        "text": "Sony WH-1000XM5 wireless headphones cost $349 and offer 30 hours of battery life. They feature active noise cancellation and weigh only 250 grams.",
        "schema": "name, brand, price_usd, battery_hours, noise_cancellation, weight_grams",
        "reference": {
            "name": "WH-1000XM5",
            "brand": "Sony",
            "price_usd": 349,
            "battery_hours": 30,
            "noise_cancellation": True,
            "weight_grams": 250
        }
    },
    {
        "id": "person-3",
        "text": "Emily Chen, 28, works as a data analyst at DataFlow Inc in Seattle. She holds a Master's degree in Statistics and earns an annual salary of $95,000.",
        "schema": "name, age, occupation, company, city, degree, salary_usd",
        "reference": {
            "name": "Emily Chen",
            "age": 28,
            "occupation": "data analyst",
            "company": "DataFlow Inc",
            "city": "Seattle",
            "degree": "Master's in Statistics",
            "salary_usd": 95000
        }
    },
    {
        "id": "place-3",
        "text": "The Grand Canyon National Park in Arizona covers 1,217,262 acres. It was established in 1919 and receives about 6 million visitors per year. The canyon is up to 18 miles wide.",
        "schema": "name, state, size_acres, year_established, annual_visitors, max_width_miles",
        "reference": {
            "name": "Grand Canyon National Park",
            "state": "Arizona",
            "size_acres": 1217262,
            "year_established": 1919,
            "annual_visitors": 6000000,
            "max_width_miles": 18
        }
    },
    {
        "id": "product-3",
        "text": "The Tesla Model 3 is an electric vehicle with a range of 272 miles. It accelerates from 0-60 mph in 5.8 seconds and has a starting price of $38,990. Seats 5 passengers.",
        "schema": "name, type, range_miles, acceleration_0_60, price_usd, seating_capacity",
        "reference": {
            "name": "Tesla Model 3",
            "type": "electric vehicle",
            "range_miles": 272,
            "acceleration_0_60": 5.8,
            "price_usd": 38990,
            "seating_capacity": 5
        }
    },
    {
        "id": "person-4",
        "text": "Chef Antonio Rossi, 55, owns three Italian restaurants in Chicago. He trained in Rome for 10 years and has won 2 Michelin stars. His signature dish is handmade pasta.",
        "schema": "name, age, occupation, num_restaurants, city, training_location, training_years, michelin_stars, signature_dish",
        "reference": {
            "name": "Antonio Rossi",
            "age": 55,
            "occupation": "chef",
            "num_restaurants": 3,
            "city": "Chicago",
            "training_location": "Rome",
            "training_years": 10,
            "michelin_stars": 2,
            "signature_dish": "handmade pasta"
        }
    },
    {
        "id": "person-5",
        "text": "Dr. James Wilson, 48, is a neurosurgeon at Mayo Clinic in Rochester. He completed his residency at Johns Hopkins and has performed over 2000 surgeries.",
        "schema": "name, age, occupation, workplace, city, residency, surgeries_performed",
        "reference": {"name": "James Wilson", "age": 48, "occupation": "neurosurgeon", "workplace": "Mayo Clinic", "city": "Rochester", "residency": "Johns Hopkins", "surgeries_performed": 2000}
    },
    {
        "id": "person-6",
        "text": "Sarah Kim, a 31-year-old graphic designer, freelances from Austin, Texas. She has 12 years of experience and charges $85 per hour. Her portfolio includes 200 projects.",
        "schema": "name, age, occupation, city, state, experience_years, hourly_rate_usd, portfolio_projects",
        "reference": {"name": "Sarah Kim", "age": 31, "occupation": "graphic designer", "city": "Austin", "state": "Texas", "experience_years": 12, "hourly_rate_usd": 85, "portfolio_projects": 200}
    },
    {
        "id": "person-7",
        "text": "Professor Li Wei, 60, teaches physics at MIT. He has authored 8 textbooks and holds 15 patents. He received his PhD from Cambridge University in 1990.",
        "schema": "name, age, occupation, university, textbooks, patents, phd_university, phd_year",
        "reference": {"name": "Li Wei", "age": 60, "occupation": "physics professor", "university": "MIT", "textbooks": 8, "patents": 15, "phd_university": "Cambridge University", "phd_year": 1990}
    },
    {
        "id": "person-8",
        "text": "Olympic swimmer Maya Johnson, 24, from Sydney, Australia, has won 5 gold medals. She trains 6 hours daily and holds the 200m freestyle world record at 1:52.3.",
        "schema": "name, age, sport, city, country, gold_medals, training_hours_daily, world_record_event, world_record_time",
        "reference": {"name": "Maya Johnson", "age": 24, "sport": "swimming", "city": "Sydney", "country": "Australia", "gold_medals": 5, "training_hours_daily": 6, "world_record_event": "200m freestyle", "world_record_time": "1:52.3"}
    },
    {
        "id": "place-4",
        "text": "The Colosseum in Rome, Italy, was completed in 80 AD and could seat 50,000 spectators. It is 189 meters long and 156 meters wide. It is a UNESCO World Heritage Site.",
        "schema": "name, city, country, year_completed, capacity, length_meters, width_meters, heritage_status",
        "reference": {"name": "Colosseum", "city": "Rome", "country": "Italy", "year_completed": 80, "capacity": 50000, "length_meters": 189, "width_meters": 156, "heritage_status": "UNESCO World Heritage Site"}
    },
    {
        "id": "place-5",
        "text": "Lake Baikal in Siberia, Russia, is the deepest lake in the world at 1,642 meters. It contains 20% of the world's unfrozen fresh water and is approximately 25 million years old.",
        "schema": "name, region, country, depth_meters, freshwater_percentage, age_million_years",
        "reference": {"name": "Lake Baikal", "region": "Siberia", "country": "Russia", "depth_meters": 1642, "freshwater_percentage": 20, "age_million_years": 25}
    },
    {
        "id": "place-6",
        "text": "The Burj Khalifa in Dubai, UAE, stands 828 meters tall with 163 floors. It was completed in 2010 and cost $1.5 billion to build. It has 57 elevators.",
        "schema": "name, city, country, height_meters, floors, year_completed, cost_billion_usd, elevators",
        "reference": {"name": "Burj Khalifa", "city": "Dubai", "country": "UAE", "height_meters": 828, "floors": 163, "year_completed": 2010, "cost_billion_usd": 1.5, "elevators": 57}
    },
    {
        "id": "product-4",
        "text": "The MacBook Pro 16-inch by Apple features an M3 Max chip and 36GB of RAM. It has a 16.2-inch Liquid Retina XDR display, 1TB SSD, and costs $3,499. Battery life is up to 22 hours.",
        "schema": "name, manufacturer, processor, ram_gb, display_inches, storage_tb, price_usd, battery_hours",
        "reference": {"name": "MacBook Pro 16-inch", "manufacturer": "Apple", "processor": "M3 Max", "ram_gb": 36, "display_inches": 16.2, "storage_tb": 1, "price_usd": 3499, "battery_hours": 22}
    },
    {
        "id": "product-5",
        "text": "The Samsung Galaxy S24 Ultra has a 6.8-inch display, 200MP camera, and 5000mAh battery. It runs on Snapdragon 8 Gen 3 processor and starts at $1,299 with 256GB storage.",
        "schema": "name, display_inches, camera_mp, battery_mah, processor, price_usd, storage_gb",
        "reference": {"name": "Samsung Galaxy S24 Ultra", "display_inches": 6.8, "camera_mp": 200, "battery_mah": 5000, "processor": "Snapdragon 8 Gen 3", "price_usd": 1299, "storage_gb": 256}
    },
    {
        "id": "product-6",
        "text": "The Dyson V15 Detect vacuum weighs 3.1 kg and provides up to 60 minutes of runtime. It has a 0.76 liter bin capacity, uses a 660W motor, and retails for $749.",
        "schema": "name, weight_kg, runtime_minutes, bin_capacity_liters, motor_watts, price_usd",
        "reference": {"name": "Dyson V15 Detect", "weight_kg": 3.1, "runtime_minutes": 60, "bin_capacity_liters": 0.76, "motor_watts": 660, "price_usd": 749}
    },
    {
        "id": "person-9",
        "text": "Dr. Anika Patel, 39, is a pediatrician in Denver, Colorado. She graduated from Stanford Medical School and has been practicing for 11 years. She sees about 30 patients per day.",
        "schema": "name, age, occupation, city, state, medical_school, years_practicing, patients_per_day",
        "reference": {"name": "Anika Patel", "age": 39, "occupation": "pediatrician", "city": "Denver", "state": "Colorado", "medical_school": "Stanford Medical School", "years_practicing": 11, "patients_per_day": 30}
    },
    {
        "id": "person-10",
        "text": "Marcus Thompson, 45, is a civil engineer who built 12 bridges across Oregon. He works for StructureCo, earns $120,000 annually, and has a Professional Engineer license.",
        "schema": "name, age, occupation, bridges_built, state, company, salary_usd, license",
        "reference": {"name": "Marcus Thompson", "age": 45, "occupation": "civil engineer", "bridges_built": 12, "state": "Oregon", "company": "StructureCo", "salary_usd": 120000, "license": "Professional Engineer"}
    },
    {
        "id": "place-7",
        "text": "Yellowstone National Park spans 2,219,789 acres across Wyoming, Montana, and Idaho. It was established in 1872 as the first national park. It has over 500 active geysers.",
        "schema": "name, size_acres, states, year_established, distinction, active_geysers",
        "reference": {"name": "Yellowstone National Park", "size_acres": 2219789, "states": "Wyoming, Montana, and Idaho", "year_established": 1872, "distinction": "first national park", "active_geysers": 500}
    },
    {
        "id": "place-8",
        "text": "The Great Wall of China stretches 21,196 kilometers. Construction began in the 7th century BC. It is visible from low Earth orbit and attracts 10 million visitors annually.",
        "schema": "name, length_km, construction_started, annual_visitors",
        "reference": {"name": "Great Wall of China", "length_km": 21196, "construction_started": "7th century BC", "annual_visitors": 10000000}
    },
    {
        "id": "product-7",
        "text": "The Nintendo Switch OLED has a 7-inch OLED screen with 64GB internal storage. It weighs 420 grams, costs $349, and has a battery life of 4.5 to 9 hours. Supports up to 8 players.",
        "schema": "name, screen_inches, storage_gb, weight_grams, price_usd, battery_hours_max, max_players",
        "reference": {"name": "Nintendo Switch OLED", "screen_inches": 7, "storage_gb": 64, "weight_grams": 420, "price_usd": 349, "battery_hours_max": 9, "max_players": 8}
    },
    {
        "id": "product-8",
        "text": "The Bose QuietComfort Ultra earbuds offer 6 hours of battery life with ANC enabled. They are IPX4 water resistant, cost $299, and weigh 6.24 grams per earbud.",
        "schema": "name, brand, battery_hours, water_resistance, price_usd, weight_grams_each",
        "reference": {"name": "QuietComfort Ultra", "brand": "Bose", "battery_hours": 6, "water_resistance": "IPX4", "price_usd": 299, "weight_grams_each": 6.24}
    },
    {
        "id": "person-11",
        "text": "Journalist Rosa Martinez, 33, writes for The Washington Post in Washington, DC. She has published 450 articles and won 3 journalism awards. She covers climate policy.",
        "schema": "name, age, occupation, employer, city, articles_published, awards, beat",
        "reference": {"name": "Rosa Martinez", "age": 33, "occupation": "journalist", "employer": "The Washington Post", "city": "Washington, DC", "articles_published": 450, "awards": 3, "beat": "climate policy"}
    },
    {
        "id": "person-12",
        "text": "Firefighter David Park, 41, has served 18 years at Station 7 in Portland. He has responded to over 3,000 emergency calls and earned the Medal of Valor in 2019.",
        "schema": "name, age, occupation, years_served, station, city, emergency_calls, medal, medal_year",
        "reference": {"name": "David Park", "age": 41, "occupation": "firefighter", "years_served": 18, "station": "Station 7", "city": "Portland", "emergency_calls": 3000, "medal": "Medal of Valor", "medal_year": 2019}
    },
    {
        "id": "place-9",
        "text": "Mount Everest stands at 8,849 meters in the Himalayas on the Nepal-Tibet border. The first successful summit was in 1953 by Edmund Hillary. Over 6,000 people have reached the top.",
        "schema": "name, height_meters, mountain_range, border, first_summit_year, first_climber, total_summits",
        "reference": {"name": "Mount Everest", "height_meters": 8849, "mountain_range": "Himalayas", "border": "Nepal-Tibet", "first_summit_year": 1953, "first_climber": "Edmund Hillary", "total_summits": 6000}
    },
    {
        "id": "place-10",
        "text": "The Louvre Museum in Paris, France, houses 380,000 objects including the Mona Lisa. It covers 72,735 square meters, was established in 1793, and receives 7.8 million visitors annually.",
        "schema": "name, city, country, total_objects, famous_work, area_sqm, year_established, annual_visitors",
        "reference": {"name": "Louvre Museum", "city": "Paris", "country": "France", "total_objects": 380000, "famous_work": "Mona Lisa", "area_sqm": 72735, "year_established": 1793, "annual_visitors": 7800000}
    },
    {
        "id": "product-9",
        "text": "The LG C3 65-inch OLED TV has a 4K resolution with 120Hz refresh rate. It supports Dolby Vision and costs $1,499. Power consumption is 118 watts. It weighs 18.2 kg.",
        "schema": "name, screen_inches, resolution, refresh_rate_hz, hdr_format, price_usd, power_watts, weight_kg",
        "reference": {"name": "LG C3 OLED", "screen_inches": 65, "resolution": "4K", "refresh_rate_hz": 120, "hdr_format": "Dolby Vision", "price_usd": 1499, "power_watts": 118, "weight_kg": 18.2}
    },
    {
        "id": "product-10",
        "text": "The Kindle Paperwhite by Amazon has a 6.8-inch display with 300 PPI. It holds up to 16GB of storage, costs $149, is IPX8 waterproof, and lasts up to 10 weeks on a single charge.",
        "schema": "name, manufacturer, display_inches, ppi, storage_gb, price_usd, water_resistance, battery_weeks",
        "reference": {"name": "Kindle Paperwhite", "manufacturer": "Amazon", "display_inches": 6.8, "ppi": 300, "storage_gb": 16, "price_usd": 149, "water_resistance": "IPX8", "battery_weeks": 10}
    },
    {
        "id": "person-13",
        "text": "Architect Yuki Tanaka, 52, designed the Tokyo Sky Tower and 30 other buildings. She founded Tanaka Design Studio in 2005 with 45 employees. She won the Pritzker Prize in 2021.",
        "schema": "name, age, occupation, notable_work, buildings_designed, company, founded_year, employees, award, award_year",
        "reference": {"name": "Yuki Tanaka", "age": 52, "occupation": "architect", "notable_work": "Tokyo Sky Tower", "buildings_designed": 30, "company": "Tanaka Design Studio", "founded_year": 2005, "employees": 45, "award": "Pritzker Prize", "award_year": 2021}
    },
    {
        "id": "person-14",
        "text": "Veterinarian Carlos Ruiz, 37, runs an animal clinic in Miami treating 25 animals daily. He specializes in exotic pets and has treated over 8,000 animals in his 9-year career.",
        "schema": "name, age, occupation, city, patients_daily, specialty, total_patients, career_years",
        "reference": {"name": "Carlos Ruiz", "age": 37, "occupation": "veterinarian", "city": "Miami", "patients_daily": 25, "specialty": "exotic pets", "total_patients": 8000, "career_years": 9}
    },
    {
        "id": "place-11",
        "text": "Machu Picchu sits at 2,430 meters altitude in the Andes of Peru. Built around 1450 by the Incas, it was rediscovered in 1911 by Hiram Bingham. It covers about 13 square kilometers.",
        "schema": "name, altitude_meters, mountain_range, country, year_built, civilization, rediscovered_year, discoverer, area_sqkm",
        "reference": {"name": "Machu Picchu", "altitude_meters": 2430, "mountain_range": "Andes", "country": "Peru", "year_built": 1450, "civilization": "Incas", "rediscovered_year": 1911, "discoverer": "Hiram Bingham", "area_sqkm": 13}
    },
    {
        "id": "place-12",
        "text": "The Sydney Opera House in Sydney, Australia, was designed by Jorn Utzon and opened in 1973. It hosts over 1,500 performances annually and cost $102 million to build.",
        "schema": "name, city, country, architect, year_opened, annual_performances, construction_cost_million",
        "reference": {"name": "Sydney Opera House", "city": "Sydney", "country": "Australia", "architect": "Jorn Utzon", "year_opened": 1973, "annual_performances": 1500, "construction_cost_million": 102}
    },
    {
        "id": "product-11",
        "text": "The GoPro Hero 12 Black shoots 5.3K video at 60fps. It is waterproof to 10 meters, weighs 154 grams, costs $399, and has a 1720mAh battery lasting approximately 70 minutes.",
        "schema": "name, video_resolution, fps, waterproof_meters, weight_grams, price_usd, battery_mah, recording_minutes",
        "reference": {"name": "GoPro Hero 12 Black", "video_resolution": "5.3K", "fps": 60, "waterproof_meters": 10, "weight_grams": 154, "price_usd": 399, "battery_mah": 1720, "recording_minutes": 70}
    },
    {
        "id": "product-12",
        "text": "The Roomba j7+ robot vacuum by iRobot has a self-emptying base, maps rooms with PrecisionVision navigation, runs for 75 minutes per charge, and costs $599.",
        "schema": "name, manufacturer, self_emptying, navigation_system, runtime_minutes, price_usd",
        "reference": {"name": "Roomba j7+", "manufacturer": "iRobot", "self_emptying": True, "navigation_system": "PrecisionVision", "runtime_minutes": 75, "price_usd": 599}
    },
    {
        "id": "person-15",
        "text": "Pilot Hannah Okafor, 34, flies Boeing 787s for United Airlines. She has logged 8,500 flight hours across 45 countries and has been flying commercially for 10 years.",
        "schema": "name, age, occupation, aircraft, airline, flight_hours, countries_visited, career_years",
        "reference": {"name": "Hannah Okafor", "age": 34, "occupation": "pilot", "aircraft": "Boeing 787", "airline": "United Airlines", "flight_hours": 8500, "countries_visited": 45, "career_years": 10}
    },
    {
        "id": "person-16",
        "text": "Baker Sophie Laurent, 29, owns a patisserie in Lyon, France. She produces 500 pastries daily with a team of 6, and her shop has a 4.9 star rating from 2,000 reviews.",
        "schema": "name, age, occupation, city, country, daily_production, team_size, rating, num_reviews",
        "reference": {"name": "Sophie Laurent", "age": 29, "occupation": "baker", "city": "Lyon", "country": "France", "daily_production": 500, "team_size": 6, "rating": 4.9, "num_reviews": 2000}
    },
    {
        "id": "place-13",
        "text": "The Amazon Rainforest covers 5.5 million square kilometers across 9 countries. It produces 20% of the world's oxygen and is home to approximately 10% of all species on Earth.",
        "schema": "name, area_sqkm, countries_count, oxygen_percentage, species_percentage",
        "reference": {"name": "Amazon Rainforest", "area_sqkm": 5500000, "countries_count": 9, "oxygen_percentage": 20, "species_percentage": 10}
    },
    {
        "id": "place-14",
        "text": "The International Space Station orbits Earth at 408 kilometers altitude, traveling at 28,000 km/h. It was launched in 1998, weighs 420,000 kg, and has been continuously occupied since 2000.",
        "schema": "name, altitude_km, speed_kmh, launch_year, weight_kg, occupied_since",
        "reference": {"name": "International Space Station", "altitude_km": 408, "speed_kmh": 28000, "launch_year": 1998, "weight_kg": 420000, "occupied_since": 2000}
    },
    {
        "id": "product-13",
        "text": "The Peloton Bike+ features a 23.8-inch rotating touchscreen, 24 resistance levels, and built-in speakers. It costs $2,495, weighs 64 kg, and requires a $44/month subscription.",
        "schema": "name, screen_inches, resistance_levels, price_usd, weight_kg, monthly_subscription_usd",
        "reference": {"name": "Peloton Bike+", "screen_inches": 23.8, "resistance_levels": 24, "price_usd": 2495, "weight_kg": 64, "monthly_subscription_usd": 44}
    },
    {
        "id": "product-14",
        "text": "The DJI Mini 4 Pro drone weighs 249 grams and shoots 4K video at 100fps. It has a 34-minute flight time, 20km transmission range, and costs $759. Obstacle sensing in all directions.",
        "schema": "name, weight_grams, video_resolution, fps, flight_time_minutes, range_km, price_usd, obstacle_sensing",
        "reference": {"name": "DJI Mini 4 Pro", "weight_grams": 249, "video_resolution": "4K", "fps": 100, "flight_time_minutes": 34, "range_km": 20, "price_usd": 759, "obstacle_sensing": "all directions"}
    },
    {
        "id": "person-17",
        "text": "Marine biologist Dr. Nadia Scott, 44, works at the Monterey Bay Aquarium Research Institute. She has discovered 7 new species and led 25 deep-sea expeditions over 16 years.",
        "schema": "name, age, occupation, institution, species_discovered, expeditions, career_years",
        "reference": {"name": "Nadia Scott", "age": 44, "occupation": "marine biologist", "institution": "Monterey Bay Aquarium Research Institute", "species_discovered": 7, "expeditions": 25, "career_years": 16}
    },
    {
        "id": "person-18",
        "text": "Photographer Alex Rivera, 38, has won 4 Pulitzer Prizes. Based in New York, he has covered conflicts in 12 countries and his work has appeared in National Geographic 15 times.",
        "schema": "name, age, occupation, awards, award_name, city, countries_covered, publication, publication_appearances",
        "reference": {"name": "Alex Rivera", "age": 38, "occupation": "photographer", "awards": 4, "award_name": "Pulitzer Prize", "city": "New York", "countries_covered": 12, "publication": "National Geographic", "publication_appearances": 15}
    },
    {
        "id": "place-15",
        "text": "Venice, Italy, is built on 118 small islands connected by 400 bridges. The city has 177 canals, was founded in 421 AD, and receives approximately 30 million tourists per year.",
        "schema": "name, country, islands, bridges, canals, year_founded, annual_tourists",
        "reference": {"name": "Venice", "country": "Italy", "islands": 118, "bridges": 400, "canals": 177, "year_founded": 421, "annual_tourists": 30000000}
    },
    {
        "id": "place-16",
        "text": "The Sahara Desert covers 9.2 million square kilometers across 11 countries in North Africa. Temperatures can reach 58 degrees Celsius, and it receives less than 25mm of rain annually.",
        "schema": "name, area_sqkm, countries_count, region, max_temperature_celsius, annual_rainfall_mm",
        "reference": {"name": "Sahara Desert", "area_sqkm": 9200000, "countries_count": 11, "region": "North Africa", "max_temperature_celsius": 58, "annual_rainfall_mm": 25}
    },
    {
        "id": "product-15",
        "text": "The Sonos Era 300 speaker delivers spatial audio with Dolby Atmos support. It costs $449, weighs 4.47 kg, connects via WiFi 6 and Bluetooth 5.2, and supports AirPlay 2.",
        "schema": "name, audio_feature, dolby_support, price_usd, weight_kg, wifi_version, bluetooth_version, airplay",
        "reference": {"name": "Sonos Era 300", "audio_feature": "spatial audio", "dolby_support": "Dolby Atmos", "price_usd": 449, "weight_kg": 4.47, "wifi_version": "WiFi 6", "bluetooth_version": "Bluetooth 5.2", "airplay": True}
    },
    {
        "id": "product-16",
        "text": "The Vitamix A3500 blender has a 2.2 HP motor with 10 variable speeds. It holds 64 ounces, costs $649, and comes with a 10-year warranty. It features wireless connectivity.",
        "schema": "name, motor_hp, speeds, capacity_oz, price_usd, warranty_years, wireless",
        "reference": {"name": "Vitamix A3500", "motor_hp": 2.2, "speeds": 10, "capacity_oz": 64, "price_usd": 649, "warranty_years": 10, "wireless": True}
    },
    {
        "id": "person-19",
        "text": "Robotics engineer Priya Sharma, 36, leads a team of 20 at Boston Dynamics. She holds 9 patents, earned her PhD from Carnegie Mellon, and has published 35 research papers.",
        "schema": "name, age, occupation, team_size, company, patents, phd_university, publications",
        "reference": {"name": "Priya Sharma", "age": 36, "occupation": "robotics engineer", "team_size": 20, "company": "Boston Dynamics", "patents": 9, "phd_university": "Carnegie Mellon", "publications": 35}
    },
    {
        "id": "person-20",
        "text": "Sommelier Jean-Pierre Dubois, 50, manages the wine cellar at Le Bernardin in New York. The collection includes 15,000 bottles from 22 countries. He has 28 years of experience.",
        "schema": "name, age, occupation, restaurant, city, bottles, countries, experience_years",
        "reference": {"name": "Jean-Pierre Dubois", "age": 50, "occupation": "sommelier", "restaurant": "Le Bernardin", "city": "New York", "bottles": 15000, "countries": 22, "experience_years": 28}
    },
]


def load_samples(cfg: Dict[str, Any]) -> List[Sample]:
    """
    Load JSON extraction samples.
    
    Supports multiple sources:
    - "toy": Use built-in toy dataset (50 samples) - clean, reliable ground truth
    - "ner": Use CoNLL-2003 NER dataset from HuggingFace (entities extraction)
    - "json_struct": Use MasterControlAIML/JSON-Unstructured-Structured from HuggingFace
    """
    dataset_cfg = cfg.get("dataset", {})
    source = dataset_cfg.get("source", "toy")
    n = int(dataset_cfg.get("n_samples", 10))
    
    if source == "toy":
        samples = _load_toy_samples(n)
    elif source == "ner":
        samples = _load_ner_samples(n)
    elif source == "json_struct":
        samples = _load_json_struct_samples(n)
    else:
        raise ValueError(f"json_extraction supports source: toy, ner, json_struct. Got: {source}")

    if len(samples) < n:
        logger.warning("Requested %d samples but only %d available (source=%s)", n, len(samples), source)
    return samples


def _load_toy_samples(n: int) -> List[Sample]:
    """Load from built-in toy dataset."""
    samples: List[Sample] = []
    for i, item in enumerate(TOY_DATASET):
        if i >= n:
            break
        samples.append(Sample(
            sid=item["id"],
            text=item["text"],
            schema=item["schema"],
            reference=json.dumps(item["reference"], indent=2),
        ))
    return samples


def _load_json_struct_samples(n: int) -> List[Sample]:
    """
    Load from MasterControlAIML/JSON-Unstructured-Structured dataset.
    
    This dataset contains text with expected JSON structure output.
    Falls back to toy dataset if loading fails.
    """
    try:
        dataset = load_dataset(
            "MasterControlAIML/JSON-Unstructured-Structured", 
            split="train"
        )
    except Exception as e:
        print(f"Warning: Could not load JSON-Unstructured-Structured dataset: {e}")
        print("Falling back to toy dataset...")
        return _load_toy_samples(n)
    
    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if len(samples) >= n:
            break
        
        try:
            # the dataset has 'unstructured_text' and 'structured_json' fields
            text = item.get("unstructured_text", item.get("text", ""))
            structured = item.get("structured_json", item.get("json", ""))
            
            if not text or not structured:
                continue
            
            # parse the structured JSON to extract schema
            if isinstance(structured, str):
                try:
                    parsed = json.loads(structured)
                except json.JSONDecodeError:
                    continue
            else:
                parsed = structured
            
            # extract schema from keys
            if isinstance(parsed, dict):
                schema = ", ".join(parsed.keys())
                reference = json.dumps(parsed, indent=2)
            else:
                continue
            
            # skip if text is too long (>500 chars) for reasonable inference
            if len(text) > 500:
                continue
            
            samples.append(Sample(
                sid=f"json-struct-{i}",
                text=text,
                schema=schema,
                reference=reference,
            ))
        except Exception:
            continue
    
    # if we didn't get enough samples, supplement with toy data
    if len(samples) < n:
        print(f"Only got {len(samples)} samples from HuggingFace, supplementing with toy data...")
        toy_samples = _load_toy_samples(n - len(samples))
        samples.extend(toy_samples)
    
    return samples


def _load_ner_samples(n: int) -> List[Sample]:
    """
    Load from CoNLL-2003 NER dataset.
    
    Task: Extract named entities (persons, organizations, locations) from text.
    Falls back to toy dataset if HuggingFace dataset fails.
    """
    # try to load CoNLL-2003 dataset
    try:
        dataset = load_dataset("conll2003", split="test")
    except Exception as e1:
        try:
            # try alternate source
            dataset = load_dataset("eriktks/conll2003", split="test")
        except Exception as e2:
            print(f"Warning: Could not load CoNLL-2003 dataset, falling back to toy data. Error: {e2}")
            return _load_toy_samples(n)
    
    # nER tag mapping for CoNLL-2003
    # tags: O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC, B-MISC, I-MISC
    tag_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    
    samples: List[Sample] = []
    for i, item in enumerate(dataset):
        if i >= n:
            break
        
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]
        
        # reconstruct text
        text = " ".join(tokens)
        
        # extract entities
        entities = {"persons": [], "organizations": [], "locations": [], "misc": []}
        current_entity = []
        current_type = None
        
        for token, tag_id in zip(tokens, ner_tags):
            tag = tag_names[tag_id]
            
            if tag.startswith("B-"):
                # save previous entity if exists
                if current_entity and current_type:
                    entity_text = " ".join(current_entity)
                    if current_type == "PER":
                        entities["persons"].append(entity_text)
                    elif current_type == "ORG":
                        entities["organizations"].append(entity_text)
                    elif current_type == "LOC":
                        entities["locations"].append(entity_text)
                    else:
                        entities["misc"].append(entity_text)
                
                # start new entity
                current_entity = [token]
                current_type = tag[2:]  # remove "B-" prefix
            elif tag.startswith("I-") and current_type == tag[2:]:
                # continue current entity
                current_entity.append(token)
            else:
                # end current entity
                if current_entity and current_type:
                    entity_text = " ".join(current_entity)
                    if current_type == "PER":
                        entities["persons"].append(entity_text)
                    elif current_type == "ORG":
                        entities["organizations"].append(entity_text)
                    elif current_type == "LOC":
                        entities["locations"].append(entity_text)
                    else:
                        entities["misc"].append(entity_text)
                current_entity = []
                current_type = None
        
        # don't forget last entity
        if current_entity and current_type:
            entity_text = " ".join(current_entity)
            if current_type == "PER":
                entities["persons"].append(entity_text)
            elif current_type == "ORG":
                entities["organizations"].append(entity_text)
            elif current_type == "LOC":
                entities["locations"].append(entity_text)
            else:
                entities["misc"].append(entity_text)
        
        # skip samples with no entities
        if not any(entities.values()):
            continue
        
        samples.append(Sample(
            sid=f"conll-{i}",
            text=text,
            schema="persons, organizations, locations, misc",
            reference=json.dumps(entities, indent=2),
        ))
        
        if len(samples) >= n:
            break
    
    return samples


def extract_json_from_prediction(prediction: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from model prediction.
    
    Tries multiple strategies:
    1. Parse the entire response as JSON
    2. Find JSON block in markdown code fence
    3. Find JSON object pattern { ... }
    """
    prediction = prediction.strip()
    
    # strategy 1: Try parsing the entire response
    try:
        return json.loads(prediction)
    except json.JSONDecodeError:
        pass
    
    # strategy 2: Look for JSON in markdown code block
    code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", prediction, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # strategy 3: Find JSON object pattern
    json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", prediction, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def _normalize_value(val) -> str:
    """Normalize a value for comparison (lowercase, strip whitespace)."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val.lower().strip()
    if isinstance(val, list):
        return str(sorted([_normalize_value(v) for v in val]))
    if isinstance(val, dict):
        return str({k: _normalize_value(v) for k, v in sorted(val.items())})
    return str(val).lower().strip()


def accuracy_check(prediction: str, reference: str) -> bool:
    """
    Check if the prediction contains valid JSON with correct field values.
    
    Accuracy criteria (STRICT - to differentiate model quality):
    1. Must produce valid JSON
    2. Must have all required fields
    3. At least 90% of field values must match EXACTLY (stricter threshold)
    
    Note: The toy dataset is relatively easy (explicit facts in text).
    Use stricter matching to better differentiate model quality.
    For harder evaluation, use source: "ner" or "json_struct" in config.yaml.
    
    Args:
        prediction: The model's full response text
        reference: The expected JSON string
    
    Returns:
        True if valid JSON with >= 90% correct field values, False otherwise
    """
    # parse the reference to get expected fields
    try:
        ref_dict = json.loads(reference)
    except json.JSONDecodeError:
        return False
    
    # extract JSON from prediction
    pred_dict = extract_json_from_prediction(prediction)
    
    if pred_dict is None or not isinstance(pred_dict, dict):
        return False
    
    # check if all required fields are present
    required_fields = set(ref_dict.keys())
    present_fields = set(pred_dict.keys())
    
    # all required fields must be present
    if not required_fields.issubset(present_fields):
        return False
    
    # count matching values - use STRICT matching
    matches = 0
    total = len(ref_dict)
    
    for field, ref_val in ref_dict.items():
        pred_val = pred_dict.get(field)
        if _values_match_strict(pred_val, ref_val):
            matches += 1
    
    # require at least 90% of values to match exactly
    return (matches / total) >= 0.90


def _values_match_strict(pred_val, ref_val) -> bool:
    """
    STRICT value matching for differentiating model quality.
    
    This helps differentiate model quality on the toy dataset.
    """
    # normalize both values
    pred_norm = _normalize_value(pred_val)
    ref_norm = _normalize_value(ref_val)
    
    # exact match after normalization
    if pred_norm == ref_norm:
        return True
    
    # for strings, require exact match or exact substring (no partial)
    if isinstance(ref_val, str) and isinstance(pred_val, str):
        ref_lower = ref_val.lower().strip()
        pred_lower = pred_val.lower().strip()
        # only allow if prediction exactly equals reference (case-insensitive)
        # or if one is a title variant (Dr., Mr., etc.)
        if ref_lower == pred_lower:
            return True
        # allow "Dr. Maria Garcia" to match "Maria Garcia" but not vice versa
        if pred_lower.replace("dr. ", "").replace("mr. ", "").replace("ms. ", "") == ref_lower:
            return True
        if ref_lower.replace("dr. ", "").replace("mr. ", "").replace("ms. ", "") == pred_lower:
            return True
        return False
    
    # for numbers, require exact match (no tolerance)
    if isinstance(ref_val, (int, float)) and isinstance(pred_val, (int, float)):
        # allow int/float type differences (35 == 35.0)
        return float(pred_val) == float(ref_val)
    
    # for booleans
    if isinstance(ref_val, bool) and isinstance(pred_val, bool):
        return ref_val == pred_val
    
    return False
