import os
from typing import Optional
from crewai.tools import BaseTool
from pydantic import Field
from serpapi import GoogleSearch
import folium

class HospitalSearchByCoordinatesTool(BaseTool):
    name: str = "Hospital Search by Coordinates"
    description: str = (
        "Finds hospitals near specified latitude/longitude coordinates. "
        "Returns names, addresses, distances, and emergency status."
    )

    def _run(
        self,
        latitude: float = Field(..., description="Latitude of the search center"),
        longitude: float = Field(..., description="Longitude of the search center"),
        radius: int = Field(5000, description="Search radius in meters (default: 5000 = 3.1 miles)"),
        limit: int = Field(5, description="Max results to return (default: 5)"),
    ) -> dict:
        """
        Searches for hospitals using SerpAPI's Google Maps engine via coordinates.
        
        Returns:
            {
                "hospitals": [
                    {
                        "name": "Hospital Name",
                        "address": "123 Main St",
                        "phone": "+1 234-567-890",
                        "distance": "2.3 miles",
                        "emergency": True,
                        "latitude": 40.7128,
                        "longitude": -74.0060
                    }
                ],
                "user_coordinates": [lat, lng]
            }
        """
        params = {
            "engine": "google_maps",
            "q": "hospitals",
            "ll": f"@{latitude},{longitude},14z",  # Key change: Uses exact coordinates
            "type": "search",
            "radius": radius,
            "hl": "en",
            "api_key": os.getenv("SERP_API")  # From your .env
        }

        try:
            results = GoogleSearch(params).get_dict()
            hospitals = []
            
            # Validate and standardize the results
            for place in results.get("local_results", [])[:limit]:
                # Ensure required fields exist
                if not all(key in place for key in ['title', 'address']):
                    continue
                    
                hospital_data = {
                    "name": place.get("title", "Unknown Hospital"),
                    "address": place.get("address", "Address not available"),
                    "phone": place.get("phone", ""),
                    "distance": place.get("distance", "Distance not available"),
                    "emergency": "emergency" in place.get("title", "").lower() or 
                                "emergency" in place.get("description", "").lower(),
                    "latitude": place.get("gps_coordinates", {}).get("latitude"),
                    "longitude": place.get("gps_coordinates", {}).get("longitude")
                }
                
                # Only include hospitals with coordinates
                if hospital_data["latitude"] and hospital_data["longitude"]:
                    hospitals.append(hospital_data)
            
            return {
                "hospitals": hospitals,
                "user_coordinates": [latitude, longitude],
                "search_radius": radius,
                "status": "success",
                "count": len(hospitals)
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Hospital search failed: {str(e)}",
                "hospitals": [],
                "user_coordinates": [latitude, longitude]
            }
