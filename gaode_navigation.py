#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:17:14 2024

@author: jyb
"""
import folium
import requests
import json
import webbrowser

amap_key = 'fd7b485413894d6ee80ba02b8166cfb5'


def search_nearby(keywords, location = '', radius=5000):
    '''gaode : https://lbs.amap.com/api/webservice/guide/api-advanced/search#around'''
    if location == '':
        location = get_location_x_y("南京")
    
    url = 'https://restapi.amap.com/v3/place/around'
    params = {
        'key': amap_key,
        'keywords': keywords,
        'location': location,
        'radius': radius,
        'types': '',
        'output': 'json'
    }
    response = requests.get(url, params=params)
    data = response.json()
    first_poi = None
    for poi in data['pois']:
        print(poi['name'], poi['address'])
        if first_poi == None:
            first_poi = poi
    print("select bearby : ", first_poi['name'])
    
    return first_poi

def get_location_x_y(place):
    
    url = 'https://restapi.amap.com/v3/geocode/geo?parameters'
    parameters = {
        'key': amap_key,
        'address':'%s' % place
    }
    page_resource = requests.get(url,params=parameters)
    text = page_resource.text       #get json data
    data = json.loads(text)         #convert to dict
    if (data["info"] != "OK"):
        return None
    location = data["geocodes"][0]['location']
    #print(place, "->location : " ,location)
    return location

def get_route_frome_reponse(response):
    data = response.json()
    if data['status'] == '1':
        route = data['route']['paths'][0]
        return route['steps']
    else:
        return None
    
def gen_path(route):
    path = ''
    for step in route:
        p = step['instruction']
        path = path + "\n" + p
    return path
    
def draw_route_on_map(start_location, end_location, route_data):
    if route_data:
        # Create map centered around the starting location
        m = folium.Map(location=start_location.split(','), zoom_start=12)

        # Add markers for start and end locations
        folium.Marker(location=start_location.split(','), popup='Start').add_to(m)
        folium.Marker(location=end_location.split(','), popup='End').add_to(m)

        for step in route_data:
            print("step :\n", len(step) , "\n", step, "================================\n\n\n")
            # Add polyline for the route
            points = [[float(coord.split(',')[1]), float(coord.split(',')[0])] for coord in step['polyline'].split(';')]
            folium.PolyLine(points, color='blue', weight=5, opacity=0.7).add_to(m)

        if m:
            html_file_path = 'route_map.html'
            m.save(html_file_path)
            print('Route map saved as {}'.format(html_file_path))
            webbrowser.open_new_tab(html_file_path)
            print('webbrowser draw map:  {}'.format(html_file_path))
        else:
            print('Failed to get directions or draw route on map')
    
def route_planning(from_place : str, to_place: str):
    from_location = get_location_x_y(from_place)
    
    if from_location is None:
        #print("No location found for {}, Please regenerate the location name".format(from_place))
        return "No location found from_location for {}, Please regenerate the location name".format(from_place)
    
    to_poi = search_nearby(to_place)
    to_location = get_location_x_y(to_poi["name"])
    if to_location is None:
        print("No location found for {}, Please regenerate the location name".format(to_poi["name"]))
        to_location = get_location_x_y(to_poi['address'])
        if to_location is None:
            print("No location found for {}, Please regenerate the location name".format(to_poi["address"]))
            return None

            

    url = 'https://restapi.amap.com/v3/direction/driving?parameters'
    
    
    # url="https://restapi.amap.com"
    # if type=="1":
    #     url = url+ "/v3/direction/transit/integrated"
    # elif type=="2":
    #     url = url + "/v3/direction/walking"
    # elif type=="3":
    #     url = url + "/v3/direction/driving"
    # elif type == "4":
    #     url = url + "/v4/direction/bicycling"
        
    
    parameters = {
        'key': '5b075bd243a18155fbc164db0c3e426b',
        'origin': str(from_location),
        'destination': str(to_location),
        'output': 'json'
    }
    
    response = requests.get(url, parameters)
    route = get_route_frome_reponse(response)
    return gen_path(route)
    #draw_route_on_map(from_location, to_location, route)


if __name__ == '__main__':
    route_planning()
