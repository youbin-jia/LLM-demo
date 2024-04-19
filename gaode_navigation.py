#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 17:17:14 2024

@author: jyb
"""

import requests
import json

amap_key = 'fd7b485413894d6ee80ba02b8166cfb5'

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


def route_planning(from_place : str, to_place: str):
    from_location = get_location_x_y(from_place)
    if from_location is None:
        print("No location found for {}, Please regenerate the location name".format(from_place))
        return "No location found for {}, Please regenerate the location name".format(from_place)
    to_location = get_location_x_y(to_place)
    if to_location is None:
        print("No location found for {}, Please regenerate the location name".format(to_location))
        return "No location found for {}, Please regenerate the location name".format(to_location)

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
        'destination': str(to_location)
    }
    
    response = requests.get(url, parameters)
    txt = response.text
    return txt


if __name__ == '__main__':
    route_planning()
