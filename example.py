#!/usr/bin/env python
# coding: utf-8

from main import main

if __name__ == "__main__":
        # process individual participant

        args = {'path_to_satellite_img' : './data/raster_aerial/AdinaKamonShoshana.png', 
                'path_to_building_img'  : './data/raster_buildings/AdinaKamonShoshana.png', 
                'output_path' : './Output_images/', # if not exist, well create one
                'path_to_gps_points' : './data/gps_data/Cluster_2K_AdinaKamonShoshana.csv'
        }
        main(**args)