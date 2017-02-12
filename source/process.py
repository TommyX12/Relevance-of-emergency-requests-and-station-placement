# -*- coding: utf-8 -*-

#author: TommyX

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, Polygon, MultiPoint, MultiPolygon
from shapely.prepared import prep
import fiona
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
import json
import datetime
import csv
from time import sleep


draw_image = True
data_info_type = 1 # 0 - points; 1 - per neighborhood
sector_name = 'AREA_S_CD'
choropleth_color_type = 1 # 0 - discrete; 1 - continuous
compute_data = False # only when image_type = 2 (custom)

out_data = [[0 for j in range(5)] for i in range(141)]
out_column_ID = 0
out_data[0][0] = "ID"
out_column_Area = 1
out_data[0][1] = "Area(m2)"
out_column_AmbDist = 2
out_data[0][2] = "Avg_Dist_to_Ambulance_Stn"
out_column_FireDist = 3
out_data[0][3] = "Avg_Dist_to_Fire_Stn"
out_column_PoliceDist = 4
out_data[0][4] = "Avg_Dist_to_Police_Stn"
            
sleep(0.05)
print "Parsing data..."
csvfile  = open('data/compiled_data.csv', "rb")
reader = csv.reader(csvfile)
data = [0 for n in range(150)]
rownum = 0
for row in reader:
    if rownum > 140: break
    if rownum > 0:
        if row[1] == "": continue
        data[rownum] = float(row[16]) # row[13] - ambulance calls; row[16] - fire service calls; row[19] - major crimes
    rownum += 1
        
if draw_image:
    
    sleep(0.05)
    print "Parsing shapefile..."
    
    #shapefilename = 'data/nybb_15d/nybb'
    #shapefilename = 'data/Neighborhoods/WGS84/Neighborhoods'
    #shapefilename = 'data/nycd_15d/nycd'
    #shapefilename = 'data/zillow/ZillowNeighborhoods-NY'
    #shapefilename = 'data/nycd_15d/nycd_wgs84'
    shapefilename = 'data/toronto_neighborhoods/tn_wgs84'
    shp = fiona.open(shapefilename+'.shp')
    coords = shp.bounds#[-74.257159, 40.495992, -73.699215, 40.915568]
    westBound = coords[0]
    eastBound = coords[2]
    northBound = coords[3]
    southBound = coords[1]
    #west south east north
    shp.close()
    
    w, h = eastBound - westBound, northBound - southBound
    extra = 0.05
    
    print "\tWest bound: lon " + str(westBound)
    print "\tEast bound: lon " + str(eastBound)
    print "\tNorth bound: lat " + str(northBound)
    print "\tSouth bound: lat " + str(southBound)
    
    sleep(0.05)
    print "Initializing basemap..."
    m = Basemap(
        projection='tmerc', ellps='WGS84',
        lon_0=np.mean([westBound, eastBound]),
        lat_0=np.mean([southBound, northBound]),
        llcrnrlon=westBound - extra * w,
        llcrnrlat=southBound - (extra * h), 
        urcrnrlon=eastBound + extra * w,
        urcrnrlat=northBound + (extra * h),
        resolution='i',  suppress_ticks=True)
    
    _out = m.readshapefile(shapefilename, name='city', drawbounds=False, color='black', zorder=2)
    
    #plt.show()
    #raw_input()
    
    sleep(0.05)
    print "Filtering datapoints..."
    # set up a map dataframe
    df_map = pd.DataFrame({
        'poly': [Polygon(hood_points) for hood_points in m.city],
        'name': [hood[sector_name] for hood in m.city_info],
    })
    
    def get_area(poly):
        return poly.area
    
    df_map['area'] = df_map['poly'].apply(get_area, args=())
    
    hood_polygons = prep(MultiPolygon(list(df_map['poly'].values)))
    
    if data_info_type == 0:
        # Convert our latitude and longitude into Basemap cartesian map coordinates
        mapped_points = [Point(m(mapped_x, mapped_y)) for mapped_x, mapped_y in data]
        all_points = MultiPoint(mapped_points)
        # Use prep to optimize polygons for faster computation
        # Filter out the points that do not fall within the map we're making
        city_points = filter(hood_polygons.contains, all_points)
    
    #print len(city_points)

    sleep(0.05)
    print "Analyzing datapoints..."
    
    def get_count_points(apolygon, city_points):
        return int(len(filter(prep(apolygon).contains, city_points)))
        
    def get_count_data(aname):
        return data[int(aname)]

    if data_info_type == 0: 
        df_map['hood_count'] = df_map['poly'].apply(get_count_points, args=(city_points,))
    elif data_info_type == 1: 
        df_map['hood_count'] = df_map['name'].apply(get_count_data, args=())
    
    # We'll only use a handful of distinct colors for our choropleth. So pick where
    # you want your cutoffs to occur. Leave zero and ~infinity alone.
    num_of_sections = 5
    highest_section = df_map.hood_count.max() * 0.95
    breaks = []
    for i in range(num_of_sections):
        breaks.append(float(int(i * float(highest_section) / float(num_of_sections-1))))
    breaks.append(1e20)
    
    def self_categorize_discrete(entry):
        for i in range(len(breaks)-1):
            if entry > breaks[i] and entry <= breaks[i+1]:
                return float(i+1)/num_of_sections
        return 0.0
        
    def self_categorize_continuous(entry):
        return float(entry) / highest_section;
        
    if choropleth_color_type == 0: df_map['jenks_bins'] = df_map.hood_count.apply(self_categorize_discrete, args=())
    elif choropleth_color_type == 1: df_map['jenks_bins'] = df_map.hood_count.apply(self_categorize_continuous, args=())        
    
    labels = ['None']+["> %d"%(perc) for perc in breaks[:-1]]
    
    # Or, you could always use Natural_Breaks to calculate your breaks for you:
    # from pysal.esda.mapclassify import Natural_Breaks
    # breaks = Natural_Breaks(df_map[df_map['hood_hours'] > 0].hood_hours, initial=300, k=3)
    # df_map['jenks_bins'] = -1 #default value if no data exists for this bin
    # df_map['jenks_bins'][df_map.hood_count > 0] = breaks.yb
    # 
    # jenks_labels = ['Never been here', "> 0 hours"]+["> %d hours"%(perc) for perc in breaks.bins[:-1]]
    
    sleep(0.05)
    print "Drawing choropleth..."    
    
    def custom_colorbar(cmap, ncolors, labels, **kwargs):    
        """Create a custom, discretized colorbar with correctly formatted/aligned labels.
        
        cmap: the matplotlib colormap object you plan on using for your graph
        ncolors: (int) the number of discrete colors available
        labels: the list of labels for the colorbar. Should be the same length as ncolors.
        """
        from matplotlib.colors import BoundaryNorm
        from matplotlib.cm import ScalarMappable
            
        norm = BoundaryNorm(range(0, ncolors), cmap.N)
        mappable = ScalarMappable(cmap=cmap, norm=norm)
        mappable.set_array([])
        mappable.set_clim(-0.5, ncolors+0.5)
        colorbar = plt.colorbar(mappable, **kwargs)
        colorbar.set_ticks(np.linspace(0, ncolors, ncolors+1)+0.5)
        colorbar.set_ticklabels(range(0, ncolors))
        colorbar.set_ticklabels(labels)
        return colorbar
    
    figwidth = 14
    fig = plt.figure(figsize=(figwidth, figwidth*h/w))
    ax = fig.add_subplot(111, axisbg='w', frame_on=False)
    
    cmap = plt.get_cmap('Reds')
    # draw neighborhoods with grey outlines
    df_map['patches'] = df_map['poly'].map(lambda x: PolygonPatch(x, ec='#111111', lw=.8, alpha=1., zorder=4))
    pc = PatchCollection(df_map['patches'], match_original=True)
    # apply our custom color values onto the patch collection
    cmap_list = [cmap(val) for val in df_map.jenks_bins.values]
    pc.set_facecolor(cmap_list)
    ax.add_collection(pc)
    
    #Draw a map scale
    m.drawmapscale(coords[0] + 0.08, coords[1] + -0.01,
        coords[0], coords[1], 10.,
        fontsize=16, barstyle='fancy', labelstyle='simple',
        fillcolor1='w', fillcolor2='#555555', fontcolor='#555555',
        zorder=5, ax=ax,)
        
        
    if compute_data:
        for i in range(len(df_map.name)):
            out_data[int(df_map.name[i])][out_column_ID] = df_map.name[i]
            out_data[int(df_map.name[i])][out_column_Area] = df_map.area[i]
    
    def draw_additional_points(isCSV, path, color, edgeColor):
        
        sleep(0.05)
        print "Reading additional data..."            
        
        if isCSV:
            
            csvfile  = open(path+'.csv', "rb")
            reader = csv.reader(csvfile)
            data_in = []
            rownum = 0
            for row in reader:
                if rownum > 0:
                    data_in.append((float(row[0]), float(row[1])))
                rownum += 1
            mapped_points = [Point(m(stn[0], stn[1])) for stn in data_in]
            geo_points = [Point(stn[0], stn[1]) for stn in data_in]
            
        else:
            
            shp = fiona.open(path+'.shp')
            coords = shp.bounds#[-74.257159, 40.495992, -73.699215, 40.915568]
            westBound = coords[0]
            eastBound = coords[2]
            northBound = coords[3]
            southBound = coords[1]
            #west south east north
            shp.close()
            
            _w, _h = eastBound - westBound, northBound - southBound
            extra = 0.05
            
            mm = Basemap(
                projection='tmerc', ellps='WGS84',
                lon_0=np.mean([westBound, eastBound]),
                lat_0=np.mean([southBound, northBound]),
                llcrnrlon=westBound - extra * _w,
                llcrnrlat=southBound - (extra * _h), 
                urcrnrlon=eastBound + extra * _w,
                urcrnrlat=northBound + (extra * _h),
                resolution='i',  suppress_ticks=True)
            
            mm.readshapefile(path, name='pt', drawbounds=False, color='black', zorder=2)
    
            mapped_points = [Point(m(stn['LONGITUDE'], stn['LATITUDE'])) for stn in mm.pt_info]
            geo_points = [Point(stn['LONGITUDE'], stn['LATITUDE']) for stn in mm.pt_info]
            
        sleep(0.05)
        print "Filtering additional data..."
        all_points = MultiPoint(mapped_points)
        # Filter out the points that do not fall within the map we're making
        city_points = filter(hood_polygons.contains, all_points)        
        
        sleep(0.05)
        print "Drawing hexbin..."
        numhexbins = 75
        m.hexbin(
            np.array([geom.x for geom in city_points]),
            np.array([geom.y for geom in city_points]),
            gridsize=(numhexbins, int(numhexbins*h/w)), #critical to get regular hexagon, must stretch to map dimensions
            bins='log', mincnt=1, edgecolor=edgeColor, alpha=1.,
            cmap=plt.get_cmap(color))
        
        return geo_points
    
        
    amb_points = draw_additional_points(False, 'data/ambulance_facility_wgs84/AMBULANCE_FACILITY_WGS84', 'brg_r', 'green')
    
    fire_points = draw_additional_points(False, 'data/fire_hall_locations_wgs84/FIRE_FACILITY_WGS84_X', 'spring_r', 'red')
    
    police_points = draw_additional_points(True, 'data/Police_Facilities_WGS84_latitude_longitude/Toronto_Police_Facilities_WGS84', 'rainbow_r', 'purple')

        
    if compute_data:
        
        sleep(0.05)
        print "Computing data..."
        
        def resize(arg):
            return 0.0
        
        def compute_avg_shortest_dist(svc_points, column_target):
            dist_svc = [0 for n in range(len(df_map['name']))] 
            dist_svc_cnt = [0 for n in range(len(df_map['name']))]  
            
            num_of_sample = 100
            sample_cnt = 0
            for sample_x in range(0, num_of_sample+1):
                for sample_y in range(0, num_of_sample+1):
                    sample_pt = Point(westBound+(eastBound-westBound)*sample_x/float(num_of_sample), southBound+(northBound-southBound)*sample_y/float(num_of_sample))
                    minDist = -1
                    for svc_pt in svc_points:
                        _d = sample_pt.distance(svc_pt)
                        if minDist == -1 or _d < minDist:
                            minDist = _d
                    for i in range(len(df_map.name)):
                        if df_map.poly[i].contains(Point(m(sample_pt.x, sample_pt.y))):
                            dist_svc[i] += minDist
                            dist_svc_cnt[i] += 1
                            #print df_map.dist_svc[i]
                            break
                    sample_cnt += 1
                    if sample_cnt % 100 == 0:
                        print "\t", str(sample_cnt/float((num_of_sample+1)**2)*100) + "%"
                    
            for i in range(len(df_map.name)):
                out_data[int(df_map.name[i])][column_target] = dist_svc[i] / dist_svc_cnt[i]
        
        compute_avg_shortest_dist(amb_points, out_column_AmbDist)
        compute_avg_shortest_dist(fire_points, out_column_FireDist)
        compute_avg_shortest_dist(police_points, out_column_PoliceDist)
        
    
    
    # ncolors+1 because we're using a "zero-th" color
    cbar = custom_colorbar(cmap, ncolors=len(labels)+1, labels=labels, shrink=0.5)
    cbar.ax.tick_params(labelsize=16)
    
    fig.suptitle("Number of Fire Service Calls per Square Kilometer for each Neighborhood in 2011", fontdict={'size':20, 'fontweight':'bold'}, y=0.98)
    ax.set_title("plus Location of Emergency Service Stations", fontsize=14, y=0.98)
    ax.text(1.35, 0.06, "data from Wellbeing Toronto", ha='right', color='#555555', style='italic', transform=ax.transAxes)
    ax.text(1.35, 0.01, "map and location from Toronto Open Data", color='#555555', fontsize=16, ha='right', transform=ax.transAxes)
    
    sleep(0.05)
    print "Saving image..."
    plt.savefig('image_out.png', dpi=128, frameon=False, bbox_inches='tight', pad_inches=0.5, facecolor='#F2F2F2')
    
    if compute_data:
        sleep(0.05)
        print "Writing data output..."
        fp = open('data_out.csv', 'wb')
        a = csv.writer(fp, delimiter=',')
        a.writerows(out_data)
        fp.close()
    
    plt.show()
    
sleep(0.05)
print "Done."