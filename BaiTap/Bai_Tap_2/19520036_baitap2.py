import pandas as pd
import geopandas as gpd

data = gpd.read_file('Data/GIS/Population/population_HCMC/population_shapefile/Population_Ward_Level.shp')
data.info()

max_area = data['Shape_Area'].idxmax()
print('Phường có diện tích lớn nhất là: ' \
        + data['Com_Name'][max_area] + ', ' + data['Dist_Name'][max_area])

max_pop_2019 = data['Pop_2019'].idxmax()
print('Phường có dân số cao nhất năm 2019 là: ' \
        + data['Com_Name'][max_pop_2019] + ', ' + data['Dist_Name'][max_pop_2019])

min_area = data['Shape_Area'].idxmin()
print('Phường có diện tích nhỏ nhất là: ' \
        + data['Com_Name'][min_area] + ', ' + data['Dist_Name'][min_area])

min_pop_2019 = data['Pop_2019'].idxmin()
print('Phường có dân số thấp nhất năm 2019 là: ' \
        + data['Com_Name'][min_pop_2019]+', ' + data['Dist_Name'][min_pop_2019])

max_speed_growth = (data['Pop_2019']/data['Pop_2009']).idxmax()
print('Phường có tốc độ tăng trưởng dân số nhanh nhất là: ' \
        + data['Com_Name'][max_speed_growth] + ', ' + data['Dist_Name'][max_speed_growth])

min_speed_growth = (data['Pop_2019']/data['Pop_2009']).idxmin()
print('Phường có tốc độ tăng trưởng dân số thấp nhất là: ' \
        + data['Com_Name'][min_speed_growth] + ', ' + data['Dist_Name'][min_speed_growth])

max_volatility = (abs(data['Pop_2019']-data['Pop_2009'])/data['Pop_2009']).idxmax()
print('Phường có biến động dân số nhanh nhất là: ' \
        + data['Com_Name'][max_volatility] + ', ' + data['Dist_Name'][max_volatility])

min_volatility = (abs(data['Pop_2019']-data['Pop_2009'])/ data['Pop_2009']).idxmin()
print('Phường có biến động dân số thấp nhất là: ' \
        + data['Com_Name'][min_volatility] + ', ' + data['Dist_Name'][min_volatility])

max_density = (data['Den_2019']).idxmax()
print('Phường có mật độ dân số cao nhất là: ' \
        + data['Com_Name'][max_density] + ', ' + data['Dist_Name'][max_density])

min_density = (data['Den_2019']).idxmin()
print('Phường có mật độ dân số thấp nhất là: ' \
        + data['Com_Name'][min_density] + ', ' + data['Dist_Name'][min_density])