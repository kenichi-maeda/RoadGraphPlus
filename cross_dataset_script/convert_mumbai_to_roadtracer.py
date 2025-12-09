import json
import os
from pathlib import Path
import tifffile
import numpy as np
from PIL import Image

def get_geotiff_params(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        tags = tif.pages[0].tags
        
        pixel_scale_tag = tags.get('ModelPixelScaleTag')
        pixel_scale = pixel_scale_tag.value[0] if pixel_scale_tag else None
        
        tiepoint_tag = tags.get('ModelTiepointTag')
        if tiepoint_tag:
            tie_points = tiepoint_tag.value
            top_left_lon = tie_points[3]
            top_left_lat = tie_points[4]
        else:
            top_left_lon = None
            top_left_lat = None
            
        return top_left_lon, top_left_lat, pixel_scale

def lonlat_to_pixel(lon, lat, top_left_lon, top_left_lat, pixel_scale):
    x = (lon - top_left_lon) / pixel_scale
    y = (top_left_lat - lat) / pixel_scale
    return x, y

def convert_mumbai_to_roadtracer(
    tif_dir='AOI_8_Mumbai/PS-RGB',
    geojson_dir='AOI_8_Mumbai/geojson_roads_speed',
    output_dir='mumbai_roadtracer_fixed',  
    image_size=512
):
    
    output_images = Path(output_dir) / 'images'
    output_labels = Path(output_dir) / 'labels'
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)
    
    tif_files = sorted(Path(tif_dir).glob('*.tif'))
    
    success_count = 0
    fail_count = 0
    
    for tif_path in tif_files:
        chip_id = tif_path.stem.replace('SN5_roads_train_AOI_8_Mumbai_PS-RGB_', '')
        geojson_path = Path(geojson_dir) / f'SN5_roads_train_AOI_8_Mumbai_geojson_roads_speed_{chip_id}.geojson'
        
        if not geojson_path.exists():
            fail_count += 1
            continue
        
        with open(geojson_path) as f:
            geojson = json.load(f)
        
        if not geojson['features']:
            fail_count += 1
            continue
        
        try:
            top_left_lon, top_left_lat, pixel_scale = get_geotiff_params(tif_path)
            
            with tifffile.TiffFile(tif_path) as tif:
                image_array = tif.asarray()
        except Exception as e:
            print(f"Error reading {chip_id}: {e}")
            fail_count += 1
            continue
        
        if image_array.shape[-1] == 3:
            image = Image.fromarray(image_array)
        else:
            image = Image.fromarray(image_array[:, :, :3])
        
        orig_height, orig_width = image.size
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        
        nodes = []
        edges = []
        node_id_map = {}
        current_node_idx = 0
        
        for feature in geojson['features']:
            if feature['geometry']['type'] != 'LineString':
                continue
            
            coords = feature['geometry']['coordinates']
            
            for i, (lon, lat) in enumerate(coords):
                x, y = lonlat_to_pixel(lon, lat, top_left_lon, top_left_lat, pixel_scale)
                
                x_scaled = (x / orig_width) * image_size
                y_scaled = (y / orig_height) * image_size
                
                x_scaled = max(0, min(image_size, x_scaled))
                y_scaled = max(0, min(image_size, y_scaled))
                
                node_key = (round(x_scaled, 2), round(y_scaled, 2))
                
                if node_key not in node_id_map:
                    node_id_map[node_key] = current_node_idx
                    nodes.append({
                        'idx': current_node_idx,
                        'x': x_scaled,
                        'y': y_scaled,
                        'border': False
                    })
                    current_node_idx += 1
                
                if i > 0:
                    prev_key = (round(prev_x, 2), round(prev_y, 2))
                    src_idx = node_id_map[prev_key]
                    dst_idx = node_id_map[node_key]
                    edges.append({
                        'src_idx': src_idx,
                        'dst_idx': dst_idx
                    })
                
                prev_x, prev_y = x_scaled, y_scaled
        
        if len(nodes) == 0:
            fail_count += 1
            continue
        
        output_image_path = output_images / f'mumbai_{chip_id}.png'
        image.save(output_image_path)
        
        label_data = {
            'region': 'mumbai',
            'image_path': str(output_image_path.absolute()),
            'crop': {'out_size': image_size},
            'coord_convention': 'pixel_y_down',
            'num_nodes': len(nodes),
            'num_edges': len(edges),
            'nodes': nodes,
            'edges': edges
        }
        
        output_json_path = output_labels / f'mumbai_{chip_id}.json'
        with open(output_json_path, 'w') as f:
            json.dump(label_data, f, indent=2)
        
        success_count += 1
        
        if success_count % 100 == 0:
            print(f'Processed {success_count} successful conversions, Failed: {fail_count}')
    
    print(f'\nConversion complete!')
    print(f'Total success: {success_count}')
    print(f'Total failed: {fail_count}')
    print(f'Output directory: {output_dir}')

if __name__ == '__main__':
    convert_mumbai_to_roadtracer()