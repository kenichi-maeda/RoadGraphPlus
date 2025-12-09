#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image
import tifffile

SOURCE_SIZE = 1300
TARGET_SIZE = 512

def get_geotiff_params(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        tags = tif.pages[0].tags
        
        if 'ModelPixelScaleTag' in tags:
            scale = tags['ModelPixelScaleTag'].value
            pixel_scale = scale[0]  
        else:
            raise ValueError(f"No ModelPixelScaleTag in {tif_path}")
        
        if 'ModelTiepointTag' in tags:
            tiepoint = tags['ModelTiepointTag'].value
            top_left_lon = tiepoint[3]
            top_left_lat = tiepoint[4]
        else:
            raise ValueError(f"No ModelTiepointTag in {tif_path}")
        
        return top_left_lon, top_left_lat, pixel_scale

def lonlat_to_pixel(lon, lat, top_left_lon, top_left_lat, pixel_scale):
    x_1300 = (lon - top_left_lon) / pixel_scale
    y_1300 = (top_left_lat - lat) / pixel_scale
    scale = TARGET_SIZE / SOURCE_SIZE
    return x_1300 * scale, y_1300 * scale

def build_graph(geojson_path, top_left_lon, top_left_lat, pixel_scale):
    with open(geojson_path, 'r') as f:
        data = json.load(f)
    
    node_to_idx = {}
    nodes = []
    edges = []
    
    def get_node(x, y):
        key = (round(x, 6), round(y, 6))
        if key not in node_to_idx:
            idx = len(nodes)
            node_to_idx[key] = idx
            nodes.append({"idx": idx, "x": key[0], "y": key[1], "border": False})
        return node_to_idx[key]
    
    for feature in data['features']:
        if feature['geometry']['type'] != 'LineString':
            continue
        coords = feature['geometry']['coordinates']
        
        pixel_coords = []
        for lon, lat in coords:
            x, y = lonlat_to_pixel(lon, lat, top_left_lon, top_left_lat, pixel_scale)
            if 0 <= x < TARGET_SIZE and 0 <= y < TARGET_SIZE:
                pixel_coords.append((x, y))
        
        for i in range(len(pixel_coords) - 1):
            idx1 = get_node(*pixel_coords[i])
            idx2 = get_node(*pixel_coords[i + 1])
            edges.append({"src_idx": idx1, "dst_idx": idx2})
            edges.append({"src_idx": idx2, "dst_idx": idx1})
    
    return nodes, edges

def convert_chip(chip_num, spacenet_dir, output_dir):
    spacenet = Path(spacenet_dir)
    output = Path(output_dir)
    
    tif = spacenet / 'Moscow/PS-RGB' / f'SN5_roads_train_AOI_7_Moscow_PS-RGB_chip{chip_num}.tif'
    geo = spacenet / 'Moscow/geojson_roads_speed' / f'SN5_roads_train_AOI_7_Moscow_geojson_roads_speed_chip{chip_num}.geojson'
    
    if not tif.exists() or not geo.exists():
        return False
    
    out_img = output / 'images' / f'moscow_{chip_num}.png'
    out_label = output / 'labels' / f'moscow_{chip_num}.json'
    
    try:
        top_left_lon, top_left_lat, pixel_scale = get_geotiff_params(tif)
    except Exception as e:
        print(f"Error reading geotiff params for chip{chip_num}: {e}")
        return False
    
    img = Image.open(tif).resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    img.save(out_img)
    
    nodes, edges = build_graph(geo, top_left_lon, top_left_lat, pixel_scale)
    
    if len(nodes) == 0:
        out_img.unlink()
        return False
    
    label = {
        "region": "moscow",
        "image_path": str(out_img.absolute()),
        "crop": {
            "out_size": TARGET_SIZE
        },
        "coord_convention": "pixel_y_down",
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges
    }
    
    with open(out_label, 'w') as f:
        json.dump(label, f, indent=2)
    
    return True

def main():
    SPACENET_DIR = '/Users/lingzhichen/Desktop/spacenet_moscow'
    OUTPUT_DIR = '/Users/lingzhichen/Desktop/moscow_roadtracer'
    NUM_SAMPLES = 200
    
    output = Path(OUTPUT_DIR)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    (output / 'labels').mkdir(parents=True, exist_ok=True)
    
    print(f"Converting {NUM_SAMPLES} samples...")
    success = 0
    
    for i in range(NUM_SAMPLES):
        if convert_chip(i, SPACENET_DIR, OUTPUT_DIR):
            success += 1
            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{NUM_SAMPLES} - Success: {success}")
    
    print(f"\nDone! Success: {success}/{NUM_SAMPLES}")
    print(f"Output: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()