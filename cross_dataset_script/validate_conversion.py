#!/usr/bin/env python3
import json
from pathlib import Path
from PIL import Image
import statistics

def validate_sample(label_path, image_path):
    errors = []
    
    if not label_path.exists():
        return [f"Missing label"]
    if not image_path.exists():
        return [f"Missing image"]
    
    try:
        img = Image.open(image_path)
        if img.size != (512, 512):
            errors.append(f"Wrong size: {img.size}")
        if img.mode != 'RGB':
            errors.append(f"Wrong mode: {img.mode}")
    except Exception as e:
        errors.append(f"Image error: {e}")
    
    try:
        with open(label_path) as f:
            data = json.load(f)
        
        required = ['image_path', 'crop', 'nodes', 'edges', 'num_nodes', 'num_edges']
        for field in required:
            if field not in data:
                errors.append(f"Missing field: {field}")
        
        if 'nodes' in data:
            out_of_bounds = []
            for node in data['nodes']:
                x, y = node['x'], node['y']
                if not (0 <= x < 512 and 0 <= y < 512):
                    out_of_bounds.append((x, y))
            if out_of_bounds:
                errors.append(f"{len(out_of_bounds)} nodes out of bounds")
        
        if data.get('num_nodes') != len(data.get('nodes', [])):
            errors.append(f"Node count mismatch")
        if data.get('num_edges') != len(data.get('edges', [])):
            errors.append(f"Edge count mismatch")
            
    except Exception as e:
        errors.append(f"JSON error: {e}")
    
    return errors

def main():
    base = Path('/Users/lingzhichen/Desktop/moscow_roadtracer')
    labels_dir = base / 'labels'
    images_dir = base / 'images'
    
    total = 0
    valid = 0
    node_counts = []
    edge_counts = []
    all_errors = []
    
    print("🔍 Validating Moscow RoadTracer dataset...\n")
    
    for label_file in sorted(labels_dir.glob('moscow_*.json')):
        total += 1
        chip_num = label_file.stem.split('_')[1]
        image_file = images_dir / f'moscow_{chip_num}.png'
        
        errors = validate_sample(label_file, image_file)
        
        if errors:
            all_errors.append((chip_num, errors))
        else:
            valid += 1
            with open(label_file) as f:
                data = json.load(f)
                node_counts.append(data['num_nodes'])
                edge_counts.append(data['num_edges'])
    
    print(f"{'='*60}")
    print(f"📊 Validation Results:")
    print(f"{'='*60}")
    print(f"Total samples:     {total}")
    print(f"Valid samples:     {valid} ({valid/total*100:.1f}%)")
    print(f"Invalid samples:   {total - valid}")
    
    if node_counts:
        print(f"\n📈 Graph Statistics:")
        print(f"Nodes per graph:   {statistics.mean(node_counts):.1f} (avg)")
        print(f"                   {min(node_counts)} ~ {max(node_counts)} (range)")
        print(f"                   {statistics.median(node_counts):.0f} (median)")
        print(f"Edges per graph:   {statistics.mean(edge_counts):.1f} (avg)")
        print(f"                   {min(edge_counts)} ~ {max(edge_counts)} (range)")
        print(f"                   {statistics.median(edge_counts):.0f} (median)")
    
    if all_errors:
        print(f"\n❌ Errors Found ({len(all_errors)} samples):")
        for chip_num, errors in all_errors[:10]:
            print(f"  moscow_{chip_num}: {', '.join(errors)}")
        if len(all_errors) > 10:
            print(f"  ... and {len(all_errors) - 10} more")
    else:
        print(f"\n✅ All samples passed validation!")
    
    print(f"\n{'='*60}")
    
    print(f"\n📋 Sample Details (first 3):")
    for label_file in sorted(labels_dir.glob('moscow_*.json'))[:3]:
        with open(label_file) as f:
            data = json.load(f)
        chip_num = label_file.stem.split('_')[1]
        print(f"\nmoscow_{chip_num}:")
        print(f"  Nodes: {data['num_nodes']}")
        print(f"  Edges: {data['num_edges']}")
        if data['nodes']:
            x_coords = [n['x'] for n in data['nodes']]
            y_coords = [n['y'] for n in data['nodes']]
            print(f"  X range: {min(x_coords):.1f} - {max(x_coords):.1f}")
            print(f"  Y range: {min(y_coords):.1f} - {max(y_coords):.1f}")
    
    return valid == total

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)