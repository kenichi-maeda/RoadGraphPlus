import os, json, glob, argparse
from typing import Dict, Tuple, List, Optional
from PIL import Image

def _clip_line_to_rect(x0, y0, x1, y1, rx0, ry0, rx1, ry1) -> Optional[Tuple[float,float,float,float]]:
    """
    Canonical Liang-Barsky segment clip to axis-aligned rect.
    Returns (cx0, cy0, cx1, cy1) or None.
    """
    eps = 1e-9 
    rx0 -= eps; ry0 -= eps; rx1 += eps; ry1 += eps

    dx, dy = x1 - x0, y1 - y0
    p = [-dx,  dx, -dy,  dy]
    q = [x0 - rx0, rx1 - x0, y0 - ry0, ry1 - y0]

    u1, u2 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if pi == 0:
            if qi < 0:
                return None 
            continue
        r = qi / pi
        if pi < 0:    
            if r > u2: 
                return None
            if r > u1:
                u1 = r
        else: 
            if r < u1:
                return None
            if r < u2:
                u2 = r

    cx0, cy0 = x0 + u1*dx, y0 + u1*dy
    cx1, cy1 = x0 + u2*dx, y0 + u2*dy
    return (cx0, cy0, cx1, cy1)

def _inside(x,y, rx0,ry0,rx1,ry1):
    return (rx0 <= x < rx1) and (ry0 <= y < ry1)

def load_tile_gt(gt_path: str):
    with open(gt_path, "r") as f:
        gt = json.load(f)
    # tile-local coords (0..tile_size)
    nodes = { int(n["nid"]): (float(n["x"]), float(n["y"])) for n in gt["nodes"] }

    # map idx -> nid then rebuild edges as (eid, src_nid, dst_nid)
    idx2nid = { int(n["idx"]): int(n["nid"]) for n in gt["nodes"] }
    edges = []
    for e in gt["edges"]:
        s = idx2nid[int(e["src_idx"])]
        d = idx2nid[int(e["dst_idx"])]
        edges.append((int(e.get("eid", -1)), s, d))
    return gt, nodes, edges

def process_one_tile(img_path: str, gt_path: str,
                     out_img_dir: str, out_lbl_dir: str,
                     crop_size: int, stride: int, out_size: int,
                     keep_empty: bool):
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    img = Image.open(img_path) 
    W, H = img.size
    assert W == H, "Expect square tiles (e.g., 4096x4096)."
    assert crop_size <= W, "crop_size must be <= tile size."
    assert stride > 0, "stride must be > 0."

    gt, node_coord, edges = load_tile_gt(gt_path)
    nid_list = list(node_coord.keys())

    # grid over the tile
    n_x = 1 + max(0, (W - crop_size) // stride)
    n_y = 1 + max(0, (H - crop_size) // stride)

    written = 0
    base = os.path.basename(img_path).replace("_sat.png", "")
    try:
        region, tx, ty = base.rsplit("_", 2)
    except Exception:
        region, tx, ty = base, "0", "0"

    for iy in range(n_y):
        for ix in range(n_x):
            x0 = ix*stride
            y0 = iy*stride
            x1 = x0 + crop_size
            y1 = y0 + crop_size
            if x1 > W or y1 > H:
                continue

            # 1) crop image
            chip = img.crop((x0, y0, x1, y1))
            scale = 1.0
            if out_size != crop_size:
                chip = chip.resize((out_size, out_size), Image.BILINEAR)
                scale = out_size / crop_size

            # 2) nodes + edges (with clipping + synthetic border nodes)
            local_nodes: Dict[int, Tuple[float,float]] = {}
            border_nodes: List[Tuple[float,float,int]] = []

            # original nodes that fall inside
            for nid in nid_list:
                gx, gy = node_coord[nid]
                if _inside(gx, gy, x0, y0, x1, y1):
                    local_nodes[nid] = (gx - x0, gy - y0)

            rect = (x0, y0, x1, y1)
            kept_edges = []
            synth_id_seed = -1
            border_kd: Dict[Tuple[int,int], int] = {}  # rounded coord key -> synthetic id

            def ensure_nid_for_point(px, py, use_orig: Optional[int]):
                # if original endpoint is inside, use it
                if use_orig is not None and _inside(px, py, *rect):
                    if use_orig not in local_nodes:
                        local_nodes[use_orig] = (px - x0, py - y0)
                    return use_orig
                # else create/get synthetic node at the boundary point
                key = (int(round((px - x0)*1000)), int(round((py - y0)*1000)))
                sid = border_kd.get(key)
                if sid is not None:
                    return sid
                nonlocal synth_id_seed
                sid = synth_id_seed
                synth_id_seed -= 1
                border_kd[key] = sid
                border_nodes.append((px - x0, py - y0, sid))
                return sid

            for eid, s, d in edges:
                sx, sy = node_coord[s]
                dx, dy = node_coord[d]
                clipped = _clip_line_to_rect(sx, sy, dx, dy, *rect)
                if clipped is None:
                    continue
                cx0, cy0, cx1, cy1 = clipped
                a_id = ensure_nid_for_point(cx0, cy0, s if _inside(sx, sy, *rect) else None)
                b_id = ensure_nid_for_point(cx1, cy1, d if _inside(dx, dy, *rect) else None)
                if a_id != b_id:
                    kept_edges.append((eid, a_id, b_id))

            if not kept_edges and not local_nodes:
                if not keep_empty:
                    continue

            for px, py, sid in border_nodes:
                local_nodes[sid] = (px, py)

            # 3) reindex and write
            H = out_size 
            nid_sorted = sorted(local_nodes.keys())
            idmap = {nid: i for i, nid in enumerate(nid_sorted)}

            nodes_out = [{
                "nid": int(nid),
                "idx": idmap[nid],
                "x": local_nodes[nid][0] * scale,
                "y": (H - local_nodes[nid][1] * scale),
                "border": (nid < 0),
            } for nid in nid_sorted]

            edges_out = [{
                "eid": int(eid),
                "src_idx": idmap[a],
                "dst_idx": idmap[b],
            } for (eid, a, b) in kept_edges if a in idmap and b in idmap and a != b]

            if not edges_out and not nodes_out and not keep_empty:
                continue

            out_name = f"{region}_{tx}_{ty}_i{iy}_j{ix}"
            out_img_path = os.path.join(out_img_dir, f"{out_name}.png")
            out_lbl_path = os.path.join(out_lbl_dir, f"{out_name}.json")

            chip.save(out_img_path, format="PNG")

            label = {
                "region": region,
                "parent_tile": f"{region}_{tx}_{ty}",
                "tile_size": W,
                "crop": {
                    "i": iy, "j": ix,
                    "x0": x0, "y0": y0, "size": crop_size,
                    "stride": stride,
                    "out_size": out_size,
                },
                "image_path": out_img_path,
                "coord_convention": "pixel_y_down",
                "num_nodes": len(nodes_out),
                "num_edges": len(edges_out),
                "nodes": nodes_out,
                "edges": edges_out,
            }
            with open(out_lbl_path, "w") as f:
                json.dump(label, f, indent=2)

            written += 1
    return written

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imagery_dir", default="data/imagery")
    ap.add_argument("--labels_dir",  default="data/labels") 
    ap.add_argument("--out_img_dir", default="chips/images")
    ap.add_argument("--out_lbl_dir", default="chips/labels")
    ap.add_argument("--crop_size", type=int, default=512)
    ap.add_argument("--stride",    type=int, default=512, help="<= crop_size for overlap")
    ap.add_argument("--out_size",  type=int, default=512, help="resize each crop to this size")
    ap.add_argument("--keep_empty", action="store_true", help="also emit crops with no graph")
    args = ap.parse_args()

    # Gather tiles & their GT jsons
    tile_pngs = glob.glob(os.path.join(args.imagery_dir, "*_*_*_sat.png"))
    if not tile_pngs:
        raise SystemExit(f"No tiles found in {args.imagery_dir}")

    total = 0
    for img_path in sorted(tile_pngs):
        # find its GT json
        base = os.path.basename(img_path).replace("_sat.png", "")  # region_tx_ty
        gt_path = os.path.join(args.labels_dir, f"{base}_graph.json")
        if not os.path.exists(gt_path):
            print(f"SKIP (no GT): {img_path}")
            continue
        n = process_one_tile(img_path, gt_path, args.out_img_dir, args.out_lbl_dir,
                             args.crop_size, args.stride, args.out_size,
                             args.keep_empty)
        print(f"{os.path.basename(img_path)} -> {n} chips")
        total += n

    print(f"Done. Wrote {total} crops.")

if __name__ == "__main__":
    main()
