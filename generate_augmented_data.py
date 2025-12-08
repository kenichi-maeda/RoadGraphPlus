import argparse
import json
import os
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image


# Supported geometric transforms.
# Naming is stable and used in output filenames and label metadata.
ALLOWED_TRANSFORMS = ("hflip", "vflip", "rot90", "rot180", "rot270")


@dataclass(frozen=True)
class TransformSpec:
    """
    Lightweight descriptor for a geometric augmentation.

    The `name` must be one of ALLOWED_TRANSFORMS.
    """
    name: str

    def __post_init__(self) -> None:
        if self.name not in ALLOWED_TRANSFORMS:
            raise ValueError(
                f"Unsupported transform '{self.name}'. "
                f"Allowed: {', '.join(ALLOWED_TRANSFORMS)}"
            )



# Signature for a coordinate transform function operating on a canvas.
PointTransform = Callable[[float, float, float, float], Tuple[float, float]]


# Signature for an image transform callable.
ImageTransform = Callable[[Image.Image], Image.Image]



def print_error(context: str, exc: BaseException) -> None:
    """
    Print a readable error block with stack trace.

    The user requested printer-style error output on any runtime problems.
    """
    print("\n" + "=" * 80)
    print(f"ERROR: {context}")
    print(f"{type(exc).__name__}: {exc}")
    print("-" * 80)
    print(traceback.format_exc().rstrip())
    print("=" * 80 + "\n")



def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def list_json_files(labels_dir: str) -> List[str]:
    files: List[str] = []
    if not os.path.isdir(labels_dir):
        return files

    for fn in os.listdir(labels_dir):
        if fn.lower().endswith(".json"):
            files.append(os.path.join(labels_dir, fn))

    files.sort()
    return files



def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4, sort_keys=False)



def infer_image_filename(label_obj: Dict, label_path: str) -> str:
    """
    Infer a PNG filename paired with the JSON.

    This assumes a common (and typical) dataset convention:
      foo.json <-> foo.png
    """
    base = os.path.splitext(os.path.basename(label_path))[0]
    return base + ".png"



def resolve_image_path(
    images_dir: str,
    label_obj: Dict,
    label_path: str
) -> str:
    """
    Resolve the corresponding image path.

    Preference order:
      1) The basename from label_obj["image_path"], if it exists in images_dir.
      2) A filename inferred from the JSON filename.
    """
    image_path_in_json = label_obj.get("image_path", "")

    if isinstance(image_path_in_json, str) and image_path_in_json:
        tail = os.path.basename(image_path_in_json)
        candidate = os.path.join(images_dir, tail)
        if os.path.exists(candidate):
            return candidate

    return os.path.join(images_dir, infer_image_filename(label_obj, label_path))



def get_label_out_size(label_obj: Dict) -> Optional[int]:
    """
    Extract the label's coordinate-space size.

    The label example suggests nodes are expressed in the cropped image
    coordinate system, where `crop.out_size` is the authoritative canvas size.
    """
    crop = label_obj.get("crop")
    if not isinstance(crop, dict):
        return None

    out_size = crop.get("out_size")
    if isinstance(out_size, int):
        return out_size

    size = crop.get("size")
    if isinstance(size, int):
        return size

    return None



def get_image_size(img: Image.Image) -> Tuple[int, int]:
    return img.size  # (W, H)



def compute_canvas_size(label_obj: Dict, img: Image.Image) -> Tuple[float, float]:
    """
    Determine the coordinate canvas for node transforms.

    Priority:
      1) crop.out_size (preferred, matches label coordinate space)
      2) crop.size
      3) actual image dimensions (fallback)
    """
    W_img, H_img = get_image_size(img)
    out_size = get_label_out_size(label_obj)

    if out_size is not None:
        return float(out_size), float(out_size)

    return float(W_img), float(H_img)



def build_augmented_filename(base_name: str, suffix: str, ext: str) -> str:
    return f"{base_name}__{suffix}{ext}"



def update_image_path_field(
    label_obj: Dict,
    new_image_path: str,
    preserve_style: bool = True
) -> None:
    """
    Update the label's image_path field.

    If preserve_style is True and the old image_path looks like it contains
    ".../data/images/...", attempt to rewrite it to ".../data_augmented/images/...".
    Otherwise, write the literal new output path.
    """
    old = label_obj.get("image_path")

    if not isinstance(old, str) or not old:
        label_obj["image_path"] = new_image_path
        return

    if not preserve_style:
        label_obj["image_path"] = new_image_path
        return

    marker_old = os.path.join("data", "images")
    marker_new = os.path.join("data_augmented", "images")

    if marker_old in old:
        prefix, _ = old.split(marker_old, 1)
        tail = os.path.basename(new_image_path)
        label_obj["image_path"] = os.path.join(prefix, marker_new, tail)
        return

    label_obj["image_path"] = new_image_path



def add_augmentation_metadata(label_obj: Dict, transform: TransformSpec) -> None:
    """
    Add a small metadata block so downstream consumers can detect augmentations.
    """
    label_obj["augmentation"] = {
        "name": transform.name
    }



def _pt_hflip(x: float, y: float, W: float, H: float) -> Tuple[float, float]:
    # Horizontal flip around the vertical midline of an inclusive [0, W] canvas.
    return (W - x), y



def _pt_vflip(x: float, y: float, W: float, H: float) -> Tuple[float, float]:
    # Vertical flip around the horizontal midline of an inclusive [0, H] canvas.
    return x, (H - y)



def _pt_rot90(x: float, y: float, W: float, H: float) -> Tuple[float, float]:
    # 90 degrees clockwise.
    return (H - y), x



def _pt_rot180(x: float, y: float, W: float, H: float) -> Tuple[float, float]:
    # 180-degree rotation.
    return (W - x), (H - y)



def _pt_rot270(x: float, y: float, W: float, H: float) -> Tuple[float, float]:
    # 270 degrees clockwise (i.e., 90 degrees counterclockwise).
    return y, (W - x)



POINT_TRANSFORMS: Dict[str, PointTransform] = {
    "hflip": _pt_hflip,
    "vflip": _pt_vflip,
    "rot90": _pt_rot90,
    "rot180": _pt_rot180,
    "rot270": _pt_rot270,
}



def _img_hflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_LEFT_RIGHT)



def _img_vflip(img: Image.Image) -> Image.Image:
    return img.transpose(Image.FLIP_TOP_BOTTOM)



def _img_rot90(img: Image.Image) -> Image.Image:
    # PIL constant for 90 degrees clockwise.
    return img.transpose(Image.ROTATE_270)



def _img_rot180(img: Image.Image) -> Image.Image:
    return img.transpose(Image.ROTATE_180)



def _img_rot270(img: Image.Image) -> Image.Image:
    return img.transpose(Image.ROTATE_90)



IMAGE_TRANSFORMS: Dict[str, ImageTransform] = {
    "hflip": _img_hflip,
    "vflip": _img_vflip,
    "rot90": _img_rot90,
    "rot180": _img_rot180,
    "rot270": _img_rot270,
}



def apply_transform_to_nodes(
    label_obj: Dict,
    transform: TransformSpec,
    W: float,
    H: float
) -> None:
    """
    Apply a geometric transform to all node coordinates in-place.

    Edges are defined via node indices (src_idx/dst_idx) and do not need
    rewriting as long as node order and idx fields remain unchanged.
    """
    nodes = label_obj.get("nodes", [])
    if not isinstance(nodes, list) or not nodes:
        return

    pt_fn = POINT_TRANSFORMS[transform.name]

    for node in nodes:
        if not isinstance(node, dict):
            continue

        if "x" not in node or "y" not in node:
            continue

        try:
            x = float(node["x"])
            y = float(node["y"])
        except (TypeError, ValueError):
            continue

        nx, ny = pt_fn(x, y, W, H)

        node["x"] = nx
        node["y"] = ny



def apply_transform_to_image(img: Image.Image, transform: TransformSpec) -> Image.Image:
    """
    Apply a geometric transform to the image via dispatch table.
    """
    return IMAGE_TRANSFORMS[transform.name](img)



def default_transforms() -> List[TransformSpec]:
    """
    Default augmentation set.
    """
    return [TransformSpec(name) for name in ALLOWED_TRANSFORMS]



def parse_transforms(names: Optional[List[str]]) -> List[TransformSpec]:
    """
    Parse user-specified transform names.

    If none provided, returns the default set.
    """
    if not names:
        return default_transforms()

    normalized = [n.strip().lower() for n in names if n and n.strip()]
    return [TransformSpec(n) for n in normalized]



def verify_label_and_coordinate_assumptions(
    label_obj: Dict,
    W: float,
    H: float,
    *,
    tol: float = 1e-2
) -> Dict[str, object]:
    """
    Validate label structure and heuristically verify the coordinate convention.

    The augmentation math assumes an inclusive canvas:
        x in [0, W], y in [0, H]

    This function performs two categories of checks:

    1) Structure checks (lightweight schema validation):
       - Top-level types
       - crop structure if present
       - nodes list of dicts with x/y/idx
       - edges list of dicts with src_idx/dst_idx
       - optional consistency of num_nodes/num_edges
       - edge indices refer to valid node idx values

    2) Coordinate-space heuristic:
       - Whether maxima appear closer to W (inclusive) vs W-1 (pixel-index)

    Returns a diagnostics dict.
    """
    structure_errors: List[str] = []

    if not isinstance(label_obj, dict):
        structure_errors.append("Label root is not a JSON object.")
        return {
            "structure_ok": False,
            "structure_errors": structure_errors,
            "ok_range": False,
            "recommended": "inclusive",
            "reason": "Invalid root object.",
        }

    # Basic optional fields type checks.
    if "region" in label_obj and not isinstance(label_obj["region"], str):
        structure_errors.append("Field 'region' exists but is not a string.")

    if "parent_tile" in label_obj and not isinstance(label_obj["parent_tile"], str):
        structure_errors.append("Field 'parent_tile' exists but is not a string.")

    if "tile_size" in label_obj and not isinstance(label_obj["tile_size"], int):
        structure_errors.append("Field 'tile_size' exists but is not an int.")

    crop = label_obj.get("crop")
    if crop is not None and not isinstance(crop, dict):
        structure_errors.append("Field 'crop' exists but is not an object.")
    elif isinstance(crop, dict):
        for k in ("i", "j", "x0", "y0", "size", "stride", "out_size"):
            if k in crop and not isinstance(crop[k], int):
                structure_errors.append(f"crop.{k} exists but is not an int.")

    nodes = label_obj.get("nodes", [])
    edges = label_obj.get("edges", [])

    if not isinstance(nodes, list):
        structure_errors.append("Field 'nodes' is not a list.")
        nodes = []

    if not isinstance(edges, list):
        structure_errors.append("Field 'edges' is not a list.")
        edges = []

    # Nodes validation.
    xs: List[float] = []
    ys: List[float] = []
    idx_values: List[int] = []

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            structure_errors.append(f"nodes[{i}] is not an object.")
            continue

        if "idx" not in node:
            structure_errors.append(f"nodes[{i}] missing 'idx'.")
        else:
            if not isinstance(node["idx"], int):
                structure_errors.append(f"nodes[{i}].idx is not an int.")
            else:
                idx_values.append(node["idx"])

        if "x" not in node or "y" not in node:
            structure_errors.append(f"nodes[{i}] missing 'x' or 'y'.")
            continue

        try:
            xs.append(float(node["x"]))
            ys.append(float(node["y"]))
        except (TypeError, ValueError):
            structure_errors.append(f"nodes[{i}].x or .y is not numeric.")

    # Optional num_nodes consistency.
    if "num_nodes" in label_obj:
        num_nodes = label_obj.get("num_nodes")
        if not isinstance(num_nodes, int):
            structure_errors.append("Field 'num_nodes' exists but is not an int.")
        else:
            if len(nodes) != num_nodes:
                structure_errors.append(
                    f"num_nodes={num_nodes} but len(nodes)={len(nodes)}."
                )

    # idx uniqueness check.
    if idx_values:
        if len(set(idx_values)) != len(idx_values):
            structure_errors.append("Duplicate node 'idx' values detected.")

    idx_set = set(idx_values)

    # Edges validation.
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            structure_errors.append(f"edges[{i}] is not an object.")
            continue

        if "src_idx" not in edge or "dst_idx" not in edge:
            structure_errors.append(f"edges[{i}] missing 'src_idx' or 'dst_idx'.")
            continue

        src = edge.get("src_idx")
        dst = edge.get("dst_idx")

        if not isinstance(src, int) or not isinstance(dst, int):
            structure_errors.append(f"edges[{i}].src_idx or .dst_idx is not an int.")
            continue

        # Validate that edges refer to existing node indices.
        # Your example uses node list indices in edges, not 'nid'.
        if idx_set:
            if src not in idx_set:
                structure_errors.append(
                    f"edges[{i}].src_idx={src} does not match any node.idx."
                )
            if dst not in idx_set:
                structure_errors.append(
                    f"edges[{i}].dst_idx={dst} does not match any node.idx."
                )

    # Optional num_edges consistency.
    if "num_edges" in label_obj:
        num_edges = label_obj.get("num_edges")
        if not isinstance(num_edges, int):
            structure_errors.append("Field 'num_edges' exists but is not an int.")
        else:
            if len(edges) != num_edges:
                structure_errors.append(
                    f"num_edges={num_edges} but len(edges)={len(edges)}."
                )

    structure_ok = len(structure_errors) == 0

    # Coordinate checks.
    if not xs or not ys:
        return {
            "structure_ok": structure_ok,
            "structure_errors": structure_errors,
            "ok_range": True,
            "recommended": "inclusive",
            "reason": "No valid node coordinates found.",
            "min_x": None,
            "max_x": None,
            "min_y": None,
            "max_y": None,
            "near_W": False,
            "near_Wm1": False,
            "near_H": False,
            "near_Hm1": False,
        }

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    ok_range = (
        all((-tol) <= x <= (W + tol) for x in xs) and
        all((-tol) <= y <= (H + tol) for y in ys)
    )

    def any_near(values: List[float], target: float) -> bool:
        return any(abs(v - target) <= tol for v in values)

    near_0_x = abs(min_x - 0.0) <= tol
    near_0_y = abs(min_y - 0.0) <= tol

    near_W = abs(max_x - W) <= tol or any_near(xs, W)
    near_H = abs(max_y - H) <= tol or any_near(ys, H)

    Wm1 = W - 1.0
    Hm1 = H - 1.0

    near_Wm1 = abs(max_x - Wm1) <= tol or any_near(xs, Wm1)
    near_Hm1 = abs(max_y - Hm1) <= tol or any_near(ys, Hm1)

    inclusive_score = int(near_0_x) + int(near_0_y) + int(near_W) + int(near_H)
    pixel_score = int(near_0_x) + int(near_0_y) + int(near_Wm1) + int(near_Hm1)

    if inclusive_score > pixel_score:
        recommended = "inclusive"
        reason = "Border evidence favors [0, S] style maxima."
    elif pixel_score > inclusive_score:
        recommended = "pixel-index"
        reason = "Border evidence favors [0, S-1] style maxima."
    else:
        recommended = "inclusive"
        reason = "Ambiguous border evidence; defaulting to inclusive."

    return {
        "structure_ok": structure_ok,
        "structure_errors": structure_errors,
        "ok_range": ok_range,
        "recommended": recommended,
        "reason": reason,
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "near_W": near_W,
        "near_Wm1": near_Wm1,
        "near_H": near_H,
        "near_Hm1": near_Hm1,
    }



def print_verification_report(
    diag: Dict[str, object],
    label_path: str
) -> None:
    """
    Emit warnings when structure or coordinate assumptions look questionable.
    """
    base = os.path.basename(label_path)

    structure_ok = bool(diag.get("structure_ok", True))
    ok_range = bool(diag.get("ok_range", True))
    recommended = diag.get("recommended", "inclusive")

    if structure_ok and ok_range and recommended == "inclusive":
        return

    print(f"[verify] {base}")

    if not structure_ok:
        print("  - Structure issues:")
        errs = diag.get("structure_errors", [])
        if isinstance(errs, list):
            for e in errs:
                print(f"    * {e}")

    if not ok_range:
        print(
            "  - Coordinate range issue: "
            f"min_x={diag.get('min_x')}, max_x={diag.get('max_x')}, "
            f"min_y={diag.get('min_y')}, max_y={diag.get('max_y')}"
        )

    if recommended != "inclusive":
        print(
            f"  - Recommended convention: {recommended} "
            f"({diag.get('reason')})"
        )



def augment_one_pair(
    image_path: str,
    label_path: str,
    out_images_dir: str,
    out_labels_dir: str,
    transforms: List[TransformSpec],
    preserve_image_path_style: bool = True,
    verify_assumptions_flag: bool = False
) -> int:
    """
    Augment one (image, label) pair with the specified transforms.

    Returns the number of augmented samples successfully created.
    """
    try:
        label_obj = load_json(label_path)
    except Exception as e:
        print_error(f"Failed to load JSON: {label_path}", e)
        return 0

    try:
        with Image.open(image_path) as img_in:
            # Normalize to a consistent mode for saving.
            img = img_in.convert("RGBA") if img_in.mode in ("P", "LA") else img_in.convert("RGB")

            Wc, Hc = compute_canvas_size(label_obj, img)

            if verify_assumptions_flag:
                diag = verify_label_and_coordinate_assumptions(label_obj, Wc, Hc)
                print_verification_report(diag, label_path)

            base_name = os.path.splitext(os.path.basename(image_path))[0]

            created = 0

            for t in transforms:
                try:
                    aug_img = apply_transform_to_image(img, t)

                    new_img_name = build_augmented_filename(base_name, t.name, ".png")
                    new_lbl_name = build_augmented_filename(base_name, t.name, ".json")

                    new_img_path = os.path.join(out_images_dir, new_img_name)
                    new_lbl_path = os.path.join(out_labels_dir, new_lbl_name)

                    aug_img.save(new_img_path)

                    # Deep copy to avoid mutating the original label object across transforms.
                    new_label_obj = json.loads(json.dumps(label_obj))

                    apply_transform_to_nodes(new_label_obj, t, Wc, Hc)

                    update_image_path_field(
                        new_label_obj,
                        new_img_path,
                        preserve_style=preserve_image_path_style
                    )

                    add_augmentation_metadata(new_label_obj, t)

                    save_json(new_label_obj, new_lbl_path)

                    created += 1

                except Exception as e:
                    print_error(
                        f"Failed transform '{t.name}' for pair "
                        f"(image={image_path}, label={label_path})",
                        e
                    )

            return created

    except Exception as e:
        print_error(f"Failed to open/process image: {image_path}", e)
        return 0



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment satellite images and RoadGraphPlus-style labels with geometric transforms."
    )
    parser.add_argument("--images_dir", default="data/images")
    parser.add_argument("--labels_dir", default="data/labels")
    parser.add_argument("--out_images_dir", default="data_augmented/images")
    parser.add_argument("--out_labels_dir", default="data_augmented/labels")
    parser.add_argument(
        "--transforms",
        nargs="*",
        help="Subset of transforms: hflip vflip rot90 rot180 rot270. Default is all."
    )
    parser.add_argument(
        "--preserve_image_path_style",
        action="store_true",
        help="Rewrite image_path by replacing data/images with data_augmented/images when possible."
    )
    parser.add_argument(
        "--no_preserve_image_path_style",
        action="store_true",
        help="Always write image_path as the literal new output image path."
    )
    parser.add_argument(
        "--verify_assumptions",
        action="store_true",
        help="Validate JSON structure and heuristically verify label coordinate assumptions."
    )

    args = parser.parse_args()

    # Default is to preserve style unless explicitly disabled.
    preserve_style = True
    if args.no_preserve_image_path_style:
        preserve_style = False
    elif args.preserve_image_path_style:
        preserve_style = True

    try:
        transforms = parse_transforms(args.transforms)
    except Exception as e:
        print_error("Invalid --transforms argument.", e)
        return

    ensure_dir(args.out_images_dir)
    ensure_dir(args.out_labels_dir)

    json_files = list_json_files(args.labels_dir)

    total_created = 0
    total_seen = 0
    missing_images = 0

    if not json_files:
        print(f"No JSON files found in: {args.labels_dir}")
        return

    for label_path in json_files:
        total_seen += 1

        try:
            label_obj = load_json(label_path)
        except Exception as e:
            print_error(f"Failed to load JSON during scan: {label_path}", e)
            continue

        image_path = resolve_image_path(args.images_dir, label_obj, label_path)

        if not os.path.exists(image_path):
            missing_images += 1
            print(f"[warn] Missing image for label: {os.path.basename(label_path)} -> {image_path}")
            continue

        created = augment_one_pair(
            image_path=image_path,
            label_path=label_path,
            out_images_dir=args.out_images_dir,
            out_labels_dir=args.out_labels_dir,
            transforms=transforms,
            preserve_image_path_style=preserve_style,
            verify_assumptions_flag=args.verify_assumptions
        )

        total_created += created

    print("\n" + "-" * 80)
    print(f"Labels scanned: {total_seen}")
    print(f"Missing images: {missing_images}")
    print(f"Augmented samples created: {total_created}")
    print(f"Output images: {args.out_images_dir}")
    print(f"Output labels: {args.out_labels_dir}")
    print("-" * 80 + "\n")



if __name__ == "__main__":
    main()
