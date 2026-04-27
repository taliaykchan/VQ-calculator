from __future__ import annotations

import io
import uuid
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.graph_objects as go
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label as connected_components
from streamlit_image_coordinates import streamlit_image_coordinates
from transformers import pipeline, AutoProcessor, AutoModelForMaskGeneration
import transformers

# Suppress HuggingFace warnings in the terminal
transformers.logging.set_verbosity_error()

st.set_page_config(page_title="BEAM Plus VQA Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .main { background: #f8fafc; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
        border-color: #e2e8f0;
        background: #ffffff;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
    }
    .loading-banner {
        padding: 2rem;
        background-color: #e0f2fe;
        border: 2px solid #bae6fd;
        border-radius: 12px;
        text-align: center;
        color: #0369a1;
        margin-bottom: 2rem;
    }
    /* Clean up default Streamlit table styling to look modern */
    table {
        border-collapse: collapse;
        width: 100%;
        color: #334155;
    }
    th {
        background-color: #f1f5f9;
        font-weight: 600;
        color: #475569;
    }
    td, th {
        border: 1px solid #e2e8f0;
        padding: 8px 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "b2": {
        "id": "nvidia/segformer-b2-finetuned-ade-512-512",
        "label": "B2 Default",
        "resize_target": 800,
    },
    "b3": {
        "id": "nvidia/segformer-b3-finetuned-ade-512-512",
        "label": "B3 Higher Detail",
        "resize_target": 920,
    },
}

DEFAULT_MODEL_KEY = "b2"
SAM_MODEL_ID = "facebook/sam-vit-base"
MAX_PHOTOS = 3
MIN_INSTANCE_PIXELS = 20

# W1 is Dark Grey, W2 is Light Pink
WEIGHT_MAP = {
    5: {"label": "External Nature (Trees, Water)", "color": (16, 185, 129)},
    4: {"label": "External Nature (Sky, Sand)", "color": (59, 130, 246)},
    3: {"label": "Internal Biophilia (Plants)", "color": (245, 158, 11)},
    2: {"label": "Internal Art (Nature Photos / Drawings)", "color": (255, 182, 193)},
    1: {"label": "Architectural / Hardscape", "color": (105, 105, 105)},
}

CATEGORY_DESCRIPTIONS = {
    5: "External Nature (Trees, Water)",
    4: "External Nature (Sky, Sand)",
    3: "Internal Biophilia (Plants)",
    2: "Internal Art (Nature Photos / Drawings)",
    1: "Architectural / Hardscape",
}

def init_state() -> None:
    st.session_state.setdefault("pending_files", [])
    st.session_state.setdefault("runs", {})
    st.session_state.setdefault("uploader_nonce", 0)
    st.session_state.setdefault("status_message", f"Engine ready. Add up to {MAX_PHOTOS} source images.")
    st.session_state.setdefault("status_level", "info")
    st.session_state.setdefault("interaction_points", {})

def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"

def get_device() -> int:
    return 0 if torch.cuda.is_available() else -1

@st.cache_resource(show_spinner=False)
def get_semantic_segmenter(model_id: str):
    return pipeline("image-segmentation", model=model_id, device=get_device())

@st.cache_resource(show_spinner=False)
def get_sam_interactive():
    processor = AutoProcessor.from_pretrained(SAM_MODEL_ID)
    model = AutoModelForMaskGeneration.from_pretrained(SAM_MODEL_ID).to("cuda" if get_device() == 0 else "cpu")
    return processor, model

def normalize_label(label: str) -> str:
    return (label or "").lower()

def is_aperture_label(label: str) -> bool:
    return any(token in normalize_label(label) for token in ["windowpane", "window", "glass", "door"])

def is_strong_nature_label(label: str) -> bool:
    tokens = ["tree", "grass", "earth", "mountain", "water", "sea", "river", "lake", "waterfall", "forest"]
    return any(token in normalize_label(label) for token in tokens)

def is_soft_nature_label(label: str) -> bool:
    return any(token in normalize_label(label) for token in ["sky", "cloud", "sand", "rock", "snow"])

def get_domain_for_weight(weight: int) -> str:
    if weight >= 4: return "External Nature"
    if weight == 3: return "Internal Biophilia"
    if weight == 2: return "Internal Art / Biomorphic"
    return "Architectural / Hardscape"

def get_category_info(label: str) -> dict[str, Any]:
    if is_strong_nature_label(label): return {"weight": 5, "domain": "External Nature"}
    if is_soft_nature_label(label): return {"weight": 4, "domain": "External Nature"}
    if is_aperture_label(label): return {"weight": 1, "domain": "Aperture (Window)"}
    return {"weight": 1, "domain": "Architectural / Hardscape"}

def apply_context_aware_classification(instances: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classified: list[dict[str, Any]] = []
    for inst in instances:
        label = normalize_label(inst["label"])
        updated = dict(inst)
        if "window frame" in label or is_aperture_label(label):
            updated["domain"] = "Aperture (Window)"
            updated["weight"] = 1
        elif "balcony" in label or "hardscape" in label:
            updated["domain"] = "Architectural / Hardscape"
            updated["weight"] = 1
        elif is_strong_nature_label(label):
            updated["domain"] = "External Nature"
            updated["weight"] = 5
        elif is_soft_nature_label(label):
            updated["domain"] = "External Nature"
            updated["weight"] = 4

        if updated["pixels"] > MIN_INSTANCE_PIXELS:
            classified.append(updated)
    return classified

def string_to_color(label: str) -> tuple[int, int, int]:
    hashed = sum(ord(c) * (i+1) for i, c in enumerate(label))
    return ((hashed & 0xFF), ((hashed >> 8) & 0xFF), ((hashed >> 16) & 0xFF))

def image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()

def open_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")

def resize_image_for_analysis(image: Image.Image, max_dim: int) -> Image.Image:
    resized = image.copy()
    width, height = resized.size
    if max(width, height) <= max_dim:
        return resized
    if width >= height:
        return resized.resize((max_dim, int(height * (max_dim / width))), Image.Resampling.LANCZOS)
    return resized.resize((int(width * (max_dim / height)), max_dim), Image.Resampling.LANCZOS)

def mask_to_bool_array(mask_like: Any, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    mask_array = np.array(mask_like)
    if mask_array.ndim == 3: mask_array = mask_array[..., 0]
    bool_mask = mask_array > 0 if mask_array.dtype != np.bool_ else mask_array.copy()
    if expected_shape is not None and bool_mask.shape != expected_shape:
        bool_mask = np.array(Image.fromarray(bool_mask.astype(np.uint8)*255).resize((expected_shape[1], expected_shape[0]), Image.Resampling.NEAREST)) > 0
    return bool_mask

def clone_instance(instance: dict[str, Any], mask: np.ndarray, pixels: int) -> dict[str, Any]:
    return {
        "id": instance.get("id", new_id("inst")),
        "label": instance["label"],
        "domain": instance["domain"],
        "weight": int(instance["weight"]),
        "pixels": int(pixels),
        "mask": mask,
        "raw_color": tuple(instance["raw_color"]),
        "is_locked": instance.get("is_locked", False), 
    }

def architecture_instance(height: int, width: int, mask: np.ndarray | None = None) -> dict[str, Any]:
    fill_mask = np.ones((height, width), dtype=bool) if mask is None else mask
    return {
        "id": new_id("inst"),
        "label": "Residual Architectural Fill",
        "domain": "Architectural / Hardscape",
        "weight": 1,
        "pixels": int(fill_mask.sum()),
        "mask": fill_mask,
        "raw_color": string_to_color("Residual Architectural Fill"),
        "is_locked": False,
    }

def extract_semantic_instances(segmentation_output: list[dict[str, Any]], height: int, width: int) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    structure = np.ones((3, 3), dtype=int)
    for result in segmentation_output:
        label_text = result["label"]
        category = get_category_info(label_text)
        base_mask = mask_to_bool_array(result["mask"], (height, width))
        labeled_array, feature_count = connected_components(base_mask, structure=structure)
        for feature_index in range(1, feature_count + 1):
            component_mask = labeled_array == feature_index
            pixel_count = int(component_mask.sum())
            if pixel_count > MIN_INSTANCE_PIXELS:
                instances.append({
                    "id": new_id("inst"), "label": label_text, "domain": category["domain"],
                    "weight": category["weight"], "pixels": pixel_count, "mask": component_mask,
                    "raw_color": string_to_color(label_text),
                    "is_locked": False,
                })
    return sorted(instances, key=lambda item: item["pixels"])

def build_semantic_coverage(instances: list[dict[str, Any]], height: int, width: int) -> tuple[list[dict[str, Any]], np.ndarray]:
    if not instances:
        return [architecture_instance(height, width)], np.zeros((height, width), dtype=np.int32)
    owner = np.full((height, width), -1, dtype=np.int32)
    claimed: list[dict[str, Any]] = []
    for instance in sorted(instances, key=lambda item: item["pixels"]):
        unique_mask = instance["mask"] & (owner == -1)
        if unique_mask.sum() > MIN_INSTANCE_PIXELS:
            owner[unique_mask] = len(claimed)
            claimed.append(clone_instance(instance, unique_mask, int(unique_mask.sum())))
            
    assigned = owner != -1
    if not assigned.all():
        _, indices = distance_transform_edt(~assigned, return_indices=True)
        nearest_owner = owner[indices[0], indices[1]]
        owner[~assigned] = nearest_owner[~assigned]
        
    rebuilt = [clone_instance(inst, owner == i, int((owner == i).sum())) for i, inst in enumerate(claimed)]
    return rebuilt, owner

def ensure_full_coverage(instances: list[dict[str, Any]], height: int, width: int) -> list[dict[str, Any]]:
    occupied = np.zeros((height, width), dtype=bool)
    normalized: list[dict[str, Any]] = []
    for instance in instances:
        unique_mask = instance["mask"] & ~occupied
        if unique_mask.sum() > 0:
            occupied[unique_mask] = True
            normalized.append(clone_instance(instance, unique_mask, int(unique_mask.sum())))
    if not occupied.all():
        normalized.append(architecture_instance(height, width, ~occupied))
    return normalized

def recalculate_run(run: dict[str, Any]) -> None:
    run["instances"].sort(key=lambda x: (not x.get("is_locked", False), x["pixels"]))
    run["instances"] = ensure_full_coverage(run["instances"], run["height"], run["width"])
    
    total_pixels = run["height"] * run["width"]
    weighted_sum = sum(instance["pixels"] * int(instance["weight"]) for instance in run["instances"])
    run["pixel_sum"] = total_pixels
    run["score"] = round(weighted_sum / total_pixels, 3) if total_pixels else 0.0

def build_overlay_image(run: dict[str, Any], view_mode: str) -> Image.Image:
    base = np.array(open_image_from_bytes(run["display_bytes"]), dtype=np.uint8)
    overlay = np.zeros_like(base)
    alpha_mask = np.zeros((run["height"], run["width"]), dtype=np.float32)
    for instance in run["instances"]:
        color = instance["raw_color"] if view_mode == "raw" else WEIGHT_MAP[int(instance["weight"])]["color"]
        overlay[instance["mask"]] = color
        alpha_mask[instance["mask"]] = 140 / 255.0
    blended = (base.astype(np.float32) * (1.0 - alpha_mask[..., None])) + (overlay.astype(np.float32) * alpha_mask[..., None])
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def analyze_image_bytes(filename: str, source_bytes: bytes, model_key: str) -> dict[str, Any]:
    model_config = MODEL_OPTIONS[model_key]
    original_image = open_image_from_bytes(source_bytes)
    resized_image = resize_image_for_analysis(original_image, model_config["resize_target"])
    
    semantic_segmenter = get_semantic_segmenter(model_config["id"])
    semantic_output = semantic_segmenter(resized_image)
    width, height = resized_image.size
    semantic_instances = apply_context_aware_classification(extract_semantic_instances(semantic_output, height, width))
    final_instances, _ = build_semantic_coverage(semantic_instances, height, width)

    processor, sam_model = get_sam_interactive()
    inputs = processor(images=resized_image, return_tensors="pt").to(sam_model.device)
    with torch.no_grad():
        embeddings = sam_model.get_image_embeddings(inputs["pixel_values"]).cpu()

    run = {
        "file_id": new_id("file"),
        "filename": filename,
        "source_bytes": source_bytes,
        "display_bytes": image_to_bytes(resized_image, fmt="PNG"),
        "width": width, 
        "height": height, 
        "model_key": model_key,
        "instances": final_instances,
        "sam_embeddings": embeddings, 
        "draft": None, 
    }
    recalculate_run(run)
    return run

def set_status(message: str, level: str = "info") -> None:
    st.session_state.status_message = message
    st.session_state.status_level = level

def render_status() -> None:
    level, message = st.session_state.status_level, st.session_state.status_message
    if level == "success": st.success(message)
    elif level == "warning": st.warning(message)
    elif level == "error": st.error(message)
    else: st.info(message)

def process_pending_files(model_key: str) -> None:
    if not st.session_state.pending_files: return
    
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
        <div class="loading-banner">
            <h2>Processing Visual Quality Assessment...</h2>
            <p>Please wait while the AI analyzes the architecture and computes precise masks.</p>
        </div>
    """, unsafe_allow_html=True)
    
    progress = st.progress(0.0)
    total = len(st.session_state.pending_files)
    
    try:
        get_semantic_segmenter(MODEL_OPTIONS[model_key]["id"])
        get_sam_interactive() 
        
        for index, pending in enumerate(st.session_state.pending_files, start=1):
            run = analyze_image_bytes(pending["name"], pending["bytes"], model_key)
            st.session_state.runs[run["file_id"]] = run
            st.session_state.interaction_points[run["file_id"]] = [] 
            progress.progress(index / total)
            
        st.session_state.pending_files = []
        set_status("Analyses & Pre-computations complete.", "success")
    except Exception as exc:
        set_status(f"Analysis failed: {exc}", "error")
    finally:
        loading_placeholder.empty()
        progress.empty()

def preview_interactive_sam(file_id: str, run: dict[str, Any], target_weight: int) -> None:
    points_data = st.session_state.interaction_points.get(file_id, [])
    if not points_data: return
    
    processor, model = get_sam_interactive()
    image = open_image_from_bytes(run["display_bytes"])
    
    pts = [p["coords"] for p in points_data]
    lbls = [1 if p["is_foreground"] else 0 for p in points_data]
    
    inputs = processor(images=image, input_points=[pts], input_labels=[lbls], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            image_embeddings=run["sam_embeddings"].to(model.device),
            input_points=inputs["input_points"],
            input_labels=inputs["input_labels"],
            multimask_output=True 
        )
    
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )
    
    best_mask_idx = outputs.iou_scores[0][0].argmax()
    best_mask = masks[0][0][best_mask_idx].numpy() > 0
    
    locked_mask = np.zeros((run["height"], run["width"]), dtype=bool)
    for inst in run["instances"]:
        if inst.get("is_locked", False):
            locked_mask |= inst["mask"]
            
    best_mask = best_mask & ~locked_mask
    
    if best_mask.sum() > MIN_INSTANCE_PIXELS:
        run["draft"] = {
            "id": new_id("inst"),
            "label": f"Manual Extraction (Weight {target_weight})",
            "domain": get_domain_for_weight(target_weight),
            "weight": target_weight,
            "pixels": int(best_mask.sum()),
            "mask": mask_to_bool_array(best_mask, (run["height"], run["width"])),
            "raw_color": WEIGHT_MAP[target_weight]["color"],
            "is_locked": True, 
        }

def confirm_draft(file_id: str, run: dict[str, Any]) -> None:
    new_inst = run["draft"]
    if not new_inst: return
    
    for inst in run["instances"]:
        if not inst.get("is_locked", False):
            inst["mask"] = inst["mask"] & ~new_inst["mask"]
            inst["pixels"] = int(inst["mask"].sum())
            
    run["instances"] = [i for i in run["instances"] if i["pixels"] > MIN_INSTANCE_PIXELS]
    run["instances"].append(new_inst)
    run["draft"] = None
    
    recalculate_run(run)
    set_status("Mask locked and successfully applied to the calculation.", "success")

def confirm_all_vq_elements(run: dict[str, Any]) -> None:
    run["instances"] = [inst for inst in run["instances"] if inst.get("is_locked", False)]
    run["draft"] = None
    recalculate_run(run)
    set_status("All unselected areas have been successfully assigned to Weight 1.", "success")

def draw_canvas_image(image: Image.Image, points_data: list[dict], draft: dict | None) -> Image.Image:
    base = np.array(image, dtype=np.uint8)
    
    if draft is not None:
        overlay = np.zeros_like(base)
        overlay[draft["mask"]] = draft["raw_color"]
        alpha = np.zeros((base.shape[0], base.shape[1]), dtype=np.float32)
        alpha[draft["mask"]] = 0.55 
        base = (base.astype(np.float32) * (1.0 - alpha[..., None]) + overlay.astype(np.float32) * alpha[..., None]).astype(np.uint8)

    img_copy = Image.fromarray(base)
    draw = ImageDraw.Draw(img_copy)
    
    # Modernized simple dot without complex crosshairs to prevent tracking offset lag
    radius = max(image.width // 150, 4) 
    
    for point in points_data:
        x, y = point["coords"]
        color = "lime" if point["is_foreground"] else "red"
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="white", width=2)
        
    return img_copy

def build_plotly_live_result(run: dict[str, Any]) -> go.Figure:
    blended_np = np.array(build_overlay_image(run, "weighted"))
    
    hover_text = np.full((run["height"], run["width"]), "W1 - Architectural / Hardscape", dtype=object)
    for inst in run["instances"]:
        weight = int(inst["weight"])
        label = WEIGHT_MAP[weight]['label']
        hover_text[inst["mask"]] = f"Weight {weight}: {label}"
        
    fig = go.Figure(go.Image(
        z=blended_np,
        customdata=hover_text,
        hovertemplate="<b>%{customdata}</b><extra></extra>", 
    ))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False, fixedrange=True),
        yaxis=dict(visible=False, fixedrange=True, scaleanchor="x", scaleratio=1), 
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="closest",
        dragmode=False 
    )
    return fig


init_state()

st.title("BEAM Plus Visual Quality Assessment")
render_status()

with st.container(border=True):
    col_uploader, col_process = st.columns([3, 1])
    with col_uploader:
        uploaded_files = st.file_uploader(
            f"Upload Project Images (Max {MAX_PHOTOS} photos per assessment)", 
            type=["jpg", "png", "jpeg"], 
            accept_multiple_files=True,
            key=f"uploader-{st.session_state.uploader_nonce}" 
        )
        
        if st.session_state.pending_files:
            st.markdown("**Pending Uploads:**")
            for idx, pending in enumerate(st.session_state.pending_files):
                p_col1, p_col2 = st.columns([5, 1])
                p_col1.write(pending["name"])
                if p_col2.button("Remove", key=f"del_pen_{idx}"):
                    st.session_state.pending_files.pop(idx)
                    st.rerun()

    with col_process:
        st.write("") 
        st.write("") 
        if st.button("Process Segmentation", use_container_width=True, type="primary"):
            process_pending_files(DEFAULT_MODEL_KEY)
            st.rerun()

if uploaded_files:
    for f in uploaded_files:
        if len(st.session_state.pending_files) + len(st.session_state.runs) < MAX_PHOTOS:
            st.session_state.pending_files.append({"file_id": new_id("pen"), "name": f.name, "bytes": f.getvalue()})
    st.session_state.uploader_nonce += 1
    st.rerun()

if st.session_state.runs:
    st.markdown("---")
    
    run_list = list(st.session_state.runs.values())
    tab_titles = ["Summary"] + [run["filename"] for run in run_list]
    tabs = st.tabs(tab_titles)
    
    # ---------------- SUMMARY TAB ----------------
    with tabs[0]:
        st.subheader("Assessment Summary")
        
        total_vqs = 0.0
        summary_rows = []
        
        for run in run_list:
            summary_rows.append({
                "Image Name": run["filename"],
                "VQS for Frame": f"{run['score']:.3f}"
            })
            total_vqs += run["score"]
            
        avg_vqs = total_vqs / len(run_list) if run_list else 0.0
        
        met_col1, met_col2 = st.columns(2)
        met_col1.metric("Average VQS for Viewpoint", f"{avg_vqs:.3f}")
        
        st.markdown("**Individual Frame Scores**")
        st.table(pd.DataFrame(summary_rows))

    # ---------------- IMAGE EDIT TABS ----------------
    for idx, (file_id, run) in enumerate(st.session_state.runs.items(), start=1):
        with tabs[idx]:
            # File Header
            sum_col1, sum_col2 = st.columns([4, 1])
            sum_col1.subheader(run["filename"])
            if sum_col2.button("Remove Photo", key=f"remove-{file_id}", use_container_width=True):
                st.session_state.runs.pop(file_id, None)
                st.session_state.interaction_points.pop(file_id, None)
                st.rerun()

            st.write("")
            
            # ---------------- ROW 1: UI LAYER (Tables & Controls) ----------------
            ui_col1, ui_col2 = st.columns(2)
            
            with ui_col1:
                # Modern, horizontal layout for VQS display
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: baseline; gap: 12px; margin-bottom: 1.5rem;">
                        <span style="font-size: 1.8rem; font-weight: 600; color: #334155;">VQS for Frame</span>
                        <span style="font-size: 1.8rem; font-weight: 700; color: #0284c7; background-color: rgba(2, 132, 199, 0.1); padding: 0.2rem 1rem; border-radius: 8px;">{run['score']:.3f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                st.markdown("**Category Summary**")
                
                pixels_by_weight = {5: 0, 4: 0, 3: 0, 2: 0, 1: 0}
                for inst in run["instances"]:
                    pixels_by_weight[int(inst["weight"])] += int(inst["pixels"])
                    
                rows = [{"Category": CATEGORY_DESCRIPTIONS[w], "Weight": f"W{w}", "Pixels": pixels_by_weight[w]} for w in [5,4,3,2,1]]
                total_analyzed = sum(pixels_by_weight.values())
                rows.append({"Category": "Total Analyzed Pixels", "Weight": "ALL", "Pixels": total_analyzed})
                
                df_summary = pd.DataFrame(rows)
                st.table(df_summary)
                
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    csv_data = df_summary.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Summary CSV", data=csv_data, file_name=f"{run['filename']}_summary.csv", mime="text/csv", use_container_width=True)
                with btn_col2:
                    blended_img = build_overlay_image(run, "weighted")
                    buf = io.BytesIO()
                    blended_img.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                    st.download_button("Download Live Result Image", data=img_bytes, file_name=f"{run['filename']}_result.png", mime="image/png", use_container_width=True)

            with ui_col2:
                # Enlarged headers and instructions
                st.markdown("<h3 style='margin-bottom: 0.5rem; color: #1e293b; font-size: 1.6rem;'>Fast Mask Refining (SAM Decoder) Controls</h3>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 1.05rem; color: #475569; margin-bottom: 1.5rem;'>Select target weight and drop pins on the Interactive Canvas to refine masks.</p>", unsafe_allow_html=True)
                
                ctrl_col1, ctrl_col2 = st.columns([2, 1.5])
                with ctrl_col1:
                    target_w = st.selectbox("Assign Weight to Click", options=[5,4,3,2,1], format_func=lambda v: f"W{v} - {WEIGHT_MAP[v]['label']}", key=f"w-{file_id}")
                with ctrl_col2:
                    pin_mode = st.radio("Pin Type", options=[True, False], format_func=lambda v: "Include" if v else "Exclude", horizontal=True, key=f"pin_mode_{file_id}")
                
                st.markdown("<p style='color: #0284c7; font-weight: 500; margin-top: -0.2rem;'>Pro tip: Start assigning weight for W4 - Sky for best results.</p>", unsafe_allow_html=True)

                draft = run.get("draft")
                points_data = st.session_state.interaction_points.get(file_id, [])
                
                # Dynamic action buttons
                st.write("") 
                if points_data:
                    act_col1, act_col2, act_col3 = st.columns(3)
                    if draft is not None:
                        if act_col1.button("Confirm Mask", type="primary", use_container_width=True, key=f"conf-{file_id}"):
                            confirm_draft(file_id, run)
                            st.session_state.interaction_points[file_id] = []
                            st.rerun()
                    if act_col2.button("Undo Last Pin", use_container_width=True, key=f"undo-{file_id}"):
                        st.session_state.interaction_points[file_id].pop()
                        if st.session_state.interaction_points[file_id]:
                            preview_interactive_sam(file_id, run, target_w)
                        else:
                            run["draft"] = None
                        st.rerun()
                    if act_col3.button("Cancel Draft", use_container_width=True, key=f"canc-{file_id}"):
                        st.session_state.interaction_points[file_id] = []
                        run["draft"] = None
                        st.rerun()
                
                st.divider()
                st.markdown("**Finalize Assessment:** Click below to lock current inputs and auto-assign Weight 1 to unselected areas.")
                if st.button("Confirm All VQ Elements Selected", type="primary", use_container_width=True, key=f"finish-{file_id}"):
                    confirm_all_vq_elements(run)
                    st.rerun()

            # ---------------- ROW 2: IMAGE LAYER (Guaranteed Alignment) ----------------
            st.write("") # Spacer between controls and images
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**Live Result** *(Hover over pixels to see weight)*")
                fig = build_plotly_live_result(run)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            with img_col2:
                st.markdown("**Interactive Canvas** *(Click to drop a pin)*")
                
                if draft is not None and draft["weight"] != target_w and points_data:
                    preview_interactive_sam(file_id, run, target_w)
                    st.rerun()
                
                base_image = open_image_from_bytes(run["display_bytes"])
                display_image = draw_canvas_image(base_image, points_data, draft)
                
                # standard use_column_width ensures 1:1 mapping of coordinates without arbitrary offset padding
                value = streamlit_image_coordinates(
                    display_image,
                    key=f"canvas-{file_id}",
                    use_column_width=True, 
                )
                
                tracker_key = f"last_click_{file_id}"
                if tracker_key not in st.session_state:
                    st.session_state[tracker_key] = None

                if value is not None and value != st.session_state[tracker_key]:
                    st.session_state[tracker_key] = value
                    point = [value["x"], value["y"]]
                    st.session_state.interaction_points[file_id].append({"coords": point, "is_foreground": pin_mode})
                    preview_interactive_sam(file_id, run, target_w)
                    st.rerun()