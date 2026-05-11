import streamlit as st
import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.pdfgen import canvas
from io import BytesIO

MIN_RELIABLE_CONFIDENCE = 0.45
STRICT_RELIABLE_CONFIDENCE = 0.60
COMPREHENSIVE_CONFIDENCE = 0.08   # Low floor — model is undertrained, retrain for higher scores
BALANCED_CONFIDENCE = 0.20

# ============== PAGE CONFIGURATION ==============
st.set_page_config(
    page_title="ADVIS - AI Vehicle Damage Inspector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== CUSTOM CSS ==============
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .element-container {
        color: white;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .header-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: 2px;
    }
    
    .header-subtitle {
        color: #a8d5ff;
        font-size: 1.3rem;
        text-align: center;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
        color: #333;
        min-height: 150px;
    }
    
    .metric-card h3 {
        color: #1e3c72;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.3rem;
    }
    
    .metric-card p {
        color: #555;
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    .metric-card strong {
        color: #1e3c72;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        min-height: 200px;
    }
    
    .info-box h4 {
        color: white;
        margin-top: 0;
        margin-bottom: 1rem;
    }
    
    .info-box ul {
        color: white;
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .info-box li {
        color: white;
        margin: 0.5rem 0;
    }
    
    /* Metric containers */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255,255,255,0.05);
        padding: 10px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255,255,255,0.1);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ============== HEADER ==============
st.markdown("""
<div class="header-container">
    <h1 class="header-title">🚗 ADVIS</h1>
    <p class="header-subtitle">Automated Deep Visual Inspection System for Vehicle Damage Detection</p>
</div>
""", unsafe_allow_html=True)

# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("### ⚙️ System Configuration")
    
    st.markdown("---")
    st.markdown("### 📊 Detection Settings")
    
    detection_mode = st.select_slider(
        "Detection Mode",
        options=["Comprehensive", "Balanced", "Reliable", "Strict"],
        value="Balanced",
        help="Balanced gives the best accuracy. Comprehensive may show more false positives. Reliable and Strict keep only the strongest predictions."
    )

    confidence_defaults = {
        "Comprehensive": COMPREHENSIVE_CONFIDENCE,
        "Balanced": BALANCED_CONFIDENCE,
        "Reliable": MIN_RELIABLE_CONFIDENCE,
        "Strict": STRICT_RELIABLE_CONFIDENCE,
    }

    confidence_threshold = st.slider(
        "Minimum Reported Confidence",
        min_value=0.05,
        max_value=0.90,
        value=confidence_defaults[detection_mode],
        step=0.01,
        help="Only detections at or above this confidence are shown. Lower = more detections. Higher = fewer but more certain."
    )
    high_precision_mode = st.checkbox(
        "High Precision Mode",
        value=False,
        help="Tighter filtering for fewer false positives. Turn this off when you want broader damage coverage."
    )

    comprehensive_mode = detection_mode == "Comprehensive"

    effective_confidence_floor = confidence_threshold

    st.caption(f"Active confidence floor: {effective_confidence_floor:.0%}")
    
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_masks = st.checkbox("Show Segmentation Masks", value=True)
    
    st.markdown("---")
    st.markdown("### 📈 Model Information")
    st.info(f"""
    **Model:** Mask R-CNN ResNet-50
    **Classes:** 4 Damage Types
    **Device:** CPU
    **Mode:** {detection_mode}
    **Confidence Floor:** {effective_confidence_floor:.0%}
    **Status:** ✅ Active
    """)
    
    st.markdown("---")
    st.markdown("### 📋 Damage Categories")
    damage_categories = [
        "🔴 Dent",
        "🟠 Scratch", 
        "🟡 Glass Break",
        "🟢 Smash",
        "🔵 Combined"
    ]
    for cat in damage_categories:
        st.markdown(f"• {cat}")

# ---------------- MODEL CONFIG ---------------- #

@st.cache_resource
def load_model():

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    cfg.MODEL.WEIGHTS = "TRAIN/model_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05   # Let model return candidates; UI slider filters further
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.40     # Tighter NMS — cleaner, less overlapping results
    cfg.TEST.DETECTIONS_PER_IMAGE = 30
    cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)

    # Clear any stale metadata from previous runs before setting
    try:
        MetadataCatalog.remove("__unused")
    except KeyError:
        pass
    # 4 classes matching the checkpoint
    MetadataCatalog.get("__unused").thing_classes = ["dent", "glass_break", "scratch", "smash"]

    return predictor


with st.spinner("🔄 Loading AI Model..."):
    predictor = load_model()

# ---------------- FUNCTIONS ---------------- #

def create_damage_chart(damage_types):
    """Create a pie chart for damage distribution"""
    if not damage_types:
        return None
    
    labels = list(damage_types.keys())
    values = list(damage_types.values())
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.4,
        marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    )])
    
    fig.update_layout(
        title_text="Damage Type Distribution",
        showlegend=True,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12)
    )
    
    return fig

def create_confidence_chart(instances):
    """Create a bar chart for detection confidence scores"""
    if len(instances) == 0:
        return None
    
    scores = instances.scores.numpy()
    class_names = MetadataCatalog.get("__unused").thing_classes
    pred_classes = instances.pred_classes.numpy()
    
    labels = [f"{class_names[cls][:15]}" for cls in pred_classes[:10]]  # Top 10
    scores_list = scores[:10].tolist()
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=scores_list,
            marker_color='rgba(76, 175, 80, 0.8)',
            text=[f'{s:.2%}' for s in scores_list],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title_text="Detection Confidence Scores (Top 10)",
        xaxis_title="Damage Type",
        yaxis_title="Confidence",
        yaxis=dict(range=[0, 1]),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12)
    )
    
    return fig

def _resize_to_short(image, target_short):
    """Resize image so shortest side == target_short."""
    h, w = image.shape[:2]
    scale = target_short / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interp)

def _apply_clahe(image, clip=1.5):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def preprocess_image(image):
    """Resize to 800px short-side and apply mild CLAHE."""
    image = _resize_to_short(image, 800)
    return _apply_clahe(image, clip=1.5)

def create_image_variants(image, comprehensive_mode=False):
    """
    Return list of (name, img, x_offset, y_offset, scale_x, scale_y).
    Offsets and scales map detected boxes back to base-image coordinates.
    """
    h, w = image.shape[:2]
    variants = [("base", image, 0, 0, 1.0, 1.0)]

    if comprehensive_mode:
        # ── Photometric variants (full image) ──────────────────
        sharpen_k  = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
        sharpened  = cv2.filter2D(image, -1, sharpen_k)
        brightened = cv2.convertScaleAbs(image, alpha=1.10, beta=15)
        darkened   = cv2.convertScaleAbs(image, alpha=0.85, beta=-10)
        clahe_hi   = _apply_clahe(image, clip=3.0)

        variants += [
            ("sharpened",  sharpened,  0, 0, 1.0, 1.0),
            ("brightened", brightened, 0, 0, 1.0, 1.0),
            ("darkened",   darkened,   0, 0, 1.0, 1.0),
            ("clahe_hi",   clahe_hi,   0, 0, 1.0, 1.0),
        ]

        # ── Multi-scale full-image passes ──────────────────────
        for short_side in [640, 1024, 1200]:
            scaled = _resize_to_short(image, short_side)
            sf = min(h, w) / short_side
            variants.append((f"scale_{short_side}", scaled, 0, 0, sf, sf))

        # ── Tiled crop passes (overlapping quadrants) ──────────
        tile_w = int(w * 0.65)
        tile_h = int(h * 0.65)
        stride_x = int(w * 0.35)
        stride_y = int(h * 0.35)

        for ty in range(0, h - tile_h + 1, stride_y):
            for tx in range(0, w - tile_w + 1, stride_x):
                tile = image[ty:ty+tile_h, tx:tx+tile_w]
                tile_up = _resize_to_short(tile, 800)
                sfx = tile_w / tile_up.shape[1]
                sfy = tile_h / tile_up.shape[0]
                variants.append((f"tile_{tx}_{ty}", tile_up, tx, ty, sfx, sfy))

    return variants

def get_effective_confidence_threshold(conf_threshold, high_precision, comprehensive_mode=False):
    """Return the confidence threshold the user selected."""
    if comprehensive_mode:
        return min(conf_threshold, BALANCED_CONFIDENCE)
    return conf_threshold

def run_tta(image, comprehensive_mode=False):
    """
    Run inference on multiple variants (photometric + multi-scale + tiled crops)
    and horizontal flips, then remap all boxes to base-image coordinates and
    merge with class-aware NMS.
    """
    from detectron2.structures import Instances, Boxes
    import torch
    from torchvision.ops import batched_nms

    base_h, base_w = image.shape[:2]

    def infer(img):
        out = predictor(img)
        return out["instances"].to("cpu")

    def remap_boxes(inst, ox, oy, sx, sy, img_w):
        """Translate + scale boxes from variant-space → base-image space."""
        if len(inst) == 0:
            return inst
        boxes = inst.pred_boxes.tensor.clone().float()
        # sx/sy may be a scalar (uniform) or already per-axis
        if isinstance(sx, (int, float)):
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * sx + ox
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * sy + oy
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * sx + ox
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * sy + oy
        # Clamp to base image bounds
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, base_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, base_h)
        inst.pred_boxes = Boxes(boxes)
        # Masks are in variant-space — drop them after remapping to avoid mismatch
        if inst.has("pred_masks"):
            inst.remove("pred_masks")
        return inst

    all_instances = []

    for name, variant, ox, oy, sx, sy in create_image_variants(image, comprehensive_mode=comprehensive_mode):
        v_h, v_w = variant.shape[:2]

        # Forward pass
        inst = infer(variant)
        inst = remap_boxes(inst, ox, oy, sx, sy, v_w)
        all_instances.append(inst)

        # Horizontal flip pass
        flipped = cv2.flip(variant, 1)
        inst_f  = infer(flipped)
        if len(inst_f) > 0:
            # Un-flip boxes in variant space before remapping
            fb = inst_f.pred_boxes.tensor.clone()
            fb[:, [0, 2]] = v_w - fb[:, [2, 0]]
            inst_f.pred_boxes.tensor = fb
            if inst_f.has("pred_masks"):
                inst_f.pred_masks = inst_f.pred_masks.flip(-1)
        inst_f = remap_boxes(inst_f, ox, oy, sx, sy, v_w)
        all_instances.append(inst_f)

    non_empty = [i for i in all_instances if len(i) > 0]
    if not non_empty:
        # Return empty Instances in base image size
        empty = Instances((base_h, base_w))
        empty.pred_boxes   = Boxes(torch.zeros((0, 4)))
        empty.scores       = torch.zeros(0)
        empty.pred_classes = torch.zeros(0, dtype=torch.long)
        return empty

    # Merge all predictions into a single Instances at base resolution
    combined = Instances((base_h, base_w))
    combined.scores       = torch.cat([i.scores       for i in non_empty])
    combined.pred_classes = torch.cat([i.pred_classes for i in non_empty])
    combined.pred_boxes   = Boxes(torch.cat([i.pred_boxes.tensor for i in non_empty]))

    # Class-aware NMS to deduplicate across variants
    keep = batched_nms(
        combined.pred_boxes.tensor,
        combined.scores,
        combined.pred_classes,
        iou_threshold=0.40 if comprehensive_mode else 0.35,
    )
    return combined[keep]

def rescue_class_coverage(instances, fallback_instances, min_score):
    from detectron2.structures import Instances

    if len(fallback_instances) == 0:
        return instances

    missing_classes = set(fallback_instances.pred_classes.numpy().tolist())
    if len(instances) > 0:
        missing_classes -= set(instances.pred_classes.numpy().tolist())

    if not missing_classes:
        return instances

    recovered_indices = []
    fallback_scores = fallback_instances.scores.numpy()
    fallback_classes = fallback_instances.pred_classes.numpy()
    for cls_idx in sorted(missing_classes):
        candidate_indices = np.where((fallback_classes == cls_idx) & (fallback_scores >= min_score))[0]
        if candidate_indices.size > 0:
            best_idx = candidate_indices[np.argmax(fallback_scores[candidate_indices])]
            recovered_indices.append(int(best_idx))

    if not recovered_indices:
        return instances

    recovered = fallback_instances[recovered_indices]
    if len(instances) == 0:
        return recovered

    merged = Instances(instances.image_size)
    merged.scores = torch.cat([instances.scores, recovered.scores])
    merged.pred_classes = torch.cat([instances.pred_classes, recovered.pred_classes])
    merged.pred_boxes = type(instances.pred_boxes).cat([instances.pred_boxes, recovered.pred_boxes])
    if instances.has("pred_masks") and recovered.has("pred_masks"):
        merged.pred_masks = torch.cat([instances.pred_masks, recovered.pred_masks])
    return merged

def detect_damage(image, conf_threshold=0.08, high_precision=True, comprehensive_mode=False):
    # Preprocess base image (resize + CLAHE) — used for model input
    proc_image = preprocess_image(image.copy())

    # run_tta now returns all boxes in proc_image coordinate space
    raw_instances = run_tta(proc_image, comprehensive_mode=comprehensive_mode)
    instances = raw_instances

    effective_threshold = get_effective_confidence_threshold(conf_threshold, high_precision, comprehensive_mode)

    # Filter by confidence threshold
    if len(instances) > 0:
        keep_mask = instances.scores >= effective_threshold
        instances = instances[keep_mask]

    if comprehensive_mode:
        instances = rescue_class_coverage(instances, raw_instances, min_score=max(0.05, effective_threshold - 0.10))

    # Remove tiny boxes (common false positives on reflections/textures)
    if len(instances) > 0:
        img_h, img_w = proc_image.shape[:2]
        image_area = float(img_h * img_w)
        boxes = instances.pred_boxes.tensor
        widths  = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
        heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        box_areas = widths * heights
        min_area_ratio = 0.001 if comprehensive_mode else (0.003 if high_precision else 0.002)
        keep_area = (box_areas / image_area) > min_area_ratio
        instances = instances[keep_area]

    # Keep strongest detections
    max_detections = 12 if comprehensive_mode else (5 if high_precision else 8)
    if len(instances) > max_detections:
        topk_indices = instances.scores.topk(max_detections).indices
        instances = instances[topk_indices]

    damage_count = len(instances)

    # Aggregate damage types and confidence
    damage_types   = {}
    avg_confidence = 0.0

    if damage_count > 0:
        pred_classes = instances.pred_classes.numpy()
        class_names  = MetadataCatalog.get("__unused").thing_classes
        for cls_idx in pred_classes:
            cls_name = class_names[cls_idx]
            damage_types[cls_name] = damage_types.get(cls_name, 0) + 1
        avg_confidence = float(instances.scores.mean())

    if damage_count == 0:
        severity = "No Damage";  severity_icon = "🟢"; severity_color = "#4CAF50"
    elif damage_count <= 2:
        severity = "Low";        severity_icon = "🟡"; severity_color = "#FFC107"
    elif damage_count <= 5:
        severity = "Medium";     severity_icon = "🟠"; severity_color = "#FF9800"
    else:
        severity = "High";       severity_icon = "🔴"; severity_color = "#F44336"

    # ── Scale boxes back to ORIGINAL (unprocessed) image for display ──
    draw_instances = instances
    if proc_image.shape[:2] != image.shape[:2] and len(instances) > 0:
        ph, pw = proc_image.shape[:2]
        oh, ow = image.shape[:2]
        sx, sy = ow / pw, oh / ph
        from detectron2.structures import Boxes
        import torch
        scaled_boxes = instances.pred_boxes.tensor.clone().float()
        scaled_boxes[:, [0, 2]] *= sx
        scaled_boxes[:, [1, 3]] *= sy
        scaled_boxes[:, [0, 2]] = scaled_boxes[:, [0, 2]].clamp(0, ow)
        scaled_boxes[:, [1, 3]] = scaled_boxes[:, [1, 3]].clamp(0, oh)
        # Build a fresh Instances so we don't mutate the cached ones
        from detectron2.structures import Instances
        draw_instances = Instances((oh, ow))
        draw_instances.pred_boxes   = Boxes(scaled_boxes)
        draw_instances.scores       = instances.scores.clone()
        draw_instances.pred_classes = instances.pred_classes.clone()
        # Masks are in proc_image space — drop to avoid size mismatch
        # (they would need bicubic upscaling which is out of scope here)

    v = Visualizer(
        image[:, :, ::-1],
        metadata=MetadataCatalog.get("__unused"),
        scale=1.2,
    )
    out = v.draw_instance_predictions(draw_instances)
    result_image = out.get_image()[:, :, ::-1]

    return {
        'result_image':   result_image,
        'damage_count':   damage_count,
        'severity':       severity,
        'severity_icon':  severity_icon,
        'severity_color': severity_color,
        'damage_types':   damage_types,
        'avg_confidence': avg_confidence,
        'confidence_floor': effective_threshold,
        'detection_mode': 'Comprehensive' if comprehensive_mode else ('Strict' if high_precision else 'Balanced'),
        'instances':      instances,
    }

def render_confidence_guidance(results):
    floor_pct = int(round(results['confidence_floor'] * 100))
    avg_pct = int(round(results['avg_confidence'] * 100))
    detection_mode = results.get('detection_mode', 'Balanced')

    if results['damage_count'] == 0:
        st.warning(
            f"No detections met the {floor_pct}% confidence floor in {detection_mode.lower()} mode. Lower the threshold or use Comprehensive mode for broader damage coverage."
        )
        return

    if detection_mode == 'Comprehensive':
        st.success(f"Comprehensive mode is active. The app is keeping broader damage coverage with an average confidence of {avg_pct}% at a {floor_pct}% floor.")
    elif results['avg_confidence'] >= STRICT_RELIABLE_CONFIDENCE:
        st.success(f"Strong detections confirmed. Average confidence is {avg_pct}% at a {floor_pct}% floor.")
    else:
        st.info(f"{detection_mode} mode kept detections at a {floor_pct}% floor. Current average confidence is {avg_pct}%.")

def generate_report(results, image_name):
    """Generate a text report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
╔══════════════════════════════════════════════════════════╗
║         ADVIS - DAMAGE INSPECTION REPORT                ║
╚══════════════════════════════════════════════════════════╝

Report Generated: {timestamp}
Image: {image_name}

─────────────────────────────────────────────────────────
INSPECTION SUMMARY
─────────────────────────────────────────────────────────
Total Damages Detected: {results['damage_count']}
Severity Level: {results['severity_icon']} {results['severity']}
Average Confidence: {results['avg_confidence']:.2%}
Confidence Floor Applied: {results['confidence_floor']:.0%}

─────────────────────────────────────────────────────────
DAMAGE TYPE BREAKDOWN
─────────────────────────────────────────────────────────
"""
    
    if results['damage_types']:
        for damage_type, count in results['damage_types'].items():
            report += f"  • {damage_type}: {count}\n"
    else:
        report += "  No damage detected\n"
    
    report += """
─────────────────────────────────────────────────────────
RECOMMENDATIONS
─────────────────────────────────────────────────────────
"""
    
    if results['damage_count'] == 0:
        report += "  ✓ Vehicle appears to be in good condition\n"
        report += "  ✓ No repairs required\n"
    elif results['severity'] == "Low":
        report += "  ⚠ Minor repairs recommended\n"
        report += "  ⚠ Estimate: $500 - $1,500\n"
    elif results['severity'] == "Medium":
        report += "  ⚠ Moderate repairs required\n"
        report += "  ⚠ Estimate: $1,500 - $5,000\n"
    else:
        report += "  🚨 Significant repairs required\n"
        report += "  🚨 Estimate: $5,000+\n"
        report += "  🚨 Professional assessment recommended\n"
    
    report += "\n═══════════════════════════════════════════════════════════\n"
    report += "     Generated by ADVIS - AI Damage Inspection System\n"
    report += "═══════════════════════════════════════════════════════════\n"
    
    return report

def generate_pdf_report(results, image_name, result_image):
    """Generate a PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e3c72'),
        spaceAfter=30,
        alignment=1  # Center
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2a5298'),
        spaceAfter=12,
        spaceBefore=12
    )
    normal_style = styles['Normal']
    
    # Title
    title = Paragraph("ADVIS - DAMAGE INSPECTION REPORT", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Report Info
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_data = [
        ['Report Generated:', timestamp],
        ['Image File:', image_name],
    ]
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Inspection Summary
    elements.append(Paragraph("INSPECTION SUMMARY", heading_style))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Damages Detected', str(results['damage_count'])],
        ['Severity Level', f"{results['severity']}"],
        ['Average Confidence', f"{results['avg_confidence']:.2%}"],
    ]
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Damage Type Breakdown
    if results['damage_types']:
        elements.append(Paragraph("DAMAGE TYPE BREAKDOWN", heading_style))
        damage_data = [['Damage Type', 'Count', 'Percentage']]
        for damage_type, count in results['damage_types'].items():
            percentage = (count / results['damage_count']) * 100
            damage_data.append([damage_type, str(count), f"{percentage:.1f}%"])
        
        damage_table = Table(damage_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        damage_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a5298')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(damage_table)
        elements.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    elements.append(Paragraph("RECOMMENDATIONS", heading_style))
    recommendations = []
    if results['damage_count'] == 0:
        recommendations = [
            "• Vehicle appears to be in good condition",
            "• No repairs required"
        ]
    elif results['severity'] == "Low":
        recommendations = [
            "• Minor repairs recommended",
            "• Estimated Cost: $500 - $1,500"
        ]
    elif results['severity'] == "Medium":
        recommendations = [
            "• Moderate repairs required",
            "• Estimated Cost: $1,500 - $5,000"
        ]
    else:
        recommendations = [
            "• Significant repairs required",
            "• Estimated Cost: $5,000+",
            "• Professional assessment recommended"
        ]
    
    for rec in recommendations:
        elements.append(Paragraph(rec, normal_style))
        elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Add result image if available
    if result_image is not None:
        try:
            # Convert result image to format compatible with ReportLab
            img_buffer = BytesIO()
            _, buffer_arr = cv2.imencode('.jpg', result_image)
            img_buffer.write(buffer_arr)
            img_buffer.seek(0)
            
            elements.append(Paragraph("DETECTION RESULTS", heading_style))
            img = RLImage(img_buffer, width=5*inch, height=3.75*inch)
            elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"Note: Unable to embed result image", normal_style))
    
    # Footer
    elements.append(Spacer(1, 0.3*inch))
    footer = Paragraph(
        "<para align=center><i>Generated by ADVIS - AI Damage Inspection System</i></para>",
        normal_style
    )
    elements.append(footer)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


def aggregate_multi_angle_results(captures):
    """Aggregate results from multiple angle captures"""
    total_damages = 0
    all_damage_types = {}
    total_confidence = 0
    total_detections = 0
    
    for capture in captures:
        results = capture['results']
        total_damages += results['damage_count']
        
        # Aggregate damage types
        for damage_type, count in results['damage_types'].items():
            all_damage_types[damage_type] = all_damage_types.get(damage_type, 0) + count
        
        # Sum confidence scores
        total_confidence += results['avg_confidence'] * results['damage_count']
        total_detections += results['damage_count']
    
    # Calculate overall metrics
    avg_confidence = total_confidence / total_detections if total_detections > 0 else 0
    
    # Determine overall severity
    if total_damages == 0:
        severity = "No Damage"
        severity_color = "#4CAF50"
        severity_icon = "✅"
    elif total_damages <= 3:
        severity = "Low"
        severity_color = "#FFC107"
        severity_icon = "⚠️"
    elif total_damages <= 7:
        severity = "Medium"
        severity_color = "#FF9800"
        severity_icon = "⚠️"
    else:
        severity = "High"
        severity_color = "#F44336"
        severity_icon = "🚨"
    
    return {
        'damage_count': total_damages,
        'damage_types': all_damage_types,
        'avg_confidence': avg_confidence,
        'severity': severity,
        'severity_color': severity_color,
        'severity_icon': severity_icon,
        'total_angles': len(captures)
    }

def generate_multi_angle_report(captures, aggregated_results):
    """Generate a comprehensive text report for multi-angle inspection"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
╔══════════════════════════════════════════════════════════╗
║   ADVIS - COMPREHENSIVE MULTI-ANGLE INSPECTION REPORT   ║
╚══════════════════════════════════════════════════════════╝

Report Generated: {timestamp}
Inspection Type: Multi-Angle Vehicle Assessment
Angles Captured: {aggregated_results['total_angles']}

─────────────────────────────────────────────────────────
OVERALL INSPECTION SUMMARY
─────────────────────────────────────────────────────────
Total Damages Detected: {aggregated_results['damage_count']}
Overall Severity: {aggregated_results['severity_icon']} {aggregated_results['severity']}
Average Confidence: {aggregated_results['avg_confidence']:.2%}

─────────────────────────────────────────────────────────
ANGLE-BY-ANGLE BREAKDOWN
─────────────────────────────────────────────────────────
"""
    
    for i, capture in enumerate(captures, 1):
        angle = capture['angle']
        results = capture['results']
        report += f"\n{i}. {angle} View:\n"
        report += f"   Damages: {results['damage_count']}"
        if results['damage_types']:
            report += f" ({', '.join([f'{k}:{v}' for k, v in results['damage_types'].items()])})"
        report += f"\n   Confidence: {results['avg_confidence']:.2%}\n"
        report += f"   Confidence Floor: {results['confidence_floor']:.0%}\n"
    
    report += """
─────────────────────────────────────────────────────────
COMBINED DAMAGE TYPE BREAKDOWN
─────────────────────────────────────────────────────────
"""
    
    if aggregated_results['damage_types']:
        for damage_type, count in aggregated_results['damage_types'].items():
            percentage = (count / aggregated_results['damage_count']) * 100
            report += f"  • {damage_type}: {count} ({percentage:.1f}%)\n"
    else:
        report += "  No damage detected across all angles\n"
    
    report += """
─────────────────────────────────────────────────────────
RECOMMENDATIONS
─────────────────────────────────────────────────────────
"""
    
    if aggregated_results['damage_count'] == 0:
        report += "  ✓ Vehicle appears to be in excellent condition\n"
        report += "  ✓ No repairs required\n"
        report += "  ✓ All angles inspected show no damage\n"
    elif aggregated_results['severity'] == "Low":
        report += "  ⚠ Minor repairs recommended\n"
        report += "  ⚠ Estimated Cost: $500 - $2,000\n"
        report += "  ⚠ Schedule cosmetic repairs at convenience\n"
    elif aggregated_results['severity'] == "Medium":
        report += "  ⚠ Moderate repairs required\n"
        report += "  ⚠ Estimated Cost: $2,000 - $7,000\n"
        report += "  ⚠ Repairs should be scheduled soon\n"
    else:
        report += "  🚨 Significant repairs required\n"
        report += "  🚨 Estimated Cost: $7,000+\n"
        report += "  🚨 Professional assessment strongly recommended\n"
        report += "  🚨 Multiple areas of concern detected\n"
    
    report += "\n═══════════════════════════════════════════════════════════\n"
    report += "     Generated by ADVIS - AI Damage Inspection System\n"
    report += "            Comprehensive Multi-Angle Analysis\n"
    report += "═══════════════════════════════════════════════════════════\n"
    
    return report

def generate_multi_angle_pdf_report(captures, aggregated_results):
    """Generate a comprehensive PDF report for multi-angle inspection"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=colors.HexColor('#1e3c72'),
        spaceAfter=20,
        alignment=1
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2a5298'),
        spaceAfter=12,
        spaceBefore=12
    )
    normal_style = styles['Normal']
    
    # Title
    title = Paragraph("ADVIS - COMPREHENSIVE MULTI-ANGLE<br/>VEHICLE INSPECTION REPORT", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.2*inch))
    
    # Report Info
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    info_data = [
        ['Report Generated:', timestamp],
        ['Inspection Type:', 'Multi-Angle Vehicle Assessment'],
        ['Angles Captured:', str(aggregated_results['total_angles'])],
    ]
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f4f8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Overall Summary
    elements.append(Paragraph("OVERALL INSPECTION SUMMARY", heading_style))
    summary_data = [
        ['Metric', 'Value'],
        ['Total Damages Detected', str(aggregated_results['damage_count'])],
        ['Overall Severity', aggregated_results['severity']],
        ['Average Confidence', f"{aggregated_results['avg_confidence']:.2%}"],
    ]
    summary_table = Table(summary_data, colWidths=[3*inch, 3*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4CAF50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Angle-by-Angle Breakdown
    elements.append(Paragraph("ANGLE-BY-ANGLE BREAKDOWN", heading_style))
    angle_data = [['Angle', 'Damages', 'Confidence', 'Status']]
    for capture in captures:
        angle = capture['angle']
        results = capture['results']
        status = "Clear" if results['damage_count'] == 0 else results['severity']
        angle_data.append([
            angle,
            str(results['damage_count']),
            f"{results['avg_confidence']:.1%}",
            status
        ])
    
    angle_table = Table(angle_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    angle_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a5298')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(angle_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Combined Damage Type Breakdown
    if aggregated_results['damage_types']:
        elements.append(Paragraph("COMBINED DAMAGE TYPE BREAKDOWN", heading_style))
        damage_data = [['Damage Type', 'Count', 'Percentage']]
        for damage_type, count in aggregated_results['damage_types'].items():
            percentage = (count / aggregated_results['damage_count']) * 100
            damage_data.append([damage_type, str(count), f"{percentage:.1f}%"])
        
        damage_table = Table(damage_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
        damage_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2a5298')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(damage_table)
        elements.append(Spacer(1, 0.2*inch))
    
    # Recommendations
    elements.append(Paragraph("RECOMMENDATIONS", heading_style))
    recommendations = []
    if aggregated_results['damage_count'] == 0:
        recommendations = [
            "• Vehicle appears to be in excellent condition",
            "• No repairs required",
            "• All angles inspected show no damage"
        ]
    elif aggregated_results['severity'] == "Low":
        recommendations = [
            "• Minor repairs recommended",
            "• Estimated Cost: $500 - $2,000",
            "• Schedule cosmetic repairs at convenience"
        ]
    elif aggregated_results['severity'] == "Medium":
        recommendations = [
            "• Moderate repairs required",
            "• Estimated Cost: $2,000 - $7,000",
            "• Repairs should be scheduled soon"
        ]
    else:
        recommendations = [
            "• Significant repairs required",
            "• Estimated Cost: $7,000+",
            "• Professional assessment strongly recommended",
            "• Multiple areas of concern detected"
        ]
    
    for rec in recommendations:
        elements.append(Paragraph(rec, normal_style))
        elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Add images from each angle (thumbnails)
    elements.append(Paragraph("DETECTION RESULTS BY ANGLE", heading_style))
    for i, capture in enumerate(captures):
        try:
            img_buffer = BytesIO()
            _, buffer_arr = cv2.imencode('.jpg', capture['result_image'])
            img_buffer.write(buffer_arr)
            img_buffer.seek(0)
            
            elements.append(Paragraph(f"<b>{capture['angle']} View</b>", normal_style))
            img = RLImage(img_buffer, width=4*inch, height=3*inch)
            elements.append(img)
            elements.append(Spacer(1, 0.15*inch))
            
            # Page break after every 2 images to avoid overflow
            if (i + 1) % 2 == 0 and i < len(captures) - 1:
                elements.append(PageBreak())
        except Exception as e:
            elements.append(Paragraph(f"Note: Unable to embed {capture['angle']} image", normal_style))
    
    # Footer
    elements.append(Spacer(1, 0.2*inch))
    footer = Paragraph(
        "<para align=center><i>Generated by ADVIS - AI Damage Inspection System<br/>Comprehensive Multi-Angle Analysis</i></para>",
        normal_style
    )
    elements.append(footer)
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ============== MAIN APPLICATION ==============

# Initialize session state for tracking analysis history
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'latest_analysis' not in st.session_state:
    st.session_state.latest_analysis = None

# Initialize multi-capture session state
if 'capture_mode' not in st.session_state:
    st.session_state.capture_mode = 'single'
if 'multi_captures' not in st.session_state:
    st.session_state.multi_captures = []
if 'current_angle_index' not in st.session_state:
    st.session_state.current_angle_index = 0
if 'angles' not in st.session_state:
    st.session_state.angles = ['Front View', 'Rear View', 'Left Side', 'Right Side']

# Create tabs for different input methods
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Live Camera", "📊 Dashboard"])

# ============== TAB 1: IMAGE UPLOAD ==============
with tab1:
    st.markdown("### 📁 Upload Vehicle Image for Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "png", "jpeg", "bmp"],
            help="Upload a clear image of the vehicle damage"
        )
    
    with col2:
        st.info("""
        **Tips for best results:**
        - Clear, well-lit images
        - Focus on damaged area
        - Multiple angles recommended
        """)
    
    if uploaded_file is not None:
        # Read and process image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Create columns for original and result
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            st.image(image, channels="BGR", use_container_width=True)
        
        with col2:
            with st.spinner("🔍 Analyzing damage... Please wait"):
                results = detect_damage(image, confidence_threshold, high_precision_mode, comprehensive_mode)
            
            # Store results in session state
            st.session_state.latest_analysis = {
                'original_image': image.copy(),
                'result_image': results['result_image'].copy(),
                'results': results,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': uploaded_file.name
            }
            
            st.markdown("#### Detection Results")
            st.image(results['result_image'], channels="BGR", use_container_width=True)
        
        # Display metrics
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="🔍 Damages Detected",
                value=results['damage_count'],
                delta=None
            )
        
        with metric_col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: {results['severity_color']}; 
                        border-radius: 10px; color: white; font-weight: bold;'>
                <h3 style='margin:0; color: white;'>{results['severity_icon']} {results['severity']}</h3>
                <p style='margin:0; font-size: 0.9rem;'>Severity Level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.metric(
                label="🎯 Avg Detection Confidence",
                value=f"{results['avg_confidence']:.1%}",
                delta=None
            )
        
        with metric_col4:
            st.metric(
                label="📋 Damage Types",
                value=len(results['damage_types']),
                delta=None
            )
        
        # Detailed analysis
        st.markdown("---")
        render_confidence_guidance(results)
        st.caption(f"Only detections at or above {results['confidence_floor']:.0%} are included in this analysis.")
        
        if results['damage_types']:
            detail_col1, detail_col2 = st.columns(2)
            
            with detail_col1:
                st.markdown("#### 📋 Damage Type Breakdown")
                for damage_type, count in results['damage_types'].items():
                    percentage = (count / results['damage_count']) * 100
                    st.markdown(f"""
                    <div style='background: linear-gradient(90deg, #4CAF50 0%, #45a049 {percentage}%, transparent {percentage}%);
                                padding: 0.75rem; margin: 0.5rem 0; border-radius: 8px; color: white; font-weight: bold;'>
                        {damage_type}: {count} ({percentage:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)
            
            with detail_col2:
                # Show charts
                if results['damage_types']:
                    fig = create_damage_chart(results['damage_types'])
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="upload_damage_chart")
        
        # Show confidence chart
        if len(results['instances']) > 0 and show_confidence:
            st.markdown("---")
            fig_conf = create_confidence_chart(results['instances'])
            if fig_conf:
                st.plotly_chart(fig_conf, use_container_width=True, key="upload_confidence_chart")
        
        # Generate and download report
        st.markdown("---")
        report_text = generate_report(results, uploaded_file.name)
        pdf_buffer = generate_pdf_report(results, uploaded_file.name, results['result_image'])
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.download_button(
                label="📥 Download TXT Report",
                data=report_text,
                file_name=f"ADVIS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name=f"ADVIS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        
        with col3:
            # Save result image
            _, buffer = cv2.imencode('.jpg', results['result_image'])
            st.download_button(
                label="💾 Download Image",
                data=buffer.tobytes(),
                file_name=f"ADVIS_Result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                mime="image/jpeg",
                use_container_width=True
            )

# ============== TAB 2: LIVE CAMERA ==============
with tab2:
    st.markdown("### 📷 Live Camera Damage Detection")
    
    # Initialize session state for camera
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    
    # Mode selection
    st.markdown("#### Inspection Mode")
    mode_col1, mode_col2, mode_col3 = st.columns([2, 2, 2])
    
    with mode_col1:
        if st.button("📷 Single Capture Mode", use_container_width=True, 
                    type="primary" if st.session_state.capture_mode == 'single' else "secondary"):
            st.session_state.capture_mode = 'single'
            st.session_state.multi_captures = []
            st.session_state.current_angle_index = 0
            st.rerun()
    
    with mode_col2:
        if st.button("🔄 Multi-Angle Mode", use_container_width=True, 
                    type="primary" if st.session_state.capture_mode == 'multi' else "secondary"):
            st.session_state.capture_mode = 'multi'
            st.session_state.multi_captures = []
            st.session_state.current_angle_index = 0
            st.session_state.camera_active = False
            st.rerun()
    
    with mode_col3:
        if st.session_state.capture_mode == 'multi' and len(st.session_state.multi_captures) > 0:
            if st.button("🔄 Reset Multi-Capture", use_container_width=True):
                st.session_state.multi_captures = []
                st.session_state.current_angle_index = 0
                st.session_state.camera_active = False
                st.rerun()
    
    st.markdown("---")
    
    # ========== SINGLE CAPTURE MODE ==========
    if st.session_state.capture_mode == 'single':
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📷 Activate Camera", use_container_width=True, type="primary"):
                st.session_state.camera_active = True
        
        with col2:
            if st.button("❌ Deactivate Camera", use_container_width=True):
                st.session_state.camera_active = False
        
        if st.session_state.camera_active:
            camera = st.camera_input("Capture vehicle damage image")
            
            if camera is not None:
                file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Captured Image")
                    st.image(image, channels="BGR", use_container_width=True)
                
                with col2:
                    with st.spinner("🔍 Analyzing damage..."):
                        results = detect_damage(image, confidence_threshold, high_precision_mode, comprehensive_mode)
                    
                    # Store results in session state
                    st.session_state.latest_analysis = {
                        'original_image': image.copy(),
                        'result_image': results['result_image'].copy(),
                        'results': results,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'filename': f"camera_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    }
                    
                    st.markdown("#### Detection Results")
                    st.image(results['result_image'], channels="BGR", use_container_width=True)
                
                # Display metrics
                st.markdown("---")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("🔍 Damages", results['damage_count'])
                
                with metric_col2:
                    st.markdown(f"""
                    <div style='text-align: center; padding: 1rem; background: {results['severity_color']}; 
                                border-radius: 10px; color: white; font-weight: bold;'>
                        <h3 style='margin:0; color: white;'>{results['severity_icon']} {results['severity']}</h3>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_col3:
                    st.metric("🎯 Detection Confidence", f"{results['avg_confidence']:.1%}")
                
                with metric_col4:
                    st.metric("📋 Types", len(results['damage_types']))

                render_confidence_guidance(results)
                st.caption(f"Only detections at or above {results['confidence_floor']:.0%} are included in this analysis.")
                
                # Show damage breakdown
                if results['damage_types']:
                    st.markdown("---")
                    st.markdown("#### 📋 Detected Damage Types")
                    for damage_type, count in results['damage_types'].items():
                        st.markdown(f"• **{damage_type}**: {count}")
                    
                    # Generate reports
                    camera_filename = f"camera_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    report_text = generate_report(results, camera_filename)
                    pdf_buffer = generate_pdf_report(results, camera_filename, results['result_image'])
                    
                    cam_col1, cam_col2 = st.columns(2)
                    with cam_col1:
                        st.download_button(
                            label="📥 Download TXT Report",
                            data=report_text,
                            file_name=f"ADVIS_Camera_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    
                    with cam_col2:
                        st.download_button(
                            label="📄 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"ADVIS_Camera_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
        else:
            st.info("👆 Click 'Activate Camera' button to start damage detection")
            st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 15px; color: white; text-align: center;'>
                <h2>📱 Single Capture Mode</h2>
                <p>Quick inspection of a specific vehicle area</p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== MULTI-ANGLE MODE ==========
    else:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%); 
                    padding: 1.5rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 1rem;'>
            <h3 style='margin:0; color: white;'>🔄 Multi-Angle Comprehensive Inspection</h3>
            <p style='margin:0.5rem 0 0 0;'>Capture all angles for complete vehicle damage assessment</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress indicator
        total_angles = len(st.session_state.angles)
        captured_angles = len(st.session_state.multi_captures)
        progress = captured_angles / total_angles
        
        st.progress(progress, text=f"Progress: {captured_angles}/{total_angles} angles captured")
        
        # Show captured angles status
        st.markdown("#### 📋 Capture Checklist")
        checklist_cols = st.columns(4)
        for i, angle in enumerate(st.session_state.angles):
            with checklist_cols[i]:
                is_captured = i < len(st.session_state.multi_captures)
                status_icon = "✅" if is_captured else "⏳"
                status_color = "#4CAF50" if is_captured else "#FFA726"
                st.markdown(f"""
                <div style='text-align: center; padding: 0.5rem; background: {status_color}; 
                            border-radius: 10px; color: white; font-weight: bold;'>
                    <div style='font-size: 2rem;'>{status_icon}</div>
                    <div>{angle}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Check if all angles captured
        if captured_angles < total_angles:
            # Show current angle to capture
            current_angle = st.session_state.angles[st.session_state.current_angle_index]
            
            st.markdown(f"""
            <div style='background: #2196F3; padding: 1rem; border-radius: 10px; 
                        color: white; text-align: center; font-size: 1.2rem; margin: 1rem 0;'>
                🎯 <b>Next:</b> Capture <b>{current_angle}</b>
            </div>
            """, unsafe_allow_html=True)
            
            # Camera controls
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("📷 Activate Camera", use_container_width=True, type="primary", key="multi_activate"):
                    st.session_state.camera_active = True
            
            with col2:
                if st.button("❌ Deactivate Camera", use_container_width=True, key="multi_deactivate"):
                    st.session_state.camera_active = False
            
            if st.session_state.camera_active:
                camera = st.camera_input(f"Capture {current_angle}")
                
                if camera is not None:
                    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, 1)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"#### {current_angle} - Captured")
                        st.image(image, channels="BGR", use_container_width=True)
                    
                    with col2:
                        with st.spinner("🔍 Analyzing damage..."):
                            results = detect_damage(image, confidence_threshold, high_precision_mode, comprehensive_mode)
                        
                        st.markdown(f"#### {current_angle} - Results")
                        st.image(results['result_image'], channels="BGR", use_container_width=True)
                    
                    # Show quick metrics
                    st.markdown("---")
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    
                    with metric_col1:
                        st.metric(f"🔍 Damages ({current_angle})", results['damage_count'])
                    
                    with metric_col2:
                        st.metric("🎯 Detection Confidence", f"{results['avg_confidence']:.1%}")
                    
                    with metric_col3:
                        st.markdown(f"""
                        <div style='text-align: center; padding: 1rem; background: {results['severity_color']}; 
                                    border-radius: 10px; color: white; font-weight: bold;'>
                            <h4 style='margin:0; color: white;'>{results['severity_icon']} {results['severity']}</h4>
                        </div>
                        """, unsafe_allow_html=True)

                    render_confidence_guidance(results)
                    st.caption(f"Only detections at or above {results['confidence_floor']:.0%} are included in this angle analysis.")
                    
                    # Save this capture
                    if st.button("✅ Confirm & Save This Angle", type="primary", use_container_width=True):
                        st.session_state.multi_captures.append({
                            'angle': current_angle,
                            'original_image': image.copy(),
                            'result_image': results['result_image'].copy(),
                            'results': results,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        st.session_state.current_angle_index += 1
                        st.session_state.camera_active = False
                        st.success(f"✅ {current_angle} saved! Moving to next angle...")
                        st.rerun()
        
        # All angles captured - show comprehensive results
        else:
            st.success("🎉 All angles captured! Generating comprehensive report...")
            
            # Aggregate results
            aggregated = aggregate_multi_angle_results(st.session_state.multi_captures)
            
            # Display overall metrics
            st.markdown("### 📊 Comprehensive Inspection Results")
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("🔍 Total Damages", aggregated['damage_count'])
            
            with metric_col2:
                st.markdown(f"""
                <div style='text-align: center; padding: 1rem; background: {aggregated['severity_color']}; 
                            border-radius: 10px; color: white; font-weight: bold;'>
                    <h3 style='margin:0; color: white;'>{aggregated['severity_icon']} {aggregated['severity']}</h3>
                    <p style='margin:0; color: white; font-size: 0.9rem;'>Overall Severity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.metric("🎯 Avg Detection Confidence", f"{aggregated['avg_confidence']:.1%}")
            
            with metric_col4:
                st.metric("📐 Angles Inspected", aggregated['total_angles'])
            
            # Combined damage types
            if aggregated['damage_types']:
                st.markdown("---")
                st.markdown("#### 📋 Combined Damage Type Breakdown")
                damage_cols = st.columns(len(aggregated['damage_types']))
                for i, (damage_type, count) in enumerate(aggregated['damage_types'].items()):
                    with damage_cols[i]:
                        percentage = (count / aggregated['damage_count']) * 100
                        st.metric(damage_type, count, f"{percentage:.1f}%")
            
            # Show all captured angles with thumbnails
            st.markdown("---")
            st.markdown("#### 🖼️ All Captured Angles")
            for capture in st.session_state.multi_captures:
                with st.expander(f"{capture['angle']} - {capture['results']['damage_count']} damages"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(capture['original_image'], channels="BGR", caption="Original", use_container_width=True)
                    with col2:
                        st.image(capture['result_image'], channels="BGR", caption="Detection Results", use_container_width=True)
            
            # Generate comprehensive reports
            st.markdown("---")
            st.markdown("### 📥 Download Comprehensive Reports")
            
            multi_report_text = generate_multi_angle_report(st.session_state.multi_captures, aggregated)
            multi_pdf_buffer = generate_multi_angle_pdf_report(st.session_state.multi_captures, aggregated)
            
            report_col1, report_col2 = st.columns(2)
            with report_col1:
                st.download_button(
                    label="📥 Download Multi-Angle TXT Report",
                    data=multi_report_text,
                    file_name=f"ADVIS_MultiAngle_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with report_col2:
                st.download_button(
                    label="📄 Download Multi-Angle PDF Report",
                    data=multi_pdf_buffer,
                    file_name=f"ADVIS_MultiAngle_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            # Store in session state for dashboard
            st.session_state.latest_analysis = {
                'original_image': st.session_state.multi_captures[0]['original_image'].copy(),
                'result_image': st.session_state.multi_captures[0]['result_image'].copy(),
                'results': aggregated,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': f"multi_angle_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'is_multi_angle': True,
                'multi_captures': st.session_state.multi_captures
            }

# ============== TAB 3: DASHBOARD ==============
with tab3:
    st.markdown("### 📊 System Dashboard & Information")
    
    # Display latest analysis if available
    if st.session_state.latest_analysis is not None:
        st.markdown("---")
        st.markdown("### 🔍 Latest Analysis Results")
        
        analysis = st.session_state.latest_analysis
        results = analysis['results']
        
        # Display timestamp and filename
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"📅 **Analyzed:** {analysis['timestamp']}")
        with col_info2:
            st.info(f"📁 **File:** {analysis['filename']}")
        
        # Display images side by side
        img_col1, img_col2 = st.columns(2)
        
        with img_col1:
            st.markdown("#### 📷 Original Image")
            st.image(analysis['original_image'], channels="BGR", use_container_width=True)
        
        with img_col2:
            st.markdown("#### 🎯 Detection Results")
            st.image(analysis['result_image'], channels="BGR", use_container_width=True)
        
        # Display key metrics
        st.markdown("#### 📊 Quick Stats")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        
        with metric_col1:
            st.metric(
                label="🔍 Total Damages",
                value=results['damage_count']
            )
        
        with metric_col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: {results['severity_color']}; 
                        border-radius: 10px; color: white;'>
                <h3 style='margin:0; color: white;'>{results['severity_icon']} {results['severity']}</h3>
                <p style='margin:0; font-size: 0.8rem;'>Severity</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_col3:
            st.metric(
                label="🎯 Avg Detection Confidence",
                value=f"{results['avg_confidence']:.1%}"
            )

            render_confidence_guidance(results)
            st.caption(f"Only detections at or above {results['confidence_floor']:.0%} are included in the latest analysis.")
        
        with metric_col4:
            st.metric(
                label="📋 Damage Types",
                value=len(results['damage_types'])
            )
        
        # Display damage breakdown with charts
        if results['damage_types']:
            st.markdown("#### 📋 Damage Type Analysis")
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                # Show damage list
                for damage_type, count in results['damage_types'].items():
                    percentage = (count / results['damage_count']) * 100
                    st.markdown(f"""
                    <div style='background: linear-gradient(90deg, #4CAF50 0%, #45a049 {percentage}%, rgba(255,255,255,0.1) {percentage}%);
                                padding: 0.75rem; margin: 0.5rem 0; border-radius: 8px; color: white; font-weight: bold;'>
                        {damage_type}: {count} ({percentage:.1f}%)
                    </div>
                    """, unsafe_allow_html=True)
            
            with chart_col2:
                # Show pie chart
                fig = create_damage_chart(results['damage_types'])
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="dashboard_damage_chart")
        
        # Show confidence scores if available
        if len(results['instances']) > 0:
            st.markdown("#### 📈 Confidence Score Distribution")
            fig_conf = create_confidence_chart(results['instances'])
            if fig_conf:
                st.plotly_chart(fig_conf, use_container_width=True, key="dashboard_confidence_chart")
        
        st.markdown("---")
    else:
        st.info("📸 No analysis data available yet. Upload an image or use the camera to perform an analysis.")
        st.markdown("---")
    
    # System stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>🤖 AI Model</h3>
            <p><strong>Architecture:</strong> Mask R-CNN</p>
            <p><strong>Backbone:</strong> ResNet-50 + FPN</p>
            <p><strong>Framework:</strong> Detectron2</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>🎯 Capabilities</h3>
            <p><strong>Detection Types:</strong> 5 Classes</p>
            <p><strong>Instance Segmentation:</strong> ✅</p>
            <p><strong>Real-time Processing:</strong> ✅</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card'>
            <h3>⚡ Performance</h3>
            <p><strong>Device:</strong> CPU</p>
            <p><strong>Avg Speed:</strong> ~2-3s/image</p>
            <p><strong>Accuracy:</strong> High</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feature showcase
    st.markdown("### ✨ Key Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        <div class='info-box'>
            <h4>🔍 Advanced Detection</h4>
            <ul>
                <li>Multi-class damage recognition</li>
                <li>Instance segmentation masks</li>
                <li>Confidence score analysis</li>
                <li>Duplicate detection removal</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
            <h4>📊 Comprehensive Reporting</h4>
            <ul>
                <li>Automated severity assessment</li>
                <li>Damage type breakdown</li>
                <li>Visual analytics & charts</li>
                <li>Downloadable reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with feature_col2:
        st.markdown("""
        <div class='info-box'>
            <h4>🚀 Easy to Use</h4>
            <ul>
                <li>Simple upload interface</li>
                <li>Live camera support</li>
                <li>Real-time analysis</li>
                <li>No technical expertise needed</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='info-box'>
            <h4>🎨 Industry Grade UI</h4>
            <ul>
                <li>Modern, responsive design</li>
                <li>Interactive visualizations</li>
                <li>Customizable settings</li>
                <li>Professional reporting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # About section
    st.markdown("### ℹ️ About ADVIS")
    st.markdown("""
    <div style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
        <p style='color: #333; font-size: 1.1rem; line-height: 1.8;'>
        <strong>ADVIS (Automated Deep Visual Inspection System)</strong> is an enterprise-grade AI-powered solution 
        for automated vehicle damage detection and classification. Using state-of-the-art deep learning technology, 
        ADVIS can accurately identify and classify various types of vehicle damage including dents, scratches, 
        glass breaks, and smashes.
        </p>
        <p style='color: #333; font-size: 1.1rem; line-height: 1.8;'>
        Perfect for insurance companies, auto repair shops, car rental services, and vehicle inspection centers, 
        ADVIS streamlines the damage assessment process, reduces manual inspection time, and provides consistent, 
        objective damage evaluations.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ============== FOOTER ==============
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 2rem; font-size: 0.9rem;'>
    <p>🚗 <strong>ADVIS</strong> - Automated Deep Visual Inspection System</p>
    <p>Powered by Detectron2 & Streamlit | © 2026 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)