# import dlib
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
import timm
from torchvision import transforms
import face_recognition
from lime import lime_image
from skimage.color import label2rgb
from skimage.segmentation import mark_boundaries

# -----------------------------
# CONFIG
# -----------------------------
WEIGHTS_PATH = "ffpp_c23.pth"
IMG_SIZE = 299
NUM_SAMPLES = 600
NUM_FEATURES = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# MODEL INITIALIZATION
# -----------------------------


@st.cache_resource
def load_model(weights_path=WEIGHTS_PATH, device=DEVICE):
    model = timm.create_model("xception", pretrained=True)
    num_ftrs = model.get_classifier().in_features
    model.fc = nn.Linear(num_ftrs, 2)
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval()


model = load_model()

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


def batch_predict(images):
    batch = []
    for img_np in images:
        img_pil = Image.fromarray((img_np*255).astype(np.uint8))
        batch.append(preprocess(img_pil))
    batch_tensor = torch.stack(batch).to(DEVICE)
    with torch.no_grad():
        logits = model(batch_tensor)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()

# -----------------------------
# FACE CROP & UTILS
# -----------------------------


def crop_ffpp_style(img_bgr, expand_ratio=0.3, target_size=IMG_SIZE):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(img_rgb)
    if not faces:
        return cv2.resize(img_rgb, (target_size, target_size))
    areas = [(b-t)*(r-l) for (t, r, b, l) in faces]
    idx = areas.index(max(areas))
    t, r, b, l = faces[idx]
    h_exp, w_exp = int(expand_ratio*(b-t)), int(expand_ratio*(r-l))
    t, b = max(0, t-h_exp), min(img_bgr.shape[0], b+h_exp)
    l, r = max(0, l-w_exp), min(img_bgr.shape[1], r+w_exp)
    cx, cy = l + (r-l)//2, t + (b-t)//2
    half = max(b-t, r-l)//2
    t_sq, b_sq = max(0, cy-half), min(img_bgr.shape[0], cy+half)
    l_sq, r_sq = max(0, cx-half), min(img_bgr.shape[1], cx+half)
    crop = img_bgr[t_sq:b_sq, l_sq:r_sq]
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (target_size, target_size))


def draw_bbox(img_bgr):
    img_out = img_bgr.copy()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(img_rgb)
    for (t, r, b, l) in faces:
        cv2.rectangle(img_out, (l, t), (r, b), (0, 255, 0), 2)
    return cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)


def get_lime_mask(explanation, top_label, num_features=NUM_FEATURES, positive_only=False):
    exp = list(explanation.local_exp[top_label])  # ensure sliceable
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=positive_only,
        negative_only=False,
        num_features=min(NUM_FEATURES, len(exp)),
        hide_rest=False
    )
    return temp, mask


def visualize_face_lime(explanation, face_rgb_uint8, alpha=0.5):
    top_label = explanation.top_labels[0]
    _, mask = get_lime_mask(explanation, top_label,
                            NUM_FEATURES, positive_only=False)

    mask_resized = cv2.resize(mask.astype(np.uint8),
                              (face_rgb_uint8.shape[1],
                               face_rgb_uint8.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    overlay = label2rgb(mask_resized, image=face_rgb_uint8/255.0,
                        colors=[(1, 0, 0), (0, 0, 1)],
                        alpha=alpha, bg_label=0)
    overlay = (overlay*255).astype(np.uint8)
    boundaries = mark_boundaries(face_rgb_uint8/255.0, mask_resized)
    boundaries = (boundaries*255).astype(np.uint8)
    return cv2.addWeighted(overlay, 0.7, boundaries, 0.3, 0)


# dlib landmarks
# LANDMARK_PATH = "shape_predictor_68_face_landmarks.dat"
# predictor = dlib.shape_predictor(LANDMARK_PATH)
# detector = dlib.get_frontal_face_detector()


def add_landmarks(img_rgb):
    img_out = img_rgb.copy()
    # faces = detector(img_rgb, 1)
    faces = face_recognition.face_landmarks(img_rgb)
    for face in faces:
        # shape = predictor(img_rgb, face)
        # for i in range(68):
        #     x, y = shape.part(i).x, shape.part(i).y
        #     cv2.circle(img_out, (x, y), 2, (0, 0, 255), -1)
        for key, points in face.items():
            for (x, y) in points:
                cv2.circle(img_out, (x, y), 2, (0, 0, 255), -1)
    return img_out


# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("Fake Detector with LIME")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Original with bounding box
    img_bbox = draw_bbox(img_bgr)

    # Cropped face
    img_face = crop_ffpp_style(img_bgr)

    # Prediction
    pred_probs = batch_predict(np.expand_dims(img_face, 0))
    fake_prob = pred_probs[0, 1]
    class_label = "FAKE" if fake_prob > 0.5 else "REAL"
    st.markdown(f"**Prediction:** {class_label} ({fake_prob*100:.2f}%)")

    with st.spinner(show_time=True):
        # st.success("✅ LIME explanation completed!")

        # LIME explanation
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_face, classifier_fn=batch_predict,
            top_labels=1, hide_color=0.5, num_samples=NUM_SAMPLES
        )

        # Cropped + LIME + landmarks
        img_overlay = visualize_face_lime(explanation, img_face)
        img_with_landmarks = add_landmarks(img_overlay)

        # Show results
        col1, col2 = st.columns(2)
        col1.image(img_bbox, caption="Original with Bounding Box")
        col2.image(img_with_landmarks,
                   caption="Cropped Face with LIME + Landmarks")
