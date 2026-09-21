import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SolarGuard AI",
    page_icon="☀️",
    layout="centered"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main background */
    .stApp {
        background: linear-gradient(
            135deg,
            #07111f 0%,
            #0b1f35 50%,
            #06101c 100%
        );
        color: #f5f7fa;
    }

    /* Main content width */
    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main title */
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
        color: #ffffff;
    }

    /* Subtitle */
    .main-subtitle {
        text-align: center;
        font-size: 19px;
        color: #9fb6cc;
        margin-bottom: 25px;
    }

    /* Description */
    .description {
        text-align: center;
        font-size: 16px;
        color: #c7d4e2;
        margin-bottom: 30px;
    }

    /* Section headings */
    .section-title {
        font-size: 22px;
        font-weight: 600;
        color: #ffffff;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    /* Result cards */
    .result-card {
        background: rgba(20, 43, 68, 0.75);
        border: 1px solid rgba(120, 170, 210, 0.25);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        min-height: 120px;
    }

    .result-label {
        color: #9fb6cc;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .result-value {
        color: #ffffff;
        font-size: 20px;
        font-weight: 600;
    }

    /* Processing cards */
    .processing-card {
        background: rgba(20, 43, 68, 0.65);
        border: 1px solid rgba(120, 170, 210, 0.20);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
    }

    .processing-label {
        color: #8fa8bd;
        font-size: 13px;
    }

    .processing-value {
        color: #ffffff;
        font-size: 17px;
        font-weight: 600;
        margin-top: 5px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #71879b;
        font-size: 13px;
        margin-top: 40px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/SolarGuard_AI_Final.keras"
    )


model = load_model()


# =========================================================
# GRAD-CAM
# =========================================================

def generate_gradcam(model, image_array, predicted_class):

    # Your MobileNetV2 base model
    base_model = model.layers[1]

    # Find the last Conv2D layer inside MobileNetV2
    last_conv_layer = None

    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("Could not find a convolutional layer.")

    # Model that gives us:
    # 1. feature maps from the last convolutional layer
    # 2. final MobileNetV2 output
    feature_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, base_output = feature_model(
            image_array,
            training=False
        )

        # Pass MobileNetV2 output through the classifier layers
        x = base_output

        for layer in model.layers[2:]:
            x = layer(x, training=False)

        predictions = x

        class_output = predictions[:, predicted_class]

    # Calculate gradients
    gradients = tape.gradient(
        class_output,
        conv_outputs
    )

    # Average gradients across spatial dimensions
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    # Weight feature maps by their gradients
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # Keep only positive values
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize between 0 and 1
    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap /= max_value

    return heatmap.numpy()


def create_gradcam_overlay(original_image, heatmap):
    """
    Overlay the Grad-CAM heatmap on the original EL image.
    """

    # Convert PIL image to RGB
    original = np.array(
        original_image.convert("RGB")
    )

    # Resize heatmap to original image size
    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    # Convert heatmap to 0-255
    heatmap = np.uint8(
        255 * heatmap
    )

    # Apply color map
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    # Convert BGR → RGB
    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # Blend original image and heatmap
    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    return overlay


# =========================================================
# CLASS NAMES
# =========================================================

class_names = [
    "Normal",
    "Low Defect Confidence",
    "Moderate Defect Confidence",
    "High Defect Confidence"
]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">☀️ SolarGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">Solar Cell Defect Inspection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">'
    'Upload an electroluminescence (EL) image of a solar cell '
    'to analyze its defect-probability class.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload a solar cell EL image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # OPEN IMAGE
    # -----------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("L")


    # -----------------------------------------------------
    # DISPLAY IMAGE
    # -----------------------------------------------------

    st.image(
        image,
        caption="Uploaded EL Image",
        use_container_width=True
    )


    # -----------------------------------------------------
    # PREPROCESSING
    # -----------------------------------------------------

    # Resize image
    image = image.resize(
        (224, 224)
    )

    # Convert image to NumPy array
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Add grayscale channel
    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    # Convert grayscale to RGB
    image_array = np.repeat(
        image_array,
        3,
        axis=-1
    )

    # MobileNetV2 preprocessing
    image_array = preprocess_input(
        image_array
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # -----------------------------------------------------
    # MODEL PREDICTION
    # -----------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_class = int(
        np.argmax(predictions[0])
    )

    confidence = (
        float(predictions[0][predicted_class]) * 100
    )

        # =====================================================
    # GRAD-CAM EXPLANATION
    # =====================================================

    try:
        heatmap = generate_gradcam(
            model,
            image_array,
            predicted_class
        )

        gradcam_image = create_gradcam_overlay(
            image,
            heatmap
        )

    except Exception as e:
        heatmap = None
        gradcam_image = None
        st.warning(
            f"Grad-CAM explanation could not be generated: {e}"
        )


    # =====================================================
    # INSPECTION RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">🔍 Inspection Result</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)


    # Prediction card
    with col1:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Prediction
                </div>
                <div class="result-value">
                    {class_names[predicted_class]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # Confidence card
    with col2:

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">
                    Model Confidence
                </div>
                <div class="result-value">
                    {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.markdown(
        '<div class="section-title">💡 Interpretation</div>',
        unsafe_allow_html=True
    )


    if predicted_class == 0:

        st.info(
            "The model identifies this image as having a "
            "normal defect-probability assessment."
        )

    elif predicted_class == 1:

        st.warning(
            "The model identifies a low defect-probability "
            "assessment. Further inspection may be useful."
        )

    elif predicted_class == 2:

        st.warning(
            "The model identifies a moderate defect-probability "
            "assessment. Further inspection is recommended."
        )

    else:

        st.error(
            "The model identifies a high defect-probability "
            "assessment. Further inspection is recommended."
        )


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    st.markdown(
        '<div class="section-title">📊 Class Probabilities</div>',
        unsafe_allow_html=True
    )


    for i, class_name in enumerate(class_names):

        probability = (
            float(predictions[0][i]) * 100
        )

        st.write(
            f"**{class_name}** — {probability:.2f}%"
        )

        st.progress(
            float(predictions[0][i])
        )
        # =====================================================
    # GRAD-CAM EXPLANATION
    # =====================================================

    if gradcam_image is not None:

        st.markdown(
            '<div class="section-title">🔎 Grad-CAM Explanation</div>',
            unsafe_allow_html=True
        )

        st.write(
            "The highlighted regions show areas of the EL image "
            "that contributed more strongly to the model's prediction."
        )

        st.image(
            gradcam_image,
            caption="Grad-CAM: Model attention visualization",
            use_container_width=True
        )


    # =====================================================
    # MODEL PROCESSING
    # =====================================================

    st.markdown(
        '<div class="section-title">⚙️ Model Processing</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            """
            <div class="processing-card">
                <div class="processing-label">
                    Input Type
                </div>
                <div class="processing-value">
                    EL Image
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col2:

        st.markdown(
            """
            <div class="processing-card">
                <div class="processing-label">
                    Model Input
                </div>
                <div class="processing-value">
                    224 × 224
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    with col3:

        st.markdown(
            """
            <div class="processing-card">
                <div class="processing-label">
                    Channels
                </div>
                <div class="processing-value">
                    Grayscale → RGB
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.caption(
        "The uploaded EL image is converted to grayscale, resized "
        "to 224 × 224 pixels, converted to RGB channels, and "
        "preprocessed before being passed to the MobileNetV2 model."
    )

# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">🧠 Model Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        **Dataset:** ELPV Dataset

        **Model:** MobileNetV2

        **Input:** Electroluminescence (EL) Image
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        **Image Size:** 224 × 224

        **Output Classes:** 4

        **Task:** Image Classification
        """,
        unsafe_allow_html=True
    )

st.caption(
    "SolarGuard AI uses a transfer-learning-based MobileNetV2 "
    "model trained on EL images to classify defect-probability assessments."
)



# =========================================================
# ABOUT SOLARGUARD AI
# =========================================================

st.markdown(
    '<div class="section-title">☀️ About SolarGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="description" style="text-align: left;">
        <b>SolarGuard AI</b> is an image-based solar cell inspection
        prototype that uses electroluminescence (EL) images and a
        MobileNetV2 transfer-learning model to classify images into
        four defect-probability assessment classes.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# MODEL LIMITATION
# =========================================================

st.markdown(
    '<div class="section-title">⚠️ Model Limitation</div>',
    unsafe_allow_html=True
)

st.warning(
    "This prototype is intended for research and demonstration "
    "purposes. Predictions are model outputs and should not be "
    "treated as a replacement for professional solar-cell inspection. "
    "Performance may vary when images come from different sources, "
    "devices, or operating conditions."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        SolarGuard AI • Image-Based Solar Cell Defect Inspection
    </div>
    """,
    unsafe_allow_html=True
)

