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

st.markdown(
    """
    <style>

        .stApp {
            background: linear-gradient(
                135deg,
                #07111f 0%,
                #0b1f35 50%,
                #06101c 100%
            );
            color: #f5f7fa;
        }

        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main-title {
            text-align: center;
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 5px;
            color: #ffffff;
        }

        .main-subtitle {
            text-align: center;
            font-size: 19px;
            color: #9fb6cc;
            margin-bottom: 25px;
        }

        .description {
            text-align: center;
            font-size: 16px;
            color: #c7d4e2;
            margin-bottom: 30px;
        }

        .section-title {
            font-size: 22px;
            font-weight: 600;
            color: #ffffff;
            margin-top: 30px;
            margin-bottom: 15px;
        }

        .footer {
            text-align: center;
            color: #71879b;
            font-size: 13px;
            margin-top: 40px;
        }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD SOLARGUARD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "model/SolarGuard_AI_Final.keras"
    )


model = load_model()


# =========================================================
# LOAD EL IMAGE VALIDATOR
# =========================================================

@st.cache_resource
def load_el_validator():

    return tf.keras.models.load_model(
        "model/EL_Image_Validator.keras"
    )


el_validator = load_el_validator()


# =========================================================
# EL IMAGE VALIDATION
# =========================================================

def validate_el_image(original_image):

    """
    Trained binary EL-vs-Non-EL validation model.

    The validator was trained to distinguish:
        1. Solar-cell EL images
        2. Non-EL images

    Output:
        EL probability
        Non-EL probability
        predicted class
    """

    # -----------------------------------------------------
    # Validator preprocessing
    #
    # The validator was trained using RGB images.
    # -----------------------------------------------------

    image = original_image.convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = preprocess_input(
        image_array
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    prediction = el_validator.predict(
        image_array,
        verbose=0
    )

    el_probability = float(
        prediction[0][0]
    )

    non_el_probability = (
        1.0 - el_probability
    )

    # -----------------------------------------------------
    # Decision threshold
    #
    # 0.50 is the prototype decision threshold.
    # -----------------------------------------------------

    if el_probability >= 0.50:

        is_el = True

    else:

        is_el = False

    return {
        "is_el": is_el,
        "el_probability": el_probability,
        "non_el_probability": non_el_probability
    }


# =========================================================
# STANDARD GRAD-CAM
# =========================================================

def generate_gradcam(
    model,
    image_array,
    predicted_class
):

    base_model = model.layers[1]

    last_conv_layer = None

    for layer in reversed(
        base_model.layers
    ):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            last_conv_layer = layer

            break

    if last_conv_layer is None:

        raise ValueError(
            "Could not find a convolutional layer."
        )

    feature_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, base_output = (
            feature_model(
                image_array,
                training=False
            )
        )

        x = base_output

        for layer in model.layers[2:]:

            x = layer(
                x,
                training=False
            )

        predictions = x

        class_output = predictions[
            :,
            predicted_class
        ]

    gradients = tape.gradient(
        class_output,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients could not be calculated."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]

    pooled_gradients = pooled_gradients[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value) > 0:

        heatmap = (
            heatmap / max_value
        )

    return heatmap.numpy()


# =========================================================
# GRAD-CAM++-STYLE EXPLANATION
# =========================================================

def generate_gradcam_plus_plus(
    model,
    image_array,
    predicted_class
):

    """
    Grad-CAM++-style positive-gradient explanation.

    This implementation provides an AI attention
    visualization for the predicted class.

    It should be described as a Grad-CAM++-style
    explanation rather than an exact higher-order
    derivative Grad-CAM++ implementation.
    """

    base_model = model.layers[1]

    last_conv_layer = None

    for layer in reversed(
        base_model.layers
    ):

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):

            last_conv_layer = layer

            break

    if last_conv_layer is None:

        raise ValueError(
            "Could not find a convolutional layer."
        )

    feature_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, base_output = (
            feature_model(
                image_array,
                training=False
            )
        )

        x = base_output

        for layer in model.layers[2:]:

            x = layer(
                x,
                training=False
            )

        predictions = x

        class_score = predictions[
            :,
            predicted_class
        ]

    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:

        raise ValueError(
            "Gradients could not be calculated."
        )

    activations = conv_outputs[0]

    gradients = gradients[0]

    positive_gradients = tf.maximum(
        gradients,
        0
    )

    weights = tf.reduce_mean(
        positive_gradients,
        axis=(0, 1)
    )

    heatmap = tf.reduce_sum(
        activations * weights,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value) > 0:

        heatmap = (
            heatmap / max_value
        )

    return heatmap.numpy()


# =========================================================
# CREATE HEATMAP OVERLAY
# =========================================================

def create_gradcam_overlay(
    original_image,
    heatmap
):

    original = np.array(
        original_image.convert("RGB")
    )

    heatmap = cv2.resize(
        heatmap,
        (
            original.shape[1],
            original.shape[0]
        )
    )

    heatmap = np.uint8(
        255 * heatmap
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )

    return overlay


# =========================================================
# SUSPICIOUS REGION LOCALIZATION
# =========================================================

def localize_suspicious_region(
    original_image,
    heatmap
):

    original = np.array(
        original_image.convert("RGB")
    )

    height, width = original.shape[:2]

    resized_heatmap = cv2.resize(
        heatmap,
        (width, height)
    )

    # -----------------------------------------------------
    # HEATMAP THRESHOLD
    # -----------------------------------------------------

    threshold = 0.60

    binary_map = (
        resized_heatmap >= threshold
    ).astype(np.uint8) * 255

    # -----------------------------------------------------
    # REMOVE SMALL NOISY REGIONS
    # -----------------------------------------------------

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    binary_map = cv2.morphologyEx(
        binary_map,
        cv2.MORPH_OPEN,
        kernel
    )

    binary_map = cv2.morphologyEx(
        binary_map,
        cv2.MORPH_CLOSE,
        kernel
    )

    # -----------------------------------------------------
    # FIND CONNECTED REGIONS
    # -----------------------------------------------------

    contours, _ = cv2.findContours(
        binary_map,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:

        return (
            original,
            None,
            0.0,
            "No strong region detected"
        )

    # -----------------------------------------------------
    # SELECT LARGEST REGION
    # -----------------------------------------------------

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    # -----------------------------------------------------
    # RELATIVE AREA
    # -----------------------------------------------------

    image_area = (
        width * height
    )

    region_area = (
        w * h
    )

    relative_area = (
        region_area / image_area
    ) * 100

    # -----------------------------------------------------
    # APPROXIMATE LOCATION
    # -----------------------------------------------------

    center_x = x + (w / 2)

    center_y = y + (h / 2)

    if center_x < width / 3:

        horizontal = "left"

    elif center_x < (
        2 * width / 3
    ):

        horizontal = "center"

    else:

        horizontal = "right"

    if center_y < height / 3:

        vertical = "upper"

    elif center_y < (
        2 * height / 3
    ):

        vertical = "middle"

    else:

        vertical = "lower"

    if (
        vertical == "middle"
        and horizontal == "center"
    ):

        location = (
            "central region"
        )

    else:

        location = (
            f"{vertical}-{horizontal} region"
        )

    # -----------------------------------------------------
    # DRAW REGION
    # -----------------------------------------------------

    localized_image = (
        original.copy()
    )

    cv2.rectangle(
        localized_image,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        4
    )

    cv2.putText(
        localized_image,
        "Suspicious Region",
        (
            x,
            max(y - 10, 25)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
        cv2.LINE_AA
    )

    return (
        localized_image,
        (x, y, w, h),
        relative_area,
        location
    )


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
    '<div class="main-title">'
    '☀️ SolarGuard AI'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Explainable AI-Assisted Solar EL Inspection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">'
    'Upload an electroluminescence (EL) image of a '
    'solar cell to analyze its defect-probability '
    'assessment.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# IMAGE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload a solar cell EL image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# =========================================================
# MAIN ANALYSIS
# =========================================================

if uploaded_file is not None:

    # =====================================================
    # OPEN IMAGE
    # =====================================================

    original_image = Image.open(
        uploaded_file
    ).convert("L")


    # =====================================================
    # EL IMAGE VALIDATION
    # =====================================================

    validation = validate_el_image(
        original_image
    )


    # =====================================================
    # EL IMAGE VALIDATION RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🛡️ EL Image Validation'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="EL Probability",
            value=f"{validation['el_probability'] * 100:.2f}%"
        )

    with col2:

        st.metric(
            label="Non-EL Probability",
            value=f"{validation['non_el_probability'] * 100:.2f}%"
        )

    with col3:

        if validation["is_el"]:

            validation_status = "EL IMAGE"

        else:

            validation_status = "NON-EL"

        st.metric(
            label="Validation",
            value=validation_status
        )


    # =====================================================
    # NON-EL IMAGE
    # =====================================================

    if not validation["is_el"]:

        st.error(
            "❌ This image was not recognized as a "
            "solar-cell electroluminescence image."
        )

        st.warning(
            "SolarGuard AI has stopped the analysis "
            "to avoid applying the defect classifier "
            "to an unsupported image."
        )

        st.markdown(
            '<div class="section-title">'
            '🖼️ Uploaded Image'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            original_image,
            caption="Rejected Non-EL Image",
            use_container_width=True
        )

        st.caption(
            "The EL validator is a prototype binary "
            "classifier trained using ELPV EL images "
            "and a collection of unrelated Non-EL images. "
            "Its 0.50 decision threshold is a prototype "
            "threshold and is not a calibrated probability "
            "of image authenticity."
        )

        st.stop()


    # =====================================================
    # EL IMAGE ACCEPTED
    # =====================================================

    st.success(
        "✅ Image recognized as a solar-cell EL image. "
        "Proceeding with SolarGuard analysis."
    )


    # =====================================================
    # DISPLAY ORIGINAL IMAGE
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🖼️ Input Image'
        '</div>',
        unsafe_allow_html=True
    )

    st.image(
        original_image,
        caption="Validated EL Image",
        use_container_width=True
    )


    # =====================================================
    # PREPROCESS IMAGE FOR SOLARGUARD
    # =====================================================

    image = original_image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Grayscale channel
    # -----------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    # -----------------------------------------------------
    # Grayscale → RGB
    # -----------------------------------------------------

    image_array = np.repeat(
        image_array,
        3,
        axis=-1
    )

    # -----------------------------------------------------
    # MobileNetV2 preprocessing
    # -----------------------------------------------------

    image_array = preprocess_input(
        image_array
    )

    # -----------------------------------------------------
    # Batch dimension
    # -----------------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # =====================================================
    # SOLARGUARD MODEL PREDICTION
    # =====================================================

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_class = int(
        np.argmax(
            predictions[0]
        )
    )

    confidence = (
        float(
            predictions[0][
                predicted_class
            ]
        ) * 100
    )


    # =====================================================
    # UNCERTAINTY ANALYSIS
    # =====================================================

    sorted_probabilities = np.sort(
        predictions[0]
    )[::-1]

    top_probability = float(
        sorted_probabilities[0]
    )

    second_probability = float(
        sorted_probabilities[1]
    )

    prediction_margin = (
        top_probability
        - second_probability
    ) * 100


    if prediction_margin >= 30:

        uncertainty_level = (
            "Low Uncertainty"
        )

        uncertainty_message = (
            "The model shows a clear separation "
            "between its top prediction and the "
            "next most likely class."
        )

    elif prediction_margin >= 15:

        uncertainty_level = (
            "Moderate Uncertainty"
        )

        uncertainty_message = (
            "The model shows some overlap between "
            "its top predictions. Human review "
            "may be useful."
        )

    else:

        uncertainty_level = (
            "High Uncertainty"
        )

        uncertainty_message = (
            "The top predictions are close to "
            "each other. Manual verification "
            "is recommended."
        )


    # =====================================================
    # INSPECTION PRIORITY
    # =====================================================

    if (
        predicted_class == 0
        and uncertainty_level
        == "Low Uncertainty"
    ):

        inspection_priority = "LOW"

        priority_message = (
            "No strong defect-probability evidence "
            "was identified. Routine inspection "
            "is appropriate."
        )

    elif uncertainty_level == (
        "High Uncertainty"
    ):

        inspection_priority = "REVIEW"

        priority_message = (
            "The prediction is uncertain. Manual "
            "verification is recommended before "
            "making an inspection decision."
        )

    elif predicted_class in [2, 3]:

        inspection_priority = "HIGH"

        priority_message = (
            "The model indicates a moderate or high "
            "defect-probability assessment. Prioritize "
            "this image for further inspection."
        )

    else:

        inspection_priority = "REVIEW"

        priority_message = (
            "The result indicates some "
            "defect-probability evidence. "
            "Further inspection may be useful."
        )


    # =====================================================
    # GRAD-CAM++-STYLE + LOCALIZATION
    # =====================================================

    heatmap = None

    gradcam_image = None

    localized_image = None

    region_box = None

    relative_area = 0.0

    region_location = None


    try:

        heatmap = generate_gradcam_plus_plus(
            model,
            image_array,
            predicted_class
        )

        gradcam_image = (
            create_gradcam_overlay(
                original_image,
                heatmap
            )
        )

        (
            localized_image,
            region_box,
            relative_area,
            region_location
        ) = localize_suspicious_region(
            original_image,
            heatmap
        )

    except Exception as e:

        st.warning(
            "Explainability/localization could not "
            f"be generated: {e}"
        )


    # =====================================================
    # INSPECTION RESULT
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🔍 Inspection Result'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="Prediction",
            value=class_names[
                predicted_class
            ]
        )

    with col2:

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2f}%"
        )

    with col3:

        st.metric(
            label="Prediction Uncertainty",
            value=uncertainty_level
        )


    # =====================================================
    # INTERPRETATION
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '💡 Interpretation'
        '</div>',
        unsafe_allow_html=True
    )


    if predicted_class == 0:

        st.info(
            "The model identifies this image as having "
            "a normal defect-probability assessment."
        )

    elif predicted_class == 1:

        st.warning(
            "The model identifies a low defect-probability "
            "assessment. Further inspection may be useful."
        )

    elif predicted_class == 2:

        st.warning(
            "The model identifies a moderate "
            "defect-probability assessment. Further "
            "inspection is recommended."
        )

    else:

        st.error(
            "The model identifies a high "
            "defect-probability assessment. Further "
            "inspection is recommended."
        )


    # =====================================================
    # PREDICTION CERTAINTY
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '⚠️ Prediction Certainty'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"**Top-two prediction margin:** "
        f"{prediction_margin:.2f} percentage points"
    )

    st.info(
        uncertainty_message
    )

    st.caption(
        "Prediction uncertainty is estimated using "
        "a prototype top-two probability margin. "
        "It is not a calibrated uncertainty estimate."
    )


    # =====================================================
    # CLASS PROBABILITIES
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '📊 Class Probabilities'
        '</div>',
        unsafe_allow_html=True
    )


    for i, class_name in enumerate(
        class_names
    ):

        probability = (
            float(
                predictions[0][i]
            ) * 100
        )

        st.write(
            f"**{class_name}** — "
            f"{probability:.2f}%"
        )

        st.progress(
            float(
                predictions[0][i]
            )
        )


    # =====================================================
    # INSPECTION PRIORITY
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '🎯 Inspection Priority'
        '</div>',
        unsafe_allow_html=True
    )


    if inspection_priority == "HIGH":

        st.error(
            f"🔴 HIGH PRIORITY\n\n"
            f"{priority_message}"
        )

    elif inspection_priority == "REVIEW":

        st.warning(
            f"🟡 REVIEW REQUIRED\n\n"
            f"{priority_message}"
        )

    else:

        st.success(
            f"🟢 LOW PRIORITY\n\n"
            f"{priority_message}"
        )


    st.caption(
        "Inspection priority is a prototype "
        "decision-support indicator based on the "
        "model's prediction and uncertainty. It is "
        "not a direct measurement of physical "
        "failure risk."
    )


    # =====================================================
    # GRAD-CAM++-STYLE EXPLANATION
    # =====================================================

    if gradcam_image is not None:

        st.markdown(
            '<div class="section-title">'
            '🔎 Grad-CAM++-Style Explanation'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "The highlighted regions show areas of "
            "the EL image that contributed more "
            "strongly to the model's prediction."
        )

        st.image(
            gradcam_image,
            caption=(
                "Grad-CAM++-style model attention "
                "visualization"
            ),
            use_container_width=True
        )


    # =====================================================
    # SUSPICIOUS REGION LOCALIZATION
    # =====================================================

    if localized_image is not None:

        st.markdown(
            '<div class="section-title">'
            '🎯 Suspicious Region Localization'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            "The highlighted box identifies the strongest "
            "high-activation region in the explanation "
            "heatmap. This is an AI-assisted localization "
            "indicator, not a confirmed physical defect "
            "boundary."
        )

        st.image(
            localized_image,
            caption=(
                "AI-assisted suspicious-region localization"
            ),
            use_container_width=True
        )

        if region_box is not None:

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    label="Region Detected",
                    value="Yes"
                )

            with col2:

                st.metric(
                    label="Relative Area",
                    value=f"{relative_area:.2f}%"
                )

            with col3:

                st.metric(
                    label="Approx. Location",
                    value=region_location
                )

        else:

            st.info(
                "No strong high-activation region was "
                "detected using the current localization "
                "threshold."
            )


    # =====================================================
    # MODEL PROCESSING
    # =====================================================

    st.markdown(
        '<div class="section-title">'
        '⚙️ Model Processing'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            label="Input Type",
            value="EL Image"
        )

    with col2:

        st.metric(
            label="Model Input",
            value="224 × 224"
        )

    with col3:

        st.metric(
            label="Channels",
            value="Grayscale → RGB"
        )

    st.caption(
        "The uploaded EL image is first validated using "
        "the binary EL-image validator. Accepted EL images "
        "are then converted to grayscale, resized to "
        "224 × 224 pixels, converted to RGB channels, and "
        "preprocessed before being passed to the "
        "SolarGuard MobileNetV2 model."
    )


# =========================================================
# MODEL INFORMATION
# =========================================================

st.markdown(
    '<div class="section-title">'
    '🧠 Model Information'
    '</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    st.markdown(
        """
        **Dataset:** ELPV Dataset

        **Primary Model:** MobileNetV2

        **Input:** Electroluminescence (EL) Image
        """
    )

with col2:

    st.markdown(
        """
        **Image Size:** 224 × 224

        **Output Classes:** 4

        **Task:** Image Classification
        """
    )

st.caption(
    "SolarGuard AI uses a binary EL-image validator "
    "followed by a transfer-learning-based MobileNetV2 "
    "model for four-class defect-probability assessment."
)


# =========================================================
# ABOUT SOLARGUARD AI
# =========================================================

st.markdown(
    '<div class="section-title">'
    '☀️ About SolarGuard AI'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "**SolarGuard AI** is an image-based solar cell "
    "inspection prototype that uses electroluminescence "
    "(EL) images and MobileNetV2 transfer learning. "
    "A binary validation model first checks whether the "
    "uploaded image resembles an EL image. Accepted "
    "images are then analyzed by the four-class SolarGuard "
    "model and supported with AI-assisted explanation "
    "and suspicious-region localization."
)


# =========================================================
# LIMITATIONS
# =========================================================

st.markdown(
    '<div class="section-title">'
    '⚠️ Model Limitations'
    '</div>',
    unsafe_allow_html=True
)

st.warning(
    "This prototype is intended for research and "
    "demonstration purposes. The EL-image validator was "
    "trained using ELPV EL images and a collection of "
    "unrelated Non-EL images, so its real-world "
    "generalization requires further testing. The "
    "validator threshold is a prototype threshold and "
    "is not a calibrated probability. The four-class "
    "SolarGuard model predicts defect-probability "
    "assessment classes rather than physical defect "
    "severity or future failure. Explanation heatmaps "
    "and suspicious-region localization are AI-assisted "
    "indicators, not confirmed physical defect "
    "boundaries. Model predictions should not replace "
    "professional solar-cell inspection."
)


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        SolarGuard AI • Explainable AI-Assisted Solar EL Inspection
    </div>
    """,
    unsafe_allow_html=True
)