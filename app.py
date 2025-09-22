import gradio as gr
import numpy as np
import cv2
import os
from ultralytics import YOLO

# ===================================================================
#                      CONFIGURATION
# ===================================================================
# The model is now expected to be in the same directory as this script.
PATH_TO_BEST_MODEL = 'best.pt'


# ===================================================================
#  LOAD THE MODEL (This is done once when the script starts)
# ===================================================================

# Check if the model file exists before attempting to load
if not os.path.exists(PATH_TO_BEST_MODEL):
    raise FileNotFoundError(f"FATAL ERROR: Model file not found at '{PATH_TO_BEST_MODEL}'. Make sure 'best.pt' is in the same folder as this script.")

# Load the trained YOLOv8 model
try:
    model = YOLO(PATH_TO_BEST_MODEL)
    print("YOLOv8 model loaded successfully.")
except Exception as e:
    raise IOError(f"Error loading the model. Details: {e}")


# ===================================================================
#  DEFINE THE CORE PREDICTION FUNCTION
# ===================================================================

def predict_and_overlay(input_image):
    """
    Takes an uploaded image, runs YOLOv8 segmentation, and returns
    the image with a semi-transparent red mask overlaid on any detected tumor.
    """
    # 1. Run inference on the input image
    results = model.predict(source=input_image, verbose=False)

    # 2. Check if the model detected any masks
    if results[0].masks is None:
        return input_image, "No tumor was detected by the model."

    # 3. If masks are detected, create an overlay
    overlay_image = input_image.copy()
    mask_color_bgr = [0, 0, 255]  # Red color for the mask
    alpha = 0.5  # Transparency factor

    for mask_tensor in results[0].masks.data:
        # Convert the mask tensor to a NumPy array
        mask_np = mask_tensor.cpu().numpy()

        # Resize the mask to the original image's dimensions
        mask_resized = cv2.resize(mask_np, (input_image.shape[1], input_image.shape[0]))

        # Create a boolean version of the mask
        bool_mask = (mask_resized > 0.5)

        # Apply the colored overlay using cv2.addWeighted for transparency
        overlay_image[bool_mask] = cv2.addWeighted(
            overlay_image[bool_mask], 1 - alpha,
            np.full(overlay_image[bool_mask].shape, mask_color_bgr, dtype=np.uint8), alpha, 0
        )

    return overlay_image, "Tumor detected and segmented successfully."


# ===================================================================
#  CREATE AND LAUNCH THE GRADIO INTERFACE
# ===================================================================

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_and_overlay,
    inputs=gr.Image(type="numpy", label="Upload Brain MRI Scan"),
    outputs=[
        gr.Image(type="numpy", label="Segmented Result"),
        gr.Textbox(label="Detection Status")
    ],
    title="🧠 YOLOv8 Brain Tumor Segmentation",
    description="Upload an MRI scan of a brain to automatically detect and segment the tumor region. The model highlights the predicted tumor area with a red mask.",
    allow_flagging="never"
)

# Launch the application
# When running locally, share=True is not needed.
print("Launching Gradio interface... Open the local URL in your browser.")
demo.launch(debug=True,share=True)