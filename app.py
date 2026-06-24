import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
import io

class ImageProcessor:
    """Handles image processing with crop preview functionality."""
    
    def __init__(self):
        self.original_image = None
        self.bounding_box = None
        self.cropped_image = None
        self.crop_stats = None
        # New variables to track resized crop and bbox coordinates
        self.resized_cropped = None
        self.bbox_coords = None
    
    def detect_signature_bbox(self, image):
        """Detect bounding box for signature (simplified example)."""
        # In real implementation, this would use ML model
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Simple example: detect non-white regions
        # Returns: (x_min, y_min, x_max, y_max)
        margin = 20
        x_min, y_min = margin, margin
        x_max, y_max = width - margin, height - margin
        
        return (x_min, y_min, x_max, y_max)
    
    def draw_bounding_box(self, image):
        """Draw bounding box on original image."""
        img_copy = image.copy()
        draw = ImageDraw.Draw(img_copy)
        x_min, y_min, x_max, y_max = self.bounding_box
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)
        return img_copy
    
    def crop_image(self, image):
        """Crop image using detected bounding box."""
        x_min, y_min, x_max, y_max = self.bounding_box
        cropped = image.crop((x_min, y_min, x_max, y_max))
        # store cropped
        self.cropped_image = cropped
        return cropped
    
    def calculate_crop_stats(self, original, cropped):
        """Calculate statistics about the crop."""
        orig_array = np.array(original)
        crop_array = np.array(cropped)
        
        stats = {
            "Original Size": f"{original.width}x{original.height}",
            "Crop Size": f"{cropped.width}x{cropped.height}",
            "Area Removed": f"{100 * (1 - (cropped.width * cropped.height) / (original.width * original.height)):.1f}%",
            "Crop Aspect Ratio": f"{cropped.width / cropped.height:.2f}",
        }
        return stats

def process_image(image):
    """Main processing function - processes on button click."""
    if image is None:
        # match the number of outputs below (6 outputs besides the input)
        return None, None, None, None, None, None
    
    processor = ImageProcessor()
    processor.original_image = image
    
    # Detect bounding box
    processor.bounding_box = processor.detect_signature_bbox(image)
    # Store bbox coords as a readable string (new variable)
    x_min, y_min, x_max, y_max = processor.bounding_box
    processor.bbox_coords = f"x_min: {x_min}, y_min: {y_min}, x_max: {x_max}, y_max: {y_max}"
    
    # Generate preview images
    preview_with_bbox = processor.draw_bounding_box(image)
    cropped = processor.crop_image(image)
    
    # Create a resized version of the cropped image for a preview (new variable)
    try:
        resized_cropped = cropped.resize((256, 256), Image.LANCZOS)
    except Exception:
        resized_cropped = cropped.copy()
    processor.resized_cropped = resized_cropped
    
    # Calculate statistics
    stats = processor.calculate_crop_stats(image, cropped)
    stats_text = "\n".join([f"{k}: {v}" for k, v in stats.items()])
    
    # New outputs to return: bbox coordinates (text) and resized cropped preview (image)
    return preview_with_bbox, cropped, stats_text, image, processor.bbox_coords, processor.resized_cropped

def reset_app():
    """Reset all outputs."""
    # Return Nones for: input_image, preview_with_bbox, cropped_preview, crop_stats_display, final_output, bbox_coords_display, resized_cropped_preview
    return None, None, None, None, None, None, None

# Build Gradio Interface
with gr.Blocks(title="Signature Vectorizer - Crop Preview") as demo:
    gr.Markdown("# Signature Vectorizer with Crop Preview")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input")
            input_image = gr.Image(
                label="Upload Image",
                type="pil",
                sources=["upload", "webcam"]
            )
            
            with gr.Row():
                process_btn = gr.Button("Process & Preview Crop", variant="primary")
                reset_btn = gr.Button("Reset", variant="secondary")
        
        with gr.Column():
            gr.Markdown("### Stage 1: Original with Bounding Box")
            preview_with_bbox = gr.Image(
                label="Original Image + Bounding Box",
                type="pil",
                interactive=False
            )
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Stage 2: Cropped Preview (Before Resize)")
            cropped_preview = gr.Image(
                label="Cropped Image",
                type="pil",
                interactive=False
            )
            # New Gradio component: resized cropped preview
            resized_cropped_preview = gr.Image(
                label="Cropped & Resized Preview",
                type="pil",
                interactive=False
            )
        
        with gr.Column():
            gr.Markdown("### Crop Statistics")
            crop_stats_display = gr.Textbox(
                label="Statistics",
                interactive=False,
                lines=6
            )
            # New Gradio component: bounding box coordinates
            bbox_coords_display = gr.Textbox(
                label="Bounding Box Coordinates",
                interactive=False,
                lines=2
            )
    
    with gr.Row():
        gr.Markdown("### Stage 3: Processing Output")
        final_output = gr.Image(
            label="Final Output (for vectorization)",
            type="pil",
            interactive=False
        )
    
    # Wire up button clicks
    process_btn.click(
        fn=process_image,
        inputs=[input_image],
        outputs=[
            preview_with_bbox,
            cropped_preview,
            crop_stats_display,
            final_output,
            bbox_coords_display,
            resized_cropped_preview
        ]
    )
    
    reset_btn.click(
        fn=reset_app,
        outputs=[
            input_image,
            preview_with_bbox,
            cropped_preview,
            crop_stats_display,
            final_output,
            bbox_coords_display,
            resized_cropped_preview
        ]
    )

if __name__ == "__main__":
    demo.launch(share=True)
