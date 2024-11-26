import gradio as gr
import torch
from diffusers import FluxInpaintPipeline
from PIL import Image, ImageFile

# Uncomment if you encounter issues with truncated images
# ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize the pipeline
pipe = FluxInpaintPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16
)
pipe.to("cuda")
pipe.load_lora_weights(
    "ali-vilab/In-Context-LoRA", 
    weight_name="visual-identity-design.safetensors"
)

def square_center_crop(img, target_size=768):
    if img.mode in ('RGBA', 'P'):
        img = img.convert('RGB')

    width, height = img.size
    crop_size = min(width, height)

    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    img_cropped = img.crop((left, top, right, bottom))
    return img_cropped.resize((target_size, target_size), Image.Resampling.LANCZOS)

def duplicate_horizontally(img):
    width, height = img.size
    if width != height:
        raise ValueError(f"Input image must be square, got {width}x{height}")

    new_image = Image.new('RGB', (width * 2, height))
    new_image.paste(img, (0, 0))
    new_image.paste(img, (width, 0))
    return new_image

def generate(image, prompt_description, prompt_user, progress=gr.Progress(track_tqdm=True)):
    prompt_structure = "The two-panel image showcases the logo on the left and the application on the right, [LEFT] the left panel is showing " + prompt_description + " [RIGHT] this logo is applied to "
    prompt = prompt_structure + prompt_user
    
    mask = Image.open("mask_square.png")
    cropped_image = square_center_crop(image)
    logo_dupli = duplicate_horizontally(cropped_image)

    out = pipe(
        prompt=prompt,
        image=logo_dupli,
        mask_image=mask,
        guidance_scale=3.5,
        height=768,
        width=1536,
        num_inference_steps=28,
        max_sequence_length=256,
        strength=1
    ).images[0]

    width, height = out.size
    half_width = width // 2
    image_2 = out.crop((half_width, 0, width, height))
    return image_2, out

with gr.Blocks() as demo:
    gr.Markdown("# Logo in Context")
    gr.Markdown("### Apply your logo to various contexts using AI-powered image generation.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Upload Logo Image",
                type="pil",
                height=384
            )
            prompt_description = gr.Textbox(
                label="Describe your logo",
                placeholder="A Hugging Face emoji logo",
            )
            prompt_input = gr.Textbox(
                label="Where should the logo be applied?",
                placeholder="e.g., a coffee cup on a wooden table"
            )
            generate_btn = gr.Button("Generate Application", variant="primary")
    
        with gr.Column():
            output_image = gr.Image(label="Generated Application")
            output_side = gr.Image(label="Side by side")
        
    generate_btn.click(
        fn=generate,
        inputs=[input_image, prompt_description, prompt_input],
        outputs=[output_image, output_side]
    )

demo.launch(server_name="0.0.0.0", server_port=8000)
