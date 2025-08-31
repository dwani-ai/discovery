from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import io

app = FastAPI()

# Model setup (load once at startup)
MID = "apple/FastVLM-0.5B"
IMAGE_TOKEN_INDEX = -200

tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)

@app.post("/describe-image")
async def describe_image(file: UploadFile = File(...)):
    try:
        # Read and process image
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Build chat template
        messages = [
            {"role": "user", "content": "<image>\nDescribe this image in detail."}
        ]
        rendered = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        
        # Split around image token
        pre, post = rendered.split("<image>", 1)
        
        # Tokenize text
        pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
        
        # Insert image token
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
        attention_mask = torch.ones_like(input_ids, device=model.device)
        
        # Process image
        px = model.get_vision_tower().image_processor(
            images=img, 
            return_tensors="pt"
        )["pixel_values"]
        px = px.to(model.device, dtype=model.dtype)
        
        # Generate description
        with torch.no_grad():
            out = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=128,
            )
        
        description = tok.decode(out[0], skip_special_tokens=True)
        
        return JSONResponse({
            "description": description,
            "status": "success"
        })
    
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "status": "error"
        }, status_code=500)

# Run with: uvicorn filename:app --host 0.0.0.0 --port 8000