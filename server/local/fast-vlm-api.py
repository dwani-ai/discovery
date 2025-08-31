# main.py

import argparse
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import io
import uvicorn

def create_app(device_choice):
    app = FastAPI()
    MID = "apple/FastVLM-1.5B"
    IMAGE_TOKEN_INDEX = -200

    if device_choice == "cpu":
        device = "cpu"
        dtype = torch.float32
    elif device_choice == "cuda" and torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MID,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model.to("cpu")

    # Store in app state for endpoint usage
    app.state.tok = tok
    app.state.model = model

    @app.post("/describe-image")
    async def describe_image(file: UploadFile = File(...)):
        try:
            image_data = await file.read()
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
            messages = [{"role": "user", "content": "<image>\nDescribe this image in detail."}]
            rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            pre, post = rendered.split("<image>", 1)

            pre_ids = tok(pre, return_tensors="pt", add_special_tokens=False).input_ids
            post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids

            img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(model.device)
            attention_mask = torch.ones_like(input_ids, device=model.device)

            px = model.get_vision_tower().image_processor(images=img, return_tensors="pt")["pixel_values"]
            px = px.to(model.device, dtype=model.dtype)

            with torch.no_grad():
                out = model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=px,
                    max_new_tokens=600,
                )

            description = tok.decode(out, skip_special_tokens=True)
            return JSONResponse({"description": description, "status": "success"})
        except Exception as e:
            return JSONResponse({"error": str(e), "status": "error"}, status_code=500)

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start FastVLM API server")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "auto"], default="auto", help="Device to run model")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for server")
    parser.add_argument("--port", type=int, default=8000, help="Port for server")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload")
    args = parser.parse_args()

    app = create_app(args.device)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
