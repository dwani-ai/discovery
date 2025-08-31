from fastapi import FastAPI, File, UploadFile, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io
import torch
from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI(title="Image Analysis API")

# Check for CUDA availability and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the model
try:
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map={"": device},
    )
except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}")

@app.post("/caption/")
async def image_caption(file: UploadFile = File(...), length: str = "short"):
    """
    Generate a caption for the uploaded image.
    Args:
        file: Uploaded image file
        length: Caption length ("short" or "normal")
    Returns:
        JSON response with caption
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        encoded_image = model.encode_image(image)

        # Generate caption
        if length not in ["short", "normal"]:
            raise HTTPException(status_code=400, detail="Invalid length parameter. Use 'short' or 'normal'.")

        if length == "short":
            caption = model.caption(encoded_image, length="short")["caption"]
        else:
            caption = ""
            for t in model.caption(encoded_image, length="normal", stream=True)["caption"]:
                caption += t

        return JSONResponse(content={"caption": caption})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing caption: {str(e)}")

@app.post("/query/")
async def visual_question_answering(file: UploadFile = File(...), question: str = ""):
    """
    Answer a question about the uploaded image.
    Args:
        file: Uploaded image file
        question: Question to ask about the image
    Returns:
        JSON response with answer
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        if not question:
            raise HTTPException(status_code=400, detail="Question parameter is required.")

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        encoded_image = model.encode_image(image)

        answer = model.query(encoded_image, question)["answer"]
        return JSONResponse(content={"answer": answer})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/detect/")
async def object_detection(file: UploadFile = File(...), object_type: str = "face"):
    """
    Detect objects in the uploaded image.
    Args:
        file: Uploaded image file
        object_type: Type of object to detect (default: "face")
    Returns:
        JSON response with number of detected objects
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        encoded_image = model.encode_image(image)

        objects = model.detect(encoded_image, object_type)["objects"]
        return JSONResponse(content={"detected_objects": len(objects), "object_type": object_type})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting objects: {str(e)}")

@app.post("/point/")
async def visual_pointing(file: UploadFile = File(...), object_type: str = "person"):
    """
    Locate objects in the uploaded image.
    Args:
        file: Uploaded image file
        object_type: Type of object to locate (default: "person")
    Returns:
        JSON response with number of located objects
    """
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        encoded_image = model.encode_image(image)

        points = model.point(encoded_image, object_type)["points"]
        return JSONResponse(content={"located_objects": len(points), "object_type": object_type})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error locating objects: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)