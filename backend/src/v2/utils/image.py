from io import BytesIO
from PIL import Image
from pdf2image import convert_from_bytes

def encode_image(image: Image.Image) -> str:
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

async def pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    try:
        return convert_from_bytes(pdf_bytes, fmt="png")
    except Exception as e:
        raise ValueError(f"PDF to image conversion failed: {e}")