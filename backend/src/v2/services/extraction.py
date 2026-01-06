from client.openai_client import get_openai_client
from utils.image import encode_image, pdf_to_images

async def extract_text_from_images_per_page(images: list[Image.Image]) -> list[str]:
    client = get_openai_client()
    page_texts = []

    for img in images:
        base64_img = encode_image(img)
        messages = [
            {"role": "system", "content": "You are an expert OCR assistant. Extract accurate plain text only."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
                    {"type": "text", "text": "Extract clean, accurate plain text from this page. Preserve structure."},
                ],
            },
        ]

        response = await client.chat.completions.create(
            model="gemma3",
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
        )
        page_texts.append(response.choices[0].message.content.strip())

    return page_texts