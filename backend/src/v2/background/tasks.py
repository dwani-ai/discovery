from database.models import FileRecord, FileStatus
from services.extraction import extract_text_from_images_per_page
from services.embedding import store_embeddings
from utils.image import pdf_to_images

async def background_extraction_task(file_id: str, pdf_bytes: bytes, filename: str, db):
    record = db.query(FileRecord).filter(FileRecord.id == file_id).first()
    if not record:
        return

    record.status = FileStatus.PROCESSING
    db.commit()

    try:
        images = await pdf_to_images(pdf_bytes)
        page_texts = await extract_text_from_images_per_page(images)
        full_text = "\n\n".join(page_texts)

        record.extracted_text = full_text
        record.status = FileStatus.COMPLETED
        db.commit()

        await store_embeddings(file_id, filename, page_texts)

    except Exception as e:
        record.status = FileStatus.FAILED
        record.error_message = str(e)
        db.commit()
        raise