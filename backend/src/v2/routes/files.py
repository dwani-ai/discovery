from fastapi import APIRouter, UploadFile, BackgroundTasks, Depends, HTTPException
from database.session import get_db
from database.models import FileRecord, FileStatus
from background.tasks import background_extraction_task
from services.pdf_generation import generate_pdf_from_text, generate_merged_pdf
import uuid

router = APIRouter(prefix="/files", tags=["Files"])

# Include all /files/* endpoints here using the refactored services