Build a Document Graph 

Complete text 

Summary - full doc 

Summary- per page 

Tokenized text per page 

---

Pdf - read text 
Pdf - ocr 

Pdf - ocr - text only - vlm

Pdf - ocr - figure extract - vlm


-
Multi- vllm / LLamacpp/ olmocr - runner

-
Scheduler / Router based on availability 

-

Document to image - code - 
Extract from olmocr - reduce dependency 
and setup time


--

GH200 - architecture understanding 


---

Find context length of models

Split image extraction logic - Fit to 75% size of GPU memory

Make batch calls to extract the text.


Step 1- Extract All Text

Step 2- Use Prompts + Extracted Text

---


Automatic Analysis

- PDF  : TExt + Images + Table + Figures

- A : OCR - Extract TExt Only

- B - Image/ Tables/ Figures
    - Map the multidimensional data to reference in Text

- C - OCR + Multimodal

- D - Merge Data  : Original + Cleaned Data

: 

Find Prompts for

- A

- B

- C

- D