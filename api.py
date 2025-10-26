import cv2
import numpy as np
import json
import io
import face_recognition # Added for Face Detection (Requires face-recognition in requirements.txt)
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PII Redaction Service (Consolidated)",
    description="Processes OCR/LLM data and redacts Text and Face in a single step."
)

@app.get("/")
def health_check():
    """A simple endpoint for Render's health check."""
    return {"status": "ok", "message": "PII Redaction Service is running."}

# --- Core Utility Functions (Optimized for Redaction) ---

def flatten_ocr_data(ocr_json: dict) -> list:
    """Flattens the OCR JSON into a simple list of word objects. (Existing Function)"""
    all_words = []
    try:
        lines = ocr_json.get("ParsedResults", [{}])[0].get("TextOverlay", {}).get("Lines", [])
        for line in lines:
            for word in line.get("Words", []):
                all_words.append({
                    "text": word.get("WordText", ""),
                    "left": word.get("Left", 0),
                    "top": word.get("Top", 0),
                    "width": word.get("Width", 0),
                    "height": word.get("Height", 0),
                })
    except (IndexError, KeyError) as e:
        print(f"Error parsing OCR data: {e}")
    return all_words

def find_text_redaction_boxes(all_words: list, pii_blocks: list) -> list:
    """Calculates merged bounding boxes for all LLM-identified text PII. (Existing Function)"""
    final_coordinates = []
    for pii_block in pii_blocks:
        pii_text = pii_block.get("text", "")
        pii_words = pii_text.strip().split()
        if not pii_words: continue

        found_words_for_block = []
        for i in range(len(all_words) - len(pii_words) + 1):
            match = True
            temp_found = []
            for j in range(len(pii_words)):
                if pii_words[j].lower() in all_words[i + j]["text"].lower():
                    temp_found.append(all_words[i + j])
                else:
                    match = False
                    break
            
            if match:
                found_words_for_block = temp_found
                break 

        if found_words_for_block:
            min_x = min(word["left"] for word in found_words_for_block)
            min_y = min(word["top"] for word in found_words_for_block)
            max_x = max(word["left"] + word["width"] for word in found_words_for_block)
            max_y = max(word["top"] + word["height"] for word in found_words_for_block)
            final_coordinates.append([min_x, min_y, max_x, max_y])
    return final_coordinates

# --- The CONSOLIDATED REDACTION ENDPOINT (New) ---

@app.post("/process-and-redact-all")
async def process_and_redact_all(
    file: UploadFile = File(..., description="The original image file."),
    ocr_data_str: str = Form(..., description="The full JSON output from the OCR service."),
    pii_data_str: str = Form(..., description="The JSON output from the Gemini LLM identifying PII blocks."),
    face_data_str: str = Form("[]", description="JSON string of face detection results (Face++).")
):
    """
    Consolidates Text Redaction (from LLM/OCR) and Face Redaction in a single API call.
    """
    # --- 1. Read and Prepare Inputs ---
    image_contents = await file.read()
    nparr = np.frombuffer(image_contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # BGR format for OpenCV

    # Parse inputs
    ocr_data = json.loads(ocr_data_str)
    pii_data = json.loads(pii_data_str)
    face_data = json.loads(face_data_str)
    
    # Prepare image for visual detection if needed (though Face++ already ran it)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Extract clean pii_blocks list from LLM output
    try:
        cleaned_text = pii_data.get("content", {}).get("parts", [{}])[0].get("text", "").replace("```json\n", "").replace("```", "")
        pii_blocks = json.loads(cleaned_text).get("pii_blocks", [])
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Could not parse PII blocks from LLM output: {e}")
        pii_blocks = []

    # --- 2. TEXT REDACTION: Map LLM PII to OCR Coordinates ---
    all_words = flatten_ocr_data(ocr_data)
    text_redaction_boxes = find_text_redaction_boxes(all_words, pii_blocks)

    # --- 3. FACE REDACTION: Collect Face++ Boxes ---
    face_redaction_boxes = []
    
    # Process Face++ data (which is in the format: faces[0].face_rectangle.{top, left, width, height})
    if face_data.get("faces"):
        for face in face_data["faces"]:
            rect = face["face_rectangle"]
            # Convert Face++ format to [x1, y1, x2, y2]
            x1 = rect["left"]
            y1 = rect["top"]
            x2 = rect["left"] + rect["width"]
            y2 = rect["top"] + rect["height"]
            face_redaction_boxes.append([x1, y1, x2, y2])

    # --- 4. DRAWING: Redact all boxes (Text and Face must NOT merge) ---
    
    # Draw TEXT boxes
    for box in text_redaction_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # Draw FACE boxes (This is a completely separate set of boxes)
    for box in face_redaction_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    
    # NOTE: QR detection can be added here if needed, but since you excluded pyzbar,
    # we leave it out.

    # --- 5. Return the Final Redacted Image ---
    _, encoded_image = cv2.imencode('.jpg', image)
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
