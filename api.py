import cv2
import numpy as np
import json
import io
import face_recognition # For easy face detection
from pyzbar import pyzbar # For simple QR code detection
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PII Redaction Service",
    description="An API for text and visual PII redaction and merging."
)

# --- Health Check Endpoint for Render ---
@app.get("/")
def health_check():
    """A simple endpoint for Render's health check."""
    return {"status": "ok", "message": "PII Redaction Service is running."}

# --- Core Helper Functions ---

def flatten_ocr_data(ocr_json: dict) -> list:
    """
    Converts the nested OCR.space JSON into a flat list of word objects.
    (Existing Function - No changes needed)
    """
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

def find_and_merge_pii_boxes(all_words: list, pii_blocks: list) -> list:
    """
    Matches PII text from the LLM to the words from the OCR data
    and calculates a single, merged bounding box for each PII block.
    (Existing Function - No changes needed)
    """
    final_coordinates = []

    for pii_block in pii_blocks:
        pii_text = pii_block.get("text", "")
        pii_words = pii_text.strip().split()
        if not pii_words:
            continue

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

# --- VISUAL DETECTION FUNCTIONS ---

def find_qr_boxes(image_cv: np.ndarray) -> list:
    """Detects QR codes using pyzbar and returns bounding boxes [x1, y1, x2, y2]."""
    # Detects QR codes; pyzbar works directly on the BGR image data
    decoded_objects = pyzbar.decode(image_cv)
    qr_boxes = []
    for obj in decoded_objects:
        if obj.type == 'QRCODE':
            # Convert (x, y, w, h) rect to (x1, y1, x2, y2)
            x, y, w, h = obj.rect
            qr_boxes.append([x, y, x + w, y + h])
    return qr_boxes

def find_face_boxes(image_rgb: np.ndarray) -> list:
    """Detects faces using face_recognition and returns bounding boxes [x1, y1, x2, y2]."""
    # Returns a list of (top, right, bottom, left) coordinates
    face_locations = face_recognition.face_locations(image_rgb)
    
    face_boxes = []
    for top, right, bottom, left in face_locations:
        # Convert (top, right, bottom, left) to [x1, y1, x2, y2]
        face_boxes.append([left, top, right, bottom])
    return face_boxes


# --- NEW ENDPOINT FOR VISUAL REDACTION ---

@app.post("/draw-boxes")
async def draw_boxes(
    file: UploadFile = File(..., description="The image with text already redacted."),
    signature_boxes_str: str = Form("[]", description="A JSON string of signature boxes from external YOLO model.")
):
    """
    Performs Face and QR detection internally, merges with external signatures, 
    and draws all visual boxes onto the image.
    """
    # --- 1. Read and Prepare Image ---
    image_contents = await file.read()
    nparr = np.frombuffer(image_contents, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR) # BGR format
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB) # For face_recognition

    # --- 2. Gather ALL Visual Boxes ---
    all_visual_boxes = []

    # A. Internal Face Detection
    all_visual_boxes.extend(find_face_boxes(image_rgb))

    # B. Internal QR Code Detection
    all_visual_boxes.extend(find_qr_boxes(image_cv))

    # C. External Signature Detection (from Hugging Face)
    try:
        if signature_boxes_str != "[]":
            # The boxes are expected to be [x1, y1, x2, y2] format
            all_visual_boxes.extend(json.loads(signature_boxes_str))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse signature boxes: {e}")

    # --- 3. Perform Final Redaction ---
    for box in all_visual_boxes:
        # Ensure box has 4 integer coordinates
        x1, y1, x2, y2 = map(int, box) 
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # --- 4. Return the Final Redacted Image ---
    _, encoded_image = cv2.imencode('.jpg', image_cv)
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")


# --- The ORIGINAL Text Redaction API Endpoint ---

@app.post("/process-and-redact")
async def process_and_redact(
    file: UploadFile = File(..., description="The original image file."),
    ocr_data_str: str = Form(..., description="The full JSON output from the OCR service."),
    pii_data_str: str = Form(..., description="The JSON output from the Gemini LLM identifying PII blocks.")
):
    """
    This endpoint finds the coordinates of the text PII and returns a text-redacted image.
    This output is then passed to the visual redaction step.
    """
    # --- 1. Read and Prepare Inputs ---
    image_contents = await file.read()
    nparr = np.frombuffer(image_contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    ocr_data = json.loads(ocr_data_str)
    pii_data = json.loads(pii_data_str)
    
    try:
        # Clean the string from markdown code fences and parse the inner JSON
        cleaned_text = pii_data.get("content", {}).get("parts", [{}])[0].get("text", "").replace("```json\n", "").replace("```", "")
        pii_blocks = json.loads(cleaned_text).get("pii_blocks", [])
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Could not parse PII blocks from LLM output: {e}")
        pii_blocks = []

    # --- 2. Perform the "Map & Merge" Logic ---
    all_words = flatten_ocr_data(ocr_data)
    redaction_boxes = find_and_merge_pii_boxes(all_words, pii_blocks)

    # --- 3. Perform the TEXT Redaction Logic ---
    for box in redaction_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # --- 4. Return the TEXT-REDACTED Image ---
    _, encoded_image = cv2.imencode('.jpg', image)
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
