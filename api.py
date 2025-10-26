import cv2
import numpy as np
import json
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PII Redaction Service",
    description="An API to find PII in OCR data and redact it on an image."
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
    Each object contains the word's text and its location.
    """
    all_words = []
    try:
        # Handle the new JSON format where data is wrapped in an array
        if isinstance(ocr_json, list):
            ocr_json = ocr_json[0]
        
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

def extract_face_boxes(ocr_json: dict, face_landmarks_json: dict = None) -> list:
    """
    Extracts face bounding boxes from both OCR response and Face++ API.
    Returns a list of [x1, y1, x2, y2] coordinates.
    """
    face_boxes = []
    
    # Extract faces from OCR data
    try:
        # Handle the new JSON format where data is wrapped in an array
        if isinstance(ocr_json, list):
            ocr_json = ocr_json[0]
        
        faces = ocr_json.get("faces", [])
        for face in faces:
            face_rect = face.get("face_rectangle", {})
            left = face_rect.get("left", 0)
            top = face_rect.get("top", 0)
            width = face_rect.get("width", 0)
            height = face_rect.get("height", 0)
            
            # Convert to [x1, y1, x2, y2] format
            x1 = left
            y1 = top
            x2 = left + width
            y2 = top + height
            
            face_boxes.append([x1, y1, x2, y2])
            print(f"Found face from OCR at: [{x1}, {y1}, {x2}, {y2}]")
    except (KeyError, TypeError) as e:
        print(f"Error extracting face boxes from OCR: {e}")
    
    # Extract faces from Face++ landmarks data
    if face_landmarks_json:
        try:
            # Handle array format
            if isinstance(face_landmarks_json, list):
                face_landmarks_json = face_landmarks_json[0]
            
            faces = face_landmarks_json.get("faces", [])
            for face in faces:
                face_rect = face.get("face_rectangle", {})
                left = face_rect.get("left", 0)
                top = face_rect.get("top", 0)
                width = face_rect.get("width", 0)
                height = face_rect.get("height", 0)
                
                # Convert to [x1, y1, x2, y2] format
                x1 = left
                y1 = top
                x2 = left + width
                y2 = top + height
                
                face_boxes.append([x1, y1, x2, y2])
                print(f"Found face from Face++ at: [{x1}, {y1}, {x2}, {y2}]")
        except (KeyError, TypeError) as e:
            print(f"Error extracting face boxes from Face++: {e}")
    
    return face_boxes

def find_and_merge_pii_boxes(all_words: list, pii_blocks: list) -> list:
    """
    Matches PII text from the LLM to the words from the OCR data
    and calculates a single, merged bounding box for each PII block.
    """
    final_coordinates = []

    for pii_block in pii_blocks:
        pii_text = pii_block.get("text", "")
        # Split the PII text into individual words for matching
        pii_words = pii_text.strip().split()
        if not pii_words:
            continue

        found_words_for_block = []

        # This is a simplified search logic. For higher accuracy, you would
        # re-implement your advanced fuzzy matching and normalization here.
        for i in range(len(all_words) - len(pii_words) + 1):
            match = True
            temp_found = []
            for j in range(len(pii_words)):
                # Simple case-insensitive check
                if pii_words[j].lower() in all_words[i + j]["text"].lower():
                    temp_found.append(all_words[i + j])
                else:
                    match = False
                    break
            
            if match:
                found_words_for_block = temp_found
                break # Found the sequence, move to the next PII block

        # If a match was found, calculate the encompassing bounding box
        if found_words_for_block:
            min_x = min(word["left"] for word in found_words_for_block)
            min_y = min(word["top"] for word in found_words_for_block)
            max_x = max(word["left"] + word["width"] for word in found_words_for_block)
            max_y = max(word["top"] + word["height"] for word in found_words_for_block)
            
            # Add the final [x1, y1, x2, y2] box
            final_coordinates.append([min_x, min_y, max_x, max_y])

    return final_coordinates

# --- The Main API Endpoint ---

@app.post("/process-and-redact")
async def process_and_redact(
    file: UploadFile = File(..., description="The original image file."),
    ocr_data_str: str = Form(..., description="The full JSON output from the OCR service."),
    pii_data_str: str = Form(..., description="The JSON output from the Gemini LLM identifying PII blocks."),
    face_landmarks_str: str = Form(..., description="The JSON output from the Face++ API with face detection and landmarks.")
):
    """
    This endpoint receives an image, its OCR data, PII blocks, and face landmarks.
    It finds the coordinates of the PII (text and faces) and returns a redacted version of the image.
    """
    # --- 1. Read and Prepare Inputs ---
    # Read the image file into an OpenCV object
    image_contents = await file.read()
    nparr = np.frombuffer(image_contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Parse the JSON strings into Python dictionaries
    ocr_data = json.loads(ocr_data_str)
    pii_data = json.loads(pii_data_str)
    face_landmarks_data = json.loads(face_landmarks_str)
    
    # Extract the PII text blocks from Gemini's response
    try:
        # Handle the new JSON format where data is wrapped in an array
        if isinstance(pii_data, list):
            pii_data = pii_data[0]
        
        # Clean the string from markdown code fences and parse the inner JSON
        cleaned_text = pii_data.get("content", {}).get("parts", [{}])[0].get("text", "").replace("```json\n", "").replace("```", "")
        pii_blocks = json.loads(cleaned_text).get("pii_blocks", [])
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Could not parse PII blocks from LLM output: {e}")
        pii_blocks = []

    # --- 2. Perform the "Map & Merge" Logic for Text PII ---
    all_words = flatten_ocr_data(ocr_data)
    text_redaction_boxes = find_and_merge_pii_boxes(all_words, pii_blocks)
    
    # --- 3. Extract Face Bounding Boxes from both OCR and Face++ ---
    face_redaction_boxes = extract_face_boxes(ocr_data, face_landmarks_data)

    # --- 4. Perform the Redaction Logic ---
    # Redact text PII
    for box in text_redaction_boxes:
        x1, y1, x2, y2 = map(int, box)
        # Draw a solid black rectangle over the identified PII
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    
    # Redact faces
    for box in face_redaction_boxes:
        x1, y1, x2, y2 = map(int, box)
        # Draw a solid black rectangle over the detected face
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # --- 5. Return the Final Redacted Image ---
    # Encode the modified image back to a JPG format in memory
    _, encoded_image = cv2.imencode('.jpg', image)
    
    # Return the image as a streaming response
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
