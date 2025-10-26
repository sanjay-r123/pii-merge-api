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
    pii_data_str: str = Form(..., description="The JSON output from the Gemini LLM identifying PII blocks.")
):
    """
    This endpoint receives an image, its OCR data, and a list of PII.
    It finds the coordinates of the PII and returns a redacted version of the image.
    """
    # --- 1. Read and Prepare Inputs ---
    # Read the image file into an OpenCV object
    image_contents = await file.read()
    nparr = np.frombuffer(image_contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Parse the JSON strings into Python dictionaries
    ocr_data = json.loads(ocr_data_str)
    pii_data = json.loads(pii_data_str)
    
    # Extract the PII text blocks from Gemini's response
    # The response is nested, so we navigate to the text part and parse it
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

    # --- 3. Perform the "Redaction" Logic ---
    for box in redaction_boxes:
        x1, y1, x2, y2 = map(int, box)
        # Draw a solid black rectangle over the identified PII
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # --- 4. Return the Final Redacted Image ---
    # Encode the modified image back to a JPG format in memory
    _, encoded_image = cv2.imencode('.jpg', image)
    
    # Return the image as a streaming response
    return StreamingResponse(io.BytesIO(encoded_image.tobytes()), media_type="image/jpeg")
