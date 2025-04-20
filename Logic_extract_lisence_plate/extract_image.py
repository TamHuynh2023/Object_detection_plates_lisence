import time
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
from google import genai
from dotenv import load_dotenv
import os
from google.genai import types
import pathlib
import PIL.Image
import requests
import anthropic


from dotenv import load_dotenv
load_dotenv("API Key.env")

class ExtractLicensePlates:
    def __init__(self, image):
        self.original_image = image
        self._init_azure_client()
        self._init_gemini_client()
        self._init_claude_client()
        self.image = self._ensure_image_path(image)

    def _init_azure_client(self):
        endpoint = "https://tamhuynh278.cognitiveservices.azure.com/"
        key = os.getenv("AZURE_OCR")
        self.azure_client = ComputerVisionClient(
            endpoint,
            CognitiveServicesCredentials(key)
        )

    def _init_gemini_client(self):
        from google import genai
        gemini_key = os.getenv("GEMINI_API_KEY")
        self.gemini_client = genai.Client(api_key = gemini_key)

    def _init_claude_client(self):
        claude_key = os.getenv("CLAUDE_API_KEY_TON")
        self.claude_client = anthropic.Anthropic(api_key=claude_key)

    def _ensure_image_path(self, image):
        if isinstance(image, str):
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        output_path = os.path.abspath("new.jpg")
        cv2.imwrite(output_path, enhanced)

        return output_path

    def extract_text_with_plate_recogniation(self):
        try:
            with open(self.image, 'rb') as fp:
                response = requests.post(
                    'https://api.platerecognizer.com/v1/plate-reader/',
                    files=dict(upload=fp),
                    headers={'Authorization': f'Token {os.getenv("PLATES_RECOGNI")}'}
                )

                if response.status_code == 201:
                    result = response.json()
                    for res in result['results']:
                        return res['plate'].upper()
                else:
                    return response.status_code, response.text
        except Exception as e:
            return None


    def extract_text_with_azure(self):
        try:
            with open(self.image , "rb") as image_stream:
                read_response = self.azure_client.read_in_stream(image_stream, raw=True)
            operation_location = read_response.headers["Operation-Location"]
            operation_id = operation_location.split("/")[-1]

            while True:
                result = self.azure_client.get_read_result(operation_id)
                if result.status.lower() not in ["notstarted", "running"]:
                    break
                time.sleep(1)

            if result.status == "succeeded":
                extracted_text = []
                if hasattr(result, 'analyze_result') and hasattr(result.analyze_result, 'read_results'):
                    for page in result.analyze_result.read_results:
                        for line in page.lines:
                            extracted_text.append(line.text)
                    return "\n".join(extracted_text) if extracted_text else None
            return None
        except Exception as e:
            return None

    def extract_text_with_claude(self):
        try:
            system_prompt = """
            You are an OCR engine specialized in reading license plates from images.

            Your task:
            - Extract the license plate only.
            - Return only the license plate as a single line of plain text.
            - Do not include any explanations, apologies, or additional messages.
            - If the license plate cannot be identified, return only: "unknown"
            - Output should never include phrases like "The extracted license plate is..." or "Unable to extract...
            """

            message_list = [
                {
                    "role": 'user',
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": get_base64_encoded_image(self.image)}},
                        {"type": "text", "text": system_prompt}
                    ]
                }
            ]

            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=600,
                messages=message_list
                )
            return response.content[0].text

        except Exception as e:
            return None

    def extract_text_with_gemini(self):
        from google import genai

        try:
            system_prompt = """
            You are an OCR engine specialized in reading license plates from images.

            Your task:
            - Extract the license plate only.
            - Return only the license plate as a single line of plain text.
            - Do not include any explanations, apologies, or additional messages.
            - If the license plate cannot be identified, return only: "unknown"
            - Output should never include phrases like "The extracted license plate is..." or "Unable to extract..."
            """
            b64_image = types.Part.from_bytes(
                data=pathlib.Path(self.image).read_bytes(),
                mime_type="image/jpeg"
            )
            response = self.gemini_client.models.generate_content(
                model="gemini-1.5-pro",
                contents=[system_prompt, b64_image ])
            return response.text
        except Exception as e:
            return None

    def analyze_plates(self, text_list):
        clean_text = []
        for text in text_list:
            if isinstance(text, str):
                clean_text.append(text)
            elif isinstance(text, list):
                clean_text.extend([str(item) for item in text])
            elif isinstance(text, tuple):
                continue
            elif isinstance(i, dict):
                continue
            else:
                clean_text.append(str(text))

        clean_text = "\n".join(clean_text)
        try:
            system_prompt = """
                You are a helpful AI assistant.
                - Extract the **best matching license plate**, even if it is **incomplete**.
                - Prioritize strings that follow **partial or full plate formats**, such as:
                - Motorcycles: XXXX XXXXX (examples: 43E1 16480, 72A2 02501, ....)
                - Cars: XXX XXXXX (examples: 29A 12345, 50C 75820, ....)
                - Please remember to put a space for me

                Instructions:
                - Return **only** the most likely license plate string.
                - Return only the license plate as a single line of plain text.
                - Do not include any explanations, apologies, or additional messages.
                """

            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=100,
                system=system_prompt,
                messages=[{"role": "user", "content": clean_text}]
            )
            return response.content[0].text
        except Exception as e:
            return None


    def run_method_OCR(self):
        extracted_text = self.extract_text_with_plate_recogniation()
        if extracted_text:
            return extracted_text

        extracted_text = self.extract_text_with_claude()
        if extracted_text:
            return extracted_text

        extracted_text = self.extract_text_with_azure()
        if extracted_text:
            return extracted_text

        extracted_text = self.extract_text_with_gemini()
        if extracted_text:
            return extracted_text

        if not extracted_text:
            return "UNKNOWN"
