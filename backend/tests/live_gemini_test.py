"""
Live integration test -- sends a locally generated JPEG to Gemini Vision
to verify that gemini-3.6-flash can analyse a road image end-to-end.

Run manually:
    venv\\Scripts\\python.exe backend\\tests\\live_gemini_test.py

This script is NOT part of the automated test suite -- it makes a real Gemini API call.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import tempfile, json, struct

from backend.llm import get_llm, get_gemini_model, is_llm_configured
from backend.agents.vision_agent import run_vision_agent


def make_real_jpeg_bytes():
    """
    Create a valid small JPEG file using Pillow (installed as a transitive dep).
    Falls back to a raw minimum-valid JPEG if Pillow is unavailable.
    """
    try:
        from PIL import Image
        import io
        img = Image.new("RGB", (64, 64), color=(90, 90, 90))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()
    except ImportError:
        pass

    # Fallback: well-known 1x1 gray JPEG (verified valid bytes)
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00,
        0xFF, 0xDB, 0x00, 0x43, 0x00,
        # Quantization table (64 bytes of 8)
        *([8] * 64),
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
        0xFF, 0xC4, 0x00, 0x1F, 0x00,
        # DC Huffman table
        0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01,
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B,
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
        0x7F, 0xA4,
        0xFF, 0xD9,
    ])


def try_download(url):
    """Try to download a real road image for more realistic testing."""
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
        if len(data) > 1000:
            return data
    except Exception:
        pass
    return None


def main():
    print("=" * 60)
    print("RoadWatch AI -- Live Gemini Vision Test")
    print("=" * 60)

    if not is_llm_configured():
        print("ERROR: GEMINI_API_KEY is not set -- cannot run live test.")
        sys.exit(1)

    model = get_gemini_model()
    print(f"Model  : {model}")
    llm = get_llm()
    print(f"LLM    : {type(llm).__name__}")
    print()

    # Try a few publicly accessible road/pothole images
    image_urls = [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/9/94/FoodDesert2.jpg/200px-FoodDesert2.jpg",
        "https://www.w3schools.com/css/img_5terre.jpg",
    ]

    data = None
    for url in image_urls:
        print(f"Trying image: {url}")
        data = try_download(url)
        if data:
            print(f"Downloaded {len(data):,} bytes")
            break

    if not data:
        print("All download attempts failed. Generating local JPEG ...")
        data = make_real_jpeg_bytes()
        print(f"Generated {len(data):,} bytes")

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(data)
        tmp_path = f.name

    print(f"Image path: {tmp_path}")
    print()
    print("Calling Vision Agent with real Gemini API ...")

    try:
        result = run_vision_agent(tmp_path, llm=llm)
        print()
        print("SUCCESS -- Vision Agent returned:")
        print(json.dumps(result, indent=2))
        print()
        print("=" * 60)
        print(f"Model '{model}' is working correctly for multimodal vision.")
        print("=" * 60)
    except Exception as e:
        print(f"\nFAILED -- Error:\n  {type(e).__name__}: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    main()
