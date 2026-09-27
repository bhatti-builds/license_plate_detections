import streamlit as st
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re

st.set_page_config(page_title="License Plate OCR", page_icon="🚗")

@st.cache_resource
def load_models():
    yolo = YOLO("license_plate_recognitions.pt")
    ocr = easyocr.Reader(["en"], gpu=False)
    return yolo, ocr

model, reader = load_models()

st.title("🚗 License Plate Recognition")

file = st.file_uploader(
    "Upload vehicle image",
    type=["jpg", "jpeg", "png"]
)

if file:

    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if st.button("Detect License Plate"):

        # YOLO
        results = model.predict(
            image,
            conf=0.25,
            verbose=False
        )

        result = results[0]
        output = image.copy()

        found = False

        for box in result.boxes.xyxy:

            found = True

            x1, y1, x2, y2 = map(int, box)

            # Crop plate
            plate = image[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            # OCR
            plate_big = cv2.resize(
                plate,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            ocr_results = reader.readtext(
                plate_big,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )

            # Get OCR candidates
            candidates = []

            for _, text, confidence in ocr_results:

                text = re.sub(
                    r"[^A-Z0-9]",
                    "",
                    text.upper()
                )

                if len(text) >= 4:
                    candidates.append(
                        (text, confidence)
                    )

            if candidates:

                text, confidence = max(
                    candidates,
                    key=lambda x: x[1]
                )

                label = f"{text} ({confidence:.0%})"

            else:

                text = "Not detected"
                label = text

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )

            
            cv2.putText(
                output,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            
            st.image(
                cv2.cvtColor(
                    plate,
                    cv2.COLOR_BGR2RGB
                ),
                caption=f"Detected Plate: {text}"
            )

            st.success(
                f"License Number: {text} | "
                f"OCR Confidence: {confidence:.1%}"
            )

        
        st.subheader("Result")

        st.image(
            cv2.cvtColor(
                output,
                cv2.COLOR_BGR2RGB
            ),
            use_container_width=True
        )

        if not found:
            st.warning("No license plate detected.")

