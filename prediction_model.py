import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageOps
import cv2
import numpy as np

# === Load Model ===
try:
    model = models.resnet50()
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(2048, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.4),
        torch.nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load("D:\\RTP\\Content Moderation\\model\\moderation_model.pth", map_location='cpu'))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: Model file not found - {e}")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# === Class Labels ===
class_names = ['Safe', 'Unsafe']

# === Transform for model (input only) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # For model input only
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# === Predict + Annotate ===
def predict_and_annotate(image_path):
    try:
        # Load image & fix orientation
        pil_img = Image.open(image_path).convert("RGB")
        pil_img = ImageOps.exif_transpose(pil_img)  # Fix orientation if needed

        # Make model input (resized)
        img = transform(pil_img).unsqueeze(0)

        # Predict
        with torch.no_grad():
            output = model(img)
            probs = F.softmax(output, dim=1)  # Probabilities for both classes
            pred = torch.argmax(probs).item()  # Predicted class index
            confidence = probs[0][pred].item()  # Confidence of predicted class
            label = class_names[pred]

        # Print detailed statistics to terminal
        print("\nPrediction Statistics:")
        for i, class_name in enumerate(class_names):
            class_confidence = probs[0][i].item() * 100
            print(f"{class_name}: {class_confidence:.1f}%")

        # Convert to OpenCV for annotation
        orig = np.array(pil_img)
        annotated = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)

        # Draw box only if UNSAFE
        if label == 'Unsafe':
            cv2.rectangle(annotated, (0, 0), (annotated.shape[1]-1, annotated.shape[0]-1), (0, 0, 255), 4)

        # Label text
        text = f"{label} ({confidence*100:.1f}%)"
        text_color = (0, 0, 255) if label == 'Unsafe' else (0, 255, 0)
        # Add a background for better text readability
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.rectangle(annotated, (20, 20), (20 + text_width, 20 + text_height + 10), (255, 255, 255), -1)
        cv2.putText(annotated, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)

        # Show result
        cv2.imshow("Moderation Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print moderation result
        print(f"Moderation Result: {label} with {confidence*100:.1f}% confidence")

    except FileNotFoundError as e:
        print(f"Error: Image not found - {e}")
    except Exception as e:
        print(f"Error processing image: {e}")

# === Try it ===
predict_and_annotate(r"C:\Users\jdpra\Downloads\IMG_20240721_194622082.jpg")