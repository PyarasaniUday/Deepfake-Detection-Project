from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

test_dir = "dataset/test"

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

# 🔥 Load ViT (main model)
model = load_model("models/vit_model.h5")

loss, acc = model.evaluate(test_data)
print(f"\nAccuracy: {acc*100:.2f}%")

# Predictions
preds = model.predict(test_data)
y_pred = (preds > 0.5).astype(int)

print("\nConfusion Matrix:")
print(confusion_matrix(test_data.classes, y_pred))

print("\nClassification Report:")
print(classification_report(test_data.classes, y_pred))