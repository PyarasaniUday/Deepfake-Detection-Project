from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.cnn_model import build_model
from models.vit_model import build_vit_model

# =========================
# SETTINGS
# =========================
USE_VIT = True   # 🔥 True = ViT, False = CNN
IMG_SIZE = (224,224)
BATCH_SIZE = 32
EPOCHS = 10

train_dir = "dataset/train"
val_dir = "dataset/val"

# =========================
# DATA GENERATOR
# =========================
datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# =========================
# MODEL SELECTION
# =========================
if USE_VIT:
    print("Using ViT Model 🚀")
    model = build_vit_model()
else:
    print("Using CNN Model")
    model = build_model()

# =========================
# TRAIN
# =========================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# =========================
# SAVE MODEL
# =========================
if USE_VIT:
    model.save("models/vit_model.h5")
else:
    model.save("models/cnn_model.h5")

print("✅ Training Complete")