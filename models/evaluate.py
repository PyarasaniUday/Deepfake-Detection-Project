from tensorflow.keras.models import load_model #type:ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore

test_dir = "dataset/test"

datagen = ImageDataGenerator(rescale=1./255)

test_data = datagen.flow_from_directory(
    test_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

model = load_model("cnn_model.h5")

loss, acc = model.evaluate(test_data)

print(f"Accuracy: {acc*100:.2f}%")