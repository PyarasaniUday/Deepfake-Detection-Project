from tensorflow.keras.preprocessing.image import ImageDataGenerator #type:ignore
from cnn_model import build_model

train_dir = "dataset/train"
val_dir = "dataset/val"

datagen = ImageDataGenerator(rescale=1./255)

train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(224,224),
    batch_size=32,
    class_mode='binary'
)

model = build_model()

model.fit(train_data, validation_data=val_data, epochs=10)

model.save("cnn_model.h5")