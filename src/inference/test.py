from src.inference.predict import load_model, preprocess_image

model = load_model("models/multitask_resnet18_best.pth")
print("Model loaded successfully")

x = preprocess_image("path/to/one/banana_image.jpg")
print(x.shape)  # should be [1, 3, 224, 224]