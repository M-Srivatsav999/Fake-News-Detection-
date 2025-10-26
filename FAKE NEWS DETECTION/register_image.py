from PIL import Image
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
import os

resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def register_reference_image(image_path, person_name):
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0)
    embedding = resnet(tensor).detach().numpy()

    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')

    np.save(f"embeddings/{person_name}.npy", embedding)
    print(f"Saved embedding for {person_name}")

# Replace with your image path
register_reference_image("MY_PASSPORT_SIZE_PHOTO.jpg", "MUKKA_SRIVATSAV")
