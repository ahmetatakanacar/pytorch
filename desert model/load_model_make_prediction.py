from model_creation import DesertClassifier
import torch
from torchvision import transforms
import setup_data
import torchvision
from pathlib import Path

MODEL_SAVE_PATH = "desert_classifier.pth"

NUM_EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_UNITS = 32
LEARNING_RATE = 0.001

train_dir = "data/desert101/train"
test_dir = "data/desert101/test"

data_transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5483, 0.4638, 0.3865], 
                                 std=[0.2173, 0.2279, 0.2263])
        ]
    )

train_dataloader, test_dataloader, class_names = setup_data.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform=data_transform,
        batch_size=BATCH_SIZE
    )

loaded_model = DesertClassifier(input_size=3, 
                                hidden_size=HIDDEN_UNITS, 
                                output_size=len(class_names))

loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))

data_path = Path("data/")
online_image_path = data_path / "baklava-online.jpg"
single_image = torchvision.io.read_image(str(online_image_path)).type(torch.float32)
single_image = single_image / 255.0
single_image_transform = transforms.Compose([
    transforms.Resize((64, 64))
])
single_image = single_image_transform(single_image)
single_image = single_image.unsqueeze(0)

with torch.inference_mode():
    logits = loaded_model(single_image)
    probs = torch.softmax(logits, dim=1)
    pred_idx = probs.argmax(dim=1).item()

print("Predicted class: ", class_names[pred_idx])