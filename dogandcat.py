from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


data_path = Path("data/")
image_path = data_path / "pets"

def split_train_test(source_dir: Path, target_dir: Path, train_ratio: float = 0.8, seed: int = 42):
      if target_dir.exists():
            print(f"{target_dir} already exists, skipping split.")
            return

      class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]

      for class_dir in class_dirs:
            images = list(class_dir.glob("*.jpg"))

            train_images, test_images = train_test_split(
                  images, train_size=train_ratio, random_state=seed, shuffle=True
            )
            splits = {"train": train_images, "test": test_images}

            for split_name, split_images in splits.items():
                  split_class_dir = target_dir / split_name / class_dir.name
                  split_class_dir.mkdir(parents=True, exist_ok=True)
                  for img_path in split_images:
                        shutil.copy2(img_path, split_class_dir / img_path.name)

split_train_test(image_path, data_path / "pets_split")

train_dir = data_path / "pets_split" / "train"
test_dir = data_path / "pets_split" / "test"