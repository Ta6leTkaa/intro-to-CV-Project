"""Fine-tune YOLO11s with exactly the same protocol as the YOLO11n baseline."""
from train_yolo11n import main


if __name__ == '__main__':
    main(default_model='yolo11s.pt')
