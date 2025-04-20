import random
import cv2
from collections import defaultdict

class Color_Pala():
    color_palette = defaultdict(lambda: (
        random.randint(0, 255),
        random.randint(0, 255),
        random.randint(0, 255)
    ))

    def __init__(self, frame):
        self.frame = frame

    def draw_color(self, track_id, bounding_box):
        x1, y1, x2, y2 = map(int, bounding_box)
        color = Color_Pala.color_palette[track_id]
        label = f"ID: {track_id}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(self.frame,
                (x1, y1),
                (x2, y2),
                color, 2)
        cv2.putText(self.frame, label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)