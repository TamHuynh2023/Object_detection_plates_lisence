class ObjectDetector:
    def __init__(self, model):
        self.model = model
        self.red_lights = []
        self.motorcycles = []
        self.cars = []
        self.vehicles = []
        self.cross_walks = []
        self.lisence_plates = []
        self.dets = []

    def detect_objects(self, image):
        results = self.model(image)
        detections = results[0].boxes.data.cpu().numpy()

        CONF_THRESHOLDS = {0: 0.4, 1: 0.3, 3: 0.6, 4: 0.4}

        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)

            if cls in CONF_THRESHOLDS and conf < CONF_THRESHOLDS[cls]:
                continue

            if cls in [0, 4]:
                width = x2 - x1
                height = y2 - y1
                self.vehicles.append(([x1, y1, width, height], conf, cls))

            if cls in [3, 6]:
                width = x2 - x1
                height = y2 - y1
                self.dets.append(([x1, y1, width, height], conf, cls))

            if cls == 3:
                self.lisence_plates.append([x1, y1, x2, y2])
            elif cls == 0:
                self.motorcycles.append([x1, y1, x2, y2])
            elif cls == 4:
                self.motorcycles.append([x1, y1, x2, y2])
            elif cls == 1:
                self.cross_walks.append([x1, y1, x2, y2])
            elif cls == 6:
                self.red_lights.append([x1, y1, x2, y2])