from google.colab.patches import cv2_imshow
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from ultralytics import YOLO
import random
from collections import defaultdict
import time
from Pipeline_objects_detections.object_detection import ObjectDetector
from Pipeline_objects_detections.tracker_object import TrackerObject
from Pipeline_objects_detections.handle_tracker_plates import HandleTracks
from Pipeline_objects_detections.handle_track_vehicle import HandleTrackVehicles
from Pipeline_objects_detections.color_pala import Color_Pala
from Logic_hande_stop_line.draw_stop_line import create_stop_line_from_crosswalk, draw_stop_line
from Logic_extract_lisence_plate.extract_image import ExtractLicensePlates

DISTANCE_THRESHOLD = 10

CONFIRMATION_FRAMES = 3
tracker = DeepSort(
    max_age=10,
    nn_budget=50,
    n_init=CONFIRMATION_FRAMES,
)

tracker_vehicles = DeepSort(
    max_age=10,
    nn_budget=50,
    n_init=CONFIRMATION_FRAMES,
)


model = YOLO("model_yolo\results_version_3\content\runs\detect\train2\weights\best.pt")


video_path = "Object_detection_plates_lisence\video_original\video_original.mp4"
cap = cv2.VideoCapture(video_path)

width, height, fpbs = (int(cap.get(x)) for x in
           (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('video_predict/output_video__version_3.avi', fourcc, fpbs,
            (width, height))

frame_count = 0

persistent_crosswalk = None
last_crosswalk_time = 0
crosswalk_timeout = 1.0

vehicle_to_license = {}


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count +=1

    OBJECT_DETECTOR = ObjectDetector(model)
    OBJECT_DETECTOR.detect_objects(frame)

    TRACKER_OBJECT = TrackerObject(tracker_vehicles, tracker, frame)
    TRACKERS_VEHICLES = TRACKER_OBJECT.deep_sort_vehicle(OBJECT_DETECTOR.vehicles)
    TRACKERS = TRACKER_OBJECT.deep_sort_dets(OBJECT_DETECTOR.dets)


    current_time = time.time()
    if len(OBJECT_DETECTOR.cross_walks) > 0:
        persistent_crosswalk = OBJECT_DETECTOR.cross_walks[0]
        last_crosswalk_time = current_time

    stop_line = None
    if persistent_crosswalk is not None and (current_time - last_crosswalk_time) < crosswalk_timeout:
        stop_line = create_stop_line_from_crosswalk(persistent_crosswalk)
        draw_stop_line(frame, stop_line)
    else:
        persistent_crosswalk = None


    height_frame, width_frame = frame.shape[:2]

    HANDLE_TRACK_VEHICLES = HandleTrackVehicles(TRACKERS_VEHICLES, frame)
    HANDLE_TRACK_VEHICLES.handle_tracks_vehicle()


    HANDLE_TRACKS = HandleTracks(TRACKERS, HANDLE_TRACK_VEHICLES.vehicle_info, OBJECT_DETECTOR.cars,
                                OBJECT_DETECTOR.motorcycles, OBJECT_DETECTOR.lisence_plates,
                                OBJECT_DETECTOR.red_lights, stop_line, frame, vehicle_to_license)

    HANDLE_TRACKS.handle_tracks()


    for track_id, lisence_img in vehicle_to_license.items():
        text_fragments  = []
        extract_plates = None
        full_text = None

        for bbox in lisence_img:
            extract_plates = ExtractLicensePlates(bbox)
            plate_number = extract_plates.run_method_OCR()
            if plate_number:
                text_fragments.append(plate_number)
            time.sleep(1)

        if extract_plates is not None and text_fragments:
            full_text = extract_plates.analyze_plates(text_fragments)

        (text_width, text_height), _ = cv2.getTextSize(full_text.strip() if full_text else "UNKNOWN" ,
                                                    cv2.FONT_HERSHEY_SIMPLEX,
                                                    1, 2)
        cv2.putText(frame, full_text.strip() if full_text else "UNKNOWN" ,
                        (width_frame - text_width - 20, height_frame - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        print(full_text)

    out.write(frame)

cap.release()
out.release()