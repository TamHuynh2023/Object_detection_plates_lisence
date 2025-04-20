import cv2

def is_vehicle_crossed_stop_line(vehicle_bbox, stop_line):
    _, _, x2, y2 = vehicle_bbox
    return y2 >= stop_line["y"] and stop_line["x_min"] <= x2 <= stop_line["x_max"]


def create_stop_line_from_crosswalk(crosswalk_bbox, offset=10, min_length=50):
    x1, y1, x2, y2 = crosswalk_bbox
    stop_line_y = y1 - offset

    length = x2 - x1
    if length < min_length:
        center = (x1 + x2)/2
        x1 = center - min_length/2
        x2 = center + min_length/2
    return {"y": stop_line_y, "x_min": int(x1), "x_max": int(x2)}



def draw_stop_line(frame, stop_line, thickness=2):
    cv2.line(frame,
             (int(stop_line["x_min"]), int(stop_line["y"])),
             (int(stop_line["x_max"]), int(stop_line["y"])),
             (0, 0, 255),
             thickness)