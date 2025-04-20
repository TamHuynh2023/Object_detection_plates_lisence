def is_valid_plate_flexible(plate_bbox, vehicle_bbox,
                          min_iou_ratio=0.2,
                          max_edge_margin=20,
                          bottom_height_ratio=0.7):

    inter_x1 = max(plate_bbox[0], vehicle_bbox[0])
    inter_y1 = max(plate_bbox[1], vehicle_bbox[1])
    inter_x2 = min(plate_bbox[2], vehicle_bbox[2])
    inter_y2 = min(plate_bbox[3], vehicle_bbox[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    plate_area = (plate_bbox[2] - plate_bbox[0]) * (plate_bbox[3] - plate_bbox[1])
    iou = inter_area / plate_area if plate_area > 0 else 0

    left_close = abs(plate_bbox[0] - vehicle_bbox[0]) <= max_edge_margin
    right_close = abs(plate_bbox[2] - vehicle_bbox[2]) <= max_edge_margin

    plate_center_y = (plate_bbox[1] + plate_bbox[3]) / 2
    vehicle_bottom = vehicle_bbox[1] + (vehicle_bbox[3] - vehicle_bbox[1]) * bottom_height_ratio
    is_at_bottom = plate_center_y >= vehicle_bottom

    return (iou >= min_iou_ratio or left_close or right_close) and is_at_bottom