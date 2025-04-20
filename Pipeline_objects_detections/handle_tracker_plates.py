from color_pala import Color_Pala
from Logic_hande_stop_line.draw_stop_line import is_vehicle_crossed_stop_line
from Logic_hande_stop_line.logic_lisence_and_vehicle import is_valid_plate_flexible  

class HandleTracks:
    def __init__(self, tracks, vehicle_info, cars, motorcycles,
                 license_plates, red_lights, stop_line, frame, vehicle_to_license):
        self.tracks = tracks
        self.vehicle_info = vehicle_info
        self.cars = cars
        self.motorcycles = motorcycles
        self.license_plates = license_plates
        self.red_lights = red_lights
        self.stop_line = stop_line
        self.frame = frame

    def handle_tracks(self):
        for track in self.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id_plate = track.track_id
            bbox_plate = track.to_tlbr()
            x1_L, y1_L, x2_L, y2_L = map(int, bbox_plate)

            self.handle_vehicle_list(x1_L, y1_L, x2_L, y2_L)

            Color_Pala(self.frame).draw_color(track_id_plate, bbox_plate)

    def handle_vehicle_list(self, x1_L, y1_L, x2_L, y2_L):
        for vehicle in self.vehicle_info:
            track_id_vehicle = vehicle['track_id']
            bbox_vehicle = vehicle['bbox']
            x1_V_current, y1_V_current, x2_V_current, y2_V_current = map(int, vehicle['bbox'])

            if (len(self.cars) > 0 or len(self.motorcycles) > 0) and len(self.license_plates) > 0 and len(self.red_lights) > 0:
                if self.stop_line is not None:
                    if is_vehicle_crossed_stop_line([x1_V_current, y1_V_current, x2_V_current, y2_V_current], self.stop_line):
                        if is_valid_plate_flexible([x1_L, y1_L, x2_L, y2_L], [x1_V_current, y1_V_current, x2_V_current, y2_V_current]):
                            crop_img = self._get_cropped_plate(x1_L, y1_L, x2_L, y2_L)

                            if track_id_vehicle not in vehicle_to_license:
                                vehicle_to_license[track_id_vehicle] = []
                            vehicle_to_license[track_id_vehicle].append(crop_img)


    def _get_cropped_plate(self, x1_L, y1_L, x2_L, y2_L):
        center_x = (x1_L + x2_L) // 2
        center_y = (y1_L + y2_L) // 2

        x1_pad = max(center_x - 335 // 2, 0)
        y1_pad = max(center_y - 462 // 2, 0)
        x2_pad = x1_pad + 335
        y2_pad = y1_pad + 462

        x2_pad = min(x2_pad, frame.shape[1])
        y2_pad = min(y2_pad, frame.shape[0])
        x1_pad = max(x2_pad - 335, 0)
        y1_pad = max(y2_pad - 462, 0)

        crop_img = self.frame[y1_pad:y2_pad, x1_pad:x2_pad]

        return crop_img
