from color_pala import Color_Pala

class HandleTrackVehicles:
    def __init__(self, track_vehicles, frame):
        self.frame = frame
        self.vehicle_info = []
        self.track_vehicles = track_vehicles

        self.track_id = None
        self.bbox = None

    def handle_tracks_vehicle(self):
        for track_v in self.track_vehicles:
            if not track_v.is_confirmed() or track_v.time_since_update > 1:
                continue
            self.track_id = track_v.track_id
            self.bbox = track_v.to_tlbr()
            x1_V, y1_V, x2_V, y2_V = map(int, self.bbox)

            self.vehicle_info.append({
                'track_id': self.track_id,
                'bbox': self.bbox
            })

            Color_Pala(self.frame).draw_color(self.track_id, self.bbox)