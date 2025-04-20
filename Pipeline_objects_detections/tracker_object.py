class TrackerObject:
    def __init__(self, tracker_vehicles, tracker, frame):
        self.tracker_vehicles = tracker_vehicles
        self.tracker = tracker
        self.frame = frame

    def deep_sort_vehicle(self, vehicle):
        tracks_vehicles = self.tracker_vehicles.update_tracks(vehicle, frame=self.frame)
        return tracks_vehicles

    def deep_sort_dets(self, dets):
        tracks = self.tracker.update_tracks(dets, frame=self.frame)
        return tracks