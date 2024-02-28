import cv2
from hikvisionapi import Client
import datetime
import os

class HikvisionCamera:
    def __init__(self, rtsp_url, camera_ip, username, password):
        self.rtsp_url = rtsp_url
        self.cam = Client(f'http://{camera_ip}', username, password, timeout=15)
        self.cap = cv2.VideoCapture(rtsp_url)
        self.last_saved_time = None

    def display_stream(self):
        cv2.namedWindow('Live Stream', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Live Stream', 1760, 1440)  # Adjust to your desired dimensions

        while True:
            ret, frame = self.cap.read()
            if ret:
                cv2.imshow('Live Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                self.maybe_save_image()

        self.cap.release()
        cv2.destroyAllWindows()
        
    def save_images_only(self, interval_seconds=60):
        while True:
            current_time = datetime.datetime.now()
            if self.last_saved_time is None or (current_time - self.last_saved_time).total_seconds() >= interval_seconds:
                self.last_saved_time = current_time
                self.save_image()
                
    def maybe_save_image(self):
        current_time = datetime.datetime.now()
        if self.last_saved_time is None or (current_time - self.last_saved_time).total_seconds() >= 60:
            self.last_saved_time = current_time
            self.save_image()

    def save_image(self):
        response = self.cam.System.Time(method='get')
        date = response["Time"]["localTime"]
        original_datetime = datetime.datetime.fromisoformat(date)
        formatted_datetime_str = original_datetime.strftime("%Y%m%d%H%M")
        name = f'{formatted_datetime_str}.jpg'
        path = 'playground/images'

        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path, name), 'wb') as f:
            for chunk in self.cam.Streaming.channels[102].picture(method='get', type='opaque_data'):
                if chunk:
                    f.write(chunk)

if __name__ == '__main__':
    camera_ip = '10.49.61.106'
    username = 'admin'
    password = 'sIgmaview124'  # Be cautious with real passwords
    rtsp_url = f'rtsp://{username}:sIgmaview124@{camera_ip}/Streaming/Channels/1'
    
    camera = HikvisionCamera(rtsp_url, camera_ip, username, password)
    camera.display_stream()

