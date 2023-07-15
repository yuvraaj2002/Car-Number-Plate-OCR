# Importing all the dependencies
from ultralytics import YOLO
import cv2
import os
import sys
import numpy as np
from sort.sort import *
from src.OCR import get_car,read_license_plate,write_csv
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


@dataclass
class Output_Storage_Config:
    output_csv_file_path = os.path.join('Output','Results.csv')


class Detection:

    def __init__(self):
        self.Output_file_path = Output_Storage_Config()


    def make_predictions_storage(self):
        """
        This method will simply use a video and store the prediction information in the csv file
        """
        try:
            logging.info("Initializing the model predictions phase")

            # Let's create a dictionary that will store the car information
            output_info = {}

            # Let's now load the pretrained coco model for car detection and our trained number plate model
            car_detection_model = YOLO('yolov8n.pt')
            number_plate_detector = YOLO('Model_Train.pt')
            logging.info("Pretrained and custom trained model loaded")

            mot_tracker = Sort()

            # Let's now load our custom testing video
            cap = cv2.VideoCapture('./sample.mp4')

            logging.info("Initializing reading frames from video and making detections")
            vehicles = [2, 3, 5, 7]

            # Let's read the frames
            frame_nmr = -1
            ret = True
            while ret:
                frame_nmr += 1
                ret, frame = cap.read()
                if ret:
                    output_info[frame_nmr] = {}
                    # detect vehicles
                    detections = car_detection_model(frame)[0]
                    detections_ = []
                    for detection in detections.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = detection
                        if int(class_id) in vehicles:
                            detections_.append([x1, y1, x2, y2, score])

                    # track vehicles
                    track_ids = mot_tracker.update(np.asarray(detections_))

                    # detect license plates
                    license_plates = number_plate_detector(frame)[0]
                    for license_plate in license_plates.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = license_plate

                        # assign license plate to car
                        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                        if car_id != -1:

                            # crop license plate
                            license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                            # process license plate
                            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255,
                                                                         cv2.THRESH_BINARY_INV)

                            # read license plate number
                            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                            if license_plate_text is not None:
                                output_info[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                              'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                                'text': license_plate_text,
                                                                                'bbox_score': score,
                                                                                'text_score': license_plate_text_score}}

            # Let's make a directory to store the csv file
            os.makedirs(os.path.dirname(self.Output_file_path.output_csv_file_path), exist_ok=True)

            # Let's now save the dictionary in the csv file and store it in the output directory
            write_csv(output_info,self.Output_file_path.output_csv_file_path)

        except Exception as e:
            raise CustomException(e,sys)
