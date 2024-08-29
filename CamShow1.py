import sys,cv2
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from pathlib import Path
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout,QMenu, QAction,QPushButton,QInputDialog,QFileDialog
from PyQt5.QtGui import QPixmap, QPainter, QImage, QPen
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread,QPoint,QRect

import time 
import numpy as np
from my_alpr4 import Ui_MainWindow
import imutils
# -----------------Counting Thread-----------------
from collections import defaultdict
from math import sqrt
import wx
import re
import os
import torch
# -------------------------------------------------
from tracking.sort import Sort
from tracking.deep_sort import DeepSort
from utils.utils import map_label, check_image_size, draw_text, check_legit_plate, \
    gettime, compute_color, argmax, BGR_COLORS, VEHICLES, crop_expanded_plate, correct_plate
    

#--------------------------------------------------
from ppocr_onnx import DetAndRecONNXPipeline as PlateReader
from tracking.sort import Sort
from ultralytics import YOLO
from PyQt5.QtCore import QThread, QWaitCondition, QMutex

#--------------------------------------------------

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    change_pixmap_signal_1 = pyqtSignal(np.ndarray)
    change_pixmap_signal_2 = pyqtSignal(np.ndarray)

    change_pixmap_signal_3 = pyqtSignal(np.ndarray)
    change_pixmap_signal_4 = pyqtSignal(np.ndarray)

    vehicle_count_signal = pyqtSignal(dict)
    plate_number_signal = pyqtSignal(str)
    plate_number_signal_1 = pyqtSignal(str)
    result_form_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.data = ""
        self.video_path=""
        self.vehicle_detector_path=""
        self.Mode = 0
        self.classes = []
        
        self._is_paused = False
        self._mutex = QMutex()
        self._condition = QWaitCondition()
        self._is_running = True


        self.vehicle_detector = YOLO("weights/vehicle_yolov8s_640.pt", task='detect')
        self.plate_detector = YOLO("weights/plate_yolov8n_320_2024.pt", task='detect')
        self.plate_reader = PlateReader(
            text_det_onnx_model="weights/ppocrv4/ch_PP-OCRv4_det_infer.onnx",
            text_rec_onnx_model="weights/ppocrv4/ch_PP-OCRv4_rec_infer.onnx",
            box_thresh=0.6)
        self.ocr_thres = 0.95
        
        
        #-------------------Tracker----------------
        self.dsort_weight = "weights/reid_model.onnx"
        self.init_tracker()
        #------------------------------------------

        self.vehicleCounter = {
            0: 0, #bus
            1: 0, #car
            2: 0, #motorcycle
            3: 0 #truck
        }

        self.plate_data = ""
        self.cnt = 0

        # Miscellaneous for displaying
        self.save_dir = "data/logs"
        self.save = True
        self.lang = "coco_vi"
        self.color = BGR_COLORS

        #print(self.data)
    @pyqtSlot(dict)
    def receivedata(self, dict_data):
        # dict = {'classes': [1,5,3,6] , 'Mode': 0, 'v' : "C:/Users/Admin/PythonLession/pic/carplate6.mp4",
        # 'pt':'C:/Users/Admin/PythonLession/yolo_dataset/best_carplate5.pt', "CountMode": False, "LinePoint": [],
        # "PolygonPoint":[]}
        print(dict_data)
        self.video_path = dict_data["v"]
        self.vehicle_detector_path = dict_data['pt']
        self.Mode = dict_data['Mode']
        self.classes = dict_data['classes']
        self._run_flag =dict_data['flag']
        self.CountMode= dict_data["CountMode"]
        self.LinePoint=dict_data["LinePoint"]
        self.PolygonPoint=dict_data["PolygonPoint"]
        self.ReadPlate=dict_data["ReadPlate"]

        print(self.video_path)
        print(self.vehicle_detector_path)
        print(self.Mode)
        print(self.classes)
        print(self._run_flag)
        print(self.CountMode)
        print(self.LinePoint)
        print(self.PolygonPoint)
        print(self.ReadPlate)
    # Mathemetic Function find a distance between a line an point


    def run(self):
        video_path = self.video_path 
        vehicle_detector_path = self.vehicle_detector_path

        if vehicle_detector_path != "" and video_path != "" and (not self.CountMode) and (not self.ReadPlate):
            model = YOLO(vehicle_detector_path)
            cap = cv2.VideoCapture(video_path)
            # Loop through the video frames
            while cap.isOpened():
                success, frame = cap.read()
                if success:
                    if self.Mode==0:
                        if self.classes == []:
                            results = model.predict(frame)
                        else:
                            results = model.predict(frame, classes=self.classes)
                    else:
                        if self.classes == []:
                            results = model.track(frame,persist=True, tracker="bytetrack.yaml")
                        else:
                            results = model.track(frame, persist=True,classes=self.classes, tracker="bytetrack.yaml")
                    annotated_frame = results[0].plot()
                    #---------------------------------------------------#
                    self.change_pixmap_signal.emit(annotated_frame)
                    if not self._run_flag:
                        break
                else:
                    # Break the loop if the end of the video is reached
                    print(" no video file")
                    cap.release()
                    break

        elif vehicle_detector_path != "" and video_path != "" and self.CountMode and (self.PolygonPoint or self.LinePoint):
            model = YOLO(vehicle_detector_path)
            cap = cv2.VideoCapture(video_path)
            track_history = defaultdict(lambda: [])
            obcross = 40  # 8 pxel, when object near the line, calculator start couting
            # crossed_objects = {}
            crossed_objects = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
            crossed_objects1 = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]

            # Loop through the video frames
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()
                if success:
                    #frame = cv2.resize(frame, (1380, 920))
                    frame = imutils.resize(frame,width=1201)
                    #results = model.track(frame, persist=True, classes=[0])  # Tracking Car only
                    if self.classes == []:
                        results = model.track(frame, persist=True, tracker="bytetrack.yaml")
                    else:
                        results = model.track(frame, persist=True, classes=self.classes, tracker="bytetrack.yaml")

                    
                    if results[0].boxes.id != None:
                        boxes = results[0].boxes.xywh.cpu()
                        # print(boxes)
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        vehicle_classes = results[0].boxes.cls.int().cpu().tolist()
                        # Visualize the results on the frame
                        # annotated_frame = results[0].plot()
                        annotated_frame = frame
                        pos_label_P = []
                        pos_label_P_count = []

                        counterP = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        counterL = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                        pos_label_L = []

                        # Plot the tracks
                        for box, track_id, cls in zip(boxes, track_ids, vehicle_classes):
                            x, y, w, h = box
                            track = track_history[track_id]
                            track.append((float(x), float(y)))  # x, y center point
                            # Draw Center point of object
                            cv2.circle(annotated_frame, (int(x), int(y)), 1, (0, 255, 0), 2)
                            if len(track) > 30:  # retain 90 tracks for 90 frames
                                track.pop(0)
                            # Check Center point (x,y) in the polygon or not. ( In =1; Out =-1, On polygon =0)
                            # --------------POLYGON COUNTER-----------------------------
                            if len(self.PolygonPoint) > 0:
                                for idx, data in enumerate(self.PolygonPoint):
                                    pts = np.asarray(data, dtype=np.int32)  # Convert to Numpy Array
                                    pts = pts.reshape(-1, 1, 2)
                                    cv2.polylines(annotated_frame, [pts], True, (255, 0, 0), 2)
                                    dist = cv2.pointPolygonTest(pts, (int(x), int(y)), False)

                                    if dist == 1:

                                        cv2.circle(annotated_frame, (int(x), int(y)), 8, (0, 0, 255), 2)
                                        cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)),
                                                      (int(x + w / 2), int(y + h / 2)),
                                                      (0, 255, 0), 2)
                                        counterP[idx] += 1
                                                
                                        # Remember Object Passing the area
                                        if track_id not in crossed_objects1[idx]:
                                            crossed_objects1[idx][track_id] = True
                                            self.vehicleCounter[cls] += 1
                                        
                                            

                                        # --------------------Final - Polygon --------------------------------------
                                        # print(counter)
                                temp = []
                                temp1 = []
                                for data in self.PolygonPoint:
                                    # print(data)
                                    temp.append(data[0])
                                    temp1.append(data[3])
                                pos_label_P = temp.copy()
                                pos_label_P_count = temp1.copy()
                                # print(pos_label_P)

                            # ---------------------------------------------------------------

                            # -----------------------LINE --------COUNTER-----------

                            distance = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                            if len(self.LinePoint) > 0:
                                for data in self.LinePoint:
                                    for idx, data1 in enumerate(data):

                                        A0 = [data1[0], data1[1]]
                                        B0 = [data1[2], data1[3]]
                                        E = [x, y]
                                        distance[idx] = self.minDistance(A0, B0, E)
                                        cv2.line(annotated_frame, (data1[0], data1[1]), (data1[2], data1[3]),
                                                 (255, 0, 0), 2)
                                        if distance[idx] < obcross:  # Assuming objects cross horizontally
                                            if track_id not in crossed_objects[idx]:
                                                crossed_objects[idx][track_id] = True
                                                self.vehicleCounter[cls] += 1
                                            
                                            # Annotate the object as it crosses the line
                                            cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)),
                                                          (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
                                            cv2.rectangle(annotated_frame, (int(x - w / 2), int(y - h / 2)),
                                                          (int(x + w / 2), int(y + h / 2)),
                                                          (0, 255, 0), 2)

                                        counterL[idx] = len(crossed_objects[idx])
                                # print(crossed_objects)
                                # print(counterL)

                                temp = []
                                # print(len(Linepoints))
                                for data in self.LinePoint:
                                    # print(data)
                                    for data1 in data:
                                        temp.append([data1[0], data1[1]])
                                pos_label_L = temp.copy()
                                # print(pos_label_L)'''
                        # -----------------Final Display -----Polygon---------And Line------

                        # for idx, pos in enumerate(pos_label_L):
                        #     # print(pos)
                        #     cv2.putText(annotated_frame, "Count In Line: " + str(counterL[idx]), (pos[0], pos[1]),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 2)
                        #     # print(counterL[idx])
                        # # --------------------------------------------------

                        # for index, pos in enumerate(pos_label_P):
                        #     cv2.putText(annotated_frame, "Count In Region: " + str(counterP[index]), (pos[0], pos[1]),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
                        #     # print(index)
                        #     # print(pos)
                        #     cv2.putText(annotated_frame, "Count Obj pass: " + str(len(crossed_objects1[index])),
                        #                 (pos[0], pos[1] + 150),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        self.change_pixmap_signal.emit(annotated_frame)
                        self.vehicle_count_signal.emit(self.vehicleCounter)
                        # display(annotated_frame) 
                else:
                    print("Video Completed")
                    break
                #cap.release()
        elif vehicle_detector_path != "" and video_path != "" and self.ReadPlate and not self.CountMode:
            cap = cv2.VideoCapture(video_path)
            log_path = self.save_dir
            frames_path = os.path.join(log_path, "frames")
            detected_objects_path = os.path.join(log_path, "objects")
            detected_plates_path = os.path.join(log_path, "plates")
            if not os.path.exists(log_path):
                os.makedirs(log_path)
                os.makedirs(frames_path)
                os.makedirs(detected_objects_path)
                os.makedirs(detected_plates_path)
            num_frame = 0 
            while cap.isOpened():
                self._mutex.lock()
                while self._is_paused:
                    self._condition.wait(self._mutex)
                self._mutex.unlock()
                ret, frame = cap.read()
                num_frame += 1
                if int(num_frame) == 500:
                    self.init_tracker()
                if frame is not None:
                    displayed_frame = frame.copy()
                else:
                    continue
                if ret:
                    """
                    --------------- VEHICLE DETECTION ---------------
                    Plate recognition include two subsections: detection and tracking
                        - Detection: Ultralytics YOLOv8
                        - Tracking: DeepSORT
                    """
                    vehicle_detection = self.vehicle_detector(
                        frame, 
                        verbose=False, 
                        imgsz=640,
                        device="0",
                        conf=0.6)[0]
                    vehicle_boxes = vehicle_detection.boxes
                    vehicle_xyxy = vehicle_boxes.xyxy
                    vehicle_labels = vehicle_boxes.cls

                    try:        
                        # outputs = self.tracker.update(vehicle_boxes.cpu().xyxy).astype(int)
                        outputs = self.tracker.update(vehicle_boxes.cpu().xywh,
                                                      vehicle_boxes.cpu().conf,
                                                      frame)
                    except BaseException:
                        continue

                    in_frame_indentities = []

                    for idx in range(len(outputs)):
                        identity = outputs[idx, -1]
                        in_frame_indentities.append(identity)
                        if str(identity) not in self.vehicles_dict:
                            self.vehicles_dict[str(identity)] = {"save": False,
                                                                "saved_plate": False,
                                                                "plate_image": None,
                                                                "vehicle_image": None}
                        self.vehicles_dict[str(
                            identity)]["bbox_xyxy"] = outputs[idx, :4]
                        vehicle_bbox = self.vehicles_dict[str(
                            identity)]["bbox_xyxy"]
                        src_point = (vehicle_bbox[0], vehicle_bbox[1])
                        dst_point = (vehicle_bbox[2], vehicle_bbox[3])
                        color = compute_color(identity)
                        cv2.rectangle(
                            displayed_frame, src_point, dst_point, color, 1)
                        
                    for index, box in enumerate(vehicle_xyxy):
                        if box is None:
                            continue
                        label_name = map_label(int(vehicle_labels[index]), VEHICLES[self.lang])
                        box = box.cpu().numpy().astype(int)
                        draw_text(img=displayed_frame, text=label_name,
                                pos=(box[0], box[1]),
                                text_color=self.color["blue"],
                                text_color_bg=self.color["green"])
                    
                    """
                    --------------- PLATE RECOGNITION ---------------
                    This section will run if --read-plate
                    Plate recognition include two subsections: detection and OCR
                        - Detection: Ultralytics YOLOv8
                        - Optical Character Recognition: Baidu PaddleOCR
                    """
                    active_vehicles = []
                    input_batch = []
                    for identity in in_frame_indentities:
                        
                        vehicle = self.vehicles_dict[str(identity)]
                        if "ocr_conf" not in vehicle:
                            vehicle["ocr_conf"] = 0.0
                            vehicle["plate_number"] = ""
                        box = vehicle["bbox_xyxy"].astype(int)
                        plate_number = self.vehicles_dict[str(
                            identity)]["plate_number"]
                        success = (vehicle["ocr_conf"] > self.ocr_thres) \
                            and len(plate_number) > 5 \
                            and check_legit_plate(plate_number)
                        if success:
                            self.cnt += 1
                            plate_number = correct_plate(plate_number)
                            if self.cnt % 2 == 0 and vehicle["vehicle_image"] is not None and vehicle["plate_image"] is not None:
                                self.plate_number_signal.emit(plate_number)
                                self.change_pixmap_signal_1.emit(vehicle["vehicle_image"])
                                self.change_pixmap_signal_2.emit(vehicle["plate_image"])
                                jilu = time.strftime("%b %d %Y %H:%M:%S",time.localtime(time.time())) +  'License Plate Number: ' + plate_number + " vehicle_id:" + str(identity)
                                self.result_form_signal.emit(jilu)
                                
                            elif self.cnt % 2 != 0 and vehicle["vehicle_image"] is not None and vehicle["plate_image"] is not None:
                                self.plate_number_signal_1.emit(plate_number)
                                self.change_pixmap_signal_3.emit(vehicle["vehicle_image"])
                                self.change_pixmap_signal_4.emit(vehicle["plate_image"])
                                jilu = time.strftime("%b %d %Y %H:%M:%S",time.localtime(time.time())) +  'License Plate Number: ' + plate_number + " vehicle_id:" + str(identity)
                                self.result_form_signal.emit(jilu)
                                
                            # jilu = time.strftime("%b %d %Y %H:%M:%S",time.localtime(time.time())) +  'License Plate Number: ' + plate_number
                            # self.result_form_signal.emit(jilu)
                            pos = (box[0], box[1] + 26)
                            draw_text(
                                img=displayed_frame,
                                text=plate_number,
                                pos=pos,
                                text_color=self.color["blue"],
                                text_color_bg=self.color["green"])
                            if self.save and not vehicle["save"]:
                                cropped_vehicle = frame[box[1]
                                    :box[3], box[0]:box[2], :]
                                if check_image_size(vehicle["plate_image"], 32, 16):
                                    cv2.imwrite(
                                        f"{detected_plates_path}/{plate_number}.jpg",
                                        vehicle["plate_image"])
                                    if cropped_vehicle is not None:
                                        cv2.imwrite(
                                            f"{detected_objects_path}/{plate_number}.jpg", cropped_vehicle)
                                    del vehicle["plate_image"]
                                    del vehicle["vehicle_image"]
                                    vehicle["vehicle_image"] = None
                                    vehicle["plate_image"] = None
                                    vehicle["save"] = True
                                    
                            continue
                        else:
                            # if box[1] < thresh_h: # Ignore vehicle out of recognition zone
                            #     in_frame_indentities.remove(identity)
                            #     continue
                            # crop vehicle image to push into the plate
                            # detector
                            cropped_vehicle = frame[box[1]
                                :box[3], box[0]:box[2], :]
                            vehicle["vehicle_image"] = cropped_vehicle
                            if not check_image_size(
                                    cropped_vehicle, 112, 112):  # ignore too small image!
                                continue
                            input_batch.append(cropped_vehicle)
                            active_vehicles.append(vehicle)
                    if len(input_batch) > 0:
                        plate_detections = self.plate_detector(
                            input_batch,
                            verbose=False,
                            imgsz=320,
                            device="0",
                            conf=0.25)
                        vehicle_having_plate = []
                        for id, detection in enumerate(plate_detections):
                            vehicle = active_vehicles[id]
                            cropped_vehicle = input_batch[id]
                            box = vehicle["bbox_xyxy"].astype(int)
                            plate_xyxy = detection.boxes.xyxy
                            if len(plate_xyxy) < 1:
                                continue
                            # Display plate detection
                            plate_xyxy = plate_xyxy[0]
                            plate_xyxy = plate_xyxy.cpu().numpy().astype(int)
                            src_point = (
                                plate_xyxy[0] + box[0], plate_xyxy[1] + box[1])
                            dst_point = (
                                plate_xyxy[2] + box[0], plate_xyxy[3] + box[1])
                            cv2.rectangle(
                                displayed_frame,
                                src_point,
                                dst_point,
                                self.color["green"],
                                thickness=2)
                            # cropped_plate = cropped_vehicle[plate_xyxy[1]:plate_xyxy[3], \
                            # plate_xyxy[0]:plate_xyxy[2], :]
                            try:
                                cropped_plate = crop_expanded_plate(
                                    plate_xyxy, cropped_vehicle, 0.15)
                            except BaseException:
                                cropped_plate = np.zeros((8, 8, 3))


                            vehicle["plate_image"] = cropped_plate
                            vehicle_having_plate.append(vehicle)

                        if len(vehicle_having_plate) > 0:
                            for vehicle in vehicle_having_plate:
                                plate_info, conf = self.extract_plate(
                                    vehicle["plate_image"])
                                cur_ocr_conf = vehicle["ocr_conf"]
                                if conf > cur_ocr_conf:
                                    vehicle["plate_number"] = plate_info
                                    vehicle["ocr_conf"] = conf
                                
                                # self.plate_data = correct_plate(plate_number)



                    self.change_pixmap_signal.emit(displayed_frame)
        else:
            print("Not Config this Mode")
    

    def extract_plate(self, plate_image):
        results = self.plate_reader.detect_and_ocr(plate_image)
        if len(results) > 0:
            plate_info = ''
            conf = []
            for result in results:
                plate_info += result.text + ' '
                conf.append(result.score)
            conf = sum(conf) / len(conf)
            return re.sub(r'[^A-Za-z0-9\-.]', '', plate_info), conf
        else:
            return '', 0
    
    def init_tracker(self):
        self.tracker = DeepSort(self.dsort_weight, max_dist=0.2,
                                    min_confidence=0.3, nms_max_overlap=0.5,
                                    max_iou_distance=0.7, max_age=70,
                                    n_init=3, nn_budget=100,
                                    use_cuda=torch.cuda.is_available())
        # self.tracker = Sort()
        self.vehicles_dict = {}

    # def stop(self):
    #     """Sets run flag to False and waits for thread to finish"""
    #     self._run_flag = False
    #     #self.wait()
    #     self.quit()
    #     self.terminate()
    def stop(self):
        self._is_running = False
        self.resume()  # Đảm bảo thoát khỏi bất kỳ chờ đợi nào

    def pause(self):
        self._mutex.lock()
        self._is_paused = True
        self._mutex.unlock()

    def resume(self):
        self._mutex.lock()
        self._is_paused = False
        self._condition.wakeAll()
        self._mutex.unlock()
    
    def continueThread(self):
        self._run_flag = True

    def countingLinepolygon(self,video_path, yolo_path, Linepoints, PolygonPoints):
        annotated_frame = None


        return annotated_frame

#----------------------------------
    def minDistance(self, A, B, E):
        # vector AB
        AB = [None, None];
        AB[0] = B[0] - A[0];
        AB[1] = B[1] - A[1];

        # vector BP
        BE = [None, None];
        BE[0] = E[0] - B[0];
        BE[1] = E[1] - B[1];

        # vector AP
        AE = [None, None];
        AE[0] = E[0] - A[0];
        AE[1] = E[1] - A[1];
        # Variables to store dot product
        # Calculating the dot product
        AB_BE = AB[0] * BE[0] + AB[1] * BE[1];
        AB_AE = AB[0] * AE[0] + AB[1] * AE[1];

        # Minimum distance from
        # point E to the line segment
        reqAns = 0;

        # Case 1
        if (AB_BE > 0):
            # Finding the magnitude
            y = E[1] - B[1];
            x = E[0] - B[0];
            reqAns = sqrt(x * x + y * y);
        # Case 2
        elif (AB_AE < 0):
            y = E[1] - A[1];
            x = E[0] - A[0];
            reqAns = sqrt(x * x + y * y);

        # Case 3
        else:
            # Finding the perpendicular distance
            x1 = AB[0];
            y1 = AB[1];
            x2 = AE[0];
            y2 = AE[1];
            mod = sqrt(x1 * x1 + y1 * y1);
            reqAns = abs(x1 * y2 - y1 * x2) / mod;
        return reqAns


class MainWindow(QMainWindow):
    dataconfig = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self)
        app = wx.App(False)
        width, height = wx.GetDisplaySize()
        print(width)
        print(height)

        # ------------------------------------------------------
        self.display_width = 1201
        self.display_height = 720
        #self.setGeometry(30, 30, 1000, 600)
        # create the label that holds the image
        self.uic.label_img.resize(self.display_width, self.display_height)

        self.uic.vid_pat.clicked.connect(self.selectVideo)
        self.uic.vid_pat.clicked.connect(self.selectPretrain)
        self.uic.vid_pat.clicked.connect(self.videoMode)

        self.uic.ShowBt.clicked.connect(self.start)

        self.uic.StopBt.clicked.connect(self.pause)
        self.uic.RecordBt.clicked.connect(self.continueBt)
        self.uic.ExitBt.clicked.connect(self.close)
        self.uic.ClearDatapoint.clicked.connect(self.cleardata)

        #-------------------------------------------
        self.thread = VideoThread()
        self.thread.start()

        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.change_pixmap_signal_1.connect(self.update_vehicle_crop)
        self.thread.change_pixmap_signal_2.connect(self.update_plate_crop)
        self.thread.plate_number_signal.connect(self.update_plate_number)
        self.thread.result_form_signal.connect(self.update_result_form)

        self.thread.change_pixmap_signal_3.connect(self.update_vehicle_crop_1)
        self.thread.change_pixmap_signal_4.connect(self.update_plate_crop_1)
        self.thread.plate_number_signal_1.connect(self.update_plate_number_1)

        self.dataconfig.connect(self.thread.receivedata)
        self.thread.vehicle_count_signal.connect(self.updateCounter)

        # ------------------------------------------------------
        self.classes=[]
        self.Mode = 0
        self.video_path =""
        self.pretrainpath =""
        self.status = False

        #------------------------------------------------
        #Draw
        self.begin, self.destination = QPoint(), QPoint()
        self.pix=None

        self.LinePoint=[]
        self.MultilinePoint=[]

        self.RecPoint=[]
        self.MultiRecpoint=[]

        self.PolygonPoint=[]
        self.MutiPolypoint = []

        self.uic.ButtonSet_region.clicked.connect(self.set_region)


#----------------------------------------------
    def videoMode(self):
        self.uic.pattle.setText("Video Mode")
        self.uic.pattle.setAlignment(Qt.AlignCenter)
    
    def cameraMode(self):
        self.uic.pattle.setText("Camera Mode")

    def start(self):
        if self.video_path!="" and self.pretrainpath!="":
            self.Mode = self.uic.Combo_Mode.currentIndex()
            data_in_classtxt = "," + self.uic.SelectClasses.text() + ","
            poscomma = []
            index = 0
            if self.uic.SelectClasses.text() == "":
                self.classes = []
            else:
                for i in data_in_classtxt:  # for element in range(0, len(string_name)):
                    if i == ',':  # print(string_name[element])
                        poscomma.append(index)
                    index += 1
                # print(poscomma)
                for j in range(0, len(poscomma) - 1):
                    if data_in_classtxt[poscomma[j] + 1:poscomma[j + 1]] != "":
                        self.classes.append(int(data_in_classtxt[poscomma[j] + 1:poscomma[j + 1]]))
                        
            if self.uic.Combo_Mode.currentIndex()==3:
                data = {'classes': self.classes, 'Mode': self.Mode, 'v': self.video_path, 'pt': self.pretrainpath,
                        'flag': True, "CountMode": False, "LinePoint": [],
                        "PolygonPoint": [], 
                        "ReadPlate": True}
                
            elif self.MultilinePoint or self.MutiPolypoint and self.uic.Combo_Mode.currentIndex()==2:
                data = {'classes': self.classes, 'Mode': self.Mode, 'v': self.video_path, 'pt': self.pretrainpath,
                        'flag': True, "CountMode": True, "LinePoint": self.MultilinePoint,
                        "PolygonPoint": self.MutiPolypoint, 
                        "ReadPlate": False}

            else:
                data = {'classes': self.classes, 'Mode': self.Mode, 'v': self.video_path, 'pt': self.pretrainpath,
                    'flag': True, "CountMode": False, "LinePoint": [],
                    "PolygonPoint": [], 
                    "ReadPlate": False}

            # print(data)
            self.dataconfig.emit(data)
            #----------------------
            self.thread.start()
            self.status =True
        else:
            print(" Please slect Video and pretrain Path")
            self.uic.Videopath_txt.setText("Please slect Video and pretrain Path")

    def selectVideo(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(r'..')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("VideoFile (*.mp4 )")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            #print(filenames)
            if filenames:
                # self.file_list.addItems([str(Path(filename)) for filename in filenames])
                self.video_path =str(Path(filenames[0]))
                self.uic.Videopath_txt.setText(str(Path(filenames[0])))


    def selectPretrain(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(r'..')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("PretrainFile (*.pt )")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            #print(filenames
            if filenames:
                # self.file_list.addItems([str(Path(filename)) for filename in filenames])
                # self.uic.source_pretrain_file.setText(str(Path(filenames[0])))
                self.pretrainpath =str(Path(filenames[0]))
                self.uic.PretrainPathTxt.setText(str(Path(filenames[0])))

                names_classes=self.classesUpdate(self.video_path,self.pretrainpath)
    
    #------------------------------------------------
    #Update counter
    @pyqtSlot(dict)
    def updateCounter(self):
        self.uic.textBrowser.setText(str(self.thread.vehicleCounter[1]))
        self.uic.textBrowser_2.setText(str(self.thread.vehicleCounter[3]))
        self.uic.textBrowser_3.setText(str(self.thread.vehicleCounter[2]))
        self.uic.textBrowser_4.setText(str(self.thread.vehicleCounter[0]))

    def closeEvent(self, event):
        self.thread.stop()
        #self.imgthread.stop()
        event.accept()

    # def pause(self):
    #     #if self.thread.isRunning():
    #     self.thread.stop()
    #     self.status = False
    
    # def continueBt(self):
    #     self.thread.continueThread()
    #     self.status = True
    def pause(self):
        self.thread.pause()
        self.status = False
    
    def continueBt(self):
        self.thread.resume()
        self.status = True

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""

        qt_img = self.convert_cv_qt(cv_img)
        self.pix = qt_img
        self.uic.label_img.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_vehicle_crop(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        scaled_qt_img = qt_img.scaled(self.uic.car_det.width(), self.uic.car_det.height(), Qt.KeepAspectRatio)
        self.pix = scaled_qt_img
        self.uic.car_det.setAlignment(Qt.AlignCenter)
        self.uic.car_det.setPixmap(scaled_qt_img)

    @pyqtSlot(np.ndarray)
    def update_plate_crop(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        scaled_qt_img = qt_img.scaled(self.uic.pr_loc.width(), self.uic.pr_loc.height(), Qt.KeepAspectRatio)
        self.pix = scaled_qt_img
        self.uic.pr_loc.setAlignment(Qt.AlignCenter)
        self.uic.pr_loc.setPixmap(scaled_qt_img)

    @pyqtSlot(str)
    def update_plate_number(self, plate_number):
        self.uic.pr_loc_4.setAlignment(Qt.AlignCenter)
        self.uic.pr_loc_4.setText(plate_number)
    
    @pyqtSlot(str)
    def update_result_form(self, info):
        self.uic.result_form.appendPlainText(info)

    @pyqtSlot(np.ndarray)
    def update_vehicle_crop_1(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        scaled_qt_img = qt_img.scaled(self.uic.car_det_2.width(), self.uic.car_det_2.height(), Qt.KeepAspectRatio)
        self.pix = scaled_qt_img
        self.uic.car_det_2.setAlignment(Qt.AlignCenter)
        self.uic.car_det_2.setPixmap(scaled_qt_img)

    @pyqtSlot(np.ndarray)
    def update_plate_crop_1(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        scaled_qt_img = qt_img.scaled(self.uic.pr_loc_2.width(), self.uic.pr_loc_2.height(), Qt.KeepAspectRatio)
        self.pix = scaled_qt_img
        self.uic.pr_loc_2.setAlignment(Qt.AlignCenter)
        self.uic.pr_loc_2.setPixmap(scaled_qt_img)

    @pyqtSlot(str)
    def update_plate_number_1(self, plate_number):
        self.uic.pr_loc_5.setAlignment(Qt.AlignCenter)
        self.uic.pr_loc_5.setText(plate_number)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
    def classesUpdate(self, video_path, yolo_path):
        names_classes = []
        model = YOLO(yolo_path)
        if video_path != "" and yolo_path!="" and not self.status:
            cap = cv2.VideoCapture(video_path)
            # Loop through the video frames
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()
                # frame=cv2.resize(frame,(480,640))
                if success:
                    # results = model.track(frame, persist=True,show=True)
                    # results = model.track(frame, persist=True)
                    results = model.predict(frame)
                    names = results[0].names

                    self.image=frame
                    self.update_image(frame) # Update Image

                    self.uic.MsgTE.setText("")
                    index=0
                    for x in names.values():
                        # print(x)
                        names_classes.append(x)
                        self.uic.MsgTE.append(str(index) + "  " + x)
                        index += 1
                    break
        return names_classes

#----------------------------------------------DRAWN------------------
    def paintEvent(self, event):
        pen = QtGui.QPen()
        pen.setWidth(3)
        pen.setColor(QtGui.QColor(255, 0, 0))
        if self.pix and not self.status:
            painter = QtGui.QPainter(self.uic.label_img.pixmap())

            painter.drawPixmap(QPoint(), self.pix)
            painter.setPen(pen)

            if not self.begin.isNull() and not self.destination.isNull() and abs(self.begin.x()-self.destination.x()) >2:
                rect = QRect(self.begin, self.destination)
                index= self.uic.ComboBox.currentIndex()
                if index==0:
                    painter.drawLine(self.begin.x(),self.begin.y(), self.destination.x(),self.destination.y())
                elif index==1:
                    painter.drawRect(rect.normalized())

                elif index == 2:
                    painter.drawLine(self.begin.x(),self.begin.y(), self.destination.x(),self.destination.y())

    def mousePressEvent(self, event):
        if event.buttons() & Qt.LeftButton and event.pos().x()<self.display_width and event.pos().y()<self.display_height:
            #print('Point 1')
            self.begin = event.pos()
            self.destination = self.begin
            if not self.PolygonPoint and self.uic.ComboBox.currentIndex()==2:
                datapoint = [self.begin.x(), self.begin.y()]
                self.PolygonPoint.append(datapoint)
            self.update()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and event.pos().x()<self.display_width and event.pos().y()<self.display_height:
            #print('Point 2')
            self.destination = event.pos()
            self.update()


    def mouseReleaseEvent(self, event):

        if event.button() & Qt.LeftButton and event.pos().x()<self.display_width and event.pos().y()<self.display_height and abs(self.begin.x()-self.destination.x()) >2 and not self.status:
            #print('Point 3')
            rect = QRect(self.begin, self.destination)


            pen = QtGui.QPen()
            pen.setWidth(3)
            pen.setColor(QtGui.QColor(255, 0, 0))
            if self.pix:
                painter = QPainter(self.pix)
                painter.drawPixmap(QPoint(), self.pix)
                painter.setPen(pen)

                index = self.uic.ComboBox.currentIndex()
                if index == 0:
                    dataPoint = [  self.begin.x(), self.begin.y() ,  self.destination.x() , self.destination.y() ]
                    self.LinePoint.append(dataPoint)
                    self.uic.PointData.setText("")
                    self.uic.PointData.setText(str(self.LinePoint))

                    painter.drawLine(self.begin.x(),self.begin.y(), self.destination.x(),self.destination.y())
                elif index == 1:
                    dataPoint = [ self.begin.x() ,self.begin.y() ,   self.destination.x() , self.destination.y() ]
                    self.RecPoint.append(dataPoint)
                    self.uic.PointData.setText("")
                    self.uic.PointData.setText(str(self.RecPoint))

                    painter.drawRect(rect.normalized())
                elif index == 2:
                    #painter.drawEllipse(self.begin.x(),self.begin.y(), self.destination.x()-self.begin.x(),self.destination.y()-self.begin.y())

                    datapoint = [self.destination.x(), self.destination.y()]
                    self.PolygonPoint.append(datapoint)
                    n = len(self.PolygonPoint)
                    painter.drawLine(self.PolygonPoint[n-2][0],self.PolygonPoint[n-2][1],self.PolygonPoint[n-1][0],self.PolygonPoint[n-1][1],)
                    self.uic.PointData.setText("")
                    self.uic.PointData.setText(str(self.PolygonPoint))

                self.begin, self.destination = QPoint(), QPoint()
                self.update()

    def mouseDoubleClickEvent(self, event):
        if self.uic.ComboBox.currentIndex()==2 and self.pix and not self.status:
            self.set_region()


    def cleardata(self):
        if not self.status and self.video_path and self.pretrainpath:
            self.RecPoint.clear()
            self.LinePoint.clear()
            self.PolygonPoint.clear()

            self.MultilinePoint.clear()
            self.MutiPolypoint.clear()
            self.MultiRecpoint.clear()


            self.uic.PointData.clear()

            self.uic.label_img.clear()
            self.uic.Videopath_txt.clear()
            self.uic.PretrainPathTxt.clear()

            self.uic.textBrowser.clear()
            self.uic.textBrowser_2.clear()
            self.uic.textBrowser_3.clear()
            self.uic.textBrowser_4.clear()

            self.uic.car_det.clear()
            self.uic.pr_loc.clear()
            self.uic.pr_loc_4.clear()

            self.uic.car_det_2.clear()
            self.uic.pr_loc_2.clear()
            self.uic.pr_loc_5.clear()

            # self.uic.pattle.setText("")
            
            self.update_image(self.image)
        else:
            print(" Please stop the QThread operation before clear data")

    def set_region(self):

        if len(self.PolygonPoint)>=1 and self.uic.ComboBox.currentIndex()==2:
            # Data  [[386, 252], [844, 269], [1002, 529], [317, 575], [386, 252]]
            #print(self.PolygonPoint)

            datapoint = self.PolygonPoint.copy()
            datapoint.pop(len(self.PolygonPoint)-1)
            #del datapoint[len(self.PolygonPoint)-1]
            datapoint.append(self.PolygonPoint[0])

            self.MutiPolypoint.append(datapoint)
            print(self.MutiPolypoint)
            self.PolygonPoint.clear()
            self.uic.PointData.clear()

        elif self.RecPoint and self.uic.ComboBox.currentIndex()==1:
            '''
            2 Rect in this array;  Converted from X1,Y1,X2,Y2 to Poligon Point
            [[[184, 270], [430, 270], [430, 434], [184, 434], [184, 270]], [[615, 289], [885, 289], [885, 485], [615, 485], [615, 289]]]

            '''
            for datapoint in self.RecPoint:
                x1=datapoint[0]
                y1=datapoint[1]
                x2=datapoint[2]
                y2=datapoint[3]
                point=[]
                #print(self.RecPoint)

                if (x1<x2 and y1<y2) or (x1>x2 and y1>y2) :
                    point.append([x1,y1])
                    point.append([x2,y1])
                    point.append([x2,y2])
                    point.append([x1,y2])
                    point.append([x1,y1])

                elif (x1<x2 and y1>y2) or(x1>x2 and y1<y2):
                    point.append([x1, y1])
                    point.append([x1, y2])
                    point.append([x2, y2])
                    point.append([x2, y1])
                    point.append([x1, y1])

                #self.MultiRecpoint.append(point)
                self.MutiPolypoint.append(point)

            print(self.MutiPolypoint)
            self.RecPoint.clear()
            self.uic.PointData.clear()

        elif self.LinePoint and self.uic.ComboBox.currentIndex()==0:  # if Index=0 LINE

            # Do nothing with Line Point  ; 3 Lines below  xy xy
            #Data [[326, 324, 944, 344], [944, 344, 951, 683], [304, 649, 650, 656]] #  [ Start, End] Line

            self.MultilinePoint.append(self.LinePoint.copy()) # Use List copy it is not affected with original one
            print(self.MultilinePoint)
            self.LinePoint.clear()
            self.uic.PointData.clear()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())