"""
Copyright (C) 2023 TonAI
"""
import argparse
import os
import cv2
import numpy as np
import torch
import re
from time import time

from ultralytics import YOLO
from tracking.deep_sort import DeepSort
from tracking.sort import Sort
from utils.utils import map_label, check_image_size, draw_text, check_legit_plate, \
    gettime, compute_color, argmax, BGR_COLORS, VEHICLES, crop_expanded_plate, correct_plate
from ppocr_onnx import DetAndRecONNXPipeline


# from ultralytics.utils.checks import check_requirements


def get_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        type=str,
        default="./test01.mp4",
        help="path to video, 0 for webcam")
    parser.add_argument("--vehicle_weight", type=str,
                        default="weights/vehicle_yolov8s_640.pt",
                        help="path to the yolov8 weight of vehicle detector")
    parser.add_argument("--plate_weight", type=str,
                        default="weights/plate_yolov8n_320_2024.pt",
                        help="path to the yolov8 weight of plate detector")
    parser.add_argument("--dsort_weight", type=str,
                        default="weights/reid_model.onnx",
                        help="path to the weight of DeepSORT tracker")
    parser.add_argument(
        "--ocr_weight",
        type=str,
        default="weights/plate_ppocr/",
        help="path to the paddle ocr weight of plate recognizer")
    parser.add_argument("--vconf", type=float, default=0.6,
                        help="confidence for vehicle detection")
    parser.add_argument(
        "--pconf",
        type=float,
        default=0.25,
        help="confidence for plate detection")
    parser.add_argument(
        "--ocr_thres",
        type=float,
        default=0.5,
        help="threshold for ocr model")
    parser.add_argument(
        "--deepsort",
        action="store_true",
        help="suse DeepSORT tracking instead of normal SORT")
    parser.add_argument(
        "--read_plate",
        action="store_true",
        help="read plate information")
    parser.add_argument(
        "--save",
        action="store_true",
        help="save output video")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="real-time monitoring")
    parser.add_argument(
        "--show_plate",
        action="store_true",
        help="zoom in detected plate")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/logs_2",
        help="saved path")
    parser.add_argument(
        "--lang",
        type=str,
        default="coco",
        help="language to show (vi, en, es, fr)")
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="cuda id if available")
    return parser.parse_args()


class TrafficCam():
    """
    License plate OCR TrafficCam
    Args:
    - opts: parsed arguments
    """

    def __init__(self, opts):
        self.opts = opts
        # Core properties
        self.video = opts.video
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(device)
        self.vehicle_detector = YOLO(opts.vehicle_weight, task='detect')
        self.plate_detector = YOLO(opts.plate_weight, task='detect').to(device=device)
        self.read_plate = opts.read_plate
        if self.read_plate:
            self.plate_reader = DetAndRecONNXPipeline(
                text_det_onnx_model="weights/ppocrv4/ch_PP-OCRv4_det_infer.onnx",
                text_rec_onnx_model="weights/ppocrv4/ch_PP-OCRv4_rec_infer.onnx",
                box_thresh=0.4
            )
        self.ocr_thres = opts.ocr_thres

        # DeepSort Tracking
        self.deepsort = opts.deepsort
        self.dsort_weight = opts.dsort_weight
        self.init_tracker()

        # Miscellaneous for displaying
        self.color = BGR_COLORS
        self.show_plate = opts.show_plate
        self.stream = opts.stream
        self.lang = opts.lang
        self.save_dir = opts.save_dir
        self.save = opts.save

    def extract_plate(self, plate_image):
        results = self.plate_reader.detect_and_ocr(plate_image)
        # kiểm tra kết quả rec, nếu có ít nhất 1 thì xử lý, không thì trả về chuỗi rỗng, độ tin cậy conf = 0
        if len(results) > 0:
            plate_info = ''
            conf = []
            # duyệt qua từng kết quả rec, sau đó ghép các kết quả lại với nhau, sau đó tính độ tin cậy trung bình conf
            for result in results:
                plate_info += result.text + ' '
                conf.append(result.score)
            conf = sum(conf) / len(conf)
            # biểu thức chính quy để loại bỏ ký tự không mong muốn, chỉ giữ lại các ký tự chữ cái, số, -, .
            return re.sub(r'[^A-Za-z0-9\-.]', '', plate_info), conf
        else:
            return '', 0

    def init_tracker(self):
        """
        Initialize tracker
        """
        if self.deepsort:
            print("Using DeepSORT Tracking")
            self.tracker = DeepSort(self.dsort_weight, max_dist=0.2,
                                    min_confidence=0.3, nms_max_overlap=0.5,
                                    max_iou_distance=0.7, max_age=70,
                                    n_init=3, nn_budget=100,
                                    use_cuda=torch.cuda.is_available())
        else:
            self.tracker = Sort()
        self.vehicles_dict = {}

    def run(self):
        """
        Run the TrafficCam end2end
        """
        # Config video properties
        vid_name = os.path.basename(self.video)
        cap = cv2.VideoCapture(self.video)
        title = "Traffic Surveillance"
        if self.stream:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.save:
            log_path = self.save_dir
            frames_path = os.path.join(log_path, "frames")
            detected_objects_path = os.path.join(log_path, "objects")
            detected_plates_path = os.path.join(log_path, "plates")
            if not os.path.exists(log_path):
                os.makedirs(log_path)
                os.makedirs(frames_path)
                os.makedirs(detected_objects_path)
                os.makedirs(detected_plates_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            vid_writer = cv2.VideoWriter(
                f"{log_path}/infered_{vid_name}", fourcc, fps, (w, h))
        num_frame = 0
        captured = 0
        thresh_h = int(h / 8)  # Limit detection zone
        print("Traffic Cam is ready!")
        while cap.isOpened():
            t0_fps = gettime()
            ret, frame = cap.read()
            num_frame += 1
            if int(num_frame) == 50:
                self.init_tracker()
            # frame = cv2.imread("data/test_samples/images/bienso/biendo15.jpg")
            if frame is not None:
                displayed_frame = frame.copy()
            else:
                continue
            if ret:
                # cv2.line(displayed_frame, (0, thresh_h), (w, thresh_h), self.color["blue"], 2)
                """
                --------------- VEHICLE DETECTION ---------------
                Plate recognition include two subsections: detection and tracking
                    - Detection: Ultralytics YOLOv8
                    - Tracking: DeepSORT
                """
                t1 = time()
                vehicle_detection = self.vehicle_detector(
                    frame,
                    verbose=False,
                    imgsz=640,
                    device=self.opts.device,
                    conf=self.opts.vconf)[0]
                # print(f"Inference time: {time() - t1}")

                # boxes có xyxy, confidence, class
                vehicle_boxes = vehicle_detection.boxes
                vehicle_xyxy = vehicle_boxes.xyxy
                vehicle_labels = vehicle_boxes.cls
                try:
                    if self.deepsort:
                        # độ dài của outputs = số đối tượng tracking
                        # outputs có dạng, trong đó 4 cái đầu là xyxy, cái cuối là class của đối tượng
                        # outputs = np.array([
                        #     [50, 30, 200, 180, 1],
                        #     [300, 50, 400, 200, 2],
                        #     [100, 150, 250, 300, 3]
                        # ])
                        outputs = self.tracker.update(vehicle_boxes.cpu().xywh,
                                                      vehicle_boxes.cpu().conf,
                                                      frame)
                        
                    else:
                        outputs = self.tracker.update(
                            vehicle_boxes.cpu().xyxy).astype(int)
                        # print(outputs)
                except BaseException:
                    continue

                in_frame_indentities = []

                for idx in range(len(outputs)):
                    # Lấy id của đối tượng
                    identity = outputs[idx, -1]
                    in_frame_indentities.append(identity)

                    # Nếu chưa có id thì append 
                    if str(identity) not in self.vehicles_dict:
                        self.vehicles_dict[str(identity)] = {"save": False,
                                                             "saved_plate": False,
                                                             "plate_image": None,
                                                             "vehicle_image": None}
                    # box của đối tượng
                    self.vehicles_dict[str(identity)]["bbox_xyxy"] = outputs[idx, :4]
                    vehicle_bbox = self.vehicles_dict[str(identity)]["bbox_xyxy"]
                    src_point = (vehicle_bbox[0], vehicle_bbox[1])
                    dst_point = (vehicle_bbox[2], vehicle_bbox[3])
                    color = compute_color(identity)
                    cv2.rectangle(
                        displayed_frame, src_point, dst_point, color, 2)

                    # # ================= DRAW ID =====================
                    # text = f"ID: {identity}"
                    # text_position = (src_point[0], src_point[1] - 10)

                    # # Get the text size and create a background rectangle
                    # (font_scale, thickness) = (0.5, 2)
                    # (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    # background_x0 = src_point[0]
                    # background_y0 = src_point[1] - text_h - 10
                    # background_x1 = src_point[0] + text_w
                    # background_y1 = src_point[1]

                    # # Draw the semi-transparent background rectangle
                    # overlay = displayed_frame.copy()
                    # cv2.rectangle(overlay, (background_x0, background_y0), (background_x1, background_y1), (0, 0, 0),
                    #               -1)
                    # alpha = 0.8  # Transparency factor
                    # cv2.addWeighted(overlay, alpha, displayed_frame, 1 - alpha, 0, displayed_frame)

                    # # Put the text on top of the background
                    # cv2.putText(displayed_frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    #             (255, 255, 255), thickness)
                    # # ================= DRAW ID =====================

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
                if self.read_plate:
                    # các xe đang xử lý
                    active_vehicles = []
                    # ảnh crop các xe
                    input_batch = []
                    for identity in in_frame_indentities:
                        vehicle = self.vehicles_dict[str(identity)]
                        # Khởi tạo giá trị mặc định cho ocr confidence và plate number nếu chưa có
                        if "ocr_conf" not in vehicle:
                            vehicle["ocr_conf"] = 0.0
                            vehicle["plate_number"] = ""
                        # Lấy tọa độ bbox của vehicle đang xử lý
                        box = vehicle["bbox_xyxy"].astype(int)
                        # Lấy plate number của nó
                        plate_number = self.vehicles_dict[str(identity)]["plate_number"]
                        # kiểm tra ocr confidence, độ dài biển số, tính legit của biển số
                        # Tính legit: 
                        # 1. bắt đầu bằng 2 chữ cái, theo sau đó là ít nhất 4 chữ số, ví dụ: AA1234_. Nếu đúng trả về true
                        # 2. tồn tại 1 chữ cái theo sau là ít nhất 4 chữ số, ví dụ: _B1234_. Nếu đúng, check điều kiện bổ sung
                        # 3. điều kiện bổ sung: chuỗi k bắt đầu bằng 2 chữ cái, tức là chuỗi bắt đầu bằng số
                        success = (vehicle["ocr_conf"] > self.ocr_thres) \
                                  and check_legit_plate(plate_number) 
                                #   and len(plate_number) > 7 \
                        # Nếu thành công
                        if success:
                            # Sửa lại thông tin biển số nếu có
                            plate_number = correct_plate(plate_number)
                            
                            pos = (box[0], box[1] + 26)
                            draw_text(
                                img=displayed_frame,
                                text=plate_number,
                                pos=pos,
                                text_color=self.color["blue"],
                                text_color_bg=self.color["green"])

                            # Nếu bật lưu, và xe chưa được lưu
                            if self.save and not vehicle["save"]:
                                # Cắt ảnh vehicle
                                cropped_vehicle = frame[box[1]
                                                        :box[3], box[0]:box[2], :]
                                # cv2.imwrite(f"{detected_objects_path}/{plate_number}.jpg", cropped_vehicle)

                                # Kiểm tra kích thước ảnh biển số xe, nếu phù hợp lưu ảnh xe + biển số xe
                                if check_image_size(vehicle["plate_image"], 32, 16):
                                    # lưu ảnh biển số
                                    cv2.imwrite(
                                        f"{detected_plates_path}/{plate_number}.jpg",
                                        vehicle["plate_image"])
                                    # lưu ảnh xe
                                    if cropped_vehicle is not None:
                                        cv2.imwrite(
                                            f"{detected_objects_path}/{plate_number}.jpg", cropped_vehicle)

                                    # xóa thông tin ảnh đã lưu để giải phóng bộ nhớ, và đánh dấu xe đã lưu
                                    del vehicle["plate_image"]          # xóa ảnh biển
                                    del vehicle["vehicle_image"]        # xóa ảnh xe
                                    vehicle["vehicle_image"] = None     # Set về None
                                    vehicle["plate_image"] = None       # Set về None
                                    vehicle["save"] = True              # Đánh dấu là đã lưu
                            continue

                        # Nếu không đạt yêu cầu OCR
                        else:
                            # Nếu không trong vùng nhận diện -> Loại khỏi `in_frame_indentities`
                            # if box[1] < thresh_h or box[3] > h - thresh_h:  # Ignore vehicle out of recognition zone
                            #     in_frame_indentities.remove(identity)
                            #     continue

                            # Nếu trong vùng nhận diện:
                            # crop ảnh xe
                            cropped_vehicle = frame[box[1]:box[3], box[0]:box[2], :]
                            # Lưu thông tin vào vehicle
                            vehicle["vehicle_image"] = cropped_vehicle

                            # Kiểm tra kích thước ảnh xe, nếu ảnh quá nhỏ thì bỏ qua
                            if not check_image_size(cropped_vehicle, 96, 96):  # ignore too small image!
                                continue

                            # Nếu ảnh đủ lớn + trong vùng recog => 
                            # input_batch chứa cropped_vehicle
                            # active_vehicles chứa vehicle
                            input_batch.append(cropped_vehicle)
                            active_vehicles.append(vehicle)

                    # NHẬN DIỆN BIỂN SỐ XE
                    # input_batch chứa ảnh đã cắt của xe trong frame hiện tại
                    if len(input_batch) > 0:
                        # khởi tạo plate_detection
                        plate_detections = self.plate_detector(
                            input_batch,
                            verbose=False,
                            imgsz=320,
                            device=self.opts.device,
                            conf=self.opts.pconf)

                        # DUYỆT QUA CÁC KẾT QUẢ SAU KHI RECOGNIZE
                        vehicle_having_plate = [] # list các biển số được phát hiện
                        # duyệt qua từng kết quả trong plate_detections
                        for id, detection in enumerate(plate_detections):
                            # thông tin của xe hiện tại trong active_vehicles
                            vehicle = active_vehicles[id]
                            # Lấy ra crop vehicle. 
                            cropped_vehicle = input_batch[id] # Hoặc: cropped_vehicle = vehicle['vehicle_image']
                            # tọa độ bbox của vehicle trong frame
                            box = vehicle["bbox_xyxy"].astype(int)
                            # tọa độ của plate trong cropped vehicle image
                            plate_xyxy = detection.boxes.xyxy
                            
                            # Kiểm tra nếu không có biển nào được phát hiện -> bỏ qua xe hiện tại
                            if len(plate_xyxy) < 1:
                                continue
                            # # Display plate detection on frame
                            # print(plate_xyxy)
                            # Lấy tọa độ biển số trong vehicle
                            plate_xyxy = plate_xyxy[0]
                            # chuyển từ tensor thành mảng numpy và kiểu số nguyên
                            plate_xyxy = plate_xyxy.cpu().numpy().astype(int)
                            # điểm trên cùng bên trái
                            src_point = (
                                plate_xyxy[0] + box[0], plate_xyxy[1] + box[1])
                            # điểm dưới cùng bên phải
                            dst_point = (
                                plate_xyxy[2] + box[0], plate_xyxy[3] + box[1])
                            # vẽ bbox cho biển số
                            cv2.rectangle(
                                displayed_frame,
                                src_point,
                                dst_point,
                                self.color["green"],
                                thickness=2)
                            
                            # CẮT VÀ LƯU BIỂN SỐ
                            try:
                                # cắt biển số từ tọa độ của nó trong frame, thêm biên 20%
                                cropped_plate = crop_expanded_plate(
                                    plate_xyxy, cropped_vehicle, 0.2)
                            # nếu xảy ra lỗi, tạo biển số mặc định là mảng 8x8 với 3 kênh màu, giá trị = 0 (màu đen)
                            except BaseException:
                                cropped_plate = np.zeros((8, 8, 3))
                            vehicle["plate_image"] = cropped_plate
                            # vehicle_having_plate chứa các xe mà biển số được phát hiện thành công
                            vehicle_having_plate.append(vehicle)
                            # nhận diện xong -> hiển thị rồi check legit hậu kỳ sau, hơn là cứ check legit liên tục rồi recog thì làm app bị chậm đi nhiều, tuy nhiên bị giảm độ chính xác

                        # TRÍCH XUẤT VÀ CẬP NHẬT THÔNG TIN BIỂN SỐ
                        # nếu có xe trong danh sách các xe đã được phát hiện biển số, trích xuất thông tin biển số
                        if len(vehicle_having_plate) > 0:
                            for vehicle in vehicle_having_plate:
                                # plate_info: string chứa thông tin plate
                                # conf: độ tin cậy của ocr
                                plate_info, conf = self.extract_plate(vehicle["plate_image"])
                                
                                # cập nhật thông tin biển số và độ tin cậy OCR nếu giá trị mới tốt hơn giá trị hiện tại
                                cur_ocr_conf = vehicle["ocr_conf"]
                                if conf > cur_ocr_conf:
                                    vehicle["plate_number"] = plate_info
                                    vehicle["ocr_conf"] = conf

                #  ---------------- MISCELLANEOUS ---------------- #
                # ids = list(map(int, list(self.vehicles_dict.keys())))
                # num_vehicle = 0 if len(ids) == 0 else max(ids)
                t = gettime() - t0_fps
                fps_info = int(round(1 / t, 0))
                global_info = f"FPS: {fps_info}"
                draw_text(img=displayed_frame, text=global_info,
                          font_scale=1, font_thickness=2,
                          text_color=self.color["blue"],
                          text_color_bg=self.color["white"])
                if self.save:  # Save inference result to file
                    vid_writer.write(displayed_frame)
                    if int(num_frame) == int(
                            fps * 10):  # save frame every 10 seconds
                        self.init_tracker()
                        cv2.imwrite(
                            f"{frames_path}/{captured}.jpg",
                            displayed_frame)
                        captured += 1
                if self.stream:
                    cv2.imshow(title, displayed_frame)
                key = cv2.waitKey(1)
                if key == ord('q'):  # Quit video
                    break
                if key == ord('r'):  # Reset tracking
                    self.init_tracker()
                if key == ord('p'):  # Pause video
                    cv2.waitKey(-1)
                del displayed_frame
                del frame
        cap.release()
        if self.save:
            vid_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    opts = get_args()
    TrafficCam = TrafficCam(opts)
    TrafficCam.run()
