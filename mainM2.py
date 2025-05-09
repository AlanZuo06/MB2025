import cv2
import numpy as np
import math
import threading
from Uart_MB import *

wide=640
height=480
timeToLine=10

def Cam_Init():
    cam = cv2.VideoCapture(1)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, wide)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cam

def Uart_send():
    global Board_send_flag,Boarder_flag,Hexagon_flag,Hexagon_send_flag,pid_output,error
    uart = Uart()
    while True:
        if not Boarder_flag and Board_send_flag:
            uart.uart_Board()
            Board_send_flag=False
        elif not Hexagon_flag and Hexagon_flag:
            uart.uart_Hexagon(error)
            Hexagon_flag=False
        else:
            uart.uart_Gostraight(pid_output)
        time.sleep(0.2)

# PID控制器类
class PIDController:
    def __init__(self, kp, ki, kd, output_min, output_max, integral_min, integral_max, integral_threshold):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        # 输出限幅
        self.output_min = output_min
        self.output_max = output_max
        # 积分限幅
        self.integral_min = integral_min
        self.integral_max = integral_max
        # 积分分离
        self.integral_threshold = integral_threshold

    def update(self, error):
        # 积分分离
        if abs(error) <= self.integral_threshold:
            self.integral += error
            # 积分限幅
            self.integral = max(self.integral_min, min(self.integral, self.integral_max))
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        # 输出限幅
        output = max(self.output_min, min(output, self.output_max))
        self.prev_error = error
        return output
class Link_reascher:
    def calculate_angle(self, x1, y1, x2, y2):
        """
        计算直线的角度
        :param x1: 直线起点的 x 坐标
        :param y1: 直线起点的 y 坐标
        :param x2: 直线终点的 x 坐标
        :param y2: 直线终点的 y 坐标
        :return: 直线角度的整数部分和小数部分
        """
        dx = x2 - x1
        dy = y2 - y1
        arctan = math.atan2(dy, dx) * 180 / math.pi
        x = int(arctan)
        y = int(arctan * 100) - int(arctan) * 100
        return x, y

    def parallel(self, frame_parallel):
        """
        在图像中检测直线并计算前两条直线的角度
        :param frame_parallel: 输入的图像帧
        :return: 两条直线角度的整数部分和小数部分
        """
        frame_parallel_candy = cv2.Canny(frame_parallel, 100, 200)
        lines = cv2.HoughLinesP(frame_parallel_candy, 1, np.pi / 180, 140, minLineLength=150, maxLineGap=800)
        if lines is not None and len(lines) >= 3:
            for i in range(3):
                x1, y1, x2, y2 = lines[i][0]
                cv2.line(frame_parallel, (x1, y1), (x2, y2), (255, 0, 255), 2)
           # cv2.imshow('frame_parallel_line', frame_parallel)
            x01, y01 = self.calculate_angle(*lines[0][0])
            x02, y02 = self.calculate_angle(*lines[2][0])
            return x01, y01, x02, y02
        else:
            return -128, -128, 128, 128
class Center_reascher:
    def find_center_in_region(self, region):
        # 找出指定区域内黑条的中心
        black_pixels = np.where(region == 0)
        rows, cols = black_pixels
        if len(rows) > 0:
            center_col = int(np.mean(cols))
            return center_col
        return None

    def split_image_and_find_centers(self, img):
        # 将图像由上到下分为六个区域，找出每个区域黑条的中心
        height, width = img.shape
        region_height = height // 8
        centers = []
        for i in range(8):
            start_y = i * region_height
            end_y = (i + 1) * region_height
            region = img[start_y:end_y, :]
            center = self.find_center_in_region(region)
            if center is not None:
                center_y = start_y + region_height // 2
                centers.append((center, center_y))
        return centers

    def generate_decreasing_sequence(self,length):
        sequence = [pow(i,3) for i in range(length, 0, -1)]
        total = sum(sequence)
        normalized_sequence = [i / total for i in sequence]
        return normalized_sequence

    def checkDx(self, centers):
        len_of_centers = len(centers)
        if len_of_centers==0:
            return 0
        sum = 0
        ROIS=self.generate_decreasing_sequence(len_of_centers)
        for i in range(len_of_centers):
            sum += centers[i][0] * ROIS[i]
        dx = sum - wide/2
        return int(dx)
class Boarder:
    def color_select_RGB(self,frame):
        # 分离通道
        b_, g_, r_ = cv2.split(frame)
        r_ = np.int16(r_)
        b_ = np.int16(b_)
        g_ = np.int16(g_)

        # 计算灰度图像
        r = (r_ + r_ - b_ - g_).clip(min=0)

        # 取绝对值并转换为uint8
        r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ret, r = cv2.threshold(r, 128, 255, cv2.THRESH_BINARY)  # TOZERO

        # 选择灰度图像
        gray = r
        # 滤波
        img_final = cv2.medianBlur(gray, 5)
        img_final = cv2.GaussianBlur(img_final, (17, 19), 0)

        cv2.imshow("1", img_final)
        return img_final

    def detect_Poly(self, image):

        gray = self.color_select_RGB(image)
        # 边缘检测
        edges = cv2.Canny(gray, 50, 150)
        # 查找轮廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # 近似轮廓
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # cv2.drawContours(image, [approx], 0, (0, 255, 0), 1)
            area = cv2.contourArea(approx)
            if area > 1000:
                return False
        return True
# 图像处理类
class ImageProcessor(Link_reascher,Center_reascher,Boarder):
    def __init__(self,cam):
        self.cam = cam
        
        # 初始化PID控制器，这里的参数需要根据实际情况进行调整，同时设置输出范围
        self.pid = PIDController(kp=0.4, ki=0.6, kd=0, output_min=-100, output_max=100,
                                 integral_min=-1000, integral_max=1000, integral_threshold=10)

    def to2(self, img):
        # cv2.imshow("img", img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        img_final = cv2.medianBlur(gray, 3)
        img_final = cv2.GaussianBlur(img_final, (5, 5), 0)
        # cv2.imshow("gray", img_final)
        # print((np.array(img_final)).shape)
        return img_final


    def img_read(self, cam):
        global globle_ret, globle_img
        while True:
            local_ret, local_img = cam.read()
            with lock:
                globle_ret = local_ret
                globle_img = local_img

if __name__ == "__main__":
    #全局变量
    globle_ret=False
    globle_img=None
    error=0
    pid_output=0
    #标志
    Hexagon_flag=True
    Hexagon_send_flag = True
    Boarder_flag=True
    Board_send_flag=True

    cam =Cam_Init()#相机初始化
    processor=ImageProcessor(cam)


    # 摄像头线程
    lock = threading.Lock()
    cam_threading = threading.Thread(target=processor.img_read, args=(cam,), name="cam_threading")
    cam_threading.daemon = True
    cam_threading.start()
    # 串口线程
    uart_threading = threading.Thread(target=Uart_send, args=(), name="uart_threading")
    uart_threading.daemon = True
    uart_threading.start()

    while True:
        img=globle_img
        ret=globle_ret
        if not ret or img is None:
            continue

        """找障碍"""
        if Boarder_flag:
            Boarder_flag = processor.detect_Poly(img)
            if Boarder_flag:
                cv2.putText(img, "NO Board", (300, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        """找中心"""
        frame = processor.to2(img)
        # 分割图像并找出每个区域的中心
        centers = processor.split_image_and_find_centers(frame)
        # 画出中心点
        for point in centers:
            cv2.circle(img, point, 3, (0, 0, 255), 3)
        # 找出偏差量
        error = processor.checkDx(centers)
        # 使用PID控制器进行优化
        pid_output = processor.pid.update(error)


        """找线"""
        if abs(error)>=timeToLine and Hexagon_flag:
            angel_int1, angel_float1, angel_int2, angel_float2 = processor.parallel(frame)
            Dangel = angel_int1 - angel_int2
            cv2.putText(img, f"NO 30", (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            if abs(Dangel - 30) < 10:#找到了
                Hexagon_flag = False
        else:
            angel_int1, angel_float1, angel_int2, angel_float2, Dangel=0,0,0,0,0

        """一坨"""
        cv2.putText(img, f"dx:{error}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(img, f"pid_output:{pid_output}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        print(f"dx:{error}")
        print(f"angle1:{angel_int1}.{angel_float1},angle2:{angel_int2}.{angel_float2},\nDangel:{Dangel}")
        print(f"pid_output:{pid_output}\n")

        cv2.imshow("img", img)

        key = cv2.waitKey(100)
        if key == ord(" "):
            break
    processor.cam.release()
    cv2.destroyAllWindows()
