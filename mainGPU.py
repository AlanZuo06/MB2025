import cv2
import numpy as np
import math
import cv2.cuda  # 新增 GPU 模块

wide = 320
height = 240
timeToLine = 40


class PIDController:
    # 保持原有 PID 控制器代码不变
    def __init__(self, kp, ki, kd, output_min, output_max, integral_min, integral_max, integral_threshold):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.output_min = output_min
        self.output_max = output_max
        self.integral_min = integral_min
        self.integral_max = integral_max
        self.integral_threshold = integral_threshold

    def update(self, error):
        if abs(error) <= self.integral_threshold:
            self.integral += error
            self.integral = max(self.integral_min, min(self.integral, self.integral_max))
        derivative = error - self.prev_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(self.output_min, min(output, self.output_max))
        self.prev_error = error
        return output


class ImageProcessor:
    def __init__(self, cam):
        self.cam = cam
        self.ROIS = (0.05, 0.1, 0.25, 0.3, 0.3, 0, 0, 0)
        self.pid = PIDController(kp=0.5, ki=0.01, kd=0.05, output_min=-100, output_max=100,
                                 integral_min=-1000, integral_max=1000, integral_threshold=10)

        # GPU 初始化
        self.init_gpu()

    def init_gpu(self):
        # 检查是否有可用的 CUDA 设备（Rock5C 的 Mali GPU 支持 OpenCL/CUDA 兼容模式）
        if not cv2.cuda.getCudaEnabledDeviceCount():
            print("警告：未检测到 CUDA 设备，将使用 CPU 模式")
        else:
            self.gpu_device = cv2.cuda.getDevice(0)  # 使用第一个 GPU 设备
            self.gpu_device.set()  # 设置当前 GPU 设备

    def to2(self, img):
        """GPU 加速的图像处理函数"""
        # 将 CPU 图像转换为 GPU 矩阵（BGR 转 GRAY 前上传到 GPU）
        gpu_img = cv2.cuda_GpuMat.fromHost(img)  # 输入图像为 BGR 格式（CPU）

        # GPU 上的色彩空间转换（BGR -> GRAY）
        gpu_gray = cv2.cuda.cvtColor(gpu_img, cv2.COLOR_BGR2GRAY)

        # GPU 上的阈值处理（二值化）
        _, gpu_gray = cv2.cuda.threshold(gpu_gray, 50, 255, cv2.THRESH_BINARY)

        # GPU 上的中值滤波（3x3）
        gpu_gray = cv2.cuda.medianBlur(gpu_gray, 3)

        # GPU 上的高斯滤波（5x5）
        gpu_gray = cv2.cuda.GaussianBlur(gpu_gray, (5, 5), 0)

        # 将 GPU 结果下载回 CPU（格式为 uint8 单通道矩阵）
        img_final = gpu_gray.download()
        return img_final

    def find_center_in_region(self, region):
        # 保持原有逻辑（CPU 处理，因区域划分和中心计算轻量）
        black_pixels = np.where(region == 0)
        rows, cols = black_pixels
        if len(rows) > 0:
            center_col = int(np.mean(cols))
            return center_col
        return None

    def split_image_and_find_centers(self, img):
        # 保持原有逻辑（CPU 处理，图像分割和坐标计算轻量）
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

    def checkDx(self, centers):
        # 保持原有逻辑（PID 输入计算，CPU 处理）
        len_of_centers = len(centers)
        if len_of_centers < 8:
            return 0
        sum_x = sum(center[0] * self.ROIS[i] for i, center in enumerate(centers))
        dx = sum_x - wide / 2
        return int(dx)

    def calculate_angle(self, x1, y1, x2, y2):
        # 保持原有逻辑（角度计算，CPU 处理）
        dx = x2 - x1
        dy = y2 - y1
        arctan = math.atan2(dy, dx) * 180 / math.pi
        x = int(arctan)
        y = int(arctan * 100) - int(arctan) * 100
        return x, y

    def parallel(self, frame_parallel):
        """GPU 加速的边缘检测 + CPU 直线检测（Hough 暂不支持 GPU，仅加速 Canny）"""
        # 将输入图像上传到 GPU 并执行 Canny 边缘检测
        gpu_frame = cv2.cuda_GpuMat.fromHost(frame_parallel)
        gpu_canny = cv2.cuda.Canny(gpu_frame, 100, 200)
        canny_img = gpu_canny.download()  # 下载回 CPU 进行 Hough 变换（OpenCV 暂不支持 GPU Hough）

        lines = cv2.HoughLinesP(canny_img, 1, np.pi / 180, 140, minLineLength=150, maxLineGap=800)
        if lines is not None and len(lines) >= 3:
            for i in range(3):
                x1, y1, x2, y2 = lines[i][0]
                cv2.line(frame_parallel, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.imshow('frame_parallel_line', frame_parallel)
            x01, y01 = self.calculate_angle(*lines[0][0])
            x02, y02 = self.calculate_angle(*lines[2][0])
            return x01, y01, x02, y02
        else:
            return -128, -128, 128, 128


if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        cam = cv2.VideoCapture(1)
        if not cam.isOpened():
            print("无法打开任何摄像头")
        else:
            processor = ImageProcessor(cam)
    else:
        processor = ImageProcessor(cam)

    processor.cam.set(cv2.CAP_PROP_FRAME_WIDTH, wide)
    processor.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    while True:
        ret, img = processor.cam.read()
        if not ret:
            continue

        # GPU 加速的图像处理
        frame = processor.to2(img)

        # 分割图像并找出中心（CPU 处理）
        centers = processor.split_image_and_find_centers(frame)

        # 绘制中心点（CPU 处理）
        for point in centers:
            cv2.circle(img, point, 3, (0, 0, 255), 3)

        # 计算偏差和 PID 输出（CPU 处理）
        error = processor.checkDx(centers)
        pid_output = processor.pid.update(error)

        # 直线检测（GPU 加速 Canny，CPU 执行 Hough）
        if abs(error) >= timeToLine:
            angel_int1, angel_float1, angel_int2, angel_float2 = processor.parallel(frame)
            Dangel = abs(angel_int1 - angel_int2)
        else:
            angel_int1, angel_float1, angel_int2, angel_float2, Dangel = 0, 0, 0, 0, 0

        # 可视化和日志（CPU 处理）
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