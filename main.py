import cv2
import numpy as np
import math
import multiprocessing as mp


# ==================== PID控制器类 ====================
class PIDController:
    def __init__(self, kp, ki, kd, output_min, output_max, integral_min, integral_max, integral_threshold):
        # PID参数
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # 误差记录
        self.prev_error = 0
        self.integral = 0
        # 输出限幅（防止输出值过大）
        self.output_min = output_min
        self.output_max = output_max
        # 积分限幅（防止积分饱和）
        self.integral_min = integral_min
        self.integral_max = integral_max
        # 积分分离（误差较大时不进行积分）
        self.integral_threshold = integral_threshold

    def update(self, error):
        # 积分分离逻辑：当误差较小时才进行积分
        if abs(error) <= self.integral_threshold:
            self.integral += error
            # 对积分项进行限幅
            self.integral = max(self.integral_min, min(self.integral, self.integral_max))
        # 计算微分项
        derivative = error - self.prev_error
        # PID输出计算
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        # 对最终输出进行限幅
        output = max(self.output_min, min(output, self.output_max))
        self.prev_error = error
        return output


# ==================== 图像处理类 ====================
class ImageProcessor:
    def __init__(self):
        # ROI权重配置（用于偏差计算）
        self.ROIS = (0.05, 0.1, 0.25, 0.3, 0.3, 0, 0, 0)
        # PID控制器初始化（参数需要实际调试）
        self.pid = PIDController(kp=0.5, ki=0.01, kd=0.05, output_min=-100, output_max=100,
                                 integral_min=-1000, integral_max=1000, integral_threshold=10)

    def to2(self, img):
        """图像预处理（二值化+滤波）"""
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 二值化处理（阈值50）
        _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        # 中值滤波（降噪，核大小减小为3x3）
        img_final = cv2.medianBlur(gray, 3)
        # 高斯模糊（核大小9x9）
        img_final = cv2.GaussianBlur(img_final, (9, 9), 0)
        return img_final

    def find_center_in_region(self, region):
        """在指定区域寻找黑条中心"""
        # 找到所有黑色像素点（值为0）
        black_pixels = np.where(region == 0)
        cols = black_pixels[1]  # 获取列坐标
        if len(cols) > 0:
            # 计算列坐标平均值作为中心
            center_col = int(np.mean(cols))
            return center_col
        return None

    def split_image_and_find_centers(self, img):
        """将图像分割为8个区域并寻找每个区域中心"""
        height, width = img.shape
        region_height = height // 8  # 每个区域高度
        centers = []
        # 遍历所有区域
        for i in range(8):
            start_y = i * region_height
            end_y = (i + 1) * region_height
            # 截取当前区域
            region = img[start_y:end_y, :]
            # 寻找中心点
            center = self.find_center_in_region(region)
            if center is not None:
                # 计算中心点坐标（相对于原图）
                center_y = start_y + region_height // 2
                centers.append((center, center_y))
        return centers

    def checkDx(self, centers):
        """计算水平方向偏差量"""
        len_of_centers = len(centers)
        if len_of_centers < 8:
            return 0  # 中心点不足时不计算偏差
        sum_val = 0
        # 加权求和计算综合偏差
        for i in range(len_of_centers):
            sum_val += centers[i][0] * self.ROIS[i]
        # 计算与图像中心（320）的偏差
        dx = sum_val - 320
        return int(dx)

    def calculate_angle(self, x1, y1, x2, y2):
        """
        计算直线的角度
        :param x1: 直线起点的x坐标
        :param y1: 直线起点的y坐标
        :param x2: 直线终点的x坐标
        :param y2: 直线终点的y坐标
        :return: 角度（整数部分和小数部分）
        """
        dx = x2 - x1
        dy = y2 - y1
        # 计算反正切值并转换为角度
        arctan = math.atan2(dy, dx) * 180 / math.pi
        # 分离整数和小数部分
        x = int(arctan)
        y = int(arctan * 100) - int(arctan) * 100
        return x, y

    def parallel(self, frame_parallel):
        """
        检测直线并计算角度
        :param frame_parallel: 输入图像帧
        :return: 前两条直线的角度信息
        """
        # Canny边缘检测
        frame_parallel_candy = cv2.Canny(frame_parallel, 100, 200)
        # 霍夫变换检测直线（参数需要调试）
        lines = cv2.HoughLinesP(frame_parallel_candy, 1, np.pi / 180, 140,
                                minLineLength=150, maxLineGap=800)
        if lines is not None and len(lines) >= 3:
            # 绘制前三条检测到的直线
            for i in range(3):
                x1, y1, x2, y2 = lines[i][0]
                cv2.line(frame_parallel, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.imshow('frame_parallel_line', frame_parallel)
            # 计算第一条和第三条直线的角度
            x01, y01 = self.calculate_angle(*lines[0][0])
            x02, y02 = self.calculate_angle(*lines[2][0])
            return x01, y01, x02, y02
        else:
            # 默认返回值（检测不到直线时）
            return -128, -128, 128, 128


# ==================== 摄像头采集进程 ====================
def camera_worker(shared_array, timestamp, lock, exit_event):
    """摄像头采集子进程"""
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        while not exit_event.is_set():
            ret, frame = cam.read()
            if ret:
                # 确保图像尺寸正确
                frame = cv2.resize(frame, (640, 480))
                with lock:
                    # 将图像数据存入共享内存
                    flat_frame = frame.flatten()
                    shared_array[:] = flat_frame[:]
                    # 更新时间戳
                    timestamp.value += 1
    finally:
        cam.release()


# ==================== 主程序 ====================
if __name__ == "__main__":
    # 共享内存配置（640x480 RGB图像）
    shared_array = mp.Array('B', 640 * 480 * 3)
    timestamp = mp.Value('i', 0)  # 时间戳用于检测新帧
    lock = mp.Lock()  # 进程锁保证数据安全
    exit_event = mp.Event()  # 退出事件

    # 启动摄像头子进程
    camera_process = mp.Process(target=camera_worker,
                                args=(shared_array, timestamp, lock, exit_event))
    camera_process.start()

    # 初始化图像处理器
    processor = ImageProcessor()
    prev_timestamp = 0  # 记录已处理的时间戳

    try:
        while True:
            current_ts = 0
            frame = None
            with lock:
                current_ts = timestamp.value
                # 仅在有新帧时处理
                if current_ts != prev_timestamp:
                    # 从共享内存重建图像
                    frame = np.frombuffer(shared_array.get_obj(),
                                          dtype=np.uint8).reshape((480, 640, 3)).copy()
                    prev_timestamp = current_ts
                else:
                    continue  # 无新帧时跳过

            # 复制原始帧用于显示
            img = frame.copy()

            # 图像处理流程
            processed_frame = processor.to2(img)
            centers = processor.split_image_and_find_centers(processed_frame)

            # 绘制中心点
            for point in centers:
                cv2.circle(img, point, 3, (0, 0, 255), 3)

            # 直线检测与角度计算
            angel_int1, angel_float1, angel_int2, angel_float2 = processor.parallel(processed_frame)
            Dangel = abs(angel_int1 - angel_int2)

            # 计算偏差并进行PID控制
            error = processor.checkDx(centers)
            pid_output = processor.pid.update(error)

            # 显示调试信息
            cv2.putText(img, f"dx:{error}", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(img, f"pid_output:{pid_output}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print(f"dx:{error}, pid_output:{pid_output}")

            # 显示图像
            cv2.imshow("img", img)
            key = cv2.waitKey(50)
            if key == ord(" "):
                break
    finally:
        # 清理资源
        exit_event.set()
        camera_process.join()
        cv2.destroyAllWindows()