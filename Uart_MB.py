import time
import serial
# 任务码
Gostraight=0 #直走
hexagon=1 #进六边形
Board=2 #障碍物
class Uart:
    def __init__(self, com="/dev/ttyUSB0"):
        # 实例化串口对象
        self.uart = serial.Serial(com,9600 , timeout=10)
        if self.uart.isOpen():
            print("uart is ready")

    def close(self):
        self.uart.close()

    def wait_for_data_packet(self, timeout=10):
        global global_task,global_color
        local_task = None
        local_color = None
        time_begin = time.time()
        time_end = time_begin + timeout
        buffer = bytearray()  
        while time.time() < time_end:  
          if self.uart.inWaiting() > 0:  
              buffer.extend(self.uart.read(self.uart.inWaiting()))  # 读取所有可用数据  
              while len(buffer) >= 4:  # 至少要有一个开头字节和三个数据字节  
                  if buffer[0] == 0xBB and buffer[3] == 0xCC:  
                      local_task = buffer[1]  # 提取task数据  
                      local_color = buffer[2]  # 提取color数据 
                      print(f"task data: {local_task}")  
                      print(f"color data: {local_color}")  
                      buffer = buffer[4:]  # 移除已处理的数据  
                      return local_task,local_color
                  else:  
                      buffer.pop(0)  # 移除第一个字节，继续检查
        if local_task is None or local_color is None:
            print("Timeout waiting for data packet.")
        return None,None    
        

    def uart_head(self,param_task,param1,param2):
        print(param_task,param1,param2)
        myinput=bytes([0xFF,param_task,param1,param2,0xFE])
        self.uart.write(myinput)

    def uart_Gostraight(self,deviation):
        if deviation>0:
            direction=0
        else:
            direction=1
        abs_deviation=abs(deviation)
        final_deviation=int(abs_deviation)
        self.uart_head(Gostraight,direction,final_deviation)

    def uart_Board(self):
        self.uart_head(Board,0,0)

    def uart_Hexagon(self):
        self.uart_head(hexagon,0,0)


if __name__ == "__main__":
    uart = Uart()
    uart.uart_Gostraight(-100)
    #uart.uart_Gostraight(100)


