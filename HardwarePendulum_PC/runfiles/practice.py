import serial
# from serial_utilis import available_port_check
from time import sleep
from envs.real_circular_pendulum import angle_normalize_alpha, angle_normalize_theta

# portname = available_port_check()
# BAUDRATE = 128000
# TIMEOUT = 0.002
# ser = serial.Serial(portname, BAUDRATE, timeout=TIMEOUT)
# print("Serial Port Info:", ser)

# a = '2.250, 3.250'
# list = a.split(',')
# print(list)
# print(float(list[1]))

# input("pause")



# 得到串口名




# name = finaPort_Name()
# ser = serial.Serial(port=name[0], baudrate=128000, timeout=0.02)
# print('Successful Connection!')
# receive_Portdata()

# print(angle_normalize_alpha(1050))

# while True:
#     send_str = bytearray('123'.encode())
#     send_str.append(0x0d)
#     send_str.append(0x0a)
#     ser.write(send_str)
#     sleep(0.001)
#     if ser.in_waiting:
#         # string=ser.read(ser.in_waiting)
#         string = ser.read_all()
#         str_decoded = str(string.decode())
#         # str_decoded = 1
#         if(str=="exit"):#退出标志
#             break
#         else:
#             print("Received data：", string, 'Decoded data: ', str_decoded)




# print("---------------")
# ser.close()#关闭串口


