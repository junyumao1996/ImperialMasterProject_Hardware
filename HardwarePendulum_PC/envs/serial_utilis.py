import serial.tools.list_ports
import serial
import serial.tools.list_ports
import time

def finaPort_Name():
    port_list = list(serial.tools.list_ports.comports())
    if len(port_list) <= 0:
        print("The Serial port can't find!")
    else:
        portname_list = []
        for i in list(port_list):
            port_list_0 = list(i)
            port_serial = port_list_0[0]
            portname_list.append(port_serial)

        return portname_list


# 收发数据
def receive_Portdata(ser):
    while 1:
        str = input("请输入要发送的数据（非中文）并同时接收数据: ")
        portStr = []
        send_str = bytearray(str.encode())
        send_str.append(0x0d)
        send_str.append(0x0a)
        ser.write(send_str)

        for i in range(5):
            # ser.write('13' + '/r/n')
            # str1 = ser.readall()
            str1 = ser.read(ser.in_waiting)
            print(str1)
            if str1 == b'@_@':
                portStr.append(str1)
                print(str1.decode().strip())
                break
            elif str1 == b'':
                continue
                #break
            else:
                portStr.append(str1)
                print(str1.decode().strip())
                #break

#########################################
############ Personal-Defined ###########
#########################################

def available_port_check():
    plist = list(serial.tools.list_ports.comports())

    if len(plist) <= 0:
        print("No available ports!")
        return None
    else:
        plist_0 = list(plist[0])
        serialName = plist_0[0]
        print("Available ports: ", serialName)
        return serialName

def send_data(ser, data_string):
    send_str = bytearray(data_string.encode())
    send_str.append(0x0d)
    send_str.append(0x0a)
    ser.write(send_str)

def receive_data_one_shoot(ser):
    if ser.in_waiting:
        # string=ser.read(ser.in_waiting)
        string = ser.read_all()
        str_decoded = str(string.decode())
        print("Received data：", string, 'Decoded data: ', str_decoded)
        return str_decoded
    else:
        return None

def receive_data_waiting(ser):
    while 1:
        if ser.in_waiting:
            # string=ser.read(ser.in_waiting)
            # string = ser.read_all()
            string = ser.readline()
            try:
                str_decoded = str(string.decode())
                received_data = received_data_process(str_decoded)
                if len(received_data) == 6:
                    return received_data
            except:
                pass
            # str_decoded = 1
            # if (str == "exit"):  # 退出标志
            #     break
            # else:
            #     print("Received data：", string, 'Decoded data: ', str_decoded)
            #     return str_decoded
            # print("Received data：", string, 'Decoded data: ', str_decoded)

            # received_data = received_data_process(str_decoded)
            # if len(received_data) == 5:
            #     return received_data

def received_data_process(data):
    data_list = data.split(',')
    data_list[-1] = data_list[-1].split("\n")[0]
    # print(data_list)
    data_list_f = []
    for i in range(len(data_list)):
        data_list_f.append(int(data_list[i]))
    return data_list_f