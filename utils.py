import os
import time

PRINT_OUT = True
LOG_TO_FILE = True

path = os.getcwd()
data_path = path + os.sep + "data"
save_path = path + os.sep + "saves"
log_path = path + os.sep + "logs"

start_time = time.strftime("%m%d%H%M%S", time.gmtime())


def get_time_str():
    return time.strftime("%d.%m.%H:%M:%S", time.gmtime())


def log(msg):
    msg = get_time_str()+" "+msg
    if PRINT_OUT:
        print(msg)
    if LOG_TO_FILE:
        with open(log_path + os.sep + "log" + start_time + ".txt", "a+") as file:
            file.write(msg+"\n")
