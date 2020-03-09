from recognizeR import Recognize
from time import time

if __name__ == '__main__':
    t1 = time()
    center_loc = Recognize(device_id=-1, debug=True).capture()
    t2 = time()
    print("Center coordinate is {}, cost time is {:.5f}.".format(center_loc, (t2-t1)))