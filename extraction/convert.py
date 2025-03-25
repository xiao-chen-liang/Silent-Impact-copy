import ctypes
import datetime
import pandas as pd
import sys
from tqdm import tqdm

data = pd.DataFrame(columns=["AccX", "AccY", "AccZ", "AsX", "AsY", "AsZ", "AngX", "AngY", "AngZ", "timestamp"])
magneticFieldData = pd.DataFrame(columns=["HX", "HY", "HZ"])
quaternionData = pd.DataFrame(columns=["Q0", "Q1", "Q2", "Q3"])

TempBytes = []

def convert(file):
    for byte in tqdm(file, desc="Converting the file to a pandas DataFrame"):
        TempBytes.append(byte)
        if len(TempBytes) == 1 and TempBytes[0] != 0x55:
            del TempBytes[0]
            continue
        if len(TempBytes) == 2 and (TempBytes[1] != 0x61 and TempBytes[1] != 0x71):
            del TempBytes[0]
            continue
        if len(TempBytes) == 28:
            processData(TempBytes)
            TempBytes.clear()

def processData(Bytes):
    if Bytes[1] == 0x61:
        process_data(Bytes)
    elif Bytes[1] == 0x71 and Bytes[2] == 0x3A:
        print("Magnetic field")
        print(Bytes)
        process_magnetic_field(Bytes)
    elif Bytes[1] == 0x71 and Bytes[2] == 0x51:
        print("Quaternion")
        print(Bytes)
        process_quaternion(Bytes)
    elif Bytes[1] == 0x71 and Bytes[2] == 0x02:
        print("frequency")
        print(Bytes)

def process_data(Bytes):
    Ax, Ay, Az = [ctypes.c_int16(Bytes[i] << 8 | Bytes[i - 1]).value / 32768 * 16 for i in range(3, 8, 2)]
    Gx, Gy, Gz = [ctypes.c_int16(Bytes[i] << 8 | Bytes[i - 1]).value / 32768 * 2000 for i in range(9, 14, 2)]
    AngX, AngY, AngZ = [ctypes.c_int16(Bytes[i] << 8 | Bytes[i - 1]).value / 32768 * 180 for i in range(15, 20, 2)]
    timestamp = extract_timestamp(Bytes)

    new_row = {
        "AccX": Ax,
        "AccY": Ay,
        "AccZ": Az,
        "AsX": Gx,
        "AsY": Gy,
        "AsZ": Gz,
        "AngX": AngX,
        "AngY": AngY,
        "AngZ": AngZ,
        "timestamp": timestamp
    }

    # Append the new data to the DataFrame
    data.loc[len(data)] = new_row


def process_magnetic_field(Bytes):
    Hx, Hy, Hz = [ctypes.c_int16(Bytes[i] << 8 | Bytes[i-1]).value / 120 for i in range(5, 10, 2)]
    magneticFieldData["HX"] = Hx
    magneticFieldData["HY"] = Hy
    magneticFieldData["HZ"] = Hz
    print(magneticFieldData)

def process_quaternion(Bytes):
    Q0, Q1, Q2, Q3 = [ctypes.c_int16(Bytes[i] << 8 | Bytes[i-1]).value / 32768 for i in range(5, 12, 2)]
    quaternionData["Q0"] = Q0
    quaternionData["Q1"] = Q1
    quaternionData["Q2"] = Q2
    quaternionData["Q3"] = Q3
    print(quaternionData)

def extract_timestamp(Bytes):
    year = Bytes[20] + 2000
    month = Bytes[21]
    day = Bytes[22]
    hour = Bytes[23]
    minute = Bytes[24]
    second = Bytes[25]
    millisecond = Bytes[26] + (Bytes[27] << 8)
    return datetime.datetime(year, month, day, hour, minute, second, millisecond * 1000)

# main function
def main(file=None, appendix=None):
    with open(file, 'rb') as f:
        # read the file as a list of bytes
        file = f.read()
        print(f'number of bytes: {len(file)}, Number of records(number of bytes / 28): {len(file) / 28}')
        try:
            convert(file)
        except Exception as e:
            # raise the exception
            raise e
        finally:
            print(f'data_shape: {data.shape}')
            if appendix is not None:
                save_file = data["timestamp"].iloc[-1].strftime("%Y-%m-%d_%H-%M-%S") + f'_{appendix}.pkl'
            else:
                save_file = data["timestamp"].iloc[-1].strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
            print(f'save_file: {save_file}')
            # save the data as Pickle file
            data.to_pickle(save_file)

if __name__ == '__main__':
    # get the file name from the first argument when running the script
    file = sys.argv[1]
    if len(sys.argv) == 3:
        appendix = sys.argv[2]
    main(file, appendix)








