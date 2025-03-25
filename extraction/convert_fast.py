import numpy as np
import pandas as pd
import datetime
import sys
from tqdm import tqdm
import os


def convert(file):
    """
    Converts raw binary data into a structured pandas DataFrame using batch processing.
    """
    data_list = []
    magnetic_list = []
    quaternion_list = []

    temp_bytes = bytearray()

    for byte in tqdm(file, desc="Converting file to DataFrame"):
        temp_bytes.append(byte)

        if len(temp_bytes) == 1 and temp_bytes[0] != 0x55:
            temp_bytes.clear()
            continue
        if len(temp_bytes) == 2 and temp_bytes[1] not in (0x61, 0x71):
            temp_bytes.pop(0)
            continue
        if len(temp_bytes) == 28:
            record = process_data(temp_bytes)
            if record:
                if record["type"] == "sensor":
                    data_list.append(record["data"])
                elif record["type"] == "magnetic":
                    magnetic_list.append(record["data"])
                elif record["type"] == "quaternion":
                    quaternion_list.append(record["data"])
            temp_bytes.clear()

    # Convert collected data into DataFrames
    data_df = pd.DataFrame(data_list,
                           columns=["AccX", "AccY", "AccZ", "AsX", "AsY", "AsZ", "AngX", "AngY", "AngZ", "timestamp"])
    magnetic_df = pd.DataFrame(magnetic_list, columns=["HX", "HY", "HZ"])
    quaternion_df = pd.DataFrame(quaternion_list, columns=["Q0", "Q1", "Q2", "Q3"])

    return data_df, magnetic_df, quaternion_df


def process_data(Bytes):
    """
    Extracts sensor, magnetic field, and quaternion data from raw bytes.
    """
    packet_type = Bytes[1]

    if packet_type == 0x61:  # Sensor Data
        Ax, Ay, Az = np.frombuffer(Bytes[2:8], dtype=np.int16) / 32768 * 16
        Gx, Gy, Gz = np.frombuffer(Bytes[8:14], dtype=np.int16) / 32768 * 2000
        AngX, AngY, AngZ = np.frombuffer(Bytes[14:20], dtype=np.int16) / 32768 * 180
        timestamp = extract_timestamp(Bytes)

        return {"type": "sensor", "data": [Ax, Ay, Az, Gx, Gy, Gz, AngX, AngY, AngZ, timestamp]}

    elif packet_type == 0x71:
        sub_type = Bytes[2]

        if sub_type == 0x3A:  # Magnetic Field
            Hx, Hy, Hz = np.frombuffer(Bytes[4:10], dtype=np.int16) / 120
            return {"type": "magnetic", "data": [Hx, Hy, Hz]}

        elif sub_type == 0x51:  # Quaternion Data
            Q0, Q1, Q2, Q3 = np.frombuffer(Bytes[4:12], dtype=np.int16) / 32768
            return {"type": "quaternion", "data": [Q0, Q1, Q2, Q3]}

    return None  # Ignore other types


def extract_timestamp(Bytes):
    """
    Extracts a timestamp from the byte stream.
    """
    return datetime.datetime(
        year=Bytes[20] + 2000, month=Bytes[21], day=Bytes[22],
        hour=Bytes[23], minute=Bytes[24], second=Bytes[25],
        microsecond=(Bytes[26] + (Bytes[27] << 8)) * 1000
    )


def main(file=None, appendix=None):
    """
    Reads the binary file, processes it, and saves the parsed data to a pickle file.
    """
    with open(file, 'rb') as f:
        file_bytes = f.read()
        print(f'Number of bytes: {len(file_bytes)}, Records (bytes / 28): {len(file_bytes) / 28}')
        data_df, magnetic_df, quaternion_df = convert(file_bytes)
        print(f'Data shape: {data_df.shape}')
        timestamp = data_df["timestamp"].iloc[-1].strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs("pd", exist_ok=True)
        save_file = os.path.join("pd", f"{timestamp}_{appendix}.pkl" if appendix else f"{timestamp}.pkl")
        print(f'Saving file: {save_file}')
        data_df.to_pickle(save_file)


if __name__ == '__main__':
    file = sys.argv[1]
    appendix = sys.argv[2] if len(sys.argv) > 2 else None
    main(file, appendix)
