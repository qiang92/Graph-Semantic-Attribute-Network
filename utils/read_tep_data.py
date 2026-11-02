import numpy as np
from glob import glob

# Core sensors: XMEAS 1-22 (indices 0-21) + XMV 1-11 (indices 41-51) = 33 variables
CORE_SENSORS = list(range(0, 22)) + list(range(41, 52))  # 33 sensors

def apply_sma(x, window=3):
    """Simple Moving Average smoothing along the time axis (axis=0)"""
    if window <= 1:
        return x
    smoothed = np.copy(x)
    for i in range(window - 1, len(x)):
        smoothed[i] = np.mean(x[i - window + 1:i + 1], axis=0)
    return smoothed

def create_tep_data_with_temporal(data_list, attribute_matrix, seen_classes, unseen_classes, seq_len=10):
    """
    重写版本：返回真实的滑动时序窗口数据 (N, 52, Seq_Len)
    每个样本的真实历史窗口被保留下来。
    """
    seen_idx = [i - 1 for i in seen_classes]
    unseen_idx = [i - 1 for i in unseen_classes]

    # Build temporal sliding windows for seen classes
    X_train_list = []
    y_train_list = []
    A_train_list = []

    for idx in seen_idx:
        data = data_list[idx]  # (480, 52)
        y = np.full(data.shape[0], idx)
        attr = np.tile(attribute_matrix[idx], (data.shape[0], 1))  # (480, 20)

        # Create sliding windows: shape (480-seq_len+1, 52, seq_len)
        num_samples = data.shape[0] - seq_len + 1
        for t in range(seq_len - 1, data.shape[0]):
            window = data[t - seq_len + 1:t + 1]  # (seq_len, 52)
            X_train_list.append(window.T)  # (52, seq_len)
            y_train_list.append(y[t])
            A_train_list.append(attribute_matrix[idx])

    X_train = np.stack(X_train_list, axis=0)  # (N, 52, seq_len)
    y_train = np.array(y_train_list)
    A_train = np.stack(A_train_list, axis=0)  # (N, 20)

    # Build temporal sliding windows for unseen classes
    X_test_list = []
    y_test_list = []
    A_test_list = []

    for idx in unseen_idx:
        data = data_list[idx]  # (480, 52)
        y = np.full(data.shape[0], idx)
        attr = np.tile(attribute_matrix[idx], (data.shape[0], 1))  # (480, 20)

        # Create sliding windows: shape (480-seq_len+1, 52, seq_len)
        for t in range(seq_len - 1, data.shape[0]):
            window = data[t - seq_len + 1:t + 1]  # (seq_len, 52)
            X_test_list.append(window.T)  # (52, seq_len)
            y_test_list.append(y[t])
            A_test_list.append(attribute_matrix[idx])

    X_test = np.stack(X_test_list, axis=0)  # (M, 52, seq_len)
    y_test = np.array(y_test_list)
    A_test = np.stack(A_test_list, axis=0)  # (M, 20)

    return X_train, y_train, A_train, None, X_test, y_test, None, A_test, attribute_matrix



def load_tep_data(data_path='TEP_data/*.dat', use_core_sensors=False, sma_window=0, dual_stream=False):
    """Load TEP data with optional dual-stream support.

    Args:
        dual_stream: If True, returns (data_list_raw, data_list_filtered) for both streams
    """
    files = sorted(glob(data_path))[:15]

    if dual_stream:
        # Load raw data for Stream A
        data_list_raw = []
        # Load filtered data for Stream B
        data_list_filtered = []
        for file in files:
            data = np.loadtxt(file)
            # Stream A: raw 52-dim
            data_list_raw.append(data)
            # Stream B: core 33-dim + SMA
            data_filtered = data[:, CORE_SENSORS]
            if sma_window > 1:
                data_filtered = apply_sma(data_filtered, sma_window)
            data_list_filtered.append(data_filtered)
        return data_list_raw, data_list_filtered
    else:
        # Original single-stream loading
        data_list = []
        for file in files:
            data = np.loadtxt(file)
            if use_core_sensors:
                data = data[:, CORE_SENSORS]
            if sma_window > 1:
                data = apply_sma(data, sma_window)
            data_list.append(data)
        return data_list

def get_data(data_list, class_indices, attribute_matrix):
    X, y, A = [], [], []
    for idx in class_indices:
        data = data_list[idx]
        X.append(data)
        y += [idx] * data.shape[0]
        A += [attribute_matrix[idx]] * data.shape[0]
    return np.vstack(X), np.array(y), np.vstack(A)

def create_tep_data(data_list, attribute_matrix, seen_classes, unseen_classes):
    # 类索引转换为0-base索引
    seen_idx = [i - 1 for i in seen_classes]
    unseen_idx = [i - 1 for i in unseen_classes]

    X_train, y_train, A_train = get_data(data_list, seen_idx, attribute_matrix)
    X_test, y_test, A_test = get_data(data_list, unseen_idx, attribute_matrix)

    train_attr_matrix = attribute_matrix[seen_idx]
    test_attr_matrix = attribute_matrix[unseen_idx]

    return X_train, y_train, A_train, train_attr_matrix, \
           X_test, y_test, A_test, test_attr_matrix, attribute_matrix

# 示例用法：
if __name__ == '__main__':
    # 假设已准备好 attribute_matrix.npy
    attribute_matrix = np.load('../data/TEP_attribute_matrix.npy')  # shape: (15, 20)
    data_list = load_tep_data()

    seen_classes = [2,3,4,5,7,8,9,10,11,12,13,15]
    unseen_classes = [1,6,14]

    X_train, y_train, A_train, train_attr_matrix, \
    X_test, y_test, A_test, test_attr_matrix, attr_matrix = create_tep_data(
        data_list, attribute_matrix, seen_classes, unseen_classes
    )
