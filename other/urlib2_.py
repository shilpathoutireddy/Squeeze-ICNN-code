import pandas as pd
# from other import Blockchain
# from other import pd_
import numpy as np
import warnings

try:
    from charset_normalizer import __version__ as charset_normalizer_version
except ImportError:
    charset_normalizer_version = None

try:
    from chardet import __version__ as chardet_version
except ImportError:
    chardet_version = None


def _check_cryptography(cryptography_version):
    # cryptography < 1.3.4
    try:
        cryptography_version = list(map(int, cryptography_version.split(".")))
    except ValueError:
        return

    if cryptography_version < [1, 3, 4]:
        warning = "Old version of cryptography ({}) may cause slowdown.".format(
            cryptography_version
        )


def data_save_block_chain():
    data = pd.read_csv('dataset/dataset.csv').values
    ### BlockChain
    from numpy import inf
    data[data == -inf] = 0  # change infinitive to 0


def request():
    def Retrive_data():
        data, lb = [], []
        datasets = ['Database//thyroid+disease//new-thyroid.data', 'Database//dermatology//dermatology.data']
        for dat in range(len(datasets)):
            if dat == 0:
                data_ = pd.read_csv(datasets[dat], header=None)
                data.append(np.array((data_.drop(columns=data_.columns[0], axis=1))))
                lb.append(np.array((data_.iloc[:, 0]) - 1))  # 3 lab
            else:
                data_ = (pd.read_csv(datasets[dat], header=None)).replace('?', 0)
                data.append(np.array((data_.drop(columns=data_.columns[-1], axis=1))).astype('float'))
                lb.append(np.array((data_.iloc[:, -1]) - 1))  # 6 lab

        return data, lb

    F,L = Retrive_data()
    return F,L

def request_open(url):
    def Retrive_data():
        data, lb = [], []
        datasets = ['Database//thyroid+disease//new-thyroid.data', 'Database//dermatology//dermatology.data']
        for dat in range(len(datasets)):
            if dat == 0:
                data_ = pd.read_csv(datasets[dat], header=None)
                data.append(np.array((data_.drop(columns=data_.columns[0], axis=1))))
                # np.array((data_.drop(columns=data_.columns[-1], axis=1)))
                lb.append(np.array((data_.iloc[:, 0]) - 1))  # 3 lab
            else:
                data_ = (pd.read_csv(datasets[dat], header=None)).replace('?', 0)
                data.append(np.array((data_.drop(columns=data_.columns[-1], axis=1))).astype('float'))
                lb.append(np.array((data_.iloc[:, -1]) - 1))  # 6 lab

        return data, lb
    F, L = Retrive_data()
    return F, L


def check_compatibility(urllib3_version, chardet_version, charset_normalizer_version):
    urllib3_version = urllib3_version.split(".")
    assert urllib3_version != ["dev"]  # Verify urllib3 isn't installed from git.

    # Sometimes, urllib3 only reports its version as 16.1.
    if len(urllib3_version) == 2:
        urllib3_version.append("0")

    # Check urllib3 for compatibility.
    major, minor, patch = urllib3_version  # noqa: F811
    major, minor, patch = int(major), int(minor), int(patch)
    # urllib3 >= 1.21.1, <= 1.26
    assert major == 1
    assert minor >= 21
    assert minor <= 26

    # Check charset_normalizer for compatibility.
    if chardet_version:
        major, minor, patch = chardet_version.split(".")[:3]
        major, minor, patch = int(major), int(minor), int(patch)
        # chardet_version >= 3.0.2, < 6.0.0
        assert (3, 0, 2) <= (major, minor, patch) < (6, 0, 0)
    elif charset_normalizer_version:
        major, minor, patch = charset_normalizer_version.split(".")[:3]
        major, minor, patch = int(major), int(minor), int(patch)
        # charset_normalizer >= 2.0.0 < 3.0.0
        assert (2, 0, 0) <= (major, minor, patch) < (3, 0, 0)
    else:
        raise Exception("You need either charset_normalizer or chardet installed")
