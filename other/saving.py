
import cv2,os
import warnings,sys,os
import os
import warnings
import logging
import numpy as np
import cv2

# ---------------- SETTINGS ---------------- #
rn = 0      # 0 = do not save/load intermediate files
            # 1 = save/load intermediate files

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Create folders automatically
os.makedirs("saved data", exist_ok=True)
os.makedirs("image_results", exist_ok=True)

# ---------------- OPTIONAL IMPORTS ---------------- #
try:
    from deep_joint_segmentation import (
        deep_joint_based_image_segmentation,
        conv_deep_joint_based_image_segmentation
    )
except ImportError:
    print("deep_joint_segmentation.py not found.")
    deep_joint_based_image_segmentation = None
    conv_deep_joint_based_image_segmentation = None

# ---------------- WIENER FILTER ---------------- #
from scipy.signal import wiener

def preprocessing_wienerc(im):
    try:
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        output = wiener(im, (3, 3))
        return output.astype('uint8')

    except Exception as e:
        print("Wiener preprocessing error:", e)
        return im

# ---------------- SAFE NPY LOADING ---------------- #
def safe_load(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)

    print(f"File not found: {path}")
    return None

# ---------------- MAIN PROGRAM ---------------- #
print("saving.py execution started")

if rn == 1:
    data = safe_load("saved data/oimg_pth2.npy")

    if data is not None:
        print("Loaded successfully")
    else:
        print("Required file missing")

else:
    print("rn=0 -> Skipping loading of saved files")

print("Program executed successfully")