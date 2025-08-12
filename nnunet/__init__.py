from __future__ import absolute_import
print("\n\nPlease cite the following paper when using nnUNet:\n\nIsensee, F., Jaeger, P.F., Kohl, S.A.A. et al. "
      "\"nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.\" "
      "Nat Methods (2020). https://doi.org/10.1038/s41592-020-01008-z\n\n")
print("If you have questions or suggestions, feel free to open an issue at https://github.com/MIC-DKFZ/nnUNet\n")

print("@@ This is a modified working framework for evalution")
from . import *

import sys,os
# path = os.path.dirname(
#       os.path.abspath(__file__)
# )

# print(path)
path =os.path.abspath(
      os.path.join(os.path.dirname("__file__"), 
      os.path.pardir))
sys.path.insert(0, path)

