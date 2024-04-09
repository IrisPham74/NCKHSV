import os
import sys

dir = os.path.abspath(os.path.dirname(__file__))
# print(dir)
sys.path.append(dir)
# print(path)
from models.C2PNet import C2PNet
