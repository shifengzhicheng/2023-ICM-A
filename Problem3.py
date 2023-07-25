# 载入必要库
import jieba
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt 
import pyecharts.options as opts
from pyecharts.charts import WordCloud
from pyecharts.charts import Bar
import re
#logging
import warnings
warnings.filterwarnings('ignore')

#读入数据集
data = pd.read_excel('./Appendix I.xlsx')
data.head(10)