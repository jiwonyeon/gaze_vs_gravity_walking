import numpy as np
import pandas as pd
import matplotlib as plt

def xy_axis_format(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move left spine to x=0 with an offset
    ax.spines['left'].set_position(('axes', -0.02))  # 2% to the left of the axes
    ax.spines['bottom'].set_position(('axes', -0.02))  # 2% below the axes
    