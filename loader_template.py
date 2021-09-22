import sys
from pathlib import Path
PATH_ROOT = Path(__file__).parent

import pandas as pd
import numpy as np
import re

np.random.seed(42)

def dataLoader():
    df = pd.DataFrame( np.random.randint(0,10,(100,5)) )
    return df
