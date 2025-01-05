import logging
from auto_prep.utils import config

config.update(log_level=logging.DEBUG, raport_decimal_precision=2)

import numpy as np

from auto_prep.prep import AutoPrep
from sklearn.datasets import fetch_openml

# Load your dataset
data = fetch_openml(name="titanic", version=1, as_frame=True, parser="auto").frame
data["survived"] = data["survived"].astype(np.uint8)

# Create and run pipeline
pipeline = AutoPrep()
pipeline.run(data, target_column="survived")
