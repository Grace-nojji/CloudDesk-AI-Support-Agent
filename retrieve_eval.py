import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from retrieval_pipeline import *

cfg = load_config()
vstore = get_vector_store(cfg)
client = InferenceClient(api_key=cfg['hf_token'] if cfg.get('hf_token') else None)
