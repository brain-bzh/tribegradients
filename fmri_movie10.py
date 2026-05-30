import torch

from tribev2.studies.algonauts2025 import Algonauts2025
from neuralset.events import transforms
import numpy as np 
from tribev2 import TribeModel
from tribev2.demo_utils import get_audio_and_text_events
import argparse
import os 

parser = argparse.ArgumentParser(description='Generate predictions for Algonauts2025 movie data')
parser.add_argument('--movie', type=str, default='bourne', help='Movie name to process')
parser.add_argument('--output', type=str, default='./', help='Output path for predictions')
parser.add_argument('--data', type=str, default='./', help='path for dataset')
parser.add_argument('--features', action='store_true', help='Extract internal features')
args = parser.parse_args()

study = Algonauts2025(path=args.data,query = "task in ['movie10'] and subject == 'Algonauts2025/sub-01' ",infra_timelines={"cluster": None}) ## the code then looks for path + 'download/algonauts_2025.competitors' 

summary = study.study_summary(apply_query=True)
print(summary[["subject", "timeline"]].to_string())

events = study.run()

movie_name = args.movie
query = transforms.QueryEvents(query=f'movie == "{movie_name}" and type == "Fmri" ')
events = query(events)

print(f"QueryEvents with fMRI: {len(events)} events, types: {events.type.unique().tolist()}")

print(events[["type", "start", "duration", "timeline"]].head(8).to_string())

from neuralset.extractors import FmriExtractor, SurfaceProjector

extractor = FmriExtractor(projection=SurfaceProjector(mesh="fsaverage5"))
fmri_data = extractor(events,start=0,duration=10.0)
print(f"Extracted fMRI data shape: {fmri_data.shape}")