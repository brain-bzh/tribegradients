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
args = parser.parse_args()

study = Algonauts2025(path="/Brain/public/datasets/neuromod/",query = "task in ['movie10'] and subject == 'Algonauts2025/sub-01' ",infra={"backend": "Cached", "folder": "/Brain/public/datasets/neuromod/cache"},infra_timelines={"cluster": None}) ## the code then looks for path + 'download/algonauts_2025.competitors' 

summary = study.study_summary(apply_query=True)
print(summary[["subject", "timeline"]].to_string())

events = study.run()

movie_name = args.movie
query = transforms.QueryEvents(query=f'movie == "{movie_name}" ')
events = query(events)

query = transforms.QueryEvents(query='type != "Fmri" ')
events = query(events)
print(f"After QueryEvents: {len(events)} events, types: {events.type.unique().tolist()}")

print(events[["type", "start", "duration", "timeline"]].head(8).to_string())


events_forinference = get_audio_and_text_events(events)

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/Brain/private/nfarrugi/",device="cuda")

preds,segments = model.predict(events_forinference)

np.savez_compressed(os.path.join(args.output, f'{args.movie}_predictions.npz'), preds=preds, segments=segments)