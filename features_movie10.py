from tribev2.studies.algonauts2025 import Algonauts2025
from neuralset.events import transforms
import numpy as np 
from tribev2 import TribeModel
from tribev2.demo_utils import get_audio_and_text_events

study = Algonauts2025(path="/Brain/public/datasets/neuromod/",query = "task in ['movie10'] and subject == 'Algonauts2025/sub-01' ",infra={"backend": "Cached", "folder": "/Brain/public/datasets/neuromod/cache"},infra_timelines={"cluster": None}) ## the code then looks for path + 'download/algonauts_2025.competitors' 

summary = study.study_summary(apply_query=True)
print(summary[["subject", "timeline"]].to_string())

events = study.run()

query = transforms.QueryEvents(query='movie == "bourne" ')
events = query(events)

query = transforms.QueryEvents(query='type != "Fmri" ')
events = query(events)
print(f"After QueryEvents: {len(events)} events, types: {events.type.unique().tolist()}")

print(events[["type", "start", "duration", "timeline"]].head(8).to_string())


events_forinference = get_audio_and_text_events(events)

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/Brain/private/nfarrugi/",device="cuda")

preds = model.predict(events_forinference)
np.savez_compressed('testpreds.npz',preds=preds)