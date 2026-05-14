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
parser.add_argument('--features', action='store_true', help='Extract internal features')
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

if args.features:
    from collections import defaultdict

    fmri_enc = model.__pydantic_private__['_model']   # FmriEncoderModel
    fmri_enc.eval()

    # ── Register hooks ────────────────────────────────────────────────────────────
    features = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            features[name].append(out.detach().cpu())
        return hook

    # Modality projectors (output shape: (batch, time, 384) each)
    for modality in ['text', 'audio', 'video']:
        m = fmri_enc.projectors[modality]
        hooks.append(m.register_forward_hook(make_hook(f'projector.{modality}')))

    # Each transformer layer: even indices = Attention, odd = FeedForward
    # encoder.layers[i][1] is the actual module (index [0] is the norm, [2] is Residual)
    for i, layer_block in enumerate(fmri_enc.encoder.layers):
        submodule = layer_block[1]  # Attention or FeedForward
        layer_type = 'attn' if i % 2 == 0 else 'ffn'
        transformer_layer_idx = i // 2  # 0-7
        name = f'encoder.layer{transformer_layer_idx}.{layer_type}'
        hooks.append(submodule.register_forward_hook(make_hook(name)))

    # Final norm, low-rank head, and predictor
    hooks.append(fmri_enc.encoder.final_norm.register_forward_hook(make_hook('encoder.final_norm')))
    hooks.append(fmri_enc.low_rank_head.register_forward_hook(make_hook('low_rank_head')))
    hooks.append(fmri_enc.predictor.register_forward_hook(make_hook('predictor')))

    print(f"Registered {len(hooks)} hooks")
    with torch.no_grad():
        preds,segments = model.predict(events_forinference)
    
        # ── Remove hooks and save ─────────────────────────────────────────────────────
    for h in hooks:
        h.remove()

    all_features_dict = {}
    for layer_name, tensors in features.items():
        stacked = torch.cat(tensors, dim=0)
        safe_name = layer_name.replace('.', '_')
        all_features_dict[safe_name] = stacked.numpy()
        print(f"  {layer_name:40s}  shape={{tuple(stacked.shape)}}")

    np.savez_compressed(os.path.join(args.output, f'{args.movie}_predictions.npz'), preds=preds, segments=segments)
    np.savez_compressed(os.path.join(args.output, f'{args.movie}_allfeatures.npz'), **all_features_dict)
    print(f"Predictions saved to {os.path.join(args.output, f'{args.movie}_predictions.npz')}")
    print(f"Features saved to {os.path.join(args.output, f'{args.movie}_allfeatures.npz')}")

else:
    print("Running inference without feature extraction...")
    preds,segments = model.predict(events_forinference)

    np.savez_compressed(os.path.join(args.output, f'{args.movie}_predictions.npz'), preds=preds, segments=segments)
    print(f"Predictions saved to {os.path.join(args.output, f'{args.movie}_predictions.npz')}")