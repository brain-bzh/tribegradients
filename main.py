from xml.parsers.expat import model
import numpy as np

def main():
    print("Test run")
    print("Downloading YouTube short and processing it with TribeModel...")
    import subprocess
    import pandas as pd

    # Download the YouTube short
    url = "https://www.youtube.com/watch?v=uIxxyiiddgU"
    output_path = "video.mp4"
    subprocess.run(["yt-dlp", "-o", output_path, url], check=True)

    from tribev2 import TribeModel

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="/Brain/private/nfarrugi/",device="cuda")

    df = model.get_events_dataframe(video_path="video.mp4")
    df.to_csv('events.csv')
    #df = pd.read_csv('events.csv')
    print(df.head())
    preds, segments = model.predict(events=df)
    print(preds.shape)  # (n_timesteps, n_vertices)

    # save the predictions
    np.savez_compressed('testpreds.npz',preds=preds)


if __name__ == "__main__":
    main()
