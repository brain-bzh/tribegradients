from xml.parsers.expat import model


def main():
    print("Test run")
    print("Downloading YouTube short and processing it with TribeModel...")
    import subprocess

    # Download the YouTube short
    url = "https://www.youtube.com/shorts/3YxtTAMNJhM"
    output_path = "video.mp4"
    subprocess.run(["yt-dlp", "-o", output_path, url], check=True)

    from tribev2 import TribeModel

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

    df = model.get_events_dataframe(audio_path="video.wav")
    print(df.head())
    #preds, segments = model.predict(events=df)
    # print(preds.shape)  # (n_timesteps, n_vertices)


if __name__ == "__main__":
    main()
