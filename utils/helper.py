import pandas as pd

def get_primary_speaker_id(filepath: str) -> str:
    """
    Given a path like:
        "/path/to/fileid_12345_spkid_00054-11429-04842.wav"
    or
        "/path/to/fileid_12345_spkid_07406.wav"
    This function returns:
        "00054"  (for multiple speakers)
    or
        "07406"  (for single speaker)
    """
    spkid_part = filepath.split("spkid_")[1]
    spkid_part = spkid_part.replace(".wav", "")
    primary_speaker_id = spkid_part.split("-")[0]

    return primary_speaker_id

def add_speaker_id(df):
    df['speaker_id'] = df['clean_file_path'].apply(get_primary_speaker_id)
    return df

def label_speaker_id(df):
    add_speaker_id(df)
    unique_speakers = df['speaker_id'].unique()
    speaker2idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
    df['label'] = df['speaker_id'].map(speaker2idx)
    return df

# Example usage:
if __name__ == "__main__":
    df = pd.read_csv('/work/hdd/bdql/thsieh/TGIF-Dataset/tr/metadata.csv')
    df = label_speaker_id(df)
    print(df)