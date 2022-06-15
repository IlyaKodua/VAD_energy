from vad import VoiceActivityDetector
import glob
import os


if __name__ == "__main__":

    # path_to_save
    dir = "pics"
    os.mkdir("pics")
    list_files = glob.glob('/home/liya/study/mb/wavs_my' + '**/*.WAV', recursive=True)
    for f in list_files:
        v = VoiceActivityDetector(f)
        v.detect_speech()
        name = f.split("/")[-1].split(".")[0]
        v.plot_wav_and_prob(dir + "/" + name)
