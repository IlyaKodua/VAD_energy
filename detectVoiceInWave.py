from vad import VoiceActivityDetector
import glob
import os


if __name__ == "__main__":

# 4 noise 0.4489931273660462
# 6 speech 0.906390297803528
# 7 music 0.7905428903534732 
# 8 speech 0.871315123426183
# 11 speech 0.7950834410026628
# 13 speech 0.6127030494891371
# 14 noise 0.2962122201590884
    # path_to_save
    dir = "pics"
    os.mkdir("pics")
    list_files = glob.glob('/home/liya/study/mb/wavs_my' + '**/*.WAV', recursive=True)
    for f in list_files:
        v = VoiceActivityDetector(f)
        v.detect_speech()
        name = f.split("/")[-1].split(".")[0]
        v.plot_wav_and_prob(dir + "/" + name)
