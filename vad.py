import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.io.wavfile import write

class VAD():
    
    def __init__(self, wave_input_filename):
        self.read_wav(wave_input_filename)
        self.window = 0.02 #20 ms
        self.hop = 0.01 #10ms
        self.speech_start_band = 200
        self.speech_end_band = 500
        # self.threshold = 0.794
        self.timestamps = []
        self.speech_prob = []
        self.filename = wave_input_filename.split("/")[-1]
           
    def read_wav(self, wave_file):
        self.rate, self.data = wf.read(wave_file)
        self.channels = len(self.data.shape)
        self.filename = wave_file
        pass

    def sigmoid_by_thres(self,x, thres = 0.7855, dx = 0.791 - 0.78):
        return 1 / (1 + np.exp(-(x - thres)/dx))
    
    
    def GetFreq(self, audio_data):
        data_freq = np.fft.fftfreq(len(audio_data),1.0/self.rate)
        return data_freq    
    
    def getAmpl(self, audio_data):
        data_ampl = np.abs(np.fft.fft(audio_data))
        return data_ampl
        
    def getEnerg(self, data):
        data_amplitude = self.getAmpl(data)
        data_energy = data_amplitude ** 2
        return data_energy
        
    
    def getFreqandEnergy(self, data_freq, data_energy):
        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2
        return energy_freq
    
    def get_energy_with_freq(self, data):
        data_freq = self.GetFreq(data)
        data_energy = self.getEnerg(data)
        energy_freq = self.getFreqandEnergy(data_freq, data_energy)
        return energy_freq
    
    def sum_in_band(self,energy_frequencies, start_band, end_band):
        sum_energy = 0
        for f in energy_frequencies.keys():
            if start_band<f<end_band:
                sum_energy += energy_frequencies[f]
        return sum_energy
    
    def max_filter(self, speech_prob, win_size = 10):
        smooth_prob = np.zeros_like(speech_prob)
        half = int(win_size/2)
        smooth_prob[0:half + 1] = np.max(speech_prob[0:half + 1])
        smooth_prob[-half - 1:-1] = np.max(speech_prob[-half - 1:-1])
        for i in range(half + 1,len(smooth_prob) - half - 1):
            smooth_prob[i - half:i+half + 1] = np.max(speech_prob[i - half :i+half + 1])
        return smooth_prob

      
       
    def detect_speech(self):
        window = int(self.rate * self.window)
        hop = int(self.rate * self.hop)
        data = self.data
        sample_start = 0
        start_band = self.speech_start_band
        end_band = self.speech_end_band
        speech_prob = []
        timstamps = []
        while (sample_start < (len(data) - window)):
            sample_end = sample_start + window
            if sample_end>=len(data): sample_end = len(data)-1
            data_window = data[sample_start:sample_end]
            energy_freq = self.get_energy_with_freq(data_window)
            sum_voice_energy = self.sum_in_band(energy_freq, start_band, end_band)
            sum_full_energy = sum(energy_freq.values())
            speech_ratio = sum_voice_energy/sum_full_energy
            speech_prob.append(speech_ratio)
            timstamps.append( (sample_start + sample_end)/2 / self.rate)
            sample_start += hop

        self.timestamps = timstamps
        self.speech_prob = self.sigmoid_by_thres(self.max_filter(speech_prob, win_size=101))
        pass
    
    def plot_wav_and_prob(self, path_to_save):

        plt.plot(np.arange(len(self.data))/self.rate, self.data/np.max(self.data), 'b', label="signal")
        max_prob = np.max(self.speech_prob)
        if  max_prob < 1e-3:
            max_prob = "0.000000"
        else:
            max_prob = str(max_prob)
        max_prob = max_prob[0:5]
        plt.plot(self.timestamps, self.speech_prob, 'r', label = "speech probobility")
        plt.xlabel('time, s')
        plt.title(self.filename + ", max probability: " + max_prob)
        plt.legend(loc='lower center')
        plt.savefig(path_to_save)
        plt.show()
        pass

    def save_wav(self, path_to_save):
        f2 = interp1d(self.timestamps, self.speech_prob, kind='cubic', fill_value="extrapolate")
        wav = 10000*f2(np.arange(len(self.data))/self.rate)
        wav[wav < 0] = np.median(wav)
        write(path_to_save, self.rate, wav.astype(np.int16))
 
