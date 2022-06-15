import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt

class VoiceActivityDetector():
    """ Use signal energy to detect voice activity in wav file """
    
    def __init__(self, wave_input_filename):
        self._read_wav(wave_input_filename)._convert_to_mono()
        self.sample_window = 0.02 #20 ms
        self.sample_overlap = 0.01 #10ms
        self.speech_start_band = 200
        self.speech_end_band = 500
        # self.threshold = 0.794
        self.timestamps = []
        self.speech_prob = []
        self.filename = wave_input_filename.split("/")[-1]
           
    def _read_wav(self, wave_file):
        self.rate, self.data = wf.read(wave_file)
        self.channels = len(self.data.shape)
        self.filename = wave_file
        return self
    
    def _convert_to_mono(self):
        if self.channels == 2 :
            self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)
            self.channels = 1
        return self
    
    def _calculate_frequencies(self, audio_data):
        data_freq = np.fft.fftfreq(len(audio_data),1.0/self.rate)
        # data_freq = data_freq[1:]
        return data_freq    
    
    def _calculate_amplitude(self, audio_data):
        data_ampl = np.abs(np.fft.fft(audio_data))
        # data_ampl = data_ampl[1:]
        return data_ampl
        
    def _calculate_energy(self, data):
        data_amplitude = self._calculate_amplitude(data)
        data_energy = data_amplitude ** 2
        return data_energy
        
    
    def _connect_energy_with_frequencies(self, data_freq, data_energy):
        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2
        return energy_freq
    
    def _calculate_normalized_energy(self, data):
        data_freq = self._calculate_frequencies(data)
        data_energy = self._calculate_energy(data)
        energy_freq = self._connect_energy_with_frequencies(data_freq, data_energy)
        return energy_freq
    
    def _sum_energy_in_band(self,energy_frequencies, start_band, end_band):
        sum_energy = 0
        for f in energy_frequencies.keys():
            if start_band<f<end_band:
                sum_energy += energy_frequencies[f]
        return sum_energy
    
    def _max_filter(self, speech_prob, win_size = 10):
        smooth_prob = np.zeros_like(speech_prob)
        half = int(win_size/2)
        smooth_prob[0:half + 1] = np.max(speech_prob[0:half + 1])
        smooth_prob[-half - 1:-1] = np.max(speech_prob[-half - 1:-1])
        for i in range(half + 1,len(smooth_prob) - half - 1):
            smooth_prob[i - half:i+half + 1] = np.max(speech_prob[i - half :i+half + 1])
            
        return smooth_prob

      
       
    def detect_speech(self):
        sample_window = int(self.rate * self.sample_window)
        sample_overlap = int(self.rate * self.sample_overlap)
        data = self.data
        sample_start = 0
        start_band = self.speech_start_band
        end_band = self.speech_end_band
        speech_prob = []
        timstamps = []
        while (sample_start < (len(data) - sample_window)):
            sample_end = sample_start + sample_window
            if sample_end>=len(data): sample_end = len(data)-1
            data_window = data[sample_start:sample_end]
            energy_freq = self._calculate_normalized_energy(data_window)
            sum_voice_energy = self._sum_energy_in_band(energy_freq, start_band, end_band)
            sum_full_energy = sum(energy_freq.values())
            speech_ratio = sum_voice_energy/sum_full_energy
            # Hipothesis is that when there is a speech sequence we have ratio of energies more than Threshold
            speech_prob.append(speech_ratio)
            timstamps.append(self.rate * (sample_start + sample_end)/2)
            sample_start += sample_overlap

        self.timestamps = timstamps
        self.speech_prob = self._max_filter(speech_prob, win_size=101)
        pass
    
    def plot_wav_and_prob(self, path_to_save):

        plt.plot(self.rate * np.arange(len(self.data)), self.data/np.max(self.data), 'b', label="signal")
        max_prob = str(np.max(self.speech_prob))
        max_prob = max_prob[0:5]
        plt.plot(self.timestamps, self.speech_prob, 'r', label = "speech probobility")
        plt.xlabel('time, s')
        plt.title(self.filename + ", max probability: " + max_prob)
        plt.legend(loc='lower center')
        plt.savefig(path_to_save)
        plt.show()
        pass
 
