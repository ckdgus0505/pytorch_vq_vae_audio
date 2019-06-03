from torch.utils import data
import torch
import glob
import os
import random
import librosa
import numpy as np

# a repo with fast generation algorithm: https://github.com/vincentherrmann/pytorch-wavenet
class VCTK(data.Dataset):
    def __init__(self,
                 root,
                 receptive_field,
                 segment_length=16126,
                 chunk_size=1000,
                 classes=256):
        """
        root - path to VCTK directory
        segment_length - size of audio segment
        chunk_size - number of audios to cache together
        classes - number of classes for mu-law encoding
        """
        self.root = root
        self.segment_length = segment_length
        self.chunk_size = chunk_size
        self.classes = classes
        self.receptive_field = receptive_field
        self.cached_pt = 0
        self.processed_folder = 'processed'
        self.num_samples = 0
        self.num_speakers = 0
        
        if not os.path.exists(os.path.join(self.root, self.processed_folder)):
            self.preprocess()
        
        #read dataset length from file
        self._read_info()
        
        self.audios = torch.load(os.path.join(
            self.root, self.processed_folder, "vctk_{:04d}.pt".format(self.cached_pt)))
        
    def __getitem__(self, index):
        """
        Take audio number idx and extract random segment of length segment_length
        """
        if self.cached_pt != index // self.chunk_size:
            self.cached_pt = int(index // self.chunk_size)
            self.audios = torch.load(os.path.join(
                self.root, self.processed_folder, "vctk_{:04d}.pt".format(self.cached_pt)))
        
        index = index % self.chunk_size
        audio, speaker_id = self.audios[index]        
        while audio.shape[0] < self.segment_length:
            index += 1
            index = index % self.chunk_size
            audio, speaker_id = self.audios[index]
         #select random segment

        max_audio_start = audio.shape[0] - self.segment_length
        audio_start = random.randint(0, max_audio_start)
        audio = audio[audio_start:audio_start+self.segment_length]
        
        #divide into input and target
        audio = torch.from_numpy(audio)
        ohe_audio = torch.FloatTensor(self.classes, self.segment_length).zero_()
        ohe_audio.scatter_(0, audio.unsqueeze(0), 1.)
        
        #target need not be one-hot encoded
        target = audio[self.receptive_field:]
        
        speaker_id = torch.from_numpy(np.array(speaker_id)).unsqueeze(0).unsqueeze(0)
        ohe_speaker = torch.FloatTensor(self.num_speakers, 1).zero_()
        ohe_speaker.scatter_(0, speaker_id, 1.)
        
        return ohe_audio, target, ohe_speaker

    def __len__(self):
        return self.num_samples
    
    def _write_info(self, num_items, num_speakers):
        info_path = os.path.join(
            self.root, self.processed_folder, "vctk_info.txt")
        with open(info_path, "w") as f:
            f.write("num_samples,{}\n".format(num_items))
            f.write("num_speakers,{}\n".format(num_speakers))
    
    def _read_info(self):
        info_path = os.path.join(
            self.root, self.processed_folder, "vctk_info.txt")
        with open(info_path, "r") as f:
            self.num_samples = int(f.readline().split(",")[1])
            self.num_speakers = int(f.readline().split(",")[1])
    
    def preprocess(self):
        """
        Read all audio files in the VCTK/wav48 folder
        Shuffle and group into chunks of length chunk_size. 
        """
        print("Preprocessing...")
        
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
                
        #get a list of tuples (audiofile, speaker)
        audios = []
        for speaker_id, speaker in enumerate(glob.glob(os.path.join(self.root, 'wav48/*'))):
            print(speaker_id)
            for audiofile in glob.glob(speaker + '/*.wav'):
                try:
                    audio, sr = librosa.load(audiofile)
                except Exception as e:
                    print(e, audiofile)
                if sr != 22050:
                    raise ValueError("{} SR of {} not equal to 22050".format(sr, audiofile))
                normalized = librosa.util.normalize(audio) #divide max(abs(audio))
                quantized = self.quantize_data(normalized, self.classes)
                audios.append((quantized, speaker_id))
        
        random.shuffle(audios)
        
        for i in range((len(audios) - 1) // self.chunk_size + 1):
            batch = audios[self.chunk_size * i : self.chunk_size * (i + 1)]
            torch.save(batch,
                       os.path.join(self.root, 
                                    self.processed_folder,
                                    "vctk_{:04d}.pt".format(i)
                                   )
                      )
            
        self._write_info(len(audios), speaker_id + 1)
        print("Preprocessing done!")
    
    def mu_law_encode(self, data, mu):
        mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
        return mu_x
    
    def mu_law_decode(self, mu_x, mu):
        data = np.sign(mu_x) * (1 / mu) * ((1 + mu) ** np.abs(mu_x) - 1)
        return data                                                                                                                                                                                      
    def quantize_data(self, data, classes):
        mu_x = self.mu_law_encode(data, classes)
        bins = np.linspace(-1, 1, classes)
        quantized = np.digitize(mu_x, bins) - 1
        return quantized
#https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
#see also: WaveNet_demo.ipynb in the same repo
