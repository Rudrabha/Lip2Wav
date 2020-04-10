import synthesizer
from synthesizer import inference as sif
import numpy as np
import sys, cv2
from tqdm import tqdm

class Generator(object):
	def __init__(self,
				synthesizer_weights='logs/'):
		super(Generator, self).__init__()

		self.synthesizer = sif.Synthesizer(synthesizer_weights, verbose=False, manual_inference=True)

	def vc(self, sample, id):
		id_windows = [range(i, i + sif.hparams.T) for i in range(0, sample['till'], 
					sif.hparams.T - sif.hparams.overlap) if (i + sif.hparams.T <= sample['till'])]

		all_images = [[sample['folder'].format(id) for id in window] for window in id_windows]
		cnt = 0
		for window_fnames in tqdm(all_images):
			window = []
			for fname in window_fnames:
				img = cv2.imread(fname)
				img = cv2.resize(img, (sif.hparams.img_size, sif.hparams.img_size))
				window.append(img)

			images = np.asarray(window) / 255. # T x H x W x 3

			s = self.synthesizer.synthesize_spectrograms(images)[0]
			if cnt == 0:
				mel = s
			else:
				mel = np.concatenate((mel, s[:, sif.hparams.mel_overlap:]), axis=1)
			cnt = cnt + 1

		wav = self.synthesizer.griffin_lim(mel)
		sif.audio.save_wav(wav, 'logs/wavs/out{}.wav'.format(id), sr=sif.hparams.sample_rate)

samples = [{'folder' : '../test_data/_-DvuD8JWNau4-cut30/{}.jpg', 'till' : (768//sif.hparams.T) * sif.hparams.T, 
			'gt' : '../test_data/_-gSeEXvMZVWo-cut35/mels.npz'},

			{'folder' : '../test_data/_-FYdYkiIHvRQ-cut27/{}.jpg', 'till' : (768//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-kE9QyG6Bexw-cut16/mels.npz'},

			{'folder' : '../test_data/_-gSeEXvMZVWo-cut35/{}.jpg', 'till' : (708//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-kE9QyG6Bexw-cut22/mels.npz'},

			{'folder' : '../test_data/_-GTH1mNz_4jg-cut37/{}.jpg', 'till' : (226//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-_O3e2HdLFQ8-cut18/mels.npz'},

			{'folder' : '../test_data/_-kE9QyG6Bexw-cut16/{}.jpg', 'till' : (768//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-vydldXpA6ec-cut25/mels.npz'},




			{'folder' : '../test_data/_-kE9QyG6Bexw-cut22/{}.jpg', 'till' : (744//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-XuCL9HBm_HQ-cut11/mels.npz'},

			{'folder' : '../test_data/_-_O3e2HdLFQ8-cut18/{}.jpg', 'till' : (768//sif.hparams.T) * sif.hparams.T, 
			'gt' : '../test_data/_-gSeEXvMZVWo-cut35/mels.npz'},

			{'folder' : '../test_data/_-olagryvWnvA-cut15/{}.jpg', 'till' : (768//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-kE9QyG6Bexw-cut16/mels.npz'},


			{'folder' : '../test_data/_-VREYgGjjox0-cut43/{}.jpg', 'till' : (768//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-vydldXpA6ec-cut25/mels.npz'},

			{'folder' : '../test_data/_-vydldXpA6ec-cut25/{}.jpg', 'till' : (768//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-XuCL9HBm_HQ-cut11/mels.npz'},

			{'folder' : '../test_data/_-XuCL9HBm_HQ-cut11/{}.jpg', 'till' : (708//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-kE9QyG6Bexw-cut16/mels.npz'},

			{'folder' : '../test_data/_-XuCL9HBm_HQ-cut17/{}.jpg', 'till' : (708//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-kE9QyG6Bexw-cut22/mels.npz'},

			{'folder' : '../test_data/_-YjgvYaU3zY0-cut8/{}.jpg', 'till' : (768//sif.hparams.T) * sif.hparams.T, 
				'gt' : '../test_data/_-_O3e2HdLFQ8-cut18/mels.npz'}
		]

ids = [i + 1 for i in range(len(samples))]

if __name__ == '__main__':
	g = Generator()
	for s, id in zip(samples, ids):
		g.vc(s, id)