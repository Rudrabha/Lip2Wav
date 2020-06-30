import synthesizer
from synthesizer import inference as sif
import numpy as np
import sys, cv2, os, pickle, argparse
from tqdm import tqdm
from shutil import copy
from glob import glob

class Generator(object):
	def __init__(self):
		super(Generator, self).__init__()

		self.synthesizer = sif.Synthesizer(verbose=False)

	def read_window(self, window_fnames):
		window = []
		for fname in window_fnames:
			img = cv2.imread(fname)
			if img is None:
				raise FileNotFoundError('Frames maybe missing in {}.' 
						' Delete the video to stop this exception!'.format(sample['folder']))

			img = cv2.resize(img, (sif.hparams.img_size, sif.hparams.img_size))
			window.append(img)

		images = np.asarray(window) / 255. # T x H x W x 3
		return images

	def vc(self, sample, outfile):
		hp = sif.hparams
		id_windows = [range(i, i + hp.T) for i in range(0, (sample['till'] // hp.T) * hp.T, 
					hp.T - hp.overlap) if (i + hp.T <= (sample['till'] // hp.T) * hp.T)]

		all_windows = [[sample['folder'].format(id) for id in window] for window in id_windows]
		last_segment = [sample['folder'].format(id) for id in range(sample['till'])][-hp.T:]
		all_windows.append(last_segment)

		ref = np.load(os.path.join(os.path.dirname(sample['folder']), 'ref.npz'))['ref'][0]
		ref = np.expand_dims(ref, 0)

		for window_idx, window_fnames in enumerate(all_windows):
			images = self.read_window(window_fnames)

			s = self.synthesizer.synthesize_spectrograms(images, ref)[0]
			if window_idx == 0:
				mel = s
			elif window_idx == len(all_windows) - 1:
				remaining = ((sample['till'] - id_windows[-1][-1] + 1) // 5) * 16
				if remaining == 0:
					continue
				mel = np.concatenate((mel, s[:, -remaining:]), axis=1)
			else:
				mel = np.concatenate((mel, s[:, hp.mel_overlap:]), axis=1)
			
		wav = self.synthesizer.griffin_lim(mel)
		sif.audio.save_wav(wav, outfile, sr=hp.sample_rate)

def get_vidlist(data_root):
	test = synthesizer.hparams.get_image_list('test', data_root)
	test_vids = {}
	for x in test:
		x = x[:x.rfind('/')]
		if len(os.listdir(x)) < 30: continue
		test_vids[x] = True
	return list(test_vids.keys())

def complete(folder):
	# first check if ref file present
	if not os.path.exists(os.path.join(folder, 'ref.npz')):
		return False

	frames = glob(os.path.join(folder, '*.jpg'))
	ids = [int(os.path.splitext(os.path.basename(f))[0]) for f in frames]
	sortedids = sorted(ids)
	if sortedids[0] != 0: return False
	for i, s in enumerate(sortedids):
		if i != s:
			return False
	return True

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', "--data_root", help="Speaker folder path", required=True)
	parser.add_argument('-r', "--results_root", help="Speaker folder path", required=True)
	parser.add_argument('--checkpoint', help="Path to trained checkpoint", required=True)
	args = parser.parse_args()

	sif.hparams.set_hparam('eval_ckpt', args.checkpoint)

	videos = get_vidlist(args.data_root)

	RESULTS_ROOT = args.results_root
	if not os.path.isdir(RESULTS_ROOT):
		os.mkdir(RESULTS_ROOT)

	GTS_ROOT = os.path.join(RESULTS_ROOT, 'gts/')
	WAVS_ROOT = os.path.join(RESULTS_ROOT, 'wavs/')
	files_to_delete = []
	if not os.path.isdir(GTS_ROOT):
		os.mkdir(GTS_ROOT)
	else:
		files_to_delete = list(glob(GTS_ROOT + '*'))
	if not os.path.isdir(WAVS_ROOT):
		os.mkdir(WAVS_ROOT)
	else:
		files_to_delete.extend(list(glob(WAVS_ROOT + '*')))
	for f in files_to_delete: os.remove(f)

	hp = sif.hparams
	g = Generator()
	for vid in tqdm(videos):
		if not complete(vid):
			continue

		sample = {}
		vidpath = vid + '/'

		sample['folder'] = vidpath + '{}.jpg'

		images = glob(vidpath + '*.jpg')
		sample['till'] = (len(images) // 5) * 5

		vidname = vid.split('/')[-2] + '_' + vid.split('/')[-1]
		outfile = WAVS_ROOT + vidname + '.wav'
		g.vc(sample, outfile)

		copy(vidpath + 'audio.wav', GTS_ROOT + vidname + '.wav')