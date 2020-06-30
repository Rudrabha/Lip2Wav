import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
	raise Exception("Must be using >= Python 3.2")

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import listdir, path
import numpy as np
import argparse, os, traceback
from tqdm import tqdm
from glob import glob
import encoder, subprocess
from encoder import inference as eif
from synthesizer import audio as sa
from synthesizer import hparams as hp

encoder_weights = 'encoder/saved_models/pretrained.pt'
eif.load_model(encoder_weights)
secs = 1
k = 1

def process_video_file(afile, args):
	wav = encoder.audio.preprocess_wav(afile)
	if len(wav) < secs * encoder.audio.sampling_rate: return

	indices = np.random.choice(len(wav) - encoder.audio.sampling_rate * secs, k)
	wavs = [wav[idx : idx + encoder.audio.sampling_rate * secs] for idx in indices]
	embeddings = np.asarray([eif.embed_utterance(wav) for wav in wavs])
	np.savez_compressed(afile.replace('audio.wav', 'ref.npz'), ref=embeddings)

	wav = sa.load_wav(afile, sr=hp.hparams.sample_rate)
	lspec = sa.linearspectrogram(wav, hp.hparams)
	melspec = sa.melspectrogram(wav, hp.hparams)

	np.savez_compressed(afile.replace('audio.wav', 'mels.npz'), lspec=lspec, mel=melspec)

def mp_handler(job):
	vfile, args = job
	try:
		process_video_file(vfile, args)
	except KeyboardInterrupt:
		exit(0)
	except:
		traceback.print_exc()
		
def dump(args):
	print('Started processing for with {} CPU cores'.format(args.num_workers))

	filelist = glob(path.join(args.preprocessed_root, '*', '*', '*', 'audio.wav'))

	jobs = [(vfile, args) for vfile in filelist]
	p = ThreadPoolExecutor(args.num_workers)
	futures = [p.submit(mp_handler, j) for j in jobs]
	_ = [r.result() for r in tqdm(as_completed(futures), total=len(futures))]

parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', help='Number of workers to run in parallel', default=8, type=int)
parser.add_argument("--preprocessed_root", help="Folder where preprocessed files will reside", 
					required=True)

args = parser.parse_args()

if __name__ == '__main__':
	dump(args)