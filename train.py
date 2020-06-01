import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from synthesizer.hparams import hparams, get_image_list
from synthesizer.train import tacotron_train
from utils.argutils import print_args
#from synthesizer import infolog
import argparse
from pathlib import Path
import os

#Prepares the data.
def prepare_run(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.tf_log_level)
    run_name = args.name
    log_dir = os.path.join(args.models_dir, "logs-{}".format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    all_images = get_image_list('train', args.data_root)
    all_test_images = get_image_list('val', args.data_root)

    hparams.add_hparam('all_images', all_images)
    hparams.add_hparam('all_test_images', all_test_images)

    print('Training on {} hours'.format(len(all_images) / (3600. * hparams.fps)))
    print('Validating on {} hours'.format(len(all_test_images) / (3600. * hparams.fps)))

    return log_dir, hparams

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Name of the run and of the logging directory.")
    parser.add_argument("--data_root", help="Speaker folder path", required=True)
    parser.add_argument("--preset", help="Speaker-specific hyper-params", type=str, required=True)

    parser.add_argument("-m", "--models_dir", type=str, default="synthesizer/saved_models/", help=\
        "Path to the output directory that will contain the saved model weights and the logs.")

    parser.add_argument("--mode", default="synthesis",
                        help="mode for synthesis of tacotron after training")
    
    parser.add_argument("--GTA", default="True",
                        help="Ground truth aligned synthesis, defaults to True, only considered "
							 "in Tacotron synthesis mode")
    parser.add_argument("--restore", type=bool, default=True,
                        help="Set this to False to do a fresh training")
    parser.add_argument("--summary_interval", type=int, default=2500,
                        help="Steps between running summary ops")
    parser.add_argument("--embedding_interval", type=int, default=1000000000,
                        help="Steps between updating embeddings projection visualization")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, # Was 5000
                        help="Steps between writing checkpoints")
    parser.add_argument("--eval_interval", type=int, default=1000, # Was 10000
                        help="Steps between eval on test data")
    parser.add_argument("--tacotron_train_steps", type=int, default=2000000, # Was 100000
                        help="total number of tacotron training steps")
    parser.add_argument("--tf_log_level", type=int, default=1, help="Tensorflow C++ log level.")

    args = parser.parse_args()
    print_args(args, parser)
    
    log_dir, hparams = prepare_run(args)
    
    with open(args.preset) as f:
        hparams.parse_json(f.read())

    tacotron_train(args, log_dir, hparams)
