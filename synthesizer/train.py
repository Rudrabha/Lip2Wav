from synthesizer.utils.symbols import symbols
from synthesizer.utils.text import sequence_to_text
from synthesizer.hparams import hparams_debug_string
from synthesizer.feeder import Feeder
from synthesizer.models import create_model
from synthesizer.utils import ValueWindow, plot
from synthesizer import infolog, audio
from datetime import datetime
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import traceback
import time
import os

log = infolog.log





def add_train_stats(model, hparams):
    with tf.variable_scope("stats") as scope:
        for i in range(hparams.tacotron_num_gpus):
            tf.summary.histogram("mel_outputs %d" % i, model.tower_mel_outputs[i])
            tf.summary.histogram("mel_targets %d" % i, model.tower_mel_targets[i])
        tf.summary.scalar("before_loss", model.before_loss)
        tf.summary.scalar("after_loss", model.after_loss)
        
        if hparams.predict_linear:
            tf.summary.scalar("linear_loss", model.linear_loss)
            for i in range(hparams.tacotron_num_gpus):
                tf.summary.histogram("mel_outputs %d" % i, model.tower_linear_outputs[i])
                tf.summary.histogram("mel_targets %d" % i, model.tower_linear_targets[i])
        
        tf.summary.scalar("regularization_loss", model.regularization_loss)
        #tf.summary.scalar("stop_token_loss", model.stop_token_loss)
        tf.summary.scalar("loss", model.loss)
        tf.summary.scalar("learning_rate", model.learning_rate)  # Control learning rate decay speed
        if hparams.tacotron_teacher_forcing_mode == "scheduled":
            tf.summary.scalar("teacher_forcing_ratio", model.ratio)  # Control teacher forcing 
        # ratio decay when mode = "scheduled"
        gradient_norms = [tf.norm(grad) for grad in model.gradients]
        tf.summary.histogram("gradient_norm", gradient_norms)
        tf.summary.scalar("max_gradient_norm", tf.reduce_max(gradient_norms))  # visualize 
        # gradients (in case of explosion)
        return tf.summary.merge_all()



def time_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def model_train_mode(args, feeder, hparams, global_step):
    with tf.variable_scope("Tacotron_model", reuse=tf.AUTO_REUSE) as scope:
        model = create_model("Tacotron", hparams)
        model.initialize(feeder.inputs, feeder.input_lengths, feeder.speaker_embeddings, 
                         feeder.mel_targets, targets_lengths=feeder.targets_lengths, global_step=global_step,
                         is_training=True, split_infos=feeder.split_infos)
        print ("INITIALIZED THE MODEL.....")
        model.add_loss()
        print ("ADDED LOSS TO THE MODEL.....")
        model.add_optimizer(global_step)
        print ("ADDED OPTIMIZER.....")
        stats = add_train_stats(model, hparams)
        return model, stats


def model_test_mode(args, feeder, hparams, global_step):
    with tf.variable_scope("Tacotron_model", reuse=tf.AUTO_REUSE) as scope:
        model = create_model("Tacotron", hparams)
        model.initialize(feeder.eval_inputs, feeder.eval_input_lengths, 
                         feeder.eval_speaker_embeddings, feeder.eval_mel_targets, targets_lengths=feeder.eval_targets_lengths, 
                         global_step=global_step, is_training=False, is_evaluating=True,
                         split_infos=feeder.eval_split_infos)
        model.add_loss()
        return model


def train(log_dir, args, hparams):
    save_dir = os.path.join(log_dir, "taco_pretrained")
    plot_dir = os.path.join(log_dir, "plots")
    wav_dir = os.path.join(log_dir, "wavs")
    mel_dir = os.path.join(log_dir, "mel-spectrograms")
    eval_dir = os.path.join(log_dir, "eval-dir")
    eval_plot_dir = os.path.join(eval_dir, "plots")
    eval_wav_dir = os.path.join(eval_dir, "wavs")
    tensorboard_dir = os.path.join(log_dir, "tacotron_events")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(wav_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(eval_plot_dir, exist_ok=True)
    os.makedirs(eval_wav_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    checkpoint_fpath = os.path.join(save_dir, "tacotron_model.ckpt")
    
    log("Checkpoint path: {}".format(checkpoint_fpath))
    log("Using model: Tacotron")
    log(hparams_debug_string())
    
    # Start by setting a seed for repeatability
    tf.set_random_seed(hparams.tacotron_random_seed)
    
    # Set up data feeder
    coord = tf.train.Coordinator()
    with tf.variable_scope("datafeeder") as scope:
        feeder = Feeder(coord, hparams)
    
    # Set up model:
    global_step = tf.Variable(0, name="global_step", trainable=False)
    model, stats = model_train_mode(args, feeder, hparams, global_step)
    #eval_model = model_test_mode(args, feeder, hparams, global_step)
    
    # Book keeping
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=2)
    
    log("Tacotron training set to a maximum of {} steps".format(args.tacotron_train_steps))
    
    # Memory allocation on the GPU as needed
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    
    # Train
    with tf.Session(config=config) as sess:
        try:
            summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
            
            sess.run(tf.global_variables_initializer())
            
            # saved model restoring
            if args.restore:
                # Restore saved model if the user requested it, default = True
                try:
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                    
                    if checkpoint_state and checkpoint_state.model_checkpoint_path:
                        log("Loading checkpoint {}".format(checkpoint_state.model_checkpoint_path),
                            slack=True)
                        saver.restore(sess, checkpoint_state.model_checkpoint_path)
                    
                    else:
                        log("No model to load at {}".format(save_dir), slack=True)
                        saver.save(sess, checkpoint_fpath, global_step=global_step)
                
                except tf.errors.OutOfRangeError as e:
                    log("Cannot restore checkpoint: {}".format(e), slack=True)
            else:
                log("Starting new training!", slack=True)
                saver.save(sess, checkpoint_fpath, global_step=global_step)
            
            # initializing feeder
            feeder.start_threads(sess)
            print ("INITIALIZED FEEDER")
            
            # Training loop
            while not coord.should_stop() and step < args.tacotron_train_steps:
                start_time = time.time()
                step, loss, opt = sess.run([global_step, model.loss, model.optimize])
                time_window.append(time.time() - start_time)
                loss_window.append(loss)
                message = "Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]".format(
                    step, time_window.average, loss, loss_window.average)
                log(message, end="\r", slack=(step % args.checkpoint_interval == 0))
                print(message)
                
                if loss > 100 or np.isnan(loss):
                    log("Loss exploded to {:.5f} at step {}".format(loss, step))
                    raise Exception("Loss exploded")
                
                if step % args.summary_interval == 0:
                    log("\nWriting summary at step {}".format(step))
                    summary_writer.add_summary(sess.run(stats), step)
                
                if step % args.eval_interval == 0:
                    pass

                
                if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps or \
                        step == 300:
                    # Save model and current global step
                    saver.save(sess, checkpoint_fpath, global_step=global_step)
                    
                    log("\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..")
                    input_seq, mel_prediction, alignment, target, target_length = sess.run([
                        model.tower_inputs[0][0],
                        model.tower_mel_outputs[0][0],
                        model.tower_alignments[0][0],
                        model.tower_mel_targets[0][0],
                        model.tower_targets_lengths[0][0],
                    ])
                    
                    # save predicted mel spectrogram to disk (debug)
                    mel_filename = "mel-prediction-step-{}.npy".format(step)
                    np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T,
                            allow_pickle=False)
                    
                    # save griffin lim inverted wav for debug (mel -> wav)
                    wav = audio.inv_mel_spectrogram(mel_prediction.T, hparams)
                    audio.save_wav(wav,
                                   os.path.join(wav_dir, "step-{}-wave-from-mel.wav".format(step)),
                                   sr=hparams.sample_rate)
                    
                    # save alignment plot to disk (control purposes)
                    plot.plot_alignment(alignment,
                                        os.path.join(plot_dir, "step-{}-align.png".format(step)),
                                        title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                                                                                    time_string(),
                                                                                    step, loss),
                                        max_len=target_length // hparams.outputs_per_step)
                    # save real and predicted mel-spectrogram plot to disk (control purposes)
                    plot.plot_spectrogram(mel_prediction, os.path.join(plot_dir,
                                                                       "step-{}-mel-spectrogram.png".format(
                                                                           step)),
                                          title="{}, {}, step={}, loss={:.5f}".format("Tacotron",
                                                                                      time_string(),
                                                                                      step, loss),
                                          target_spectrogram=target,
                                          max_len=target_length)
                
                if step % args.embedding_interval == 0 or step == args.tacotron_train_steps or step == 1:
                    # Get current checkpoint state
                    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
                    

            
            log("Tacotron training complete after {} global steps!".format(
                args.tacotron_train_steps), slack=True)
            return save_dir
        
        except Exception as e:
            log("Exiting due to exception: {}".format(e), slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def tacotron_train(args, log_dir, hparams):
    return train(log_dir, args, hparams)
