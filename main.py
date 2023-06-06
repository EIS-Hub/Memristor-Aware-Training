import numpy as np
import pickle
import matplotlib.pyplot as plt

# import pytorch and utils
import torch
import random
from utils import *
from models_utils import MLP, RSNN, params
from utils_mobilenet_v2 import get_mobilenet
# define the device
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print( f'you are using a Mac-based GPU' )
elif torch.cuda().is_available():
    device = torch.cuda.device(0)
    print( 'You are using a '+str(torch.cuda.get_device_name(0)) )
else: 
    device = torch.device('cpu')
    print( f'you are using a: {device}' )

# argparser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-task", help="which task you want to train on", choices={'mnist', 'cifar10', 'cifar100', 'ecg'}, default='mnist')
parser.add_argument("-noise_inference", help="turns on noise-aware-training", type=bool, default=True)
parser.add_argument("-noise_sd", help="standard deviation (related to the max of the weights) for noise-aware-training", type=float, default=0.1)
parser.add_argument("-clip_w", help="clips the weights during training to clip_w times the standard deviation", type=float, default=None)
parser.add_argument("-epochs", help="how many epochs you want to train for", type=int, default=None, choices=range(0,500))
parser.add_argument("-file_save", help="enables to save the model you are training", action='store_true')
parser.add_argument("-file_save_path", help="the path where the model you trained will be stored in", default='/Users/filippomoro/Documents/Training_with_memristors/Models')
parser.add_argument("-file_load", help="enables to load the pre-trained model you are training with noise", action='store_true')
parser.add_argument("-file_load_path", help="the path from which the pre-trained model is loaded", default='/Users/filippomoro/Documents/Training_with_memristors/Models')
parser.add_argument("-file_load_name", help="the file_name of the pre-trained model")
parser.add_argument("-model_num", help="selects 1 out of 5 or 10 pretrained models", type=int, default=0, choices=range(1,10))
parser.add_argument("-seed", help="sets the seed for the random number generator", type=int, default=14)
parser.add_argument("-verbose", help="enables more printing from the training stats", action='store_true', default=False)
args = parser.parse_args()

#### TODO
# make it work for ECG and allow to load a pretrained model and save the trained ones


if __name__ == "__main__":

    # Reproducibility in RNG
    torch.manual_seed( args.seed )
    random.seed( args.seed )
    np.random.seed( args.seed )
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # model saving and loading options (TODO)
    training_type = {0:'normal', 1:'ste'}
    file_save = False
    filename_save = f'/{args.task}_{training_type[args.noise_inference]}_{args.model_num}'


    # import dataset
    batch_size = {'cifar10':128, 'cifar100':128, 'mnist':32, 'ecg':16}
    train_loader, test_loader = generate_dataset( task=args.task, train_batch_size=batch_size[args.task], test_batch_size=batch_size[args.task] )
    print(f'-- {args.task} dataset: loaded')


    # define model
    if args.task == 'mnist':
        model = MLP(noise_inference=args.noise_inference, noise_sd=args.noise_sd)
    elif args.task == 'cifar10':
        model = get_mobilenet( weights = 'cifar_specs', out_features=10,  noise_inference=args.noise_inference, noise_inference_bn=False, noise_sd=args.noise_sd )
    elif args.task == 'cifar100':
        model = get_mobilenet( weights = 'cifar_specs', out_features=100, noise_inference=args.noise_inference, noise_inference_bn=False, noise_sd=args.noise_sd )
    elif args.task == 'ecg':
        rsnn = RSNN(params)
        rsnn.noise_inference = args.noise_inference
        Noisy_Inference.noise_sd = args.noise_sd
        rsnn.noiser = Noisy_Inference.apply
        rsnn.noise_sd = args.noise_sd
    print(f'-- model: loaded')


    # training the network
    print(f'-- Training')
    if args.task == 'mnist':
        lr = 5e-3
        optimizer = torch.optim.Adam( model.parameters(), lr=lr )
        if args.epochs is None: epochs = 30
        else: epochs = args.epochs
    elif args.task == 'ecg' :
        lr = 1e-3
        optimizer = torch.optim.Adamax(rsnn.parameters(), lr=lr, betas=(0.9,0.999))
        if args.epochs is None: epochs = 50
        else: epochs = args.epochs
    else:
        optimizer = None
        if args.epochs is None: epochs = 200
        else: epochs = args.epochs 
        lr = 1e-2
    data_loaders = [train_loader, test_loader]
    model_trained, train_stats, test_stats = training_algo( training_type='normal', model=model, data_loaders=data_loaders,
                                                                                    optimizer=optimizer,
                                                                                    clip_w=args.clip_w, lr=lr, epochs=epochs, epochs_noise=2,
                                                                                    print_every=1, verbose=args.verbose, device=device,
                                                                                    save_checkpoint_path=None,
                                                                                    load_checkpoint_path=None )
    if not args.verbose: print( f'Test acc {test_stats[0]*100}% loss {test_stats[1]}, Train acc {train_stats[0]*100}% loss {train_stats[1]}' )

    # testing with difference degree of noise
    print('-- Testing with difference degree of noise')
    noise_sd_list = np.array([0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3])
    accuracy_with_noise = np.zeros( len( noise_sd_list ) )
    with torch.no_grad():
        for l in model.layers:
                model.layers[l].noise_inference = False
        for n, noise_sd in enumerate(noise_sd_list):
            model_trained_noisy = adding_noise_model( model=model, add_quantization=False, add_noise=True, noise_sd=noise_sd )
            acc_test_noise, _ = testing( model_trained_noisy, test_loader=test_loader, verbose=False, device=device )
            accuracy_with_noise[n] = acc_test_noise
            print(f'Inference noise {noise_sd*100:.2f}%, Test Acc {acc_test_noise*100:.2f}%')

    # if args.file_save:
    #     dict_save = { 'test_stats':test_stats, 'train_stats':train_stats, 'args':args, 'model':model }
    #     pickle.dump( dict_save, open(file_save_path, 'wb') )