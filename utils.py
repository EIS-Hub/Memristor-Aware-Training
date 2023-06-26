import copy
import numpy as np
import torch
from torchvision import datasets, transforms


def quantize( parameters, levels=None, num_levels=15, quantile=0.01, device='cpu' ):
    '''Based on an input parameter tensor (parameter), this function approximates these values to the closest available in the array <<levels>> 
    You can either specify <<levels>> explicitely or built it by default based on equally distributed <<num_levels>>.
    The <<quantile>> parameter enters in action only if <<levels>> is not defined and it sets the min and max quantized values in <<levels>>
    based on the 1-quantile and quantile values of <<parameters>> tensor
    '''
    if levels is None:
        # if levels are not specified, they assumed to be uniformely distributed and with num_levels levels
        upper_w = torch.quantile(parameters, np.clip(1-quantile, 0, 1)).item()
        lower_w = torch.quantile(parameters, np.clip(quantile, 0, 1)).item()
        levels = torch.linspace(lower_w, upper_w, num_levels).to(device)
    # bins are built from the quantized levels, each bin defined by half the space between levels
    bins = torch.tensor( [levels[l]+(levels[l]-levels[l+1]).abs()/2 for l in range(num_levels-1)], device=device )
    # returning a tensor with the indeces corresponding to the bins at each entry of parameter
    idx = torch.bucketize( parameters, boundaries=bins )
    # building the quantized tensor from the quantized values (levels)
    quant_parameters = levels[idx]
    return quant_parameters


def adding_noise_model( model, add_quantization=True, add_noise=True, levels=np.linspace(-1,1,15), num_levels=15, noise_sd=1e-2 ):
    '''Function that takes a model with his parameters and adds noise to the parameters
    model: the original model
    add_quantization: [bool] whether to quantized the weights with num_levels levels
    add_noise: [bool] wheter to apply gaussian noise with noise_sd standard deviation'''
    with torch.no_grad():
        model_noisy = copy.deepcopy( model )
        for p in model_noisy.parameters():
            if add_quantization:
                q = quantize( p, num_levels=num_levels )
                p.copy_( q )
            if add_noise:
                delta_w = 2*p.abs().max()
                #delta_w = torch.sqrt( delta_w**2 + delta_w**2 )
                n = torch.randn_like( p )*(noise_sd*delta_w)
                p.copy_( p+n )
    return model_noisy

class Noisy_Inference(torch.autograd.Function):
    """
    Function taking the weight tensor as input and applying gaussian noise with standard deviation 
    (noise_sd) and outputing the noisy version for the forward pass, but keeping track of the 
    original de-noised version of the weight for the backward pass
    """
    noise_sd = 1e-1

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we add some noise from a gaussian distribution
        """
        ctx.save_for_backward( input )
        weight = input.clone()
        delta_w = 2*torch.abs( weight ).max()
        # sd of the sum of two gaussians, given we have pos and neg devices in the chips
        #delta_w = torch.sqrt( delta_w**2 + delta_w**2 )
        noise = torch.randn_like( weight )*( Noisy_Inference.noise_sd * delta_w )
        return torch.add( weight, noise )

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we simply copy the gradient from upward in the computational graph
        """
        input, = ctx.saved_tensors
        weight = input.clone()
        return grad_output
noiser = Noisy_Inference.apply


def training_algo( training_type, model, data_loaders, optimizer=None, criterion=None, scheduler=None, out_activation=None,
                   device='cpu', lr=1e-3, clip_w=2.5, epochs=10, epochs_noise=2, 
                   noise_sd=1e-2, noise_every=100, levels=None, num_levels=15, print_every=1, verbose=False,
                   save_checkpoint_path=None, load_checkpoint_path=None  ):

    train_loader, test_loader = data_loaders
    if criterion is None: criterion = torch.nn.NLLLoss() #torch.nn.CrossEntropyLoss()
    if optimizer is None: optimizer = torch.optim.SGD( model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4 )
    if scheduler is None: scheduler = torch.optim.lr_scheduler.MultiStepLR( optimizer, milestones=[100, 150, 200], gamma=0.5 )
    if out_activation is None: out_activation = torch.nn.LogSoftmax( dim=-1 )

    if load_checkpoint_path is not None:
        checkpoint = torch.load( load_checkpoint_path, map_location='cpu' )
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch_start = checkpoint['epoch']
    else:
        model = model.to(device)
        epoch_start = 0

    losses_train, accs_train = [], []
    for e in range(epochs):
        losses = []
        correct = 0
        tot_samples = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            yhat = model( x )
            y_soft = out_activation( yhat )
            loss = criterion( y_soft, y.long() )
            loss.backward()

            if training_type == 'qat_noise' or training_type == 'qat':
                for p in list(model.parameters()):
                        if hasattr(p,'hid'):
                            p.data.copy_(p.hid)

            optimizer.step()
            if scheduler is not None: scheduler.step()
            losses.append( loss.item() )
            correct += torch.eq( torch.argmax(yhat, dim=1), y ).cpu().sum()
            tot_samples += len(y)

            if e+1 > epochs - epochs_noise and batch_idx%noise_every==0 and training_type == 'noise_fine_tuning':
               with torch.no_grad():
                   for p in model.parameters():
                       delta_w = torch.abs( p.max()-p.min() )
                       n = torch.randn_like( p )*(noise_sd*delta_w)
                       p.copy_( p+n )
            
            if clip_w is not None:
                with torch.no_grad():
                    for p in model.parameters():
                        std_w = torch.std( p )
                        p.clip_( -std_w*clip_w, +std_w*clip_w )

            if training_type == 'qat':
                for p in list(model.parameters()):  # updating the hid attribute
                    if hasattr(p,'hid'):
                        p.hid.copy_(p.data)
                    p.data = quantize( parameters=p.data, levels=levels, num_levels=num_levels, device=device )

            if training_type == 'qat_noise':
                for p in list(model.parameters()):  # updating the hid attribute
                    if hasattr(p,'hid'):
                        p.hid.copy_(p.data)
                    p.data = quantize( parameters=p.data, levels=levels, num_levels=num_levels, device=device )
                    p.data.add_( torch.randn_like(p.data)*noise_sd )

        acc_train = correct/tot_samples
        loss_train = np.mean(losses)
        if verbose and e%print_every==0:
            print( f'Train Epoch {e+1}, Train accuracy {acc_train*100:.2f}% Train loss {loss_train:.4f}' )
        accs_train.append(acc_train); losses_train.append(loss_train)

    if save_checkpoint_path is not None:
        torch.save({
            'epoch': epochs+epoch_start,
            'model_state_dict': model.to('cpu').state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss.item(),
            }, save_checkpoint_path.format( epochs+epoch_start+1 ))
        print(f'Checkpoint saved at: {save_checkpoint_path}')

    losses = []
    correct = 0
    tot_samples = 0
    model = model.to(device).eval()
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        yhat = model( x )
        y_soft = out_activation( yhat )
        loss = criterion( y_soft, y.long() )
        losses.append( loss.item() )
        correct += torch.eq( torch.argmax(yhat, dim=1), y ).cpu().sum()
        tot_samples += len(y)
    acc_test = correct/tot_samples
    loss_test = np.mean(losses)
    if verbose: print( f'Tot epochs {epochs+epoch_start} -- Test accuracy {acc_test*100:.2f}% Test loss {loss_test:.4f}' )

    return model, [accs_train, losses_train], [acc_test, loss_test]


def testing( model, test_loader, criterion=None, out_activation=None, device='cpu', verbose=True ):
    '''The function assessing the test classification accuracy of the model.
    model: model of choice
    test_loader: the test dataloader for the task of choice
    verbose: if True, makes the function output the test accuracy and loss'''
    model = model.to(device).eval()
    losses = []
    correct, tot_samples = 0, 0
    if criterion is None: criterion = torch.nn.NLLLoss()
    if out_activation is None: out_activation = torch.nn.LogSoftmax( dim=-1 )
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)
        yhat = model( x )
        y_soft = out_activation( yhat )
        loss = criterion( y_soft, y.long() )
        losses.append( loss.item() )
        correct += torch.eq( torch.argmax(yhat, dim=1), y ).cpu().sum()
        tot_samples += len(y)
    acc_test = correct/tot_samples
    loss_test = np.mean(losses)
    if verbose: print( f'-- Test accuracy {acc_test*100:.2f}% Test loss {loss_test:.4f}' )
    return acc_test, loss_test


def generate_dataset( task='mnist', train_batch_size=32, test_batch_size=32, image_size=32 ):
    '''Simple function that returns a dataloader depending on which task is specified in <<task>>.
    <<task>> can be: mnist, cifar100 or shd .
    train_batch_size and test_batch_size define the train and test batch sizes.
    '''

    if task=='mnist':
        mnist_path = "/Users/filippomoro/Documents/datasets"
        transform=transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize((0.1307,), (0.3081,))
                ])
        train_dataset = datasets.MNIST( mnist_path, train=True,  download=True, transform=transform )
        test_dataset  = datasets.MNIST( mnist_path, train=False, download=True, transform=transform )
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=test_batch_size, shuffle=False)
    
    elif task=='cifar10':
        cifar10_path = "/Users/filippomoro/Documents/datasets"
        transform=transforms.Compose([
        transforms.Resize( image_size ),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        transform_train = transforms.Compose([ #transforms.RandomCrop(128, padding=4), 
                         transforms.Resize( image_size ), 
                         transforms.RandomHorizontalFlip(p=0.5),
                         transforms.RandomRotation(10),
                         transforms.ToTensor(), 
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_dataset = datasets.CIFAR10( cifar10_path, train=True,  download=False, transform=transform_train ) ### transform_train
        test_dataset  = datasets.CIFAR10( cifar10_path, train=False, download=False, transform=transform )
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=2)
        test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=test_batch_size, shuffle=False, pin_memory=True, num_workers=2)

    elif task=='cifar100':
        cifar100_path = "/Users/filippomoro/Documents/datasets"
        transform=transforms.Compose([
        transforms.Resize( image_size ), 
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        ])
        transform_train = transforms.Compose( [ #transforms.RandomCrop(128, padding=4,padding_mode='reflect'), 
                         transforms.Resize( image_size ), 
                         transforms.RandomHorizontalFlip(p=0.5),
                         transforms.RandomRotation(10),
                         transforms.ToTensor(), 
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_dataset = datasets.CIFAR100( cifar100_path, train=True,  download=False, transform=transform_train )
        test_dataset  = datasets.CIFAR100( cifar100_path, train=False, download=False, transform=transform )
        train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, pin_memory=False, num_workers=2)
        test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=test_batch_size, shuffle=False, pin_memory=False, num_workers=2)

    else:
        print("please select a valid dataset name")
    return train_loader, test_loader
