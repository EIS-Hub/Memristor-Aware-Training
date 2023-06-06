# Memristor-Aware-Training
---

Memristors are interesting devices that can be programmed to different conductance levels and - in a In-Memory Computing (IMC) architecture - enable efficient neural networks inference. 
They can implement the weights of neural network mapping the synaptic efficacies as conductance values.
Despite the promises of memristive In-Memory system, the field has struggled to effectively implement such systems. 
One reason is the the large variability and noise that accompanies these memristive device.
When deploying a memristor-based system, the weights of a neural network are effectively pertubed by variability and noise, which heavily reduces the original computational power of the neural network.

Different solutions have been proposed both at the hardware and software side to reduce the impact of memristor variability during deployment. 
This repository aims at presenting what is currently the simplest and most effective way to deal with memristor variability: it's called Memristo-Aware-Training.
It is a simple training procedure that accounts for memristor non-idealities during training so that a neural network learns to perform well despite the variability and noise of memristors.
They way it works is that first the network is trained without variability, as in a normal case, and later a second training phase consists in adding noise to the weight in the forward pass.
This way, it is proven that the network assumes a more stable weight configuration, becoming resilient to noise and variability perturbations.
This training procedure is proven on MNIST (MLP), CIFAR10/100 (CNN, mobilenet) and ECG (RNN and SRNN).

### To run and example:
- mnist) python main.py -task mnist -noise_sd 0.2 -epochs 30 clip_w 2.5 -verbose
- cifar10) python main.py -task cifar10 -noise_sd 0.1 -epochs 100 -verbose
- cifar100) python main.py -task cifar100 -noise_sd 0.1 -epochs 100 -verbose

### Collecting Results:
- mnist) with the notebook: MNIST_MLP
- cifar10/100) with the notebook: CNN_CIFAR
- ecg) with the notebook: Test_ECG

(Filippo Moro)
