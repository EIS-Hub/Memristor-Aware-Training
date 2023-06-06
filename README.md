# Memristor-Aware-Training
---

Memristors are interesting devices that can be programmed to different conductance levels and, embedded in a In-Memory Computing (IMC) architecture, enable efficient neural networks inference [1]. 
The programmable conductances of memristors can map the weights of neural network and crossbar arrays efficiently perform the Matrix-Vector multiplication, ubiquitous in neural networks [1,2].
Despite the promises of memristive In-Memory system, the field has struggled to effectively implement such systems.
One reason is the the large variability and noise of these memristive device.
When deploying a memristor-based system, the weights of neural networks are effectively pertubed by variability and noise, which heavily reduces the original performance of the neural network model.

Different solutions have been proposed both at the hardware and software side to reduce the impact of memristor variability during deployment. 
This repository aims at presenting what is currently the simplest and most effective way to deal with memristor variability: it's called Memristor-Aware-Training.
It is a simple training procedure that accounts for memristor non-idealities during training so that a neural network learns to perform well despite the variability and noise of memristors.
They way it works is that first the network is trained without variability, as in a normal case, and later a second training phase consists in adding noise to the weight in the forward pass. The backpropagation algorithm, however, ignores the noise injection and weight updates are performed on the original weights.
This way, it is proven that the network assumes a more stable weight configuration, becoming resilient to noise and variability perturbations.
This training procedure is proven on MNIST (MLP), CIFAR10/100 (CNN, mobilenet) and ECG (RNN and SRNN).

Existing scientific publications of a similar implementations are:
- [3] applying this training method to a PCM array, demonstrating a ResNet20 CNN on CIFAR10/100 and tinyImagenet.
- [4] a work on a RRAM-based IMC architecture, deonstrated on CIFAR10.
- [5] the work that first introduced the noise injection technique.
- [6] a noise-injection training procedure applied to a RRAM-based accelerator.
- [7] a different training procedure dealing with the use of Dropout, with dropout-rates optimized by Bayesian optimization, reaching similar conclusions as the mentioned noise-injection technique.
*Note: the above mentioned works are applied on both PCM and RRAM memristor types, demonstrating to solve the variability issue of both devices. However, the method is general and can be applied to all kinds of memristors.*


### To run an example of the Memristor-Aware-Training (on your terminal):
- mnist: python main.py -task mnist -noise_sd 0.2 -epochs 30 clip_w 2.5 -verbose
- cifar10: python main.py -task cifar10 -noise_sd 0.1 -epochs 100 -verbose
- cifar100: python main.py -task cifar100 -noise_sd 0.1 -epochs 100 -verbose

### Collecting Results:
- mnist: with the notebook MNIST_MLP
- cifar10/100: with the notebook CNN_CIFAR
- ecg: with the notebook Test_ECG

(Filippo Moro)

*References*

[1] Ielmini, Daniele, and H-S. Philip Wong. "In-memory computing with resistive switching devices." Nature electronics 1.6 (2018): 333-343.\
[2] Ambrogio, Stefano, et al. "Equivalent-accuracy accelerated neural-network training using analogue memory." Nature 558.7708 (2018): 60-67.\
[3] Joshi, Vinay, et al. "Accurate deep neural network inference using computational phase-change memory." Nature communications 11.1 (2020): 2473.\
[4] Wan, Weier, et al. "A compute-in-memory chip based on resistive random-access memory." Nature 608.7923 (2022): 504-512.\
[5] Long, Yun, Xueyuan She, and Saibal Mukhopadhyay. "Design of reliable DNN accelerator with un-reliable ReRAM." 2019 Design, Automation & Test in Europe Conference & Exhibition (DATE). IEEE, 2019.\
[6] He, Zhezhi, et al. "Noise injection adaption: End-to-end ReRAM crossbar non-ideal effect adaption for neural network mapping." Proceedings of the 56th Annual Design Automation Conference 2019. 2019.\
[7] Ye, Nanyang, et al. "Improving the robustness of analog deep neural networks through a Bayes-optimized noise injection approach." Communications Engineering 2.1 (2023): 25.
