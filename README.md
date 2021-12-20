# ECE-472-DeepLearning
Repository for assignments in ECE-472: Deep Learning at The Cooper Union Fall 2021

## Assignments
1. Sine regression: Linear regression of noisy sinewave on gaussian basis functions
2. Spiral classification: Classification of spirals using multilayered perceptron
3. MNIST classification: Classification of MNIST dataset using convolutional neural network
4. CIFAR classification: Classification of CIFAR10 and CIFAR100 datasets using convolutional neural networks
5. AG News classification: Classification of AG News datast using convolutional neural network

## Midterm Assignment
Replication of results from: 
  
Alain, G. and Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes. *arXiv:1610.01644.*  
    
Experiments replicated:
- Basic example on MNIST (Section 3.5)
- ResNet-50* (Section 4.1)
    - *Note that the model used is a ResNet-56 trained on CIFAR-10 and not a ResNet-50 trained on ImageNet due to the large size of ImageNet
- Pathological behavior on skip connections (Section 5.1)

## Final Assignment
Experiments extending the NALU block to periodic functions.

NALU block source:

Trask, A., Hill, F., Reed, S., Rae, J., Dyer, C. and Blunsom P. (2018). Neural Arithmetic Logic Units. *arXiv:1808.00508.*