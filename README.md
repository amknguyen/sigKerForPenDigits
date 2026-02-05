# Signature-based Classification & Kernel Methods on UCI PenDigits

### Context:
This repository reproduces some of the elementary results from _Chevyrev & Kormilitzin_ (2016) and 
_Király & Oberhauser_ (2019). Using the iisignatures library, I explore truncated path signatures combined
with standard classifers (e.g., sklearn's RandomForest) on classifying handwritten pen digits
on UCI's PenDigits dataset (1998).

Here are some current visulations: \
2D: Here, red digits indicate the time-step ordering: \
![2D Digit Path](./2d.png) 

3D: Here, time-augmentation gives time its own axis, which greatly improves model performance; 
lighter colors represent earlier times:
![3D Digit Path 1](./3d1.png) 
![3D Digit Path 2](./3d2.png) 

### Extensions:
While the current implementation uses signature kernels as a feature map for a linear classifer,
I would like to **implement signature kernels for a Gaussian Processes**. This would allow us to
perform Bayesian inference directly on the space of trajectories, which, when coming from a 
dynamical system, allows us to capture the geometry of the dynamical system readily.

Updates / Changes: \
2/5/26:
* Cleaned up README, adding visualizations

2/1/26:
* Decided to use a random forest instead for better classification
* Added time-augmentation capabilities
* Created interactive 3D plots to plot (t,x,y) path data