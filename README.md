## Replication of Signature Method on UCI PenDigits Dataset

This is my attempt at replicating elementary results using iisignature on the aformentioned dataset.
The purpose of this is to explore how signatures can capture non-linear sequential data well,
allowing for a linear classifier to achieve high accuracy on sequential data.


Updates / Changes: \
2/1/26:
* Decided to use a random forest instead for better classification
* Added time-augmentation capabilities
* Created interactive 3D plots to plot (t,x,y) path data