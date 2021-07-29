# FMC-CNN
*Classification task: crack detection

*Full matrix Capture (FMC)

Full matrix capture (FMC) is a data acquistion technique for array ulstrasonic sensor.

Each element in the array transmit ultrasonic wave and all the element in the array received wave and record it in the data matrix.

![Capture](https://user-images.githubusercontent.com/48675751/127565549-1880b857-730a-4a63-a7c9-9b6cacbad1c3.PNG)


*FMC-CNN architecture

The designed CNN architecture is aligned with FMC structure. Input is divided into parallel sub-groups and then connected to wrapped convlutional and max pooling layer.
Weights are shared within the parallel block layer so model can cpature local spatial relationship features that are prominent inside each block.

All the parallel output from each block are concatenated in a layer and passed onto next layer to learn global features.

The final output is binary classfication. i.e. crack versus no-crack

![Picture1](https://user-images.githubusercontent.com/48675751/127563014-4cbbff02-0bde-4a65-bf98-ba49ba67a029.png)
