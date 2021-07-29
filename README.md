# FMC-CNN
Full matrix capture is a data acquistion technique for array ulstrasonic sensor.
Each element in the array transmit ultrasonic wave and all the element in the array received wave and record it in the data matrix

![Picture2](https://user-images.githubusercontent.com/48675751/127565197-fd5483f3-93bb-4448-a730-0a7727e86ecb.png)


The designed CNN architecture is aligned with FMC structure. Input is divided into parallel blocks and then connected to wrapped convlutional and max pooling layer.
Weights are shared within the block layer so model can cpature spatial relationship features that are prominent inside each block.

![Picture1](https://user-images.githubusercontent.com/48675751/127563014-4cbbff02-0bde-4a65-bf98-ba49ba67a029.png)
