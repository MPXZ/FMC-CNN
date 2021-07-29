# FMC-CNN
Full matrix capture is a data acquistion technique for array ulstrasonic sensor.
The designed CNN architecture is aligned with FMC structure. Input is divided into parallel blocks and then connected to wrapped convlutional and max pooling layer.
Weights are shared within the block layer so model can cpature spatial relationship features that are prominent inside each block.
