This is the version with parallel computation of strides as described [here](https://nbviewer.jupyter.org/github/robintibor/braindecode/blob/80d2fd3a10d9d4e0019d83648b3461d73e055561/braindecode/notebooks/explanations/Multiple_Prediction_Same_Convolutions.ipynb).

In short: 
* Normal net: 2,2-stride -> from a 2x2 output (0,0),(0,1),(1,0),(1,1), the net only uses 
the topleft offset (0,0) for the following computations
* Parallel stride: The net uses all four offsets independently in parallel (shifted to 'b' axis) and 
at the end means over all outputs from different stride offsets.

Results:

|N_res_blocks_per_super_block|network|test acc|test loss|train loss|
|----------------------------|--------|---|---|---|
|5|normal|92.67%|0.309590|0.185623|
|5|parallel stride|93.16%|0.298192|0.157280|
|9|normal|92.17%|0.326476|0.206078|
|9|parallel stride|92.99 %|0.285899|0.180039|


