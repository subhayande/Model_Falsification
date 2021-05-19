# Model Falsification in Python

For a theoretical overview on model falsification please see:   

1. De, Subhayan, et al. "Investigation of model falsification using error and likelihood bounds with application to a structural system." Journal of Engineering Mechanics 144.9 (2018): 04018078.   
https://doi.org/10.1061/(ASCE)EM.1943-7889.0001440  
2. De, Subhayan, et al. "A hybrid probabilistic framework for model validation with application to structural dynamics modeling." Mechanical Systems and Signal Processing 121 (2019): 961-980.  
https://doi.org/10.1016/j.ymssp.2018.10.014 


Download the module from https://github.com/subhayande/Model_Falsification. See the demo [fals_test1.py](fals_test1.py) for an example of the implementation. For a tutorial see [Falsification_Tutorial.pdf](Falsification_Tutorial.pdf).  

### Required packages: ###
numpy, scipy, time  
NOTE: Currently, only Gaussian distributions for residual errors are allowed and two-sided hypothesis tests are implemented.  
Report any bugs to Subhayan.De@colorado.edu 


License: Copyright (C) 2019 Subhayan De 

This program is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this
program. If not, see https://www.gnu.org/licenses/.



