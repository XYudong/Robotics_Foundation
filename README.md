# Problem Sets from Udacity AI for Robotics course


## Particle Filter
### Implementation:
* Creating *N* particles;
* Loop with movements and measurements:
    1. Moving each particle
    2. Taking measurements
    3. Updating importance **weight** for each particle
        * based on mismatch between corresponding prediction and the current **measurement**;
        * the weight is the production of probability values from measurement Gaussian distributions;
    4. Resampling *N* particles from old ones with replacement
        * normalize the weights;
        * utilize "**resampling wheel**" to realize this operation efficiently;
        
        
### Tricks:
* Using fewer particles when they track well to save time:
    * i.e. when all non-normalized importance weights are very large
    
        