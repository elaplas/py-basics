
### Acquisition functions
- Proposing sampling points in the search space is done by acquisition functions. They trade off exploitation and exploration. Exploitation means sampling where the surrogate model (Gaussian Process) predicts a high objective and exploration means sampling at locations where the surrogate model predicts the uncertainty is high. Both correspond to high acquisition function values and the goal is to maximize the acquisition function to determine the next sampling point

### Algorithm 
- Find the next sampling input point by acquisition functions
- Perform the experiment to find the function value of the input point from the previous step
- Update the Guassian process using the input value and the function value from previous step
- Repeat the previous steps 