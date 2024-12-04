# AI-DFN

This is a project to obtain the parameters of the Discrete Fracture Networks (DFN) automatically with MCTS. The main calibration elements are the length and density of 3D fractures in the rock mass, and a reward function is constructed from the density and frequency distribution of surface fracture traces.

## **File Structure**

1. `**mcts.py**`

   - Main program utilizing the **mctx library** to implement the Monte Carlo Tree Search algorithm.
   - Designed for optimizing and adjusting parameters in the DFN model.

2. `**dfn_env.py**`

   - Contains tools for generating random DFNs (Discrete Fracture Networks) and slicing them.
   - Simulates geometric and statistical properties of fracture networks.

3. `**window_method.py**`

   - Implements the traditional **circular scanline method** for evaluating fracture characteristics.
   - Used for comparison with the optimization approach based on MCTS.

4. `**points.csv**`

   - Input data file containing digitized information about a surface.

   - Includes: Trace starting and ending points.

     

