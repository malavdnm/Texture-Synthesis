# texture-synthesis
This repository is an implementation of the image quilting algorithm for texture synthesis using Python. For more details on the algorithm, refer to the original paper by Alexei A. Efros and Willian T. Freeman [here](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/papers/efros-siggraph01.pdf).

## Usage
To run the code on a given texture, run the following code:

`python main.py --image_path <image_path> --block_size <block_size> --overlap <overlap> --scale <scale> --num_outputs <num_outputs> --output_file <filename> --plot <plot> --tolerance <tolerance>`

For more details, use `python main.py -h`
