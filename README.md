### Dependency
1. Run `conda create --name <env> --file requirements.txt`
1. Install package [`PerceptualSimilarity`](https://github.com/richzhang/PerceptualSimilarity)
2. Install package [`CoordConv`](https://github.com/walsvid/CoordConv)

### Generate training data (natural home images) for PTZ module

1. Download Gibson `.glb` files and store them in a folder called `gibson`
2. To generate habitat dataset to train PTZ module, run `./gen_ptz_data.sh < train/test_env.txt` inside `gen_data`
3. Folders `habitat_train` and `habitat_test` should now have 6500 images and 2300 images in total respectively from different environments. We sample 10 locations from each environment and generate 10 images at each location via consecutive right turns.

### Generate training data (noise images) for PTZ module

1. To generate noise dataset to train PTZ module, run `./gen_noise_data.sh` inside `gen_data`
2. Folders `noise_train` and `noise_test` should now have 40k and 4k images in total respectively. Both contain 4 folders corresponding to fractal noise, perlin noise, overlaping random shapes and non-overlaping random shapes. 

### Train PTZ module

1. Run `./train_ptz.sh` to train the PTZ module
2. Run `./eval_ptz.sh` to eval the PTZ module

### Generate training data for navigation policy

1. Run `./gen_data/gen_nav_data.sh`
2. `nav_train` contains 10 environments, each has 1k validation data in `nav0` and 5k training data in `nav1`

### Train navigation policy

1. Run `./train_nav.sh`

### Run inference in testing environments

1. Build a target environment floor map by first randomly exploring the the space via `gen_ptz_data.sh`. 
2. Generate a scatter plot of all the valid locations via `load_all_states` from `nav.py`
3. Contour a floor map using a concave hull algorithm via `load_floor` from `nav.py`
4. Run `./eval_nav.sh`

### Results

Please refer to the [report](http://people.csail.mit.edu/felixw/noise2ptz/assets/noise2ptz.pdf).  