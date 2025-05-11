# krigSCD

Official implementation of "Probabilistic Spatial Interpolation of Sparse Data using Diffusion Models," submitted for review to AMS Journal: Artificial Intelligence for the Earth Systems (AIES). 

## Introduction  
The large underlying assumption of climate models today relies on the basis of a "confident" initial condition, a reasonably plausible snapshot of the Earth for which all future predictions depend on. However, given the inherently chaotic nature of our system, this assumption is complicated by sensitive dependence, where small uncertainties in initial conditions can lead to exponentially diverging outcomes over time. This challenge is particularly salient at global spatial scales and over centennial timescales, where data gaps are not just common but expected. The source of uncertainty is two-fold: (1) sparse, noisy observations from satellites and ground stations, and (2) internal variability stemming from the simplifying approximations within the models themselves.  

In practice, data assimilation methods are used to reconcile this missing information by conditioning model states on partial observations. Our work builds on this idea but operates at the extreme end of sparsity. We propose a conditional data imputation framework that reconstructs full temperature fields from as little as 1% observational coverage. The method leverages a diffusion model guided by a prekriged mask, effectively inferring the full-state fields from minimal data points. We validate our framework over the Southern Great Plains, focusing on afternoon (12:00–6:00 PM) temperature fields during the summer months of 2018–2020. Across varying observational densities—from swath data to isolated in-situ sensors—our model achieves strong reconstruction accuracy, highlighting its potential to fill in critical data gaps in both historical reanalysis and real-time forecasting pipelines.

**Framework Overview**  

Our approach is divided into three primary components:   
1. a training process utilizing a diffusion model to learn the underlying distribution of input images;
2. a mask generation process that simulates realistic observation patterns, where individual pixels represent in-situ data and randomized trajectories of varying lengths and directions correspond to satellite swath observations; and
3. a guiding process that refines the model’s focus with something we refer to as a kriging smoother, directing it to converge around regions of observational coverage.    

Finally, we assess the performance of the model on previously unseen, out-of-sample images and quantify its performance against traditional methods.  

<p align="center">
<img src="https://github.com/valtsao17/krigSCD/blob/main/framework.gif" width="65%" height="50%"/>
</p>

## Dataset
The dataset utilized in this study consists of historical U.S. temperature data at 2 meters above the surface, recorded hourly between 17:00 and 23:00 UTC during the summer months (June 1–August 31) from 2018 to 2020. These data were sourced from the National Oceanic and Atmospheric Administration (NOAA) High-Resolution Rapid Refresh (HRRR) model. All data can be downloaded from the [AWS HRRR archive](https://registry.opendata.aws/noaa-hrrr-pds/).  

## Installation
To run our experiments, first clone the repository with:  
```
git clone https://github.com/valtsao17/krigSCD.git
```

For specific library versions, please refer to the `requirements.txt` file. To download the pretrained models, see [the Dropbox link](https://duke.box.com/s/uzuolhxknhzhksoarm1iis014ydi1b0j).

For training your own model, please refer to the instructions from OpenAI's [guided diffusion](https://github.com/openai/improved-diffusion) for which our model is based on.

## Running Experiments
The masks are generated using `mask_generation.ipynb`, where two observation modalities can be created--satellite swaths and insitu observations. This code allows for you to denote a `swath_ratio` parameter specifying how much of your observations you want to come from satellite vs. insitu. For further details, please refer to the file itself.  

Our experiments are run in two steps: a smoothing step and an inpainting step. 

### Smoothing
Refer to `prekriging.py` within the `RePaint` folder. This script requires two flags: `--gt_filepath` and `--mask_filepath` that refer to the ground truth image and corresponding mask that you would like smoothed. Unless otherwise specified, it will save the new images to `/smoothed/`. 

### Inpainting
Please refer to [here](https://github.com/andreas128/RePaint) for more detailed instructions. Inpainting requires modifying a custom config file, with a sample one given in `sample_run.yml`. The parameters in this file match with those of our pretrained model. To inpaint, change `model_path` to where your model is located, `gt_path` to where your ground truth images are located, and `mask_path` to where your masks are located. The script will throw an error if the number of ground truth images does not match that of the number of masks. Please also match `max_len` to the amount of files that you want to inpaint. To run, execute the command:
```
python test.py --conf_path /path/to/conf
```
Note: having access to a GPU is highly recommended, as otherwise this script can take a while to complete. 

The output will be saved into a `/log/` folder with four subfolders:
 - `inpainted` are the reconstructed images
 - `gt_masked` are the masked images with the ground truth filled in for known values
 - `gt` are the ground truth images
 - `gt_keep_masks` are the binary masks

Since the inpainting algorithm is probabilistic, we ensemble 10 identical ground truths and masks and average the result to obtain a final reconstruction. It is important to note that all error calculations from the smoothing process are made in comparison to the original, unsmoothed ground truth.  

## License
This project is licensed under the MIT License.
