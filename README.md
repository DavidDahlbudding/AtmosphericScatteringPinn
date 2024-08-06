# **Approximate Atmospheric Scattering using PINNs** 

Code for the paper [Dahlbüdding et al. (2024)](https://arxiv.org/abs/2408.00084).

The code is based on https://github.com/mroberto166/PinnsSub (~ commit [338c540](https://github.com/mroberto166/PinnsSub/tree/338c5400080ba570fd8f53d7481539225fcf4d17)) from the paper by [Mishra & Molinario (2021)](https://arxiv.org/abs/2009.13291).

## **Data Availability**

Trained models and necessary data are available at https://zenodo.org/doi/10.5281/zenodo.10727497.

## **Run the Code**

To train a physics-informed neural network (PINN) one has to run:

    python3 PINNS2.py

Specify the network properties at the beginning of the file.

### **Absorption PINN**

Inside the ImportFile.py make sure that the correct equation file is imported:

    import EquationModels.IsothermalAtmo as Ec

To train the absorption PINN, set the `addmask` property of the `Pinns`-class to `False` in ModelClassTorch2.py.

For the evaluation of a single PINN on the 95 test spectra run `python3 eval_model.py path/example_model.pkl path/example_file_name.h5`.

For the hyperparameter search run hyperparameter.sh.

Adjust the file name in hyperparameter_eval.sh before running the evaluation.


### **Rayleigh Scattering PINN**

Inside the ImportFile.py make sure that the correct equation file is imported:

    import EquationModels.IsothermalAtmoOnlyRayleigh as Ec

To use the special architecture of the scattering PINN, set the `addmask` property of the `Pinns`-class to `True` in ModelClassTorch2.py.


## **Overview of Jupyter Notebooks**

1. Hyperparameter Search of Absorption PINN
    - Plot all Losses
    - Plot all Errors of the Test Spectra
2. Individual Analysis of Absorption PINN
    - Check for Possible Correlation of Error with Atmospheric Parameters or Wavelength
    - Test on Custom Spectrum and Analyze Individual Alpha Profiles (used for Iterative Improvement)
    - Test Randomly Generated $\alpha$-profiles
3. Plot Solution $u(x,y,\phi)$ of Scattering PINN and its Residual Loss
4. Plot Example Spectrum which uses the Scattering PINN
5. Compare Solution $u(x,y,\phi)$ with "fixed" PINN


## **Use of petitRADTRANS**

Download [petitRADTRANS](https://petitradtrans.readthedocs.io/) (Version 2.6.7 was used for the paper).

In the folder, where Python 3 is installed, replace the file `site-packages/petitRADTRANS/radtrans.py` with the file `pRT/radtrans.py` from this repository.

Additionally, in the `input_data` folder, where the opacities etc. for pRT are saved (see [here](https://petitradtrans.readthedocs.io/en/latest/content/installation.html)), create two additional folders `pinn` (for the absorption PINN) and `rayleigh_pinn` (for the scattering PINN). You can now insert the `model.pkl` file, which you wish to use, into these folders.

To actually use the PINNs, create an atmosphere with the additional arguments:

    from petitRADTRANS import Radtrans
    
    atmosphere = Radtrans([...],
                      enable_pinn=True,
                      enable_rayleigh=True)


You can turn them on or off anytime after intializing them (e.g. `atmosphere.enable_pinn = False`).

Also switching to another model is not a problem:

    atmosphere.model = torch.load('path/to/model.pkl'), map_location=atmosphere.dev) # aborption PINN
    atmosphere.rayleigh_model = torch.load('path/to/model.pkl'), map_location=atmosphere.dev) # scattering PINN

If you have any GPU memory issue or want to increase the performance when using the absorption PINN, you can adjust the used batch size with e.g. `atmosphere.batch_size = int(2**22)`.

For the scattering PINN you can additionally adjust the parameter `atmosphere.rayleigh_on`, which, if set to `False`, uses only the absorption component of the light predicted by the scattering PINN, in order to asses its accuracy.
To adjust the angular extent of the star $\Delta_*$, one can also change the parameter `atmosphere.incident_angle`, in units of $\pi$. (See the Jupyter Notebook `4_scaPINN_testRayleigh.ipynb` for an example.)






