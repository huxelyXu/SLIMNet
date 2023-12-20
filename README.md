# SLIMNet
 
```
.
├── electrolyte                    # test of electrolyte
├── small_dataset_test             # test of small dataset
|   └── slim-morgan                # SLIMNet with Morgan fingerprint 
|   └── slim-schnet                # SLIMNet with SchNet fingerprint 
├── LICENSE
└── README.md
```

All relevant data are available from the authors upon reasonable request 

### test of electrolyte:
Modified from [chemarr](https://github.com/learningmatter-mit/Chem-prop-pred).
Property prediction of solid-state polymer electrolytes 
obtained by modifying the output of chemprop using a rule based on the scaling law

#### Scripts
`pip install chemprop`

`cd electrolyte`

`python train_and_plot_cv_models.py`

run `python train_and_plot_cv_models.py --help` to see the possible commands of training.
the usage of GPU's has been by default turned off, can be enabled by having a `--gpu` argument flag.
The code uses slimnet for output by default.
To compare chemarr results, change the `--outputmode` in `argument` and `pred_args` to `arr` in the `train_and_plot_cv_models.py` file.


### test of small dataset:
The example data provided in [RadonPy](https://github.com/RadonPy/RadonPy) were selected for testing on a small dataset containing a total of 1070 amorphous polymers made up of monomers.
and four 
properties (thermal conductivity, thermal diffusivity, dielectric constant, linear expansion coefficient) were selected for prediction.
Small molecule characterisations were obtained for model construction using Morgan fingerprints and SchNet, placed in folders `slim-morgan` and `slim-morgan` respectively

#### Scripts
`pip install rdkit`

`conda install pyg`

slim-morgan：Tests on small datasets were performed using a method based on Morgan's fingerprints, 
and the testing process is detailed in document [RF-slim-test.ipynb](small_dataset_test%2Fslim-morgan%2FRF-slim-test.ipynb)
slim-schnet：Tests on small datasets were performed using a method based on SchNet,
Use file [test-schnet.py](small_dataset_test%2Fslim-schnet%2Ftest-schnet.py) to test the effect of simple schnet, 
and use file [test-slimschet.py](small_dataset_test%2Fslim-schnet%2Ftest-slimschet.py) to test the effect of schnet with slimnet added.
Use file [test-qm9.py](small_dataset_test%2Fslim-schnet%2Ftest-qm9.py) for model pre-training, 
and use file [test-transslimschnet.py](small_dataset_test%2Fslim-schnet%2Ftest-transslimschnet.py) to test the effect of slim-schnet after pre-training
