
```
WolfPack
├── code
|    ├──best_models
|    ├──processed_data
|    ├──basic_processor.py
|    ├──utils.py
|    ├──FINAL_MMD_JT_HL_CM.ipynb
|    └──README.md
|
├──hl_2a_train_repo
├──jtext_train_repo
├──training_repo
├──testing_repo
├──C-Mod_data.zip
├──HL-2A_data.zip
├──J-TEXT_data.zip
└── requirements.txt
```

### Objective
You will be provided with data from three distinct tokamaks, namely ***C-Mod, J-TEXT, and HL-2A. J-TEXT and HL-2A*** will be used as training set, and the C-Mod data set will be used for evaluation. The objective of this challenge is to develop a disruption prediction model that can be applied universally, using ***J-TEXT and HL-2A as the current devices and C-Mod*** as the future device.

### To run code:
- To set up directory for preprocessing and modelling set rootdir/root_working_dir as **WolfPack**, describe by the tree.
- Download C-Mod data.zip , HL_2A data.zip and JTEXT data.zip from the [competition website](https://zindi.africa/competitions/multi-machine-disruption-prediction-challenge/data) place it within the WolfPack dir.
- Afterwards from the root_working_dir as **WolfPack**, open any code editor of your choice and navigate to FINAL_MMD_JT_HL_CM.ipynb within the code folder and simply run all to generate and save dataframes for training and testing, and reusing the saved data to train 2 gradient boosting models on 12 folds and saves inference file over probability of 0.51% certainty

- requirements.txt - contains used libraries and their specific versions except jddb which was installed through github.

Environment Used: 
- Visual Studio code
- Python 3.10.8
- trained on CPU

Kindly Note to set zip and extracted paths correctly to ensure smooth running of the notebook.
