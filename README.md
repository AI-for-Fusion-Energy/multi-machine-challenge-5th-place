
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

### To run code:
- To set up directory for preprocessing and modelling set rootdir as **WolfPack**, describe by the tree. Afterwards, open any code editor of your choice and navigate to FINAL_MMD_JT_HL_CM.ipynb within the code folder and simply run all to generate and save dataframe for training and testing, and reusing the saved data to train 2 gradient boosting models on 12 folds and saves inference file over probability of 0.51% certainty

- requirements.txt - contains used libraries and their specific versions except jddb which was installed through github

Environment Used: Python 3.10.8 trained on CPU

Kindly Note to set zip and extracted paths correctly to ensure smooth running of the notebook.