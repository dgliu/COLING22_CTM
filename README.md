## COLING22_CTM

Experiments codes for the paper:

Dugang Liu, Weihao Du, Lei Li, Weike Pan and Zhong Ming. Augmenting Legal Judgment Prediction with Contrastive Case Relations. To appear in COLING '22.

**Please cite our COLING '22 paper if you use our codes. Thanks!**

---

## Usage

<font size=3>**The execution process of the main experiment:**</font>

* Switch to working directory:
```
cd COLING22_CTM/CTM_small

cd COLING22_CTM/CTM_big
```

* Carry out data exploration and obtain experimental data:
```
python3 data_exploration.py

python3 get_processed_data.py
python3 generate_data_structure.py

python3 get_processed_data.py -d 'big/'
python3 generate_data_structure.py -d 'big/'
```

* Tuning model (searcher='optuna'):

    ```
    CUDA_VISIBLE_DEVICES=0 nohup python3 -u tune_parameters.py -tb 'ctm_tuning_0.csv' -y 'config/ctm.yml' -s 0 >ctm_out_0 2>&1 &

    ```

- Train the model according to the best parameters, save the parameters, and output the results:

  ```
  CUDA_VISIBLE_DEVICES=0 python3 reproduce_paper_results.py
  ```

  