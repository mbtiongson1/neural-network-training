## Results

### **Ranked by F1 Macro**

| Rank | Network | Epoch (MSE<0.02) | f1\_macro | Notes |
| --- | --- | --- | --- | --- |
| **1** | **networkB_improv2** | 73 | **0.9886104267** | |
| **2** | **networkB_improv2_fast** | 51 | **0.9852983084** | |
| **3** | **networkA_improv2_fast** | 44 | **0.9835187069** | |
| **4** | **networkA_improv2_small** | 74 | **0.9810832261** | |
| **5** | **networkA_improv** | 56 | **0.9800595241** | |
| **6** | **networkA_improv2** | 74 | **0.9760872926** | |
| **7** | **networkB_improv** | 71 | **0.9612885197** | |
| **8** | **networkB_improv2_small** | DNF | **0.5564479093** | |
| **9** | **networkA** | DNF | **0.0292494481** | |
| **10** | **networkB** | DNF | **0.0250281215** | |


### Full Scores Results

| Rank | Network | Epoch (MSE<0.02) | F1 Score | MCC | Accuracy | Precision | Recall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **1** | **networkB_improv2** | 73 | **0.9886104266802997** | 0.9871434813936002 | 0.98875 | 0.9880438764134416 | 0.9893306379155435 |
| **2** | **networkB_improv2_fast** | 51 | **0.985298308383793** | 0.9828614495415673 | 0.985 | 0.9856464558169103 | 0.9851633996825357 |
| **3** | **networkA_improv2_fast** | 44 | **0.9835187068670926** | 0.9801059555388116 | 0.9825 | 0.9839927637721755 | 0.983825881940443 |
| **4** | **networkA_improv2_small** | 74 | **0.9810832261192908** | 0.9786034851115598 | 0.98125 | 0.9812392134852626 | 0.9814406841812409 |
| **5** | **networkA_improv** | 56 | **0.9800595240561725** | 0.9772495150791172 | 0.98 | 0.9804202582393017 | 0.9806557825589426 |
| **6** | **networkA_improv2** | 74 | **0.9760872925642827** | 0.9730911194363044 | 0.97625 | 0.9773195018363925 | 0.9768042419426961 |
| **7** | **networkB_improv** | 71 | **0.9612885197229641** | 0.9548088483566725 | 0.96 | 0.9646664883688096 | 0.9614403902328368 |
| **8** | **networkB_improv2_small** | DNF | **0.5564479093333913** | 0.593441665499621 | 0.6125 | 0.5601799678977939 | 0.6089298481433483 |
| **9** | **networkA** | DNF | **0.02924944812362031** | 0.0 | 0.1325 | 0.0165625 | 0.125 |
| **10** | **networkB** | DNF | **0.025028121484814397** | 0.0 | 0.11125 | 0.01390625 | 0.125 |


### Training Time Results

Adding in a value so that we can compare normalized training time to get an equivalent 0.02 MSE:

$$ \text{Time-Adjusted} = \text{Time} × (\text{Epoch}_{\text{converged}} / 100) $$

Special rules:

* If **Epoch\_converged > 100**, multiplier = **1.00**
* If **DNF**, then Time-Adjusted = **N/A**

| Rank | Network | Epoch | Time (s) | Time-Adjusted | Your Notes |
| --- | --- | --- | --- | --- | --- |
| **1** | **networkB_improv2** | 73 | 96.92362 | **70.75424** | |
| **2** | **networkB_improv2_fast** | 51 | 87.41550 | **44.58191** | |
| **3** | **networkA_improv2_fast** | 44 | 67.50442 | **29.70195** | |
| **4** | **networkA_improv2_small** | 74 | 42.35613 | **31.34354** | |
| **5** | **networkA_improv** | 56 | 52.24818 | **29.25898** | |
| **6** | **networkA_improv2** | 74 | 59.33289 | **43.90634** | |
| **7** | **networkB_improv** | 71 | 74.36175 | **52.79684** | |
| **8** | **networkB_improv2_small** | DNF | 45.96956 | **N/A** | |
| **9** | **networkA** | DNF | 44.12977 | **N/A** | |
| **10** | **networkB** | DNF | 50.06487 | **N/A** | |
