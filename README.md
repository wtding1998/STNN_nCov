# Predict the Distribution of Disease by STNN

This project is supported by the National Natural Science Foundation of China (Grant No: 11601327) and the Key Construction National “985” Program of China (Grant No: WF220426001).


## Data
### nCov2019
The folder `data/ncov/time_datas` contains the raw data, including confirmed, cured and dead data. The 233 rows correspond to the 233 timestep, and the 31 columns are the 31 space points.
The folder `data/ncov/relations` contains the spatial relations between the 31 space points. `location.csv` is a 31 by 31 adjacency matrix _A_, where _A(i, j)_ = 1 means that point _i_ is a direct neighbor of point _j_ in space, and is 0 otherwise.`distance.csv` is a 31 by 31 matrix _B_, where _B(i, j)_ means the reciprocal of distance between points _i_ and _j_.`migration.csv` is a 31 by 31 matrix _C_, where _C(i, j)_ means the migration speed from points _i_ to _j_.

## Model
#### STNN
`python train_stnn.py --dataset ncov --outputdir output_ncov --manualSeed 1932 --xp stnn`

#### STNN-A(rgument)
`python train_stnn_A.py --dataset ncov --outputdir output_ncov --manualSeed 3301 --xp stnn_a --patience 800 --l1_rel 1e-8`

#### STNN-I(nput)
`python train_stnn_I.py --dataset ncov --outputdir output_ncov --manualSeed 1290 --xp stnn_r --patience 1000 --l1_rel 3e-6`