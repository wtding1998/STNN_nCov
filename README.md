# STNN 使用说明

## 数据
`./data`目录下每一个文件夹对应一个数据集，文件夹的名称即为数据集的名称。仓库存放的数据集以及训练集的划分如下：

| 名称  | 时间段 |train | validate | test| 说明|
| --- | :--- |:--- | :--- | :---| :---|
| Jan  | 01/22-02/05| 8 | 2 | 5| 中国一月数据集|
| Feb  | 02/01-03/05| 26 | 3 | 5| 中国二月数据集|
| Mar  | 02/01-03/05| 27 | 4 | 5| 中国三月数据集|
| Peak  | 01/22-02/19| 21 | 3 | 5|中国二月波峰数据集|
| Italy  | 03/01-12/01| 250 | 10 | 15| 意大利数据集|
| USA  | 05/01-12/16| 190 | 10 | 30| 美国数据集|

每个数据集下有两个文件夹: `time_data`和 `overall_relations`，分别用来存放时间序列数据和空间关系矩阵，形如
```
dataset
├── overall_relations
│   ├── relation1.csv
│   ├── relation2.csv
│   └── ...
└── time_data
    ├── date1.csv
    ├── data2.csv
    └── ...
```

时间数据是时间序列形式的数据，以 `.csv`的格式保存在数据集目录下`time_data`文件夹中。每种时间数据对应一个文件，每个文件保存了各个地区在各个时间点中的数据，其中每行为一天中各个地区的数据。例如意大利数据集中 `./data/Italy/time_data/confirmed.csv`共有 275行，21列，代表它存放了意大利21个地区275天的现存确诊数据。

空间关系是矩阵形式的数据，以`.csv`的格式保存在数据集目录下`overall_relatinos`文件夹中。每种空间关系对应一个文件，每个文件保存了各个地区相互之间影响强弱的数据，其中第$i$行第$j$列代表第$i$个地区对第$j$个地区的影响强度。例如意大利数据集中 `./data/Italy/overall_relatinos/distance_region.csv`共21行21列，代表了意大利21个地区之间距离的远近。


## 训练模型

### 模型
模型储存在 `./stnn.py`文件夹中，其中有三个类 `SaptioTemporalNN_classical, SaptioTemporalNN_A, SaptioTemporalNN_I`， 它们分别对应`STNN, STNN-A, STNN-I`这三个模型。

### 训练模型
模型的训练是通过 `./train_stnn.py`来进行的。由于每次都需要指定不同的参数，所以在这里参数被设置为可以从命令行中传入，不传入的参数则为文件中的默认值。可以在命令行中输入
```bash
python train_stnn.py --parmaeter1 value1 --parameter2 value2 ...
```
或在 `Jupyter Notebook` 的代码块中运行
```bash
run train_stnn.py --parmaeter1 value1 --parameter2 value2 ...
```

### 参数设置
参数在 `./train_stnn.py`中通过
```python
p.add('--parameter', type=str, help='set parameter', default='parameter')
```
来设置， 最后所有的参数以字典的形式储存在变量`opt`中。可以通过`opt.parameter` 或 `opt['parameter']`来访问参数 `parameter`的值。例如
```python
p.add('--dataset', type=str, help='dataset name', default='Italy')
```
设置了参数 `dataset` 的类型为 `str` ，默认值为 `'Italy'` 。若需要修改该参数的值为`USA`，可以通过在命令行中输入
```bash
python train_stnn.py --dataset USA
```
来修改该参数的值，并在文件中通过 `opt.dataset` 或者 `opt['dataset']` 来访问。参数的说明参考代码中设置该参数时的`help`部分。

### 停止训练
由于 STNN 参数较多，非常容易过拟合，所以在合适的时间中止训练时非常必要的。并且模型的测试误差往往变得非常快，所以再快要到需要停止的轮次时，通过认为降低学习率，能够比较容易把握到停止的时机。

在训练过程中，有两种终止的情况，一种是训练的轮数超过了设定的 `nepoch` ，另一种是学习率小于等于 `1e-6` 。代码中有两处控制学习率的地方：`reduce_start` 和 `es_start` 。

`reduce_start` 是人为设置学习率降低，当训练的轮次超过这一参数之后，将学习率乘以一定的倍数，该倍数由 `reduce_factor` 来设置。当 `reduce_start` 设置为0时，它将不起作用。

`es_start` 是通过[pytorch](https://pytorch.org/)的[torch.optim.lr_scheduler.ReduceLROnPlateau](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)来设置的。当 `validate` 设置为 `True` 之后，每轮训练之后会计算 `val_sum_score` ，当训练轮次超过 `es_start` ，且该值小于等于 `es_val_bound` 时，考虑减低学习率。若在经过 `patience` 轮次之后，该值还未下降，就会将学习率乘以 `es_wd_factor` 。当`patience > 0` 且 `es_start > 0` 时，该机制生效。

## 结果

在训练结束后，结果会以目录的形式保存在 `./output/outputdir/xp` 目录下，其中 `outputdir` 和 `xp` 是设置的参数，该目录形如

```
xp
├── config.json
├── fitting.pdf
├── logs.json
├── logs.pdf
├── model.pt
├── pred_data1.txt
├── pred_data2.txt
├── ...
└── model.txt
```
其中 `config.json` 中保存了本次训练的参数，以及最终的误差， `logs.json` 保存了每轮的训练误差，验证误差，测试误差等等。 `fitting.pdf` 保存了真实数据与该模型在数据集上的拟合的对比， `logs.pdf` 是每轮的训练误差，测试误差等的曲线。 `model.pt` 是模型的文件。 `STNN.txt` 是该模型在整个数据集上的拟合结果，只包含 `indicate_data` ，  `pred_confirmed.txt` 是该模型在验证集以及测试集上的预测值。

对于单个实验，可以通过 `./result.py` 中的 `Exp` 类来进行加载:
```python
exp = Exp(xp, outputdir)
```
它的参数以及结果储存在字典 `exp.config` 中，模型储存在 `exp.model` 中。

对于多个实验若它们都存放在 `./output/outputdir` 目录下 ，可以通过 `./result.py` 中的 `FolderResult` 来显示它们的结果。
```python
result = FolderResult(outputdir)
```
通过
```python
result.get_result_df(col=['rmse_train_loss', 'rmse_test_loss'])
```
显示所有实验的 `rmse_test_loss` 和 `rmse_train_loss`。通过
```python
result.plot_fitting()
```
显示所有实验的拟合结果。
