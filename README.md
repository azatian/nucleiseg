## Nuclei Segmentation of Monkey V1 X-ray


nuclei segmentation pipeline including data ingestion, preprocessing and U-Net model deployment and inference of macaque V1-V2 XNH dataset



## Run

```bash
python pipeline.py
```

## Code

**download.py** 
  Downloads nuclei annotations from WebKnossos, the digital platform we use for painting cellular features

**pipeline.py**
  Driver code that reads in customizable model configuration yaml file and implements U-Net and saves final weights

**inferencer.py**
  Loads model weights to predict nuclei masks from new data

## Model Configuration File
  Customizable, edit here: **input/config.yaml**

```jsx
dataset:
  ann_path: data/annotations_10_17_23.pickle
  vol_path: data/wk_ids_10_16_23.csv
optimizer:
  choice: Adam
  initial_lr: 0.0001
train:
  logger: True
  id: UNet1
  logger_path: runs/
  weights_path: outputs/weights/
  batch_size: 16
  epochs: 1000
  depth: 4
  metric: Dice Coefficient
  test_size: .2
  random_state: 49
```
