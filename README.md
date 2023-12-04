## Nuclei Segmentation of Monkey V1 X-ray

nuclei segmentation pipeline including data ingestion, preprocessing and U-Net model deployment and inference of macaque V1-V2 XNH dataset

## Run

```bash
python pipeline.py
```

## Code

**download.py** 
  
  - Downloads nuclei annotations from WebKnossos, the digital platform we use for painting cellular features

**pipeline.py**

  - Driver code that reads in customizable model configuration yaml file and deploys U-Net, saving final weights

**inferencer.py**
  
  - Loads model weights to predict nuclei masks from new data

## Model Configuration File
  Customizable in that you can choose hyperparameters, edit here: **input/config.yaml**

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

## Results

![img10](https://github.com/azatian/nucleiseg/assets/9220290/befd524b-8cb4-474b-b507-f68b44211b3d)


![img15](https://github.com/azatian/nucleiseg/assets/9220290/c59bb0fe-ff3e-4008-b823-139bb6383253)
![img14](https://github.com/azatian/nucleiseg/assets/9220290/1d0f4a7e-3b3a-4a7b-b9e2-ee019847bef8)
![img16](https://github.com/azatian/nucleiseg/assets/9220290/253764e3-a863-4108-846d-9b3cd8d7ed38)

