# Gui-element Detection using YOLOv5, YOLOv6, YOLOv7, and YOLOv8

[Link to the paper](https://arxiv.org/abs/2408.03507)

[VINS Dataset](https://github.com/sbunian/VINS)

## Dataset Pre-Processing and Training
1. Create anaconda environment called **gui_det**.
   ```bash
   git clone https://github.com/SookmyungHumanAI/gui-element-detection.git
   conda create -n gui_det python==3.9
   conda activate gui_det
   conda install jupyter
   ```

2. Create a folder named `src/Dataset`.
3. Download the **VINS_Dataset** from the link above into the `src/Dataset` directory.
4. In the `src/constants.py` file, set the `VINS_DATASET_PATH` variable to the absolute path of `src/Dataset`.
5. Run the `src/data_cleaning.ipynb` notebook:<br>
   (1) Merge the four datasets (`Android`, `Rico`, `iPhone`, `Uplabs`) inside the `src/Dataset/Merged` folder, excluding the `Wireframes` dataset.<br>
   (2) Perform data preprocessing to generate the required `.txt` files for YOLO, and save them in the `yolo5_format` folder.<br>
   (3) Combine images and annotations into a folder named `yolo5_full`.<br>
6. Run the `src/yolo/yolov5.ipynb` notebook:<br>
   (1) Add the absolute path of the `src` folder to `sys.path` using `sys.path.insert(1, "absolute path to src")`.<br>
   (2) `git clone https://github.com/ultralytics/yolov5`<br>
   (3) `pip install -r requirements.txt`<br>
   (4) Check if CUDA is available:<br>
      ```bash
      python -c "import torch; print(torch.cuda.is_available())"
      >>> True
      ```
      else:
      ```bash
      pip uninstall torch torchvision
      conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
      ```
   (5) Create `train`, `validation`, and `test` folders and split the dataset accordingly.<br>
   (6) Check if the variable **TRAIN = True**
   (7) Finally, Train Yolov5 for GUI detection. (Related paper was 416 with batches of 16)<br>
      Replace yolo5.yaml to `yolov5` and rename it to dataset.yaml and change the src path.
   ```bash
   python train.py --data dataset.yaml --weights yolov5s.pt --img 416 --epochs {EPOCHS} --batch-size 16 --name {RES_DIR}
   ```

## Folder Structure
Below is the folder structure of the project.
```bash
ckpt/
src/
├── Dataset/
│   ├── Android/
│   ├── iphone/
│   ├── Merged/
│   │   ├── annotations/
│   │   ├── dataset/
│   │   ├── images/
│   │   ├── yolo5_format/
│   │   └── yolo5_full/
│   ├── Rico/
│   ├── uplabs/
│   └── Wireframes/
├── util/
└── yolo/
    ├── datasets/
    ├── yolo*/ # YOLO Repositories(if needed)
    └── yolo*.ipynb
```
Focusing on `src/Dataset/Merged`
```bash
Merged/
├── ...
├── dataset/
│   ├── images/
│   │   ├── test/
│   │   ├── train/
│   │   └── validation/
│   └── labels/
│       ├── test/
│       ├── train/
│       └── validation/
├── ...
```

## Related papers' repositories

- [74 - Object Detection for Graphical User Interface: Old Fashioned or Deep Learning or a Combination?](https://github.com/chenjshnn/Object-Detection-for-Graphical-User-Interface)
  (CenterNet, Faster-RCNN, YOLOv3, Xianyu)
- [72 - UIED: a hybrid tool for GUI element detection](https://github.com/MulongXie/UIED)
- [CenterNet v2](https://github.com/xingyizhou/CenterNet2)

### Important Papers

- 72 - UIED: a hybrid tool for GUI element detection
- 74 - Object detection for graphical user interface: old-fashioned or deep learning or a combination?
- 75 - GUI Widget Detection and Intent Generation via Image Understanding

### Datasets
- RICO -> 72 and 74 : This dataset (android only) is pretty much crap, so much preprocessing is required to make it work, and other authors who used it did not publish their preprocessed dataset 
- Image2emmet: very small dataset, both for web and android
- [The ReDraw Dataset](https://zenodo.org/record/2530277#.ZAQ5mXbMJ3g)
- [VINS Dataset](https://github.com/sbunian/VINS)


#### Report
- cleaned the data and transform xml to txt (yolo5_format folder) -> See data_cleaning file
- see readme.md in yolo directory

## Metrics
- [https://medium.com/@vijayshankerdubey550/evaluation-metrics-for-object-detection-algorithms-b0d6489879f3](https://medium.com/@vijayshankerdubey550/evaluation-metrics-for-object-detection-algorithms-b0d6489879f3)

## Citation
```
@misc{daneshvar2024guielementdetectionusing,
      title={GUI Element Detection Using SOTA YOLO Deep Learning Models}, 
      author={Seyed Shayan Daneshvar and Shaowei Wang},
      year={2024},
      eprint={2408.03507},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.03507}, 
}
```
