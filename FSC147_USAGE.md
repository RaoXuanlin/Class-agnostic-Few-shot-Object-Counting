# FSC-147 Dataset Usage Guide

This guide explains how to use the FSC-147 dataset for class-agnostic few-shot object counting with this repository.

## What is FSC-147?

FSC-147 (Few-Shot Counting) is a dataset specifically designed for few-shot object counting tasks. Unlike COCO, which is a general object detection dataset, FSC-147 is purpose-built for counting with:

- **6,135 images** with diverse object classes
- **3 exemplar bounding boxes** per image (reference objects)
- **Point annotations** for each object instance
- **Pre-defined train/val/test splits**
- **Class-agnostic design** - no category labels needed

## Dataset Structure

After downloading FSC-147, your directory should look like this:

```
FSC147_384_V2/
├── images_384_VarV2/
│   └── *.jpg (6135 images at 384x384 resolution)
├── annotation_FSC147_384.json
├── Train_Test_Val_FSC_147.json
└── ImageClasses_FSC147.txt
```

## Downloading FSC-147

1. Visit the official repository: https://github.com/cvlab-stonybrook/LearningToCountEverything
2. Follow their instructions to download FSC-147_384_V2.zip
3. Extract the dataset to your preferred location

## Configuration

### Step 1: Update config_fsc147.yaml

Edit `configs/config_fsc147.yaml` and update the `data_path`:

```yaml
data:
  dataset: fsc147
  data_path: /your/path/to/FSC147_384_V2  # Update this!
  num_references: 3
```

### Step 2: Configure Training Parameters (Optional)

You can adjust training parameters in the same config file:

```yaml
train:
  epochs: 5
  batch_size: 12
  num_workers: 2
  ssim_loss: 1.0e-5
  references: 3
  result_path: ./results
```

## Training on FSC-147

### Basic Training

```bash
python main.py --config=config_fsc147.yaml --doc=fsc147_exp --train
```

### Training with Custom Settings

```bash
python main.py --config=config_fsc147.yaml --doc=my_experiment --train --seed=42
```

Training logs and checkpoints will be saved to `exp/logs/my_experiment/`

## Testing/Evaluation on FSC-147

### Step 1: Set Checkpoint Path

Edit `configs/config_fsc147.yaml` and update the checkpoint path:

```yaml
eval:
  checkpoint: ./exp/logs/fsc147_exp/model_epoch_4.pth  # Update this!
  sample: true
  image_folder: ./eval_results
```

### Step 2: Run Evaluation

```bash
python main.py --config=config_fsc147.yaml --doc=fsc147_test --test
```

The evaluation will:
- Use the pre-defined **test split** from FSC-147
- Calculate MAE (Mean Absolute Error) and MSE (Mean Squared Error)
- Optionally save visualization samples if `sample: true`

## Key Differences from COCO

| Feature | COCO Dataset | FSC-147 Dataset |
|---------|--------------|-----------------|
| Reference Images | 500 cropped per category | 3 exemplar boxes per image |
| Annotations | Bounding boxes + segmentation | Point annotations |
| Preprocessing | Requires running crop.py | No preprocessing needed |
| Shots | 5-shot (configurable) | 3-shot (native to dataset) |
| Categories | 80 object categories | Class-agnostic |
| Training Split | Custom fold-based | Pre-defined splits |

## Understanding the Dataset Loader

The FSC-147 dataset loader (`data/fsc147.py`) handles:

1. **Loading annotations** from `annotation_FSC147_384.json`
2. **Extracting exemplar regions** from the 3 bounding boxes per image
3. **Generating density maps** from point annotations
4. **Applying transforms** to query and reference images
5. **Managing train/val/test splits** automatically

### Data Format

Each batch contains:
- **Query tensor**: `(batch_size, 3, 256, 256)` - The image to count objects in
- **Reference tensor**: `(batch_size, 3, 3, 64, 64)` - 3 exemplar crops
- **Density map**: `(batch_size, 1, 256, 256)` - Ground truth density map

## Tips for Best Results

1. **Start with pre-trained weights**: If available, use weights from COCO training as initialization
2. **Adjust learning rate**: FSC-147 is smaller than COCO, consider lower learning rate
3. **Use 3-shot**: FSC-147 provides 3 exemplars, which matches the dataset design
4. **Monitor both MAE and MSE**: These are the standard metrics for counting tasks

## Troubleshooting

### Issue: "Failed to load image"
- Check that `data_path` in config points to the correct directory
- Verify that `images_384_VarV2/` subdirectory exists

### Issue: ModuleNotFoundError
- Run `pip install -r requirements.txt`
- Ensure all dependencies are installed

### Issue: Out of memory
- Reduce `batch_size` in config_fsc147.yaml
- Use fewer `num_workers`

## Example: Quick Start

```bash
# 1. Download FSC-147 dataset
# 2. Update config
nano configs/config_fsc147.yaml  # Set your data_path

# 3. Train
python main.py --config=config_fsc147.yaml --doc=quickstart --train

# 4. Test
# First, update checkpoint path in config_fsc147.yaml
python main.py --config=config_fsc147.yaml --doc=quickstart_test --test
```

## Further Reading

- **FSC-147 Paper**: "Learning To Count Everything" (CVPR 2021)
- **Original Paper**: "Class-Agnostic Few-Shot Object Counting" (WACV 2021)
- **FSC-147 Dataset**: https://github.com/cvlab-stonybrook/LearningToCountEverything

## Support

If you encounter issues specific to FSC-147 dataset:
1. Check that your dataset structure matches the expected format
2. Verify the JSON files are valid and contain the expected fields
3. Ensure images are in the correct directory

For general model or training issues, refer to the main README.md
