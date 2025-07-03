# Name Generator with LSTM

A simple character-level name generator using PyTorch.

---

### Features
- LSTM model with 3 layers, 64 hidden units each
- Trained on [a list of English names](https://github.com/spro/practical-pytorch/blob/master/data/names/English.txt)
- Generates new names from given initials
- Optional tree-style visualization of generation path

### Training
```bash
python train.py
``` 
Trained for 100 epochs (~62 minutes), with Adam and CrossEntropyLoss.
Training results were as follows:
```
Epoch 20 | loss: 1.7444231018765282
Epoch 40 | loss: 1.4955448371032203
Epoch 60 | loss: 1.4187650525274167
Epoch 80 | loss: 1.3883606963907749
Epoch 100 | loss: 1.3742179288696437
```

### Sample Outputs
```
A -> Aylly
M -> Milding
Jo -> Joulson
M -> Mcmaotley
```

### Tree Visualization
Visualize model's top predictions step by step:
```bash
python generate.py
```
