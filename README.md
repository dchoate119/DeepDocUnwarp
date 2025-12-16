# Deep Document Unwarping

**Author:** Daniel Choate 

**Course:** Computer Vision - Final Project (Fall 2025)

**Instructor:** Roy Shilkrot

---

## GOAL 

To design and train a deep learning pipeline that reconstructs a flat, readable document from an image of a crumpled or folded page. 


## Architecture 

*add a diagram*

---
## Loss Function


---
## Results


---
## Metrics



---
### TODO 
**Milestone 1**
- [x] Load and visualize dataset 
- [ ] Implement simple encoder-decoder model
- [ ] Train on the dataset with MSE loss 
- [ ] Evaluate on validation set 

**Milestone 2**
- [ ] Integrate pretrained backbone
- [ ] Add skip connections (U-Net style)
- [ ] Experiment with different loss functions (L1, perceptual, SSIM)
- [ ] Implement proper evaluation metrics 

**Milestone 3** (if time)
- [ ] Use depth/UV information 
- [ ] Add attention mechanisms
- [ ] Implement adversarial training (GAN)
- [ ] Try transformer-based architectures
- [ ] Ensemble multiple models


**For submission**
- [ ] <model.py> torch model definition (encoder, decoder, unwarping logic)
- [ ] <train.ipynb> code used to train model, training loop and loss curves 
- [ ] <evaluate.py> loads weights and calculates SSIM on validation set 
- [ ] <best_model.pth> trained model weights 