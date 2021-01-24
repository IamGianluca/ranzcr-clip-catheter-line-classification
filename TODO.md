## Todo List

- [x] Create initial skelethon for DVC pipeline
- [x] Create step in DVC pipeline to resize images, in order to avoid having to do this on-the-fly, which would cause sub-optimal GPU utilization
- [ ] Improve cross-validation strategy as there are multiple images for patient. StratifiedGroupKFold might be a good choice
- [ ] Create submission file by rank averaging Out-Of-Folds predictions and record metric in DVC 
- [ ] Use`timm` library to construct network architectures. The library already contains good factories that allow tailoring the architecture to our needs
- [ ] Train different architectures: ResNet, ResNet v2, Xception v3, Xception v4, SE-ResNet, ResNext, SE-ResNext, ResNest, SWSL-ResNext, EfficientNet, EfficientNet-Noisy Student, ViT
- [ ] Investigate usage of image annotations
