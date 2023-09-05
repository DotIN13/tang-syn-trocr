## Training logs

The model converged faster with 50k steps on a batch size of 540, compared with training for only 10k steps on a batch size of 720 (one augmentation only for each online generated image sample).

The difference could lie in the different choice of pretained decoder, or the augmentation method employed. Under current circumstances, the next step would be to keep using the Mengzi decoder, and go back to the original augmentation method that simultaeously apply various types of visual distortions. So that we will be able to observe the possible performance differences between the two methods of data augmentation.

The experiment with the augmentation method revealed the inefficiency of the mixed augmentations. The model converged slower when applying multiple augmentations to the same image sample.

### 2023-09-03

If the training on Google cloud achieved the same result as the July 23 variant, then it should be okay to say that the new data augmentation method works as good as the previous one.

And it could also possibly indicate that the old messier dataset works better.

But it won't provide much insight on the comparison of linear and bicubic interpolation modes, as the dataset is different.

### 2023-09-03-01

The training on Google Cloud using linear interpolation showed that the dataset and interpolation mode does not affect greatly the model convergence.

So the gap in performance might lie in the new elastic method, the new text sampling technique, and the introduction of attention_mask.
