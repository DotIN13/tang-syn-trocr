## Training logs

The model converged faster with 50k steps on a batch size of 540, compared with training for only 10k steps on a batch size of 720 (one augmentation only for each online generated image sample).

The difference could lie in the different choice of pretained decoder, or the augmentation method employed. Under current circumstances, the next step would be to keep using the Mengzi decoder, and go back to the original augmentation method that simultaeously apply various types of visual distortions. So that we will be able to observe the possible performance differences between the two methods of data augmentation.

The experiment with the augmentation method revealed the inefficiency of the mixed augmentations. The model converged slower when applying multiple augmentations to the same image sample.