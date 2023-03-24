# partialpgd
PGD attack on feature space

About multiple images and attempts:
1. i thought since our final goal is to maximize the cost, so multiple images and attempts are ways to provide more cost values we could select from. so i use the max_cost to find the maximize one.

for multiple images:

2. my return from that function is a stack which is 4D. [images per class*10, 128, 16, 16]
3. before the multiple image, we only have 1 image per class, so we could get target output in this way:
  target_outputs = torch.stack([ref_outputs[i] for i in target_labels])
  while we have multiple images, the ith ref_outputs is no longer the corresponding target label one, the ref_outputs[images_per_class * i : images_per_class * i+images_per_class] is the those correspond to the target label i.
4. then the target_outputs we obtained are 5D. [16, images per class*10, 128, 16, 16]
5. the next place use the target_outputs is using loss function to compute the cost, so i iter all the images_per_class, get that each target_output per image, using:
   for j in range(images_per_class):
       cost = -loss(outputs, target_outputs[:,j,:,:,:])
6. and then save the max_cost by compare the cost each time. eventually use that max_cost to compute the grad.


for multiple attempts:

7. since we want to have 10 times of 4 iters, we want the 10 times to be different, so i thought the attempts iter should start at first before any random directions were generated. and set the max_cost and max_image at first so that could compare to obtain the largest cost with its corresponding adv_images.
9. everything happens after are the same, and i only compute the grad after all the iter ends, using the max_cost and max_image to get the final adv_images.
            
