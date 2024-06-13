# DEMAL_Model
Dual-Encoding Matching Adversarial Learning for Image Cartoonlization

## Abstract
Generative Adversarial Network (GAN)-based image cartoonization has made great progress. They 
usually use a ``single-encoding adversarial feedback architecture'' to generate cartoon image in a similar cartoon-style domain. However, this architecture cannot generate a satisfactory cartoonized image with both high style similarity and visual fidelity. In this work, to relieve this problem, we propose a novel dual-encoding matching adversarial learning dubbed DEMAL for image cartoonlization. Particularly, we first design a dual-encoding matching (DEM) by using a pair of dual encoders and a statistical matching module (SM) to match the content-style feature encodings extracted separately in the statistical space. We then construct double-structure style discriminators to adversarially learn global and local feature representations of cartoon-style via the improved loss function. Furthermore, we also propose a pre-training strategy for the DEMAL to achieve the best FID and ArtFID distance. Extensive experiments have demonstrated that our proposed DEMAL achieves high visual fidelity and style similarity compared to the previous representative baseline cartoonization methods.
## Framework
![DEMAL](https://github.com/ZYDeeplearning/DEMAL_Model/blob/main/1.png)
The overall pipeline of our DEMAL framework.

## Environment
You can visit [Kaggle ((https://www.kaggle.com))] 

## Pretrained models
You can visit[Google(https://drive.google.com/drive/folders/1Bk0H3VrkdkeLuxW7Rieki-BvhqKPxNdC?usp=drive_link)], includings the training weights on the 'Hayao', 'Shinkai'', and 'Paprika'' datasets.

## Run
Inculding 20 epoch Pretraining, you can set the ``--init_train''==True. 
```
python main.py 
```
