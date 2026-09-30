# Dataset Sources

BRIDGE-AD evaluates on MVTec AD, VAD, and MVTec LOCO AD. Please obtain these datasets from their original providers and cite the corresponding work. This repository does not host or redistribute the benchmark images.

## MVTec AD

- Dataset and download instructions: [MVTec AD official page](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).
- Paper: Paul Bergmann, Michael Fauser, David Sattlegger, and Carsten Steger. [MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Bergmann_MVTec_AD_--_A_Comprehensive_Real-World_Dataset_for_Unsupervised_Anomaly_CVPR_2019_paper.html). CVPR, 2019.
- Dataset license: CC BY-NC-SA 4.0, as specified by MVTec.

## VAD

- Dataset and download instructions: [Valeo Anomaly Dataset, authors' repository](https://github.com/abc-125/segad).
- Paper: Aimira Baitieva, David Hurych, Victor Besnier, and Olivier Bernard. [Supervised Anomaly Detection for Complex Industrial Images](https://openaccess.thecvf.com/content/CVPR2024/html/Baitieva_Supervised_Anomaly_Detection_for_Complex_Industrial_Images_CVPR_2024_paper.html). CVPR, 2024.
- Dataset license: CC BY-NC-SA 4.0, as specified in the authors' repository.

VAD here denotes the **Valeo Anomaly Dataset**, not VisA. The authors' repository provides the download link and the original dataset organization.

## MVTec LOCO AD

- Dataset and download instructions: [MVTec LOCO AD official page](https://www.mvtec.com/research-teaching/datasets/mvtec-loco-ad).
- Paper: Paul Bergmann, Kilian Batzner, Michael Fauser, David Sattlegger, and Carsten Steger. [Beyond Dents and Scratches: Logical Constraints in Unsupervised Anomaly Detection and Localization](https://doi.org/10.1007/s11263-022-01578-9). International Journal of Computer Vision, 2022.
- Dataset license: CC BY-NC-SA 4.0, as specified by MVTec.

## BRIDGE-AD Preparation

Original dataset downloads do not by themselves provide the BRIDGE-AD relation registry, training/support manifests, or prepared conditioning assets. The corresponding preparation code and configurations are part of the planned post-acceptance release. The code license of this repository does not change the original datasets' licenses.
