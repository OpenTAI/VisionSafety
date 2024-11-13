---
title: Leaderboards
blocks:
  - titleen: Adversarial Robustness Leaderboards
    titlezh: 对抗鲁棒性排行榜
    subtitleen: >-
      Black-box & white-box: evaluated the top-10 most downloaded or cited
      models in 5 popular vision tasks, on domain datasets as well as
      CC1M-Adv-C/F
    subtitlezh: OpenTAI Rank is exploring中文
    buttonTexten: Learn how it works
    buttonTextzh: 了解评测细节
    tableTitleen: Rankings
    tableTitlezh: Rankings
    tab1en: Black-box
    tab1zh: 黑盒
    tab2en: White-box
    tab2zh: 白盒
    modelsRanking:
      - titlezh: 图像分类
        titleen: Image Classification
        rankings:
          - nameen: resnet50.a1_in1k
            namezh: resnet50.a1_in1k
            paper: 'https://huggingface.co/timm/resnet50.a1_in1k'
            download: 9271420
            datasetA: 35.09%
            datasetB: 14.61%
            ranking: '4'
          - nameen: vit_small_patch16_224.augreg_in21k_ft_in1k
            namezh: vit_small_patch16_224.augreg_in21k_ft_in1k
            paper: >-
              https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
            download: 546023
            datasetA: 32.62%
            datasetB: 27.21%
            ranking: '5'
          - nameen: tf_mobilenetv3_small_minimal_100.in1k
            namezh: tf_mobilenetv3_small_minimal_100.in1k
            paper: 'https://huggingface.co/timm/tf_mobilenetv3_large_minimal_100.in1k'
            download: 783520
            datasetA: 8.99%
            datasetB: 24.83%
            ranking: '10'
          - nameen: resnet18.fb_swsl_ig1b_ft_in1k
            namezh: resnet18.fb_swsl_ig1b_ft_in1k
            paper: 'https://huggingface.co/timm/resnet18.fb_swsl_ig1b_ft_in1k'
            download: 207942
            datasetA: 24.59%
            datasetB: 12.14%
            ranking: '7'
          - nameen: mobilenetv3_large_100.ra_in1k
            namezh: mobilenetv3_large_100.ra_in1k
            paper: 'https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k'
            download: 408002
            datasetA: 23.82%
            datasetB: 12.28%
            ranking: '8'
          - nameen: efficientnet_b3.ra2_in1k
            namezh: efficientnet_b3.ra2_in1k
            paper: 'https://huggingface.co/timm/efficientnet_b3.ra2_in1k'
            download: 2391229
            datasetA: 11.30%
            datasetB: 8.18%
            ranking: '9'
          - nameen: resnet18.a1_in1k
            namezh: resnet18.a1_in1k
            paper: 'https://huggingface.co/timm/resnet18.a1_in1k'
            download: 872296
            datasetA: 27.11%
            datasetB: 14.91%
            ranking: '6'
          - nameen: davit_base.msft_in1k
            namezh: davit_base.msft_in1k
            paper: 'https://huggingface.co/timm/davit_base.msft_in1k'
            download: 1720
            datasetA: 83.25%
            datasetB: 38.06%
            ranking: '3'
          - nameen: coatnet_rmlp_nano_rw_224.sw_in1k
            namezh: GPT-4Test
            paper: 'https://huggingface.co/timm/coatnet_rmlp_nano_rw_224.sw_in1k'
            download: 467
            datasetA: 52.75%
            datasetB: 29.14%
            ranking: '2'
          - nameen: coatnet_rmlp_nano_rw_384.sw_in1k
            namezh: coatnet_rmlp_nano_rw_384.sw_in1k
            paper: 'https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
            download: 764
            datasetA: 67.49%
            datasetB: 31.58%
            ranking: '1'
      - titlezh: 目标检测
        titleen: Object Detection
        rankings:
          - nameen: atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco
            namezh: atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco
            paper: >-
              https://platform.openmmlab.com/modelzoo/mmdetection/DyHead?athId=b19bf998702a943f70e46d53b1054e51&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
            download: 646
            datasetA: '0.447'
            datasetB: '69.43'
            ranking: '1'
          - nameen: yolox_x_8x8_300e_coco
            namezh: yolox_x_8x8_300e_coco
            paper: >-
              https://platform.openmmlab.com/modelzoo/mmdetection/YOLOX?athId=e0fd346d0ae014efd2de972e6df9dea8&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
            download: 5152
            datasetA: '0.426'
            datasetB: 76.75%
            ranking: '2'
          - nameen: deformable-detr(deformable_detr_refine_r50_16x2_50e_coco)
            namezh: deformable-detr(deformable_detr_refine_r50_16x2_50e_coco)
            paper: >-
              https://platform.openmmlab.com/modelzoo/mmdetection/Deformable%20DETR?athId=45f3fa81f746aef44a5b0eb2eacb16c1&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
            download: 6510
            datasetA: '0.286'
            datasetB: 68.12%
            ranking: '3'
          - nameen: faster_rcnn_x101_64x4d_fpn_1x_coco
            namezh: faster_rcnn_x101_64x4d_fpn_1x_coco
            paper: >-
              https://platform.openmmlab.com/modelzoo/mmdetection/Faster%20R-CNN?athId=6e1c4a83606f2a559343d2c69c93d10f&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
            download: 50486
            datasetA: '0.28'
            datasetB: 71.85%
            ranking: '4'
          - nameen: deformable_detr_r50_16x2_50e_coco
            namezh: deformable_detr_r50_16x2_50e_coco
            paper: >-
              https://platform.openmmlab.com/modelzoo/mmdetection/Deformable%20DETR?athId=45f3fa81f746aef44a5b0eb2eacb16c1&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
            download: 5414
            datasetA: '0.276'
            datasetB: 66.27%
            ranking: '5'
          - nameen: Model Name
          - nameen: Model Name
          - nameen: detr-resnet-50(detr_r50_8x2_150e_coco)
            namezh: detr-resnet-50(detr_r50_8x2_150e_coco)
            paper: >-
              https://platform.openmmlab.com/modelzoo/mmdetection/DETR?athId=b609e23c7b56f32054cf4a85c0ef9c01&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
            download: 566289
            datasetA: '0.241'
            datasetB: 63.84%
            ranking: '8'
          - nameen: Model Name
      - titlezh: 实例分割
        titleen: Instance Segmentation
      - titlezh: 语义分割
        titleen: Semantic Segmentation
      - titlezh: 医学图像分类
        titleen: Medical Image Classification
    modelsRanking1:
      - titlezh: ImageNet Classfication
        titleen: ImageNet Classfication
        rankings:
          - nameen: GPT-4
            namezh: GPT-4Test
            paper: >-
              Robust Principles: Architectural DesignPrinciples for
              Adversarially Robust...
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '1'
          - nameen: Llama 3
            namezh: Llama 3
            paper: Better Diffusion Models Further lmprove Adversarial Training
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '2'
          - nameen: Claude 3
            namezh: Claude 3
            paper: >-
              MixedNUTS: Training-Free Accuracy-RobustnessBalance via
              Nonlinearly Mixed...
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '3'
          - nameen: Ernie-4.0
            namezh: Ernie-4.0
            paper: >-
              Improving the Accuracy-Robustness Trade-off ofClassifiers via
              Adaptive Smoothing
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '4'
          - nameen: GPT-3.5 turbo
            namezh: GPT-3.5 turbo
            paper: Better Diffusion Models Further lmprove Adversarial Training
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '5'
          - nameen: GPT-3.5
            namezh: GPT-3.5
            paper: >-
              MixedNUTS: Training-Free Accuracy-RobustnessBalance via
              Nonlinearly Mixed...
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '6'
          - nameen: xAI-Grok 2
            namezh: xAI-Grok 2
            paper: Better Diffusion Models Further lmprove Adversarial Training
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '7'
      - titlezh: CC1M
        titleen: CC1M
        rankings:
          - nameen: GPT-41
            namezh: GPT-4Test
            paper: >-
              Robust Principles: Architectural DesignPrinciples for
              Adversarially Robust...
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '1'
          - nameen: Llama 31
            namezh: Llama 3
            paper: Better Diffusion Models Further lmprove Adversarial Training
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '2'
          - nameen: Claude 31
            namezh: Claude 3
            paper: >-
              MixedNUTS: Training-Free Accuracy-RobustnessBalance via
              Nonlinearly Mixed...
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '3'
          - nameen: Ernie-4.01
            namezh: Ernie-4.0
            paper: >-
              Improving the Accuracy-Robustness Trade-off ofClassifiers via
              Adaptive Smoothing
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '4'
          - nameen: GPT-3.5 turbo1
            namezh: GPT-3.5 turbo
            paper: Better Diffusion Models Further lmprove Adversarial Training
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '5'
          - nameen: GPT-3.5
            namezh: GPT-3.5
            paper: >-
              MixedNUTS: Training-Free Accuracy-RobustnessBalance via
              Nonlinearly Mixed...
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '6'
          - nameen: xAI-Grok 2
            namezh: xAI-Grok 2
            paper: Better Diffusion Models Further lmprove Adversarial Training
            download: 112092
            datasetA: 76.5%
            datasetB: 76.5%
            ranking: '7'
    _template: table
---

