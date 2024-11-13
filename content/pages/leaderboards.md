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
          - nameen: vit_small_patch16_224.augreg_in21k_ft_in1k
            namezh: vit_small_patch16_224.augreg_in21k_ft_in1k
            paper: >-
              https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
            download: 546023
            datasetA: 32.62%
            datasetB: 27.21%
          - nameen: tf_mobilenetv3_small_minimal_100.in1k
            namezh: tf_mobilenetv3_small_minimal_100.in1k
            paper: 'https://huggingface.co/timm/tf_mobilenetv3_large_minimal_100.in1k'
            download: 783520
            datasetA: 8.99%
            datasetB: 24.83%
          - nameen: resnet18.fb_swsl_ig1b_ft_in1k
            namezh: resnet18.fb_swsl_ig1b_ft_in1k
            paper: 'https://huggingface.co/timm/resnet18.fb_swsl_ig1b_ft_in1k'
            download: 207942
            datasetA: 24.59%
            datasetB: 12.14%
          - nameen: mobilenetv3_large_100.ra_in1k
            namezh: mobilenetv3_large_100.ra_in1k
            paper: 'https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k'
            download: 408002
            datasetA: 23.82%
            datasetB: 12.28%
          - nameen: efficientnet_b3.ra2_in1k
            namezh: efficientnet_b3.ra2_in1k
            paper: 'https://huggingface.co/timm/efficientnet_b3.ra2_in1k'
            download: 2391229
            datasetA: 11.30%
            datasetB: 8.18%
          - nameen: resnet18.a1_in1k
            namezh: resnet18.a1_in1k
            paper: 'https://huggingface.co/timm/resnet18.a1_in1k'
            download: 872296
            datasetA: 27.11%
            datasetB: 14.91%
          - nameen: davit_base.msft_in1k
            namezh: davit_base.msft_in1k
            paper: 'https://huggingface.co/timm/davit_base.msft_in1k'
            download: 1720
            datasetA: 83.25%
            datasetB: 38.06%
          - nameen: coatnet_rmlp_nano_rw_224.sw_in1k
            namezh: GPT-4Test
            paper: 'https://huggingface.co/timm/coatnet_rmlp_nano_rw_224.sw_in1k'
            download: 467
            datasetA: 52.75%
            datasetB: 29.14%
            ranking: ''
          - nameen: coatnet_rmlp_nano_rw_384.sw_in1k
            namezh: coatnet_rmlp_nano_rw_384.sw_in1k
            paper: 'https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k'
            download: 764
            datasetA: 67.49%
            datasetB: 31.58%
      - titlezh: 目标检测
        titleen: Object Detection
        rankings:
          - nameen: detr-resnet-50(detr_r50_8x2_150e_coco)
            namezh: detr-resnet-50(detr_r50_8x2_150e_coco)
            paper: >-
              https://platform.openmmlab.com/modelzoo/mmdetection/DETR?athId=b609e23c7b56f32054cf4a85c0ef9c01&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
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

