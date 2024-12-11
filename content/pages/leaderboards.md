---
title: Leaderboards
blocks:
  - titleen: Adversarial Robustness Leaderboards
    titlezh: 对抗鲁棒性排行榜
    subtitleen: >-
      Black-box and white-box evaluations: conducted for the top 10 most
      downloaded/cited models across five popular vision tasks, using both
      domain-specific datasets and our CC1M-Adv-C/F benchmarks.
    subtitlezh: OpenTAI Rank is exploring中文
    buttonTexten: Learn how it works
    buttonTextzh: 了解评测细节
    table1:
      tab1en: Black-box
      tab1zh: 黑盒
      columnName1: Model Name
      columnName2: Link
      columnName3: Downloads
      columnName4: Adversarial Safety
      columnName4A: Domain Dataset
      columnName4B: CC1M-Adv
      columnName5: Rank
      modelsRanking1:
        - titlezh: Image Classification
          titleen: Image Classification - ImageNet
          rankings:
            - nameen: coatnet_rmlp_nano_rw_384.sw_in1k
              namezh: coatnet_rmlp_nano_rw_384.sw_in1k
              paper:
                text: >-
                  https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k
                link: >-
                  https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k
              download: 764
              datasetA: 67.49%
              datasetB: 31.58%
              ranking: '1'
            - nameen: coatnet_rmlp_nano_rw_224.sw_in1k
              namezh: GPT-4Test
              paper:
                text: 'https://huggingface.co/timm/coatnet_rmlp_nano_rw_224.sw_in1k'
                link: 'https://huggingface.co/timm/coatnet_rmlp_nano_rw_224.sw_in1k'
              download: 467
              datasetA: 52.75%
              datasetB: 29.14%
              ranking: '2'
            - nameen: davit_base.msft_in1k
              namezh: davit_base.msft_in1k
              paper:
                text: 'https://huggingface.co/timm/davit_base.msft_in1k'
                link: 'https://huggingface.co/timm/davit_base.msft_in1k'
              download: 1720
              datasetA: 83.25%
              datasetB: 38.06%
              ranking: '3'
            - nameen: resnet50.a1_in1k
              namezh: resnet50.a1_in1k
              paper:
                text: 'https://huggingface.co/timm/resnet50.a1_in1k'
                link: 'https://huggingface.co/timm/resnet50.a1_in1k'
              download: 9271420
              datasetA: 35.09%
              datasetB: 14.61%
              ranking: '4'
            - nameen: vit_small_patch16_224.augreg_in21k_ft_in1k
              namezh: vit_small_patch16_224.augreg_in21k_ft_in1k
              paper:
                text: >-
                  https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
                link: >-
                  https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k
              download: 546023
              datasetA: 32.62%
              datasetB: 27.21%
              ranking: '5'
            - nameen: resnet18.a1_in1k
              namezh: resnet18.a1_in1k
              paper:
                text: 'https://huggingface.co/timm/resnet18.a1_in1k'
                link: 'https://huggingface.co/timm/resnet18.a1_in1k'
              download: 872296
              datasetA: 27.11%
              datasetB: 14.91%
              ranking: '6'
            - nameen: resnet18.fb_swsl_ig1b_ft_in1k
              namezh: resnet18.fb_swsl_ig1b_ft_in1k
              paper:
                text: 'https://huggingface.co/timm/resnet18.fb_swsl_ig1b_ft_in1k'
                link: 'https://huggingface.co/timm/resnet18.fb_swsl_ig1b_ft_in1k'
              download: 207942
              datasetA: 24.59%
              datasetB: 12.14%
              ranking: '7'
            - nameen: mobilenetv3_large_100.ra_in1k
              namezh: mobilenetv3_large_100.ra_in1k
              paper:
                text: 'https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k'
                link: 'https://huggingface.co/timm/mobilenetv3_large_100.ra_in1k'
              download: 408002
              datasetA: 23.82%
              datasetB: 12.28%
              ranking: '8'
            - nameen: efficientnet_b3.ra2_in1k
              namezh: efficientnet_b3.ra2_in1k
              paper:
                text: 'https://huggingface.co/timm/efficientnet_b3.ra2_in1k'
                link: 'https://huggingface.co/timm/efficientnet_b3.ra2_in1k'
              download: 2391229
              datasetA: 11.30%
              datasetB: 8.18%
              ranking: '9'
            - nameen: tf_mobilenetv3_small_minimal_100.in1k
              namezh: tf_mobilenetv3_small_minimal_100.in1k
              paper:
                text: >-
                  https://huggingface.co/timm/tf_mobilenetv3_large_minimal_100.in1k
                link: >-
                  https://huggingface.co/timm/tf_mobilenetv3_large_minimal_100.in1k
              download: 783520
              datasetA: 8.99%
              datasetB: 24.83%
              ranking: '10'
            - nameen: mask-rcnn_r50_fpn_albu-1x_coco
              namezh: mask-rcnn_r50_fpn_albu-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: atss_r50_fpn_1x_coco
              namezh: atss_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: atss_r101_fpn_1x_coco
              namezh: atss_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: autoassign_r50-caffe_fpn_1x_coco
              namezh: autoassign_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: boxinst_r50_fpn_ms-90k_coco
              namezh: boxinst_r50_fpn_ms-90k_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: boxinst_r101_fpn_ms-90k_coco
              namezh: boxinst_r101_fpn_ms-90k_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_carafe_1x_coco
              namezh: faster-rcnn_r50_fpn_carafe_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_carafe_1x_coco
              namezh: mask-rcnn_r50_fpn_carafe_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_r50-caffe_fpn_1x_coco
              namezh: cascade-rcnn_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_r50_fpn_1x_coco
              namezh: cascade-rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_r50_fpn_20e_coco
              namezh: cascade-rcnn_r50_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_r101-caffe_fpn_1x_coco
              namezh: cascade-rcnn_r101-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_r101_fpn_1x_coco
              namezh: cascade-rcnn_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_r101_fpn_20e_coco
              namezh: cascade-rcnn_r101_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_x101-32x4d_fpn_1x_coco
              namezh: cascade-rcnn_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_x101-32x4d_fpn_20e_coco
              namezh: cascade-rcnn_x101-32x4d_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_x101-64x4d_fpn_1x_coco
              namezh: cascade-rcnn_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_x101_64x4d_fpn_20e_coco
              namezh: cascade-rcnn_x101_64x4d_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r50-caffe_fpn_1x_coco
              namezh: cascade-mask-rcnn_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r50_fpn_1x_coco
              namezh: cascade-mask-rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r50_fpn_20e_coco
              namezh: cascade-mask-rcnn_r50_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r101-caffe_fpn_1x_coco
              namezh: cascade-mask-rcnn_r101-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r101_fpn_1x_coco
              namezh: cascade-mask-rcnn_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r101_fpn_20e_coco
              namezh: cascade-mask-rcnn_r101_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x4d_fpn_1x_coco
              namezh: cascade-mask-rcnn_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x4d_fpn_20e_coco
              namezh: cascade-mask-rcnn_x101-32x4d_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-64x4d_fpn_1x_coco
              namezh: cascade-mask-rcnn_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-64x4d_fpn_20e_coco
              namezh: cascade-mask-rcnn_x101-64x4d_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r50-caffe_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_r50-caffe_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r50_fpn_mstrain_3x_coco
              namezh: cascade-mask-rcnn_r50_fpn_mstrain_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r101-caffe_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_r101-caffe_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r101_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_r101_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x4d_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_x101-32x4d_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x8d_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_x101-32x8d_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-64x4d_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_x101-64x4d_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rpn_fast-rcnn_r50-caffe_fpn_1x_coco
              namezh: cascade-rpn_fast-rcnn_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco
              namezh: cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: centernet_r18-dcnv2_8xb16-crop512-140e_coco
              namezh: centernet_r18-dcnv2_8xb16-crop512-140e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: centernet_r18_8xb16-crop512-140e_coco
              namezh: centernet_r18_8xb16-crop512-140e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: centernet-update_r50-caffe_fpn_ms-1x_coco
              namezh: centernet-update_r50-caffe_fpn_ms-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco
              namezh: centripetalnet_hourglass104_16xb6-crop511-210e-mstest_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: condinst_r50_fpn_ms-poly-90k_coco_instance
              namezh: condinst_r50_fpn_ms-poly-90k_coco_instance
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: conditional-detr_r50_8xb2-50e_coco.py
              namezh: conditional-detr_r50_8xb2-50e_coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cornernet_hourglass104_10xb5-crop511-210e-mstest_coco
              namezh: cornernet_hourglass104_10xb5-crop511-210e-mstest_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cornernet_hourglass104_8xb6-210e-mstest_coco
              namezh: cornernet_hourglass104_8xb6-210e-mstest_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cornernet_hourglass104_32xb3-210e-mstest_coco
              namezh: cornernet_hourglass104_32xb3-210e-mstest_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco
              namezh: mask-rcnn_convnext-t-p4-w7_fpn_amp-ms-crop-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco
              namezh: cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco
              namezh: cascade-mask-rcnn_convnext-s-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dab-detr_r50_8xb2-50e_coco.py
              namezh: dab-detr_r50_8xb2-50e_coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_dconv_c3-c5_1x_coco
              namezh: faster-rcnn_r50_fpn_dconv_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_dpool_1x_coco
              namezh: faster-rcnn_r50_fpn_dpool_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r101-dconv-c3-c5_fpn_1x_coco
              namezh: faster-rcnn_r101-dconv-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco
              namezh: faster-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_dconv_c3-c5_1x_coco
              namezh: mask-rcnn_r50_fpn_dconv_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco
              namezh: mask-rcnn_r50_fpn_fp16_dconv_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco
              namezh: mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_r50_fpn_dconv_c3-c5_1x_coco
              namezh: cascade-rcnn_r50_fpn_dconv_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_r101-dconv-c3-c5_fpn_1x_coco
              namezh: cascade-rcnn_r101-dconv-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r50_fpn_dconv_c3-c5_1x_coco
              namezh: cascade-mask-rcnn_r50_fpn_dconv_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco
              namezh: cascade-mask-rcnn_r101-dconv-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco
              namezh: cascade-mask-rcnn_x101-32x4d-dconv-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_mdconv_c3-c5_1x_coco
              namezh: faster-rcnn_r50_fpn_mdconv_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco
              namezh: faster-rcnn_r50_fpn_mdconv_c3-c5_group4_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_mdpool_1x_coco
              namezh: faster-rcnn_r50_fpn_mdpool_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_mdconv_c3-c5_1x_coco
              namezh: mask-rcnn_r50_fpn_mdconv_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco
              namezh: mask-rcnn_r50_fpn_fp16_mdconv_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ddod_r50_fpn_1x_coco
              namezh: ddod_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deformable-detr_r50_16xb2-50e_coco
              namezh: deformable-detr_r50_16xb2-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deformable-detr_refine_r50_16xb2-50e_coco
              namezh: deformable-detr_refine_r50_16xb2-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deformable-detr_refine_twostage_r50_16xb2-50e_coco
              namezh: deformable-detr_refine_twostage_r50_16xb2-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_r50-rfp_1x_coco
              namezh: cascade-rcnn_r50-rfp_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_r50-sac_1x_coco
              namezh: cascade-rcnn_r50-sac_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: detectors_cascade-rcnn_r50_1x_coco
              namezh: detectors_cascade-rcnn_r50_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_r50-rfp_1x_coco
              namezh: htc_r50-rfp_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_r50-sac_1x_coco
              namezh: htc_r50-sac_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: detectors_htc-r50_1x_coco
              namezh: detectors_htc-r50_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: detr_r50_8xb2-150e_coco
              namezh: detr_r50_8xb2-150e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dino-4scale_r50_8xb2-12e_coco.py
              namezh: dino-4scale_r50_8xb2-12e_coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dino-4scale_r50_8xb2-24e_coco.py
              namezh: dino-4scale_r50_8xb2-24e_coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dino-4scale_r50_8xb2-24e_coco.py
              namezh: dino-4scale_r50_8xb2-24e_coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dino-5scale_swin-l_8xb2-12e_coco.py
              namezh: dino-5scale_swin-l_8xb2-12e_coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dino-5scale_swin-l_8xb2-36e_coco.py
              namezh: dino-5scale_swin-l_8xb2-36e_coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dh-faster-rcnn_r50_fpn_1x_coco
              namezh: dh-faster-rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: atss_r50-caffe_fpn_dyhead_1x_coco
              namezh: atss_r50-caffe_fpn_dyhead_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: atss_r50_fpn_dyhead_1x_coco
              namezh: atss_r50_fpn_dyhead_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco
              namezh: atss_swin-l-p4-w12_fpn_dyhead_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dynamic-rcnn_r50_fpn_1x_coco
              namezh: dynamic-rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_effb3_fpn_8xb4-crop896-1x_coco
              namezh: retinanet_effb3_fpn_8xb4-crop896-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_attention_1111_1x_coco
              namezh: faster-rcnn_r50_fpn_attention_1111_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_attention_0010_1x_coco
              namezh: faster-rcnn_r50_fpn_attention_0010_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_attention_1111_dcn_1x_coco
              namezh: faster-rcnn_r50_fpn_attention_1111_dcn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_attention_0010_dcn_1x_coco
              namezh: faster-rcnn_r50_fpn_attention_0010_dcn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50-caffe-c4_1x_coco
              namezh: faster-rcnn_r50-caffe-c4_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50-caffe-c4_mstrain_1x_coco
              namezh: faster-rcnn_r50-caffe-c4_mstrain_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50-caffe-dc5_1x_coco
              namezh: faster-rcnn_r50-caffe-dc5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50-caffe_fpn_1x_coco
              namezh: faster-rcnn_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_1x_coco
              namezh: faster-rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_fp16_1x_coco
              namezh: faster-rcnn_r50_fpn_fp16_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_2x_coco
              namezh: faster-rcnn_r50_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r101-caffe_fpn_1x_coco
              namezh: faster-rcnn_r101-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r101_fpn_1x_coco
              namezh: faster-rcnn_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r101_fpn_2x_coco
              namezh: faster-rcnn_r101_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_x101-32x4d_fpn_1x_coco
              namezh: faster-rcnn_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_x101-32x4d_fpn_2x_coco
              namezh: faster-rcnn_x101-32x4d_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_x101-64x4d_fpn_1x_coco
              namezh: faster-rcnn_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_x101-64x4d_fpn_2x_coco
              namezh: faster-rcnn_x101-64x4d_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_iou_1x_coco
              namezh: faster-rcnn_r50_fpn_iou_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_giou_1x_coco
              namezh: faster-rcnn_r50_fpn_giou_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_bounded_iou_1x_coco
              namezh: faster-rcnn_r50_fpn_bounded_iou_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50-caffe-dc5_mstrain_1x_coco
              namezh: faster-rcnn_r50-caffe-dc5_mstrain_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50-caffe-dc5_mstrain_3x_coco
              namezh: faster-rcnn_r50-caffe-dc5_mstrain_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50-caffe_fpn_ms-2x_coco
              namezh: faster-rcnn_r50-caffe_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50-caffe_fpn_ms-3x_coco
              namezh: faster-rcnn_r50-caffe_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_mstrain_3x_coco
              namezh: faster-rcnn_r50_fpn_mstrain_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r101-caffe_fpn_ms-3x_coco
              namezh: faster-rcnn_r101-caffe_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r101_fpn_ms-3x_coco
              namezh: faster-rcnn_r101_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_x101-32x4d_fpn_ms-3x_coco
              namezh: faster-rcnn_x101-32x4d_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_x101-32x8d_fpn_ms-3x_coco
              namezh: faster-rcnn_x101-32x8d_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_x101-64x4d_fpn_ms-3x_coco
              namezh: faster-rcnn_x101-64x4d_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_tnr-pretrain_1x_coco
              namezh: faster-rcnn_r50_fpn_tnr-pretrain_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_r50-caffe_fpn_gn-head_1x_coco
              namezh: fcos_r50-caffe_fpn_gn-head_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco
              namezh: fcos_r50-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco
              namezh: fcos_r50-dcn-caffe_fpn_gn-head-center-normbbox-centeronreg-giou_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_r101-caffe_fpn_gn-head-1x_coco
              namezh: fcos_r101-caffe_fpn_gn-head-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco
              namezh: fcos_r50-caffe_fpn_gn-head_ms-640-800-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_r101-caffe_fpn_gn-head_ms-640-800-2x_coco
              namezh: fcos_r101-caffe_fpn_gn-head_ms-640-800-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco
              namezh: fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fovea_r50_fpn_4xb4-1x_coco
              namezh: fovea_r50_fpn_4xb4-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fovea_r50_fpn_4xb4-2x_coco
              namezh: fovea_r50_fpn_4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fovea_r50_fpn_gn-head-align_4xb4-2x_coco
              namezh: fovea_r50_fpn_gn-head-align_4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fovea_r50_fpn_gn-head-align_ms-640-800-4xb4-2x_coco
              namezh: fovea_r50_fpn_gn-head-align_ms-640-800-4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fovea_r101_fpn_4xb4-1x_coco
              namezh: fovea_r101_fpn_4xb4-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fovea_r101_fpn_4xb4-2x_coco
              namezh: fovea_r101_fpn_4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fovea_r101_fpn_gn-head-align_4xb4-2x_coco
              namezh: fovea_r101_fpn_gn-head-align_4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fovea_r101_fpn_gn-head-align_ms-640-800-4xb4-2x_coco
              namezh: fovea_r101_fpn_gn-head-align_ms-640-800-4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpg_crop640-50e_coco
              namezh: faster-rcnn_r50_fpg_crop640-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpg-chn128_crop640-50e_coco
              namezh: faster-rcnn_r50_fpg-chn128_crop640-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpg_crop640-50e_coco
              namezh: mask-rcnn_r50_fpg_crop640-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpg-chn128_crop640-50e_coco
              namezh: mask-rcnn_r50_fpg-chn128_crop640-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50_fpg_crop640_50e_coco
              namezh: retinanet_r50_fpg_crop640_50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50_fpg-chn128_crop640_50e_coco
              namezh: retinanet_r50_fpg-chn128_crop640_50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: freeanchor_r50_fpn_1x_coco
              namezh: freeanchor_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: freeanchor_r101_fpn_1x_coco
              namezh: freeanchor_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: freeanchor_x101-32x4d_fpn_1x_coco
              namezh: freeanchor_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fsaf_r50_fpn_1x_coco
              namezh: fsaf_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fsaf_r101_fpn_1x_coco
              namezh: fsaf_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fsaf_x101-64x4d_fpn_1x_coco
              namezh: fsaf_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco
              namezh: mask-rcnn_r50_fpn_r16_gcb_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco
              namezh: mask-rcnn_r50_fpn_r4_gcb_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101-gcb-r16-c3-c5_fpn_1x_coco
              namezh: mask-rcnn_r101-gcb-r16-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101-gcb-r4-c3-c5_fpn_1x_coco
              namezh: mask-rcnn_r101-gcb-r4-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_syncbn-backbone_1x_coco
              namezh: mask-rcnn_r50_fpn_syncbn-backbone_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco
              namezh: mask-rcnn_r50_fpn_syncbn-backbone_r16_gcb_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco
              namezh: mask-rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101-syncbn_fpn_1x_coco
              namezh: mask-rcnn_r101-syncbn_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101-syncbn-gcb-r16-c3-c5_fpn_1x_coco
              namezh: mask-rcnn_r101-syncbn-gcb-r16-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101-syncbn-gcb-r4-c3-c5_fpn_1x_coco
              namezh: mask-rcnn_r101-syncbn-gcb-r4-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x4d-syncbn_fpn_1x_coco
              namezh: mask-rcnn_x101-32x4d-syncbn_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x4d-syncbn-gcb-r16-c3-c5_fpn_1x_coco
              namezh: mask-rcnn_x101-32x4d-syncbn-gcb-r16-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x4d-syncbn-gcb-r4-c3-c5_fpn_1x_coco
              namezh: mask-rcnn_x101-32x4d-syncbn-gcb-r4-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x4d-syncbn_fpn_1x_coco
              namezh: cascade-mask-rcnn_x101-32x4d-syncbn_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x4d-syncbn-r16-gcb-c3-c5_fpn_1x_coco
              namezh: cascade-mask-rcnn_x101-32x4d-syncbn-r16-gcb-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x4d-syncbn-r4-gcb-c3-c5_fpn_1x_coco
              namezh: cascade-mask-rcnn_x101-32x4d-syncbn-r4-gcb-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5_fpn_1x_coco
              namezh: cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5-r16-gcb-c3-c5_fpn_1x_coco
              namezh: cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5-r16-gcb-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5-r4-gcb-c3-c5_fpn_1x_coco
              namezh: cascade-mask-rcnn_x101-32x4d-syncbn-dconv-c3-c5-r4-gcb-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gfl_r50_fpn_1x_coco
              namezh: gfl_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gfl_r50_fpn_ms-2x_coco
              namezh: gfl_r50_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gfl_r101_fpn_ms-2x_coco
              namezh: gfl_r101_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gfl_r101-dconv-c3-c5_fpn_ms-2x_coco
              namezh: gfl_r101-dconv-c3-c5_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gfl_x101-32x4d_fpn_ms-2x_coco
              namezh: gfl_x101-32x4d_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gfl_x101-32x4d-dconv-c4-c5_fpn_ms-2x_coco
              namezh: gfl_x101-32x4d-dconv-c4-c5_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50_fpn_ghm-1x_coco
              namezh: retinanet_r50_fpn_ghm-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r101_fpn_ghm-1x_coco
              namezh: retinanet_r101_fpn_ghm-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_x101-32x4d_fpn_ghm-1x_coco
              namezh: retinanet_x101-32x4d_fpn_ghm-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_x101-64x4d_fpn_ghm-1x_coco
              namezh: retinanet_x101-64x4d_fpn_ghm-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_gn-all_2x_coco
              namezh: mask-rcnn_r50_fpn_gn-all_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_gn-all_3x_coco
              namezh: mask-rcnn_r50_fpn_gn-all_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101_fpn_gn-all_2x_coco
              namezh: mask-rcnn_r101_fpn_gn-all_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101_fpn_gn-all_3x_coco
              namezh: mask-rcnn_r101_fpn_gn-all_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_gn-all_contrib_2x_coco
              namezh: mask-rcnn_r50_fpn_gn-all_contrib_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_gn-all_contrib_3x_coco
              namezh: mask-rcnn_r50_fpn_gn-all_contrib_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_gn_ws-all_1x_coco
              namezh: faster-rcnn_r50_fpn_gn_ws-all_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r101_fpn_gn-ws-all_1x_coco
              namezh: faster-rcnn_r101_fpn_gn-ws-all_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_x50-32x4d_fpn_gn-ws-all_1x_coco
              namezh: faster-rcnn_x50-32x4d_fpn_gn-ws-all_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_x101-32x4d_fpn_gn-ws-all_1x_coco
              namezh: faster-rcnn_x101-32x4d_fpn_gn-ws-all_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_gn_ws-all_2x_coco
              namezh: mask-rcnn_r50_fpn_gn_ws-all_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101_fpn_gn-ws-all_2x_coco
              namezh: mask-rcnn_r101_fpn_gn-ws-all_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x50-32x4d_fpn_gn-ws-all_2x_coco
              namezh: mask-rcnn_x50-32x4d_fpn_gn-ws-all_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x4d_fpn_gn-ws-all_2x_coco
              namezh: mask-rcnn_x101-32x4d_fpn_gn-ws-all_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_gn_ws-all_20_23_24e_coco
              namezh: mask-rcnn_r50_fpn_gn_ws-all_20_23_24e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101_fpn_gn-ws-all_20-23-24e_coco
              namezh: mask-rcnn_r101_fpn_gn-ws-all_20-23-24e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x50-32x4d_fpn_gn-ws-all_20-23-24e_coco
              namezh: mask-rcnn_x50-32x4d_fpn_gn-ws-all_20-23-24e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x4d_fpn_gn-ws-all_20-23-24e_coco
              namezh: mask-rcnn_x101-32x4d_fpn_gn-ws-all_20-23-24e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: grid-rcnn_r50_fpn_gn-head_2x_coco
              namezh: grid-rcnn_r50_fpn_gn-head_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: grid-rcnn_r101_fpn_gn-head_2x_coco
              namezh: grid-rcnn_r101_fpn_gn-head_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: grid-rcnn_x101-32x4d_fpn_gn-head_2x_coco
              namezh: grid-rcnn_x101-32x4d_fpn_gn-head_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco
              namezh: grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_groie_1x_coco
              namezh: faster-rcnn_r50_fpn_groie_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: grid-rcnn_r50_fpn_gn-head-groie_1x_coco
              namezh: grid-rcnn_r50_fpn_gn-head-groie_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_groie_1x_coco
              namezh: mask-rcnn_r50_fpn_groie_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco
              namezh: mask-rcnn_r50_fpn_syncbn-backbone_r4_gcb_c3-c5_groie_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101_fpn_syncbn-r4-gcb_c3-c5-groie_1x_coco
              namezh: mask-rcnn_r101_fpn_syncbn-r4-gcb_c3-c5-groie_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-rpn_r50-caffe_fpn_1x_coco
              namezh: ga-rpn_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-rpn_r101-caffe_fpn_1x_coco
              namezh: ga-rpn_r101-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-rpn_x101-32x4d_fpn_1x_coco
              namezh: ga-rpn_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-rpn_x101-64x4d_fpn_1x_coco
              namezh: ga-rpn_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-faster-rcnn_r50-caffe_fpn_1x_coco
              namezh: ga-faster-rcnn_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-faster-rcnn_r101-caffe_fpn_1x_coco
              namezh: ga-faster-rcnn_r101-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-faster-rcnn_x101-32x4d_fpn_1x_coco
              namezh: ga-faster-rcnn_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-faster-rcnn_x101-64x4d_fpn_1x_coco
              namezh: ga-faster-rcnn_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-retinanet_r50-caffe_fpn_1x_coco
              namezh: ga-retinanet_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-retinanet_r101-caffe_fpn_1x_coco
              namezh: ga-retinanet_r101-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-retinanet_x101-32x4d_fpn_1x_coco
              namezh: ga-retinanet_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ga-retinanet_x101-64x4d_fpn_1x_coco
              namezh: ga-retinanet_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_hrnetv2p-w18-1x_coco
              namezh: faster-rcnn_hrnetv2p-w18-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_hrnetv2p-w18-2x_coco
              namezh: faster-rcnn_hrnetv2p-w18-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_hrnetv2p-w32-1x_coco
              namezh: faster-rcnn_hrnetv2p-w32-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_hrnetv2p-w32_2x_coco
              namezh: faster-rcnn_hrnetv2p-w32_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_hrnetv2p-w40-1x_coco
              namezh: faster-rcnn_hrnetv2p-w40-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_hrnetv2p-w40_2x_coco
              namezh: faster-rcnn_hrnetv2p-w40_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_hrnetv2p-w18-1x_coco
              namezh: mask-rcnn_hrnetv2p-w18-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_hrnetv2p-w18-2x_coco
              namezh: mask-rcnn_hrnetv2p-w18-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_hrnetv2p-w32-1x_coco
              namezh: mask-rcnn_hrnetv2p-w32-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_hrnetv2p-w32-2x_coco
              namezh: mask-rcnn_hrnetv2p-w32-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_hrnetv2p-w40_1x_coco
              namezh: mask-rcnn_hrnetv2p-w40_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_hrnetv2p-w40-2x_coco
              namezh: mask-rcnn_hrnetv2p-w40-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_hrnetv2p-w18-20e_coco
              namezh: cascade-rcnn_hrnetv2p-w18-20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_hrnetv2p-w32-20e_coco
              namezh: cascade-rcnn_hrnetv2p-w32-20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_hrnetv2p-w40-20e_coco
              namezh: cascade-rcnn_hrnetv2p-w40-20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_hrnetv2p-w18_20e_coco
              namezh: cascade-mask-rcnn_hrnetv2p-w18_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_hrnetv2p-w32_20e_coco
              namezh: cascade-mask-rcnn_hrnetv2p-w32_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_hrnetv2p-w40-20e_coco
              namezh: cascade-mask-rcnn_hrnetv2p-w40-20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_hrnetv2p-w18_20e_coco
              namezh: htc_hrnetv2p-w18_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_hrnetv2p-w32_20e_coco
              namezh: htc_hrnetv2p-w32_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_hrnetv2p-w40_20e_coco
              namezh: htc_hrnetv2p-w40_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_hrnetv2p-w18-gn-head_4xb4-1x_coco
              namezh: fcos_hrnetv2p-w18-gn-head_4xb4-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_hrnetv2p-w18-gn-head_4xb4-2x_coco
              namezh: fcos_hrnetv2p-w18-gn-head_4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_hrnetv2p-w32-gn-head_4xb4-1x_coco
              namezh: fcos_hrnetv2p-w32-gn-head_4xb4-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_hrnetv2p-w32-gn-head_4xb4-2x_coco
              namezh: fcos_hrnetv2p-w32-gn-head_4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_hrnetv2p-w18-gn-head_ms-640-800-4xb4-2x_coco
              namezh: fcos_hrnetv2p-w18-gn-head_ms-640-800-4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_hrnetv2p-w32-gn-head_ms-640-800-4xb4-2x_coco
              namezh: fcos_hrnetv2p-w32-gn-head_ms-640-800-4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcos_hrnetv2p-w40-gn-head_ms-640-800-4xb4-2x_coco
              namezh: fcos_hrnetv2p-w40-gn-head_ms-640-800-4xb4-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_r50_fpn_1x_coco
              namezh: htc_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_r50_fpn_20e_coco
              namezh: htc_r50_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_r101_fpn_20e_coco
              namezh: htc_r101_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_x101-32x4d_fpn_16xb1-20e_coco
              namezh: htc_x101-32x4d_fpn_16xb1-20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_x101-64x4d_fpn_16xb1-20e_coco
              namezh: htc_x101-64x4d_fpn_16xb1-20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_x101-64x4d-dconv-c3-c5_fpn_ms-400-1400-16xb1-20e_coco
              namezh: htc_x101-64x4d-dconv-c3-c5_fpn_ms-400-1400-16xb1-20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_instaboost_4x_coco
              namezh: mask-rcnn_r50_fpn_instaboost_4x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101_fpn_instaboost-4x_coco
              namezh: mask-rcnn_r101_fpn_instaboost-4x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-64x4d_fpn_instaboost-4x_coco
              namezh: mask-rcnn_x101-64x4d_fpn_instaboost-4x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r50_fpn_instaboost_4x_coco
              namezh: cascade-mask-rcnn_r50_fpn_instaboost_4x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lad_r101-paa-r50_fpn_2xb8_coco_1x
              namezh: lad_r101-paa-r50_fpn_2xb8_coco_1x
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lad_r50-paa-r101_fpn_2xb8_coco_1x
              namezh: lad_r50-paa-r101_fpn_2xb8_coco_1x
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ld_r18-gflv1-r101_fpn_1x_coco
              namezh: ld_r18-gflv1-r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ld_r34-gflv1-r101_fpn_1x_coco
              namezh: ld_r34-gflv1-r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ld_r50-gflv1-r101_fpn_1x_coco
              namezh: ld_r50-gflv1-r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ld_r101-gflv1-r101-dcn_fpn_2x_coco
              namezh: ld_r101-gflv1-r101-dcn_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: libra-faster-rcnn_r50_fpn_1x_coco
              namezh: libra-faster-rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: libra-faster-rcnn_r101_fpn_1x_coco
              namezh: libra-faster-rcnn_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: libra-faster-rcnn_x101-64x4d_fpn_1x_coco
              namezh: libra-faster-rcnn_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: libra-retinanet_r50_fpn_1x_coco
              namezh: libra-retinanet_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic
              namezh: mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco-panoptic
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_r101_8xb2-lsj-50e_coco
              namezh: mask2former_r101_8xb2-lsj-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_r101_8xb2-lsj-50e_coco-panoptic
              namezh: mask2former_r101_8xb2-lsj-50e_coco-panoptic
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_r50_8xb2-lsj-50e_coco-panoptic
              namezh: mask2former_r50_8xb2-lsj-50e_coco-panoptic
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic
              namezh: mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco-panoptic
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_r50_8xb2-lsj-50e_coco
              namezh: mask2former_r50_8xb2-lsj-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic
              namezh: mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic
              namezh: mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic
              namezh: mask2former_swin-b-p4-w12-384_8xb2-lsj-50e_coco-panoptic
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco
              namezh: mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco
              namezh: mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50-caffe_fpn_1x_coco
              namezh: mask-rcnn_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_1x_coco
              namezh: mask-rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_fp16_1x_coco
              namezh: mask-rcnn_r50_fpn_fp16_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_2x_coco
              namezh: mask-rcnn_r50_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101-caffe_fpn_1x_coco
              namezh: mask-rcnn_r101-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101_fpn_1x_coco
              namezh: mask-rcnn_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101_fpn_2x_coco
              namezh: mask-rcnn_r101_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x4d_fpn_1x_coco
              namezh: mask-rcnn_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x4d_fpn_2x_coco
              namezh: mask-rcnn_x101-32x4d_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-64x4d_fpn_1x_coco
              namezh: mask-rcnn_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-64x4d_fpn_2x_coco
              namezh: mask-rcnn_x101-64x4d_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x8d_fpn_1x_coco
              namezh: mask-rcnn_x101-32x8d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50-caffe_fpn_ms-poly-2x_coco
              namezh: mask-rcnn_r50-caffe_fpn_ms-poly-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco
              namezh: mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_mstrain-poly_3x_coco
              namezh: mask-rcnn_r50_fpn_mstrain-poly_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101_fpn_ms-poly-3x_coco
              namezh: mask-rcnn_r101_fpn_ms-poly-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r101-caffe_fpn_ms-poly-3x_coco
              namezh: mask-rcnn_r101-caffe_fpn_ms-poly-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x4d_fpn_ms-poly-3x_coco
              namezh: mask-rcnn_x101-32x4d_fpn_ms-poly-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x8d_fpn_ms-poly-1x_coco
              namezh: mask-rcnn_x101-32x8d_fpn_ms-poly-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-32x8d_fpn_ms-poly-3x_coco
              namezh: mask-rcnn_x101-32x8d_fpn_ms-poly-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco
              namezh: mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maskformer_r50_ms-16xb1-75e_coco
              namezh: maskformer_r50_ms-16xb1-75e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maskformer_swin-l-p4-w12_64xb1-ms-300e_coco
              namezh: maskformer_swin-l-p4-w12_64xb1-ms-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ms-rcnn_r50-caffe_fpn_1x_coco
              namezh: ms-rcnn_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ms-rcnn_r50-caffe_fpn_2x_coco
              namezh: ms-rcnn_r50-caffe_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ms-rcnn_r101-caffe_fpn_1x_coco
              namezh: ms-rcnn_r101-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ms-rcnn_r101-caffe_fpn_2x_coco
              namezh: ms-rcnn_r101-caffe_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ms-rcnn_x101-32x4d_fpn_1x_coco
              namezh: ms-rcnn_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ms-rcnn_x101-64x4d_fpn_1x_coco
              namezh: ms-rcnn_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ms-rcnn_x101-64x4d_fpn_2x_coco
              namezh: ms-rcnn_x101-64x4d_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco
              namezh: nas-fcos_r50-caffe_fpn_nashead-gn-head_4xb4-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nas-fcos_r50-caffe_fpn_fcoshead-gn-head_4xb4-1x_coco
              namezh: nas-fcos_r50-caffe_fpn_fcoshead-gn-head_4xb4-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50_fpn_crop640-50e_coco
              namezh: retinanet_r50_fpn_crop640-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50_nasfpn_crop640-50e_coco
              namezh: retinanet_r50_nasfpn_crop640-50e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: paa_r50_fpn_1x_coco
              namezh: paa_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: paa_r50_fpn_1.5x_coco
              namezh: paa_r50_fpn_1.5x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: paa_r50_fpn_2x_coco
              namezh: paa_r50_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: paa_r50_fpn_mstrain_3x_coco
              namezh: paa_r50_fpn_mstrain_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: paa_r101_fpn_1x_coco
              namezh: paa_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: paa_r101_fpn_2x_coco
              namezh: paa_r101_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: paa_r101_fpn_mstrain_3x_coco
              namezh: paa_r101_fpn_mstrain_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_pafpn_1x_coco
              namezh: faster-rcnn_r50_pafpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: panoptic_fpn_r50_fpn_1x_coco
              namezh: panoptic_fpn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: panoptic_fpn_r50_fpn_mstrain_3x_coco
              namezh: panoptic_fpn_r50_fpn_mstrain_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: panoptic_fpn_r101_fpn_1x_coco
              namezh: panoptic_fpn_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: panoptic_fpn_r101_fpn_mstrain_3x_coco
              namezh: panoptic_fpn_r101_fpn_mstrain_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_pvt-t_fpn_1x_coco
              namezh: retinanet_pvt-t_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_pvt-s_fpn_1x_coco
              namezh: retinanet_pvt-s_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_pvt-m_fpn_1x_coco
              namezh: retinanet_pvt-m_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_pvtv2-b0_fpn_1x_coco
              namezh: retinanet_pvtv2-b0_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_pvtv2-b1_fpn_1x_coco
              namezh: retinanet_pvtv2-b1_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_pvtv2-b2_fpn_1x_coco
              namezh: retinanet_pvtv2-b2_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_pvtv2-b3_fpn_1x_coco
              namezh: retinanet_pvtv2-b3_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_pvtv2-b4_fpn_1x_coco
              namezh: retinanet_pvtv2-b4_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_pvtv2-b5_fpn_1x_coco
              namezh: retinanet_pvtv2-b5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pisa_faster_rcnn_r50_fpn_1x_coco
              namezh: pisa_faster_rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pisa_faster_rcnn_x101_32x4d_fpn_1x_coco
              namezh: pisa_faster_rcnn_x101_32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pisa_mask_rcnn_r50_fpn_1x_coco
              namezh: pisa_mask_rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pisa_retinanet_r50_fpn_1x_coco
              namezh: pisa_retinanet_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pisa_retinanet_x101_32x4d_fpn_1x_coco
              namezh: pisa_retinanet_x101_32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pisa_ssd300_coco
              namezh: pisa_ssd300_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pisa_ssd512_coco
              namezh: pisa_ssd512_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: point_rend_r50_caffe_fpn_mstrain_1x_coco
              namezh: point_rend_r50_caffe_fpn_mstrain_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: point_rend_r50_caffe_fpn_mstrain_3x_coco
              namezh: point_rend_r50_caffe_fpn_mstrain_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: queryinst_r50_fpn_1x_coco
              namezh: queryinst_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: queryinst_r50_fpn_ms-480-800-3x_coco
              namezh: queryinst_r50_fpn_ms-480-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: queryinst_r50_fpn_300-proposals_crop-ms-480-800-3x_coco
              namezh: queryinst_r50_fpn_300-proposals_crop-ms-480-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: queryinst_r101_fpn_ms-480-800-3x_coco
              namezh: queryinst_r101_fpn_ms-480-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: queryinst_r101_fpn_300-proposals_crop-ms-480-800-3x_coco
              namezh: queryinst_r101_fpn_300-proposals_crop-ms-480-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-3.2GF_fpn_1x_coco
              namezh: mask-rcnn_regnetx-3.2GF_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-4GF_fpn_1x_coco
              namezh: mask-rcnn_regnetx-4GF_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-6.4GF_fpn_1x_coco
              namezh: mask-rcnn_regnetx-6.4GF_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-8GF_fpn_1x_coco
              namezh: mask-rcnn_regnetx-8GF_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-12GF_fpn_1x_coco
              namezh: mask-rcnn_regnetx-12GF_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-3.2GF-mdconv-c3-c5_fpn_1x_coco
              namezh: mask-rcnn_regnetx-3.2GF-mdconv-c3-c5_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_regnetx-3.2GF_fpn_1x_coco
              namezh: faster-rcnn_regnetx-3.2GF_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_regnetx-3.2GF_fpn_2x_coco
              namezh: faster-rcnn_regnetx-3.2GF_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_regnetx-800MF_fpn_1x_coco
              namezh: retinanet_regnetx-800MF_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_regnetx-1.6GF_fpn_1x_coco
              namezh: retinanet_regnetx-1.6GF_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_regnetx-3.2GF_fpn_1x_coco
              namezh: retinanet_regnetx-3.2GF_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_regnetx-400MF_fpn_ms-3x_coco
              namezh: faster-rcnn_regnetx-400MF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_regnetx-800MF_fpn_ms-3x_coco
              namezh: faster-rcnn_regnetx-800MF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_regnetx-1.6GF_fpn_ms-3x_coco
              namezh: faster-rcnn_regnetx-1.6GF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_regnetx-3.2GF_fpn_ms-3x_coco
              namezh: faster-rcnn_regnetx-3.2GF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_regnetx-4GF_fpn_ms-3x_coco
              namezh: faster-rcnn_regnetx-4GF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco
              namezh: mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-400MF_fpn_ms-poly-3x_coco
              namezh: mask-rcnn_regnetx-400MF_fpn_ms-poly-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-800MF_fpn_ms-poly-3x_coco
              namezh: mask-rcnn_regnetx-800MF_fpn_ms-poly-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-1.6GF_fpn_ms-poly-3x_coco
              namezh: mask-rcnn_regnetx-1.6GF_fpn_ms-poly-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco
              namezh: mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_regnetx-4GF_fpn_ms-poly-3x_coco
              namezh: mask-rcnn_regnetx-4GF_fpn_ms-poly-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_regnetx-400MF_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_regnetx-400MF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_regnetx-800MF_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_regnetx-800MF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_regnetx-1.6GF_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_regnetx-1.6GF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_regnetx-3.2GF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_regnetx-4GF_fpn_ms-3x_coco
              namezh: cascade-mask-rcnn_regnetx-4GF_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco
              namezh: reppoints-bbox_r50_fpn-gn_head-gn-grid_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: reppoints-bbox_r50-center_fpn-gn_head-gn-grid_1x_coco
              namezh: reppoints-bbox_r50-center_fpn-gn_head-gn-grid_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: reppoints-moment_r50_fpn_1x_coco
              namezh: reppoints-moment_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: reppoints-moment_r50_fpn-gn_head-gn_1x_coco
              namezh: reppoints-moment_r50_fpn-gn_head-gn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: reppoints-moment_r50_fpn-gn_head-gn_2x_coco
              namezh: reppoints-moment_r50_fpn-gn_head-gn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: reppoints-moment_r101_fpn-gn_head-gn_2x_coco
              namezh: reppoints-moment_r101_fpn-gn_head-gn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: reppoints-moment_r101-dconv-c3-c5_fpn-gn_head-gn_2x_coco
              namezh: reppoints-moment_r101-dconv-c3-c5_fpn-gn_head-gn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: reppoints-moment_x101-dconv-c3-c5_fpn-gn_head-gn_2x_coco
              namezh: reppoints-moment_x101-dconv-c3-c5_fpn-gn_head-gn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_res2net-101_fpn_2x_coco
              namezh: faster-rcnn_res2net-101_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_res2net-101_fpn_2x_coco
              namezh: mask-rcnn_res2net-101_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_res2net-101_fpn_20e_coco
              namezh: cascade-rcnn_res2net-101_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_res2net-101_fpn_20e_coco
              namezh: cascade-mask-rcnn_res2net-101_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: htc_res2net-101_fpn_20e_coco
              namezh: htc_res2net-101_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco
              namezh: faster-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco
              namezh: faster-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco
              namezh: mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_s101_fpn_syncbn-backbone+head_ms-1x_coco
              namezh: mask-rcnn_s101_fpn_syncbn-backbone+head_ms-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco
              namezh: cascade-rcnn_s50_fpn_syncbn-backbone+head_ms-range-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco
              namezh: cascade-rcnn_s101_fpn_syncbn-backbone+head_ms-range-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco
              namezh: cascade-mask-rcnn_s50_fpn_syncbn-backbone+head_ms-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_s101_fpn_syncbn-backbone+head_ms-1x_coco
              namezh: cascade-mask-rcnn_s101_fpn_syncbn-backbone+head_ms-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_rsb-pretrain_1x_coco
              namezh: faster-rcnn_r50_fpn_rsb-pretrain_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cascade-mask-rcnn_r50_fpn_rsb-pretrain_1x_coco
              namezh: cascade-mask-rcnn_r50_fpn_rsb-pretrain_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50-rsb-pre_fpn_1x_coco
              namezh: retinanet_r50-rsb-pre_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_rsb-pretrain_1x_coco
              namezh: mask-rcnn_r50_fpn_rsb-pretrain_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r18_fpn_1x_coco
              namezh: retinanet_r18_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r18_fpn_1xb8-1x_coco
              namezh: retinanet_r18_fpn_1xb8-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50-caffe_fpn_1x_coco
              namezh: retinanet_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50_fpn_1x_coco
              namezh: retinanet_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50_fpn_amp-1x_coco
              namezh: retinanet_r50_fpn_amp-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50_fpn_2x_coco
              namezh: retinanet_r50_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r50_fpn_ms-640-800-3x_coco
              namezh: retinanet_r50_fpn_ms-640-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r101-caffe_fpn_1x_coco
              namezh: retinanet_r101-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r101-caffe_fpn_ms-3x_coco
              namezh: retinanet_r101-caffe_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r101_fpn_1x_coco
              namezh: retinanet_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r101_fpn_2x_coco
              namezh: retinanet_r101_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_r101_fpn_ms-640-800-3x_coco
              namezh: retinanet_r101_fpn_ms-640-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_x101-32x4d_fpn_1x_coco
              namezh: retinanet_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_x101-32x4d_fpn_2x_coco
              namezh: retinanet_x101-32x4d_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_x101-64x4d_fpn_1x_coco
              namezh: retinanet_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_x101-64x4d_fpn_2x_coco
              namezh: retinanet_x101-64x4d_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: retinanet_x101-64x4d_fpn_ms-640-800-3x_coco
              namezh: retinanet_x101-64x4d_fpn_ms-640-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rpn_r50-caffe_fpn_1x_coco
              namezh: rpn_r50-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rpn_r50_fpn_1x_coco
              namezh: rpn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rpn_r50_fpn_2x_coco
              namezh: rpn_r50_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rpn_r101-caffe_fpn_1x_coco
              namezh: rpn_r101-caffe_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rpn_x101-32x4d_fpn_1x_coco
              namezh: rpn_x101-32x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rpn_x101-32x4d_fpn_2x_coco
              namezh: rpn_x101-32x4d_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rpn_x101-64x4d_fpn_1x_coco
              namezh: rpn_x101-64x4d_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rpn_x101-64x4d_fpn_2x_coco
              namezh: rpn_x101-64x4d_fpn_2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rtmdet_tiny_8xb32-300e_coco
              namezh: rtmdet_tiny_8xb32-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rtmdet_s_8xb32-300e_coco
              namezh: rtmdet_s_8xb32-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rtmdet_m_8xb32-300e_coco
              namezh: rtmdet_m_8xb32-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rtmdet_l_8xb32-300e_coco
              namezh: rtmdet_l_8xb32-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rtmdet_x_8xb32-300e_coco
              namezh: rtmdet_x_8xb32-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rtmdet-ins_tiny_8xb32-300e_coco
              namezh: rtmdet-ins_tiny_8xb32-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rtmdet-ins_s_8xb32-300e_coco
              namezh: rtmdet-ins_s_8xb32-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rtmdet-ins_m_8xb32-300e_coco
              namezh: rtmdet-ins_m_8xb32-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rtmdet-ins_l_8xb32-300e_coco
              namezh: rtmdet-ins_l_8xb32-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rtmdet-ins_x_8xb16-300e_coco
              namezh: rtmdet-ins_x_8xb16-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sabl-faster-rcnn_r50_fpn_1x_coco
              namezh: sabl-faster-rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sabl-faster-rcnn_r101_fpn_1x_coco
              namezh: sabl-faster-rcnn_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sabl-cascade-rcnn_r50_fpn_1x_coco
              namezh: sabl-cascade-rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sabl-cascade-rcnn_r101_fpn_1x_coco
              namezh: sabl-cascade-rcnn_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sabl-retinanet_r50_fpn_1x_coco
              namezh: sabl-retinanet_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sabl-retinanet_r50-gn_fpn_1x_coco
              namezh: sabl-retinanet_r50-gn_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sabl-retinanet_r101_fpn_1x_coco
              namezh: sabl-retinanet_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sabl-retinanet_r101-gn_fpn_1x_coco
              namezh: sabl-retinanet_r101-gn_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sabl-retinanet_r101-gn_fpn_ms-640-800-2x_coco
              namezh: sabl-retinanet_r101-gn_fpn_ms-640-800-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sabl-retinanet_r101-gn_fpn_ms-480-960-2x_coco
              namezh: sabl-retinanet_r101-gn_fpn_ms-480-960-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: scnet_r50_fpn_1x_coco
              namezh: scnet_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: scnet_r50_fpn_20e_coco
              namezh: scnet_r50_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: scnet_r101_fpn_20e_coco
              namezh: scnet_r101_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: scnet_x101-64x4d_fpn_20e_coco
              namezh: scnet_x101-64x4d_fpn_20e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: faster-rcnn_r50_fpn_gn-all_scratch_6x_coco
              namezh: faster-rcnn_r50_fpn_gn-all_scratch_6x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_gn-all_scratch_6x_coco
              namezh: mask-rcnn_r50_fpn_gn-all_scratch_6x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco
              namezh: mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_270k_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco
              namezh: mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_32x2_90k_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco
              namezh: mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_270k_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_90k_coco
              namezh: mask-rcnn_r50_fpn_syncbn-all_rpn-2conv_ssj_scp_32x2_90k_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.01-coco.py
              namezh: soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.01-coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.02-coco.py
              namezh: soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.02-coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.05-coco.py
              namezh: soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.05-coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py
              namezh: soft-teacher_faster-rcnn_r50-caffe_fpn_180k_semi-0.1-coco.py
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sparse-rcnn_r50_fpn_1x_coco
              namezh: sparse-rcnn_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sparse-rcnn_r50_fpn_ms-480-800-3x_coco
              namezh: sparse-rcnn_r50_fpn_ms-480-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco
              namezh: sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sparse-rcnn_r101_fpn_ms-480-800-3x_coco
              namezh: sparse-rcnn_r101_fpn_ms-480-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco
              namezh: sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: decoupled-solo_r50_fpn_1x_coco
              namezh: decoupled-solo_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: decoupled-solo_r50_fpn_3x_coco
              namezh: decoupled-solo_r50_fpn_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: decoupled-solo-light_r50_fpn_3x_coco
              namezh: decoupled-solo-light_r50_fpn_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: solo_r50_fpn_3x_coco
              namezh: solo_r50_fpn_3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: solo_r50_fpn_1x_coco
              namezh: solo_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: solov2_r50_fpn_1x_coco
              namezh: solov2_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: solov2_r50_fpn_ms-3x_coco
              namezh: solov2_r50_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: solov2_r101-dcn_fpn_ms-3x_coco
              namezh: solov2_r101-dcn_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: solov2_x101-dcn_fpn_ms-3x_coco
              namezh: solov2_x101-dcn_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: solov2-light_r18_fpn_ms-3x_coco
              namezh: solov2-light_r18_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: solov2-light_r50_fpn_ms-3x_coco
              namezh: solov2-light_r50_fpn_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ssd300_coco
              namezh: ssd300_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ssd512_coco
              namezh: ssd512_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ssdlite_mobilenetv2-scratch_8xb24-600e_coco
              namezh: ssdlite_mobilenetv2-scratch_8xb24-600e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_r50-caffe_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco
              namezh: mask-rcnn_r50-caffe_fpn_rpn-2conv_4conv1fc_syncbn-all_lsj-100e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco
              namezh: mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco
              namezh: mask-rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_swin-t-p4-w7_fpn_1x_coco
              namezh: mask-rcnn_swin-t-p4-w7_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco
              namezh: mask-rcnn_swin-t-p4-w7_fpn_amp-ms-crop-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tridentnet_r50-caffe_1x_coco
              namezh: tridentnet_r50-caffe_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tridentnet_r50-caffe_ms-1x_coco
              namezh: tridentnet_r50-caffe_ms-1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tridentnet_r50-caffe_ms-3x_coco
              namezh: tridentnet_r50-caffe_ms-3x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tood_r101_fpn_ms-2x_coco
              namezh: tood_r101_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tood_x101-64x4d_fpn_ms-2x_coco
              namezh: tood_x101-64x4d_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tood_r101-dconv-c3-c5_fpn_ms-2x_coco
              namezh: tood_r101-dconv-c3-c5_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tood_r50_fpn_anchor-based_1x_coco
              namezh: tood_r50_fpn_anchor-based_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tood_r50_fpn_1x_coco
              namezh: tood_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tood_r50_fpn_ms-2x_coco
              namezh: tood_r50_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vfnet_r50_fpn_1x_coco
              namezh: vfnet_r50_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vfnet_r50_fpn_ms-2x_coco
              namezh: vfnet_r50_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco
              namezh: vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vfnet_r101_fpn_1x_coco
              namezh: vfnet_r101_fpn_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vfnet_r101_fpn_ms-2x_coco
              namezh: vfnet_r101_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vfnet_r101-mdconv-c3-c5_fpn_ms-2x_coco
              namezh: vfnet_r101-mdconv-c3-c5_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vfnet_x101-32x4d-mdconv-c3-c5_fpn_ms-2x_coco
              namezh: vfnet_x101-32x4d-mdconv-c3-c5_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vfnet_x101-64x4d-mdconv-c3-c5_fpn_ms-2x_coco
              namezh: vfnet_x101-64x4d-mdconv-c3-c5_fpn_ms-2x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolact_r50_1x8_coco
              namezh: yolact_r50_1x8_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolact_r50_8x8_coco
              namezh: yolact_r50_8x8_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolact_r101_1x8_coco
              namezh: yolact_r101_1x8_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolov3_d53_320_273e_coco
              namezh: yolov3_d53_320_273e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolov3_d53_mstrain-416_273e_coco
              namezh: yolov3_d53_mstrain-416_273e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolov3_d53_mstrain-608_273e_coco
              namezh: yolov3_d53_mstrain-608_273e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolov3_d53_fp16_mstrain-608_273e_coco
              namezh: yolov3_d53_fp16_mstrain-608_273e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolov3_mobilenetv2_8xb24-320-300e_coco
              namezh: yolov3_mobilenetv2_8xb24-320-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolov3_mobilenetv2_8xb24-ms-416-300e_coco
              namezh: yolov3_mobilenetv2_8xb24-ms-416-300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolof_r50_c5_8x8_1x_coco
              namezh: yolof_r50_c5_8x8_1x_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolox_s_8x8_300e_coco
              namezh: yolox_s_8x8_300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolox_l_8x8_300e_coco
              namezh: yolox_l_8x8_300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolox_x_8x8_300e_coco
              namezh: yolox_x_8x8_300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: yolox_tiny_8x8_300e_coco
              namezh: yolox_tiny_8x8_300e_coco
              paper:
                text: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmdetection/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
        - titlezh: 目标检测
          titleen: Object Detection - COCO
          rankings:
            - nameen: atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco
              namezh: atss_swin-l-p4-w12_fpn_dyhead_mstrain_2x_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/DyHead?athId=b19bf998702a943f70e46d53b1054e51&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/DyHead?athId=b19bf998702a943f70e46d53b1054e51&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
              download: 646
              datasetA: '0.447'
              datasetB: '0.6943'
              ranking: '1'
            - nameen: yolox_x_8x8_300e_coco
              namezh: yolox_x_8x8_300e_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/YOLOX?athId=e0fd346d0ae014efd2de972e6df9dea8&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/YOLOX?athId=e0fd346d0ae014efd2de972e6df9dea8&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
              download: 5152
              datasetA: '0.426'
              datasetB: '0.7675'
              ranking: '2'
            - nameen: deformable-detr(deformable_detr_refine_r50_16x2_50e_coco)
              namezh: deformable-detr(deformable_detr_refine_r50_16x2_50e_coco)
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Deformable%20DETR?athId=45f3fa81f746aef44a5b0eb2eacb16c1&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Deformable%20DETR?athId=45f3fa81f746aef44a5b0eb2eacb16c1&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
              download: 6510
              datasetA: '0.286'
              datasetB: '0.6812'
              ranking: '3'
            - nameen: faster_rcnn_x101_64x4d_fpn_1x_coco
              namezh: faster_rcnn_x101_64x4d_fpn_1x_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Faster%20R-CNN?athId=6e1c4a83606f2a559343d2c69c93d10f&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Faster%20R-CNN?athId=6e1c4a83606f2a559343d2c69c93d10f&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
              download: 50486
              datasetA: '0.28'
              datasetB: '0.7185'
              ranking: '4'
            - nameen: deformable_detr_r50_16x2_50e_coco
              namezh: deformable_detr_r50_16x2_50e_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Deformable%20DETR?athId=45f3fa81f746aef44a5b0eb2eacb16c1&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Deformable%20DETR?athId=45f3fa81f746aef44a5b0eb2eacb16c1&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
              download: 5414
              datasetA: '0.276'
              datasetB: '0.6627'
              ranking: '5'
            - nameen: cascade_mask_rcnn_r50_fpn_20e_coco
              namezh: cascade_mask_rcnn_r50_fpn_20e_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Cascade%20Mask%20R-CNN?athId=0ec20d0f9d5914e4422d251f2ddf247b&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Cascade%20Mask%20R-CNN?athId=0ec20d0f9d5914e4422d251f2ddf247b&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
              download: 1505
              datasetA: '0.257'
              datasetB: '0.7193'
              ranking: '6'
            - nameen: gfl_r50_fpn_1x_coco
              namezh: gfl_r50_fpn_1x_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Generalized%20Focal%20Loss?athId=d5b8ec1f0fa4ca080d1be245181c200d&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Generalized%20Focal%20Loss?athId=d5b8ec1f0fa4ca080d1be245181c200d&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
              download: 1190
              datasetA: '0.246'
              datasetB: '0.6724'
              ranking: '7'
            - nameen: detr-resnet-50(detr_r50_8x2_150e_coco)
              namezh: detr-resnet-50(detr_r50_8x2_150e_coco)
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/DETR?athId=b609e23c7b56f32054cf4a85c0ef9c01&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/DETR?athId=b609e23c7b56f32054cf4a85c0ef9c01&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
              download: 566289
              datasetA: '0.241'
              datasetB: '0.6384'
              ranking: '8'
            - nameen: faster_rcnn_r50_caffe_c4_1x_coco
              namezh: faster_rcnn_r50_caffe_c4_1x_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Faster%20R-CNN?athId=6e1c4a83606f2a559343d2c69c93d10f&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Faster%20R-CNN?athId=6e1c4a83606f2a559343d2c69c93d10f&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
              download: 50486
              datasetA: '0.214'
              datasetB: '0.6946'
              ranking: '9'
            - nameen: yolox_tiny_8x8_300e_coco
              namezh: yolox_tiny_8x8_300e_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/YOLOX?athId=e0fd346d0ae014efd2de972e6df9dea8&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/YOLOX?athId=e0fd346d0ae014efd2de972e6df9dea8&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Object%20Detection
              download: 5152
              datasetA: '0.182'
              datasetB: '0.6965'
              ranking: '10'
        - titlezh: 实例分割
          titleen: Instance Segmentation - COCO
          rankings:
            - nameen: htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco
              namezh: htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/HTC?athId=b42170f82908262275e7328643dcdb2f&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/HTC?athId=b42170f82908262275e7328643dcdb2f&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
              download: 1559
              datasetA: '0.334'
              datasetB: '0.6576'
              ranking: '1'
            - nameen: >-
                cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco
              namezh: >-
                cascade_mask_rcnn_convnext-s_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Cascade%20Mask%20R-CNN?athId=0ec20d0f9d5914e4422d251f2ddf247b&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Cascade%20Mask%20R-CNN?athId=0ec20d0f9d5914e4422d251f2ddf247b&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
              download: 1505
              datasetA: '0.332'
              datasetB: '0.6814'
              ranking: '2'
            - nameen: scnet_x101_64x4d_fpn_20e_coco
              namezh: scnet_x101_64x4d_fpn_20e_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/SCNet?athId=17226ceb499bc933e2b73dd6633bbc2d&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/SCNet?athId=17226ceb499bc933e2b73dd6633bbc2d&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
              download: 90
              datasetA: '0.297'
              datasetB: '0.6143'
              ranking: '3'
            - nameen: cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco
              namezh: cascade_mask_rcnn_x101_32x4d_fpn_dconv_c3-c5_1x_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Deformable%20Convolutional%20Networks?athId=4aefab1107c2b0c71c3c091cc39b721d&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Deformable%20Convolutional%20Networks?athId=4aefab1107c2b0c71c3c091cc39b721d&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
              download: 1505
              datasetA: '0.287'
              datasetB: '0.6733'
              ranking: '4'
            - nameen: cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco
              namezh: cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Cascade%20Mask%20R-CNN?athId=0ec20d0f9d5914e4422d251f2ddf247b&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Cascade%20Mask%20R-CNN?athId=0ec20d0f9d5914e4422d251f2ddf247b&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
              download: 1505
              datasetA: '0.264'
              datasetB: '0.6536'
              ranking: '5'
            - nameen: cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco
              namezh: cascade_mask_rcnn_r50_fpn_dconv_c3-c5_1x_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Deformable%20Convolutional%20Networks?athId=4aefab1107c2b0c71c3c091cc39b721d&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Deformable%20Convolutional%20Networks?athId=4aefab1107c2b0c71c3c091cc39b721d&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
              download: 1505
              datasetA: '0.26'
              datasetB: '0.6541'
              ranking: '6'
            - nameen: rfnext_search_cascade_mask_rcnn_hrnetv2p_w18_20e_coco
              namezh: rfnext_search_cascade_mask_rcnn_hrnetv2p_w18_20e_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/RF-Next?athId=e99ac3889efff20e6fe2e8ac4ed9bc25&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/RF-Next?athId=e99ac3889efff20e6fe2e8ac4ed9bc25&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
              download: 37
              datasetA: '0.236'
              datasetB: '0.6474'
              ranking: '7'
            - nameen: cascade_mask_rcnn_r50_caffe_fpn_1x_coco
              namezh: cascade_mask_rcnn_r50_caffe_fpn_1x_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Cascade%20Mask%20R-CNN?athId=0ec20d0f9d5914e4422d251f2ddf247b&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/Cascade%20Mask%20R-CNN?athId=0ec20d0f9d5914e4422d251f2ddf247b&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
              download: 1505
              datasetA: '0.219'
              datasetB: '0.6275'
              ranking: '8'
            - nameen: yolact_r101_1x8_coco
              namezh: yolact_r101_1x8_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/YOLACT?athId=1c39dd15015b6452c3f753766ddb5278&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/YOLACT?athId=1c39dd15015b6452c3f753766ddb5278&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
              download: 2569
              datasetA: '0.201'
              datasetB: '0.6253'
              ranking: '9'
            - nameen: yolact_r50_8x8_coco
              namezh: yolact_r50_8x8_coco
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/YOLACT?athId=1c39dd15015b6452c3f753766ddb5278&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmdetection/YOLACT?athId=1c39dd15015b6452c3f753766ddb5278&repo=mmdetection&repoNameId=a4e3d984ec9475ca950bb6baf2b2a8e2&task=Instance%20Segmentation
              download: 2569
              datasetA: '0.178'
              datasetB: '0.6045'
              ranking: '10'
        - titlezh: 语义分割
          titleen: Semantic Segmentation - ADE20K
          rankings:
            - nameen: knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k
              namezh: knet_s3_upernet_swin-l_8x2_512x512_adamw_80k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/KNet?athId=36dcc0bba02bb32f43af76a927e050cf&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/KNet?athId=36dcc0bba02bb32f43af76a927e050cf&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 343
              datasetA: '42.31'
              datasetB: '0.6286'
              ranking: '1'
            - nameen: >-
                upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k
              namezh: >-
                upernet_swin_large_patch4_window12_512x512_pretrain_384x384_22K_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/UPerNet?athId=6eedb26553f6ddb295adee667149f722&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/UPerNet?athId=6eedb26553f6ddb295adee667149f722&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 1967
              datasetA: '40.35'
              datasetB: '0.5528'
              ranking: '2'
            - nameen: segformer_mit-b4_512x512_160k_ade20k
              namezh: segformer_mit-b4_512x512_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/Segformer?athId=94937aa281ea263f6484a359dfa3ec4b&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/Segformer?athId=94937aa281ea263f6484a359dfa3ec4b&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 4600
              datasetA: '36.88'
              datasetB: '0.4756'
              ranking: '3'
            - nameen: setr_mla_512x512_160k_b16_ade20k
              namezh: setr_mla_512x512_160k_b16_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/SETR?athId=a0088b8a1527ee3e20b6241c2b66b496&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/SETR?athId=a0088b8a1527ee3e20b6241c2b66b496&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 3437
              datasetA: '36.67'
              datasetB: '0.5365'
              ranking: '4'
            - nameen: twins_svt-b_fpn_fpnhead_8x4_512x512_80k_ade20k
              namezh: twins_svt-b_fpn_fpnhead_8x4_512x512_80k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/FPN?athId=7f617fa591d3dfd31fb2a9a7cc0ae8ba&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/FPN?athId=7f617fa591d3dfd31fb2a9a7cc0ae8ba&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 1503
              datasetA: '31.82'
              datasetB: '0.5073'
              ranking: '5'
            - nameen: dpt_vit-b16_512x512_160k_ade20k
              namezh: dpt_vit-b16_512x512_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DPT?athId=b2c699d0fddf59a4e952cecea08b1b8b&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DPT?athId=b2c699d0fddf59a4e952cecea08b1b8b&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 1813
              datasetA: '30.61'
              datasetB: '0.4642'
              ranking: '6'
            - nameen: deeplabv3_r101-d8_512x512_160k_ade20k
              namezh: deeplabv3_r101-d8_512x512_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DeepLabV3?athId=6f315fcddecd0407b37cae1346078876&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DeepLabV3?athId=6f315fcddecd0407b37cae1346078876&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 11112
              datasetA: '30.4'
              datasetB: '0.4265'
              ranking: '7'
            - nameen: fcn_hr48_512x512_160k_ade20k
              namezh: fcn_hr48_512x512_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/FCN?athId=9cb4ee8cc5fee1e37d4418259aa76d81&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/FCN?athId=9cb4ee8cc5fee1e37d4418259aa76d81&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 52662
              datasetA: '27.4'
              datasetB: '0.4143'
              ranking: '8'
            - nameen: dnl_r50-d8_4xb4-160k_ade20k-512x512
              namezh: dnl_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DNLNet?athId=e7a94769be0d3a1b41a6e067db8e0f5d&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DNLNet?athId=e7a94769be0d3a1b41a6e067db8e0f5d&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 375
              datasetA: '26.3'
              datasetB: '0.4419'
              ranking: '9'
            - nameen: segmenter_vit-b_mask_8x1_512x512_160k_ade20k
              namezh: segmenter_vit-b_mask_8x1_512x512_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/Segmenter?athId=0a8f2e1dccdce40c26a35ebe5b074f36&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/Segmenter?athId=0a8f2e1dccdce40c26a35ebe5b074f36&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 1845
              datasetA: '22.29'
              datasetB: '0.6068'
              ranking: '10'
        - titlezh: 医学图像分类
          titleen: Medical Image Classification - CheXpert
          rankings:
            - nameen: CheXpert-5-convnextv2-tiny-384
              namezh: CheXpert-5-convnextv2-tiny-384
              paper:
                text: 'https://huggingface.co/shreydan/CheXpert-5-convnextv2-tiny-384'
                link: 'https://huggingface.co/shreydan/CheXpert-5-convnextv2-tiny-384'
              download: 14
              datasetA: 67.04%
              datasetB: 56.60%
              ranking: '1'
            - nameen: vit_small_patch16_224.medmae_CXR_mae_ft_CheXpert
              namezh: vit_small_patch16_224.medmae_CXR_mae_ft_CheXpert
              paper:
                text: >-
                  https://huggingface.co/1aurent/vit_small_patch16_224.medmae_CXR_mae_ft_CheXpert
                link: >-
                  https://huggingface.co/1aurent/vit_small_patch16_224.medmae_CXR_mae_ft_CheXpert
              download: 2
              datasetA: 66.92%
              datasetB: 89.74%
              ranking: '2'
            - nameen: vit_base_patch16_224.medmae_CXR_mae_ft_CheXpert
              namezh: vit_base_patch16_224.medmae_CXR_mae_ft_CheXpert
              paper:
                text: >-
                  https://huggingface.co/1aurent/vit_base_patch16_224.medmae_CXR_mae_ft_CheXpert/tree/main
                link: >-
                  https://huggingface.co/1aurent/vit_base_patch16_224.medmae_CXR_mae_ft_CheXpert/tree/main
              download: 4
              datasetA: 57.58%
              datasetB: 89.39%
              ranking: '3'
    table2:
      tab2en: White-box
      tab2zh: 白盒
      columnName1: Model Name
      columnName2: Paper
      columnName3: Citations
      columnName4: Evaluation Result
      columnName4A: Clean Acc
      columnName4B: Robust Acc
      columnName5: Rank
      modelsRanking2:
        - titlezh: CIFAR-10
          titleen: CIFAR-10
          rankings:
            - nameen: RaWideResNet-70-16
              namezh: RaWideResNet-70-16
              paper:
                text: >-
                  Robust Principles: Architectural Design Principles for
                  Adversarially Robust CNNs
                link: 'https://arxiv.org/abs/2308.16258'
              download: 38
              datasetA: 93.27%
              datasetB: 71.10%
              ranking: '1'
            - nameen: WideResNet-70-16
              namezh: WideResNet-70-16
              paper:
                text: Better Diffusion Models Further Improve Adversarial Training
                link: 'https://arxiv.org/abs/2302.04638'
              download: 194
              datasetA: 93.25%
              datasetB: 70.70%
              ranking: '2'
            - nameen: ResNet-152 + WideResNet-70-16 + mixing network
              namezh: ResNet-152 + WideResNet-70-16 + mixing network
              paper:
                text: >-
                  Improving the Accuracy-Robustness Trade-off of Classifiers via
                  Adaptive Smoothing
                link: 'https://arxiv.org/abs/2301.12554'
              download: 13
              datasetA: 95.23%
              datasetB: 68.06%
              ranking: '3'
            - nameen: WideResNet-28-10
              namezh: WideResNet-28-10
              paper:
                text: Decoupled Kullback-Leibler Divergence Loss
                link: 'https://arxiv.org/abs/2305.13948'
              download: 34
              datasetA: 92.16%
              datasetB: 67.75%
              ranking: '4'
            - nameen: WideResNet-28-10
              namezh: WideResNet-28-10
              paper:
                text: Better Diffusion Models Further Improve Adversarial Training
                link: 'https://arxiv.org/abs/2305.13948'
              download: 194
              datasetA: 92.44%
              datasetB: 67.31%
              ranking: '5'
            - nameen: WideResNet-70-16
              namezh: WideResNet-70-16
              paper:
                text: Fixing Data Augmentation to Improve Adversarial Robustness
                link: 'https://arxiv.org/abs/2302.04638'
              download: 285
              datasetA: 92.23%
              datasetB: 66.59%
              ranking: '6'
            - nameen: WideResNet-70-16
              namezh: WideResNet-70-16
              paper:
                text: Improving Robustness using Generated Data
                link: 'https://arxiv.org/abs/2110.09468'
              download: 287
              datasetA: 88.74%
              datasetB: 66.14%
              ranking: '7'
            - nameen: WideResNet-70-16
              namezh: WideResNet-70-16
              paper:
                text: >-
                  Uncovering the Limits of Adversarial Training against
                  Norm-Bounded Adversarial Examples
                link: 'https://arxiv.org/abs/2010.03593'
              download: 345
              datasetA: 91.10%
              datasetB: 65.89%
              ranking: '8'
            - nameen: WideResNet-A4
              namezh: WideResNet-A4
              paper:
                text: >-
                  Revisiting Residual Networks for Adversarial Robustness: An
                  Architectural Perspective
                link: 'https://arxiv.org/abs/2212.11005'
              download: 38
              datasetA: 91.59%
              datasetB: 65.78%
              ranking: '9'
            - nameen: WideResNet-106-16
              namezh: WideResNet-106-16
              paper:
                text: Fixing Data Augmentation to Improve Adversarial Robustness
                link: 'https://arxiv.org/abs/2103.01946'
              download: 285
              datasetA: 88.50%
              datasetB: 64.68%
              ranking: '10'
        - titlezh: CIFAR-100
          titleen: CIFAR-100
          rankings:
            - nameen: WideResNet-70-16
              namezh: WideResNet-70-16
              paper:
                text: Better Diffusion Models Further Improve Adversarial Training
                link: null
              download: 194
              datasetA: 75.23%
              datasetB: 42.83%
              ranking: '1'
            - nameen: WideResNet-28-10
              namezh: WideResNet-28-10
              paper:
                text: Decoupled Kullback-Leibler Divergence Loss
                link: null
              download: 34
              datasetA: 73.83%
              datasetB: 39.39%
              ranking: '2'
            - nameen: WideResNet-28-10
              namezh: WideResNet-28-10
              paper:
                text: Better Diffusion Models Further Improve Adversarial Training
                link: null
              download: 194
              datasetA: 72.58%
              datasetB: 38.92%
              ranking: '3'
            - nameen: WideResNet-70-16
              namezh: WideResNet-70-16
              paper:
                text: >-
                  Uncovering the Limits of Adversarial Training against
                  Norm-Bounded Adversarial Examples
                link: null
              download: 345
              datasetA: 69.15%
              datasetB: 37.20%
              ranking: '4'
            - nameen: XCiT-L12
              namezh: XCiT-L12
              paper:
                text: A Light Recipe to Train Robust Vision Transformers
                link: null
              download: 56
              datasetA: 70.77%
              datasetB: 35.27%
              ranking: '5'
            - nameen: WideResNet-70-16
              namezh: WideResNet-70-16
              paper:
                text: Fixing Data Augmentation to Improve Adversarial Robustness
                link: null
              download: 285
              datasetA: 63.56%
              datasetB: 34.74%
              ranking: '6'
            - nameen: XCiT-M12
              namezh: XCiT-M12
              paper:
                text: A Light Recipe to Train Robust Vision Transformers
                link: null
              download: 56
              datasetA: 69.20%
              datasetB: 34.33%
              ranking: '7'
            - nameen: WideResNet-70-16
              namezh: WideResNet-70-16
              paper:
                text: >-
                  Robustness and Accuracy Could Be Reconcilable by (Proper)
                  Definition
                link: null
              download: 136
              datasetA: 65.56%
              datasetB: 33.14%
              ranking: '8'
        - titlezh: ImageNet-1k
          titleen: ImageNet-1k
          rankings:
            - nameen: ConvNeXtV2-L + Swin-L
              namezh: ConvNeXtV2-L + Swin-L
              paper:
                text: >-
                  MixedNUTS: Training-Free Accuracy-Robustness Balance via
                  Nonlinearly Mixed Classifiers
                link: null
              download: 2
              datasetA: 81.10%
              datasetB: 58.65%
              ranking: '1'
            - nameen: Swin-L
              namezh: Swin-L
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 57
              datasetA: 78.18%
              datasetB: 57.35%
              ranking: '2'
            - nameen: ConvNeXt-L
              namezh: ConvNeXt-L
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 57
              datasetA: 77.48%
              datasetB: 56.53%
              ranking: '3'
            - nameen: ConvNeXt-L + ConvStem
              namezh: ConvNeXt-L + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 48
              datasetA: 76.79%
              datasetB: 55.94%
              ranking: '4'
            - nameen: Swin-B
              namezh: Swin-B
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 57
              datasetA: 76.22%
              datasetB: 54.41%
              ranking: '5'
            - nameen: ConvNeXt-B
              namezh: ConvNeXt-B
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 57
              datasetA: 76.38%
              datasetB: 54.13%
              ranking: '6'
            - nameen: ConvNeXt-B + ConvStem
              namezh: ConvNeXt-B + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 48
              datasetA: 75.46%
              datasetB: 53.94%
              ranking: '7'
            - nameen: ViT-B + ConvStem
              namezh: ViT-B + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 48
              datasetA: 76.12%
              datasetB: 52.82%
              ranking: '8'
            - nameen: ConvNeXt-S + ConvStem
              namezh: ConvNeXt-S + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 48
              datasetA: 73.37%
              datasetB: 49.74%
              ranking: '9'
            - nameen: RaWideResNet-101-2
              namezh: RaWideResNet-101-2
              paper:
                text: >-
                  Robust Principles: Architectural Design Principles for
                  Adversarially Robust CNNs
                link: null
              download: 38
              datasetA: 73.45%
              datasetB: 49.06%
              ranking: '10'
            - nameen: ConvNeXt-T + ConvStem
              namezh: ConvNeXt-T + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 48
              datasetA: 72.45%
              datasetB: 47.70%
              ranking: '11'
        - titlezh: CC1M
          titleen: CC1M
          rankings:
            - nameen: ConvNeXt-L + ConvStem
              namezh: ConvNeXt-L + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 48
              datasetA: 100%
              datasetB: 18.17%
              ranking: '1'
            - nameen: ConvNeXtV2-L + Swin-L
              namezh: ConvNeXtV2-L + Swin-L
              paper:
                text: >-
                  MixedNUTS: Training-Free Accuracy-Robustness Balance via
                  Nonlinearly Mixed Classifiers
                link: null
              download: 2
              datasetA: 100%
              datasetB: 17.56%
              ranking: '2'
            - nameen: Swin-L
              namezh: Swin-L
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 57
              datasetA: 100%
              datasetB: 17.23%
              ranking: '3'
            - nameen: ConvNeXt-L
              namezh: ConvNeXt-L
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 57
              datasetA: 100%
              datasetB: 17.13%
              ranking: '4'
            - nameen: Swin-B
              namezh: Swin-B
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 57
              datasetA: 100%
              datasetB: 16.78%
              ranking: '5'
    _template: table
---

