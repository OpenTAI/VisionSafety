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

            - nameen: mask-rcnn_r50_fpn_albu-1x_coco
              namezh: mask-rcnn_r50_fpn_albu-1x_coco
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
            
            - nameen: pisa_mask_rcnn_r50_fpn_1x_coco
              namezh: pisa_mask_rcnn_r50_fpn_1x_coco
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
            
            - nameen: mask-rcnn_res2net-101_fpn_2x_coco
              namezh: mask-rcnn_res2net-101_fpn_2x_coco
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
            
            - nameen: cascade-mask-rcnn_r50_fpn_rsb-pretrain_1x_coco
              namezh: cascade-mask-rcnn_r50_fpn_rsb-pretrain_1x_coco
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
              
            - nameen: ann_r50-d8_4xb4-80k_ade20k-512x512
              namezh: ann_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ann_r101-d8_4xb4-80k_ade20k-512x512
              namezh: ann_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ann_r50-d8_4xb4-160k_ade20k-512x512
              namezh: ann_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ann_r101-d8_4xb4-160k_ade20k-512x512
              namezh: ann_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: apcnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: apcnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: apcnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: apcnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: apcnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: apcnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: apcnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: apcnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: beit-base_upernet_8xb2-160k_ade20k-640x640
              namezh: beit-base_upernet_8xb2-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: beit-large_upernet_8xb1-amp-160k_ade20k-640x640
              namezh: beit-large_upernet_8xb1-amp-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ccnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: ccnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ccnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: ccnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ccnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: ccnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ccnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: ccnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext-tiny_upernet_8xb2-amp-160k_ade20k-512x512
              namezh: convnext-tiny_upernet_8xb2-amp-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext-small_upernet_8xb2-amp-160k_ade20k-512x512
              namezh: convnext-small_upernet_8xb2-amp-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext-base_upernet_8xb2-amp-160k_ade20k-512x512
              namezh: convnext-base_upernet_8xb2-amp-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext-base_upernet_8xb2-amp-160k_ade20k-640x640
              namezh: convnext-base_upernet_8xb2-amp-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext-large_upernet_8xb2-amp-160k_ade20k-640x640
              namezh: convnext-large_upernet_8xb2-amp-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext-xlarge_upernet_8xb2-amp-160k_ade20k-640x640
              namezh: convnext-xlarge_upernet_8xb2-amp-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: danet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: danet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: danet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: danet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: danet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: danet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: danet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: danet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deeplabv3_r50-d8_4xb4-80k_ade20k-512x512
              namezh: deeplabv3_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deeplabv3_r101-d8_4xb4-80k_ade20k-512x512
              namezh: deeplabv3_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deeplabv3_r50-d8_4xb4-160k_ade20k-512x512
              namezh: deeplabv3_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deeplabv3_r101-d8_4xb4-160k_ade20k-512x512
              namezh: deeplabv3_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512
              namezh: deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512
              namezh: deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512
              namezh: deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512
              namezh: deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dmnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: dmnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dmnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: dmnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dmnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: dmnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dmnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: dmnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dnl_r50-d8_4xb4-80k_ade20k-512x512
              namezh: dnl_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dnl_r101-d8_4xb4-80k_ade20k-512x512
              namezh: dnl_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dnl_r50-d8_4xb4-160k_ade20k-512x512
              namezh: dnl_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dnl_r101-d8_4xb4-160k_ade20k-512x512
              namezh: dnl_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dpt_vit-b16_8xb2-160k_ade20k-512x512
              namezh: dpt_vit-b16_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: encnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: encnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: encnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: encnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: encnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: encnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: encnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: encnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastfcn_r50-d32_jpu_aspp_4xb4-80k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_aspp_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastfcn_r50-d32_jpu_aspp_4xb4-160k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_aspp_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastfcn_r50-d32_jpu_psp_4xb4-80k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_psp_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastfcn_r50-d32_jpu_psp_4xb4-160k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_psp_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastfcn_r50-d32_jpu_enc_4xb4-80k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_enc_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastfcn_r50-d32_jpu_enc_4xb4-160k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_enc_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcn_r50-d8_4xb4-80k_ade20k-512x512
              namezh: fcn_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcn_r101-d8_4xb4-80k_ade20k-512x512
              namezh: fcn_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcn_r50-d8_4xb4-160k_ade20k-512x512
              namezh: fcn_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcn_r101-d8_4xb4-160k_ade20k-512x512
              namezh: fcn_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: gcnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: gcnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: gcnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: gcnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcn_hr18s_4xb4-80k_ade20k-512x512
              namezh: fcn_hr18s_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcn_hr18_4xb4-80k_ade20k-512x512
              namezh: fcn_hr18_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcn_hr48_4xb4-80k_ade20k-512x512
              namezh: fcn_hr48_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcn_hr18s_4xb4-160k_ade20k-512x512
              namezh: fcn_hr18s_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcn_hr18_4xb4-160k_ade20k-512x512
              namezh: fcn_hr18_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fcn_hr48_4xb4-160k_ade20k-512x512
              namezh: fcn_hr48_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: isanet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: isanet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: isanet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: isanet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: isanet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: isanet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: isanet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: isanet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: knet-s3_r50-d8_fcn_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_r50-d8_fcn_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: knet-s3_r50-d8_pspnet_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_r50-d8_pspnet_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: knet-s3_r50-d8_deeplabv3_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_r50-d8_deeplabv3_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: knet-s3_r50-d8_upernet_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_r50-d8_upernet_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-640x640
              namezh: knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mae-base_upernet_8xb2-amp-160k_ade20k-512x512
              namezh: mae-base_upernet_8xb2-amp-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_r50_8xb2-160k_ade20k-512x512
              namezh: mask2former_r50_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_r101_8xb2-160k_ade20k-512x512
              namezh: mask2former_r101_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-t_8xb2-160k_ade20k-512x512
              namezh: mask2former_swin-t_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-s_8xb2-160k_ade20k-512x512
              namezh: mask2former_swin-s_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640
              namezh: mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640
              namezh: mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640
              namezh: mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maskformer_r50-d32_8xb2-160k_ade20k-512x512
              namezh: maskformer_r50-d32_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maskformer_r101-d32_8xb2-160k_ade20k-512x512
              namezh: maskformer_r101-d32_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512
              namezh: maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maskformer_swin-s_upernet_8xb2-160k_ade20k-512x512
              namezh: maskformer_swin-s_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet-v2-d8_fcn_4xb4-160k_ade20k-512x512
              namezh: mobilenet-v2-d8_fcn_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet-v2-d8_pspnet_4xb4-160k_ade20k-512x512
              namezh: mobilenet-v2-d8_pspnet_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet-v2-d8_deeplabv3_4xb4-160k_ade20k-512x512
              namezh: mobilenet-v2-d8_deeplabv3_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet-v2-d8_deeplabv3plus_4xb4-160k_ade20k-512x512
              namezh: mobilenet-v2-d8_deeplabv3plus_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nonlocal_r50-d8_4xb4-80k_ade20k-512x512
              namezh: nonlocal_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nonlocal_r101-d8_4xb4-80k_ade20k-512x512
              namezh: nonlocal_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nonlocal_r50-d8_4xb4-160k_ade20k-512x512
              namezh: nonlocal_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nonlocal_r101-d8_4xb4-160k_ade20k-512x512
              namezh: nonlocal_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ocrnet_hr18s_4xb4-80k_ade20k-512x512
              namezh: ocrnet_hr18s_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ocrnet_hr18_4xb4-80k_ade20k-512x512
              namezh: ocrnet_hr18_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ocrnet_hr48_4xb4-80k_ade20k-512x512
              namezh: ocrnet_hr48_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ocrnet_hr18s_4xb4-80k_ade20k-512x512
              namezh: ocrnet_hr18s_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ocrnet_hr18_4xb4-80k_ade20k-512x512
              namezh: ocrnet_hr18_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ocrnet_hr48_4xb4-160k_ade20k-512x512
              namezh: ocrnet_hr48_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pointrend_r50_4xb4-160k_ade20k-512x512
              namezh: pointrend_r50_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pointrend_r101_4xb4-160k_ade20k-512x512
              namezh: pointrend_r101_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fpn_poolformer_s12_8xb4-40k_ade20k-512x512
              namezh: fpn_poolformer_s12_8xb4-40k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fpn_poolformer_s24_8xb4-40k_ade20k-512x512
              namezh: fpn_poolformer_s24_8xb4-40k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fpn_poolformer_s36_8xb4-40k_ade20k-512x512
              namezh: fpn_poolformer_s36_8xb4-40k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fpn_poolformer_m36_8xb4-40k_ade20k-512x512
              namezh: fpn_poolformer_m36_8xb4-40k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fpn_poolformer_m48_8xb4-40k_ade20k-512x512
              namezh: fpn_poolformer_m48_8xb4-40k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: psanet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: psanet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: psanet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: psanet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: psanet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: psanet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: psanet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: psanet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pspnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: pspnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pspnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: pspnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pspnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: pspnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pspnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: pspnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest_s101-d8_fcn_4xb4-160k_ade20k-512x512
              namezh: resnest_s101-d8_fcn_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest_s101-d8_pspnet_4xb4-160k_ade20k-512x512
              namezh: resnest_s101-d8_pspnet_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest_s101-d8_deeplabv3_4xb4-160k_ade20k-512x512
              namezh: resnest_s101-d8_deeplabv3_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest_s101-d8_deeplabv3plus_4xb4-160k_ade20k-512x512
              namezh: resnest_s101-d8_deeplabv3plus_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segformer_mit-b0_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b0_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segformer_mit-b1_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b1_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segformer_mit-b2_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b2_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segformer_mit-b3_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b3_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segformer_mit-b4_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b4_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segformer_mit-b5_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b5_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segformer_mit-b5_8xb2-160k_ade20k-640x640
              namezh: segformer_mit-b5_8xb2-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segmenter_vit-t_mask_8xb1-160k_ade20k-512x512
              namezh: segmenter_vit-t_mask_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segmenter_vit-s_fcn_8xb1-160k_ade20k-512x512
              namezh: segmenter_vit-s_fcn_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segmenter_vit-s_mask_8xb1-160k_ade20k-512x512
              namezh: segmenter_vit-s_mask_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segmenter_vit-b_mask_8xb1-160k_ade20k-512x512
              namezh: segmenter_vit-b_mask_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segmenter_vit-l_mask_8xb1-160k_ade20k-512x512
              namezh: segmenter_vit-l_mask_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512
              namezh: segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segnext_mscan-s_1xb16-adamw-160k_ade20k-512x512
              namezh: segnext_mscan-s_1xb16-adamw-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512
              namezh: segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512
              namezh: segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fpn_r50_4xb4-160k_ade20k-512x512
              namezh: fpn_r50_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fpn_r101_4xb4-160k_ade20k-512x512
              namezh: fpn_r101_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: setr_vit-l_naive_8xb2-160k_ade20k-512x512
              namezh: setr_vit-l_naive_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: setr_vit-l_pup_8xb2-160k_ade20k-512x512
              namezh: setr_vit-l_pup_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: setr_vit-l-mla_8xb1-160k_ade20k-512x512
              namezh: setr_vit-l-mla_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: setr_vit-l_mla_8xb2-160k_ade20k-512x512
              namezh: setr_vit-l_mla_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin-base-patch4-window12-in1k-384x384-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-base-patch4-window12-in1k-384x384-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin-base-patch4-window12-in22k-384x384-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-base-patch4-window12-in22k-384x384-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_pcpvt-s_uperhead_8xb4-160k_ade20k-512x512
              namezh: twins_pcpvt-s_uperhead_8xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_pcpvt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_pcpvt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_pcpvt-b_uperhead_8xb2-160k_ade20k-512x512
              namezh: twins_pcpvt-b_uperhead_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_pcpvt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_pcpvt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_pcpvt-l_uperhead_8xb2-160k_ade20k-512x512
              namezh: twins_pcpvt-l_uperhead_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_svt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_svt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_svt-s_uperhead_8xb2-160k_ade20k-512x512
              namezh: twins_svt-s_uperhead_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_svt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_svt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_svt-b_uperhead_8xb2-160k_ade20k-512x512
              namezh: twins_svt-b_uperhead_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_svt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_svt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_pcpvt-l_uperhead_8xb2-160k_ade20k-512x512
              namezh: twins_pcpvt-l_uperhead_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: upernet_r50_4xb4-80k_ade20k-512x512
              namezh: upernet_r50_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: upernet_r101_4xb4-80k_ade20k-512x512
              namezh: upernet_r101_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: upernet_r50_4xb4-160k_ade20k-512x512
              namezh: upernet_r50_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: upernet_r101_4xb4-160k_ade20k-512x512
              namezh: upernet_r101_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512
              namezh: vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_deit-s16_upernet_8xb2-80k_ade20k-512x512
              namezh: vit_deit-s16_upernet_8xb2-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_deit-s16_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-s16_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_deit-s16_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-s16_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_deit-b16_upernet_8xb2-80k_ade20k-512x512
              namezh: vit_deit-b16_upernet_8xb2-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_deit-b16_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-b16_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_deit-b16_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-b16_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_deit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
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
              
            - nameen: bat_resnext26ts
              namezh: bat_resnext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: beit_base_patch16_224
              namezh: beit_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: beit_base_patch16_384
              namezh: beit_base_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: beit_large_patch16_224
              namezh: beit_large_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: beit_large_patch16_384
              namezh: beit_large_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: beit_large_patch16_512
              namezh: beit_large_patch16_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: beitv2_base_patch16_224
              namezh: beitv2_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: beitv2_large_patch16_224
              namezh: beitv2_large_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: botnet26t_256
              namezh: botnet26t_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: botnet50ts_256
              namezh: botnet50ts_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: caformer_b36
              namezh: caformer_b36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: caformer_m36
              namezh: caformer_m36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: caformer_s18
              namezh: caformer_s18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: caformer_s36
              namezh: caformer_s36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cait_m36_384
              namezh: cait_m36_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cait_m48_448
              namezh: cait_m48_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cait_s24_224
              namezh: cait_s24_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cait_s24_384
              namezh: cait_s24_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cait_s36_384
              namezh: cait_s36_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cait_xs24_384
              namezh: cait_xs24_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cait_xxs24_224
              namezh: cait_xxs24_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cait_xxs24_384
              namezh: cait_xxs24_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cait_xxs36_224
              namezh: cait_xxs36_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cait_xxs36_384
              namezh: cait_xxs36_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coat_lite_medium
              namezh: coat_lite_medium
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coat_lite_medium_384
              namezh: coat_lite_medium_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coat_lite_mini
              namezh: coat_lite_mini
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coat_lite_small
              namezh: coat_lite_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coat_lite_tiny
              namezh: coat_lite_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coat_mini
              namezh: coat_mini
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coat_small
              namezh: coat_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coat_tiny
              namezh: coat_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_0_224
              namezh: coatnet_0_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_0_rw_224
              namezh: coatnet_0_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_1_224
              namezh: coatnet_1_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_1_rw_224
              namezh: coatnet_1_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_2_224
              namezh: coatnet_2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_2_rw_224
              namezh: coatnet_2_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_3_224
              namezh: coatnet_3_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_3_rw_224
              namezh: coatnet_3_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_4_224
              namezh: coatnet_4_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_5_224
              namezh: coatnet_5_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_bn_0_rw_224
              namezh: coatnet_bn_0_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_nano_cc_224
              namezh: coatnet_nano_cc_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_nano_rw_224
              namezh: coatnet_nano_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_pico_rw_224
              namezh: coatnet_pico_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_rmlp_0_rw_224
              namezh: coatnet_rmlp_0_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_rmlp_1_rw2_224
              namezh: coatnet_rmlp_1_rw2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_rmlp_1_rw_224
              namezh: coatnet_rmlp_1_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_rmlp_2_rw_224
              namezh: coatnet_rmlp_2_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_rmlp_2_rw_384
              namezh: coatnet_rmlp_2_rw_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_rmlp_3_rw_224
              namezh: coatnet_rmlp_3_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnet_rmlp_nano_rw_224
              namezh: coatnet_rmlp_nano_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: coatnext_nano_rw_224
              namezh: coatnext_nano_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convformer_b36
              namezh: convformer_b36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convformer_m36
              namezh: convformer_m36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convformer_s18
              namezh: convformer_s18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convformer_s36
              namezh: convformer_s36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convit_base
              namezh: convit_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convit_small
              namezh: convit_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convit_tiny
              namezh: convit_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convmixer_768_32
              namezh: convmixer_768_32
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convmixer_1024_20_ks9_p14
              namezh: convmixer_1024_20_ks9_p14
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convmixer_1536_20
              namezh: convmixer_1536_20
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_atto
              namezh: convnext_atto
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_atto_ols
              namezh: convnext_atto_ols
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_base
              namezh: convnext_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_femto
              namezh: convnext_femto
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_femto_ols
              namezh: convnext_femto_ols
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_large
              namezh: convnext_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_large_mlp
              namezh: convnext_large_mlp
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_nano
              namezh: convnext_nano
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_nano_ols
              namezh: convnext_nano_ols
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_pico
              namezh: convnext_pico
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_pico_ols
              namezh: convnext_pico_ols
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_small
              namezh: convnext_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_tiny
              namezh: convnext_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_tiny_hnf
              namezh: convnext_tiny_hnf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_xlarge
              namezh: convnext_xlarge
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnext_xxlarge
              namezh: convnext_xxlarge
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnextv2_atto
              namezh: convnextv2_atto
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnextv2_base
              namezh: convnextv2_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnextv2_femto
              namezh: convnextv2_femto
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnextv2_huge
              namezh: convnextv2_huge
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnextv2_large
              namezh: convnextv2_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnextv2_nano
              namezh: convnextv2_nano
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnextv2_pico
              namezh: convnextv2_pico
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnextv2_small
              namezh: convnextv2_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: convnextv2_tiny
              namezh: convnextv2_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_9_240
              namezh: crossvit_9_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_9_dagger_240
              namezh: crossvit_9_dagger_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_15_240
              namezh: crossvit_15_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_15_dagger_240
              namezh: crossvit_15_dagger_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_15_dagger_408
              namezh: crossvit_15_dagger_408
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_18_240
              namezh: crossvit_18_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_18_dagger_240
              namezh: crossvit_18_dagger_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_18_dagger_408
              namezh: crossvit_18_dagger_408
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_base_240
              namezh: crossvit_base_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_small_240
              namezh: crossvit_small_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: crossvit_tiny_240
              namezh: crossvit_tiny_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3darknet_focus_l
              namezh: cs3darknet_focus_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3darknet_focus_m
              namezh: cs3darknet_focus_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3darknet_focus_s
              namezh: cs3darknet_focus_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3darknet_focus_x
              namezh: cs3darknet_focus_x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3darknet_l
              namezh: cs3darknet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3darknet_m
              namezh: cs3darknet_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3darknet_s
              namezh: cs3darknet_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3darknet_x
              namezh: cs3darknet_x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3edgenet_x
              namezh: cs3edgenet_x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3se_edgenet_x
              namezh: cs3se_edgenet_x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3sedarknet_l
              namezh: cs3sedarknet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3sedarknet_x
              namezh: cs3sedarknet_x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cs3sedarknet_xdw
              namezh: cs3sedarknet_xdw
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cspdarknet53
              namezh: cspdarknet53
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cspresnet50
              namezh: cspresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cspresnet50d
              namezh: cspresnet50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cspresnet50w
              namezh: cspresnet50w
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: cspresnext50
              namezh: cspresnext50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: darknet17
              namezh: darknet17
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: darknet21
              namezh: darknet21
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: darknet53
              namezh: darknet53
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: darknetaa53
              namezh: darknetaa53
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: davit_base
              namezh: davit_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: davit_giant
              namezh: davit_giant
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: davit_huge
              namezh: davit_huge
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: davit_large
              namezh: davit_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: davit_small
              namezh: davit_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: davit_tiny
              namezh: davit_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit3_base_patch16_224
              namezh: deit3_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit3_base_patch16_384
              namezh: deit3_base_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit3_huge_patch14_224
              namezh: deit3_huge_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit3_large_patch16_224
              namezh: deit3_large_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit3_large_patch16_384
              namezh: deit3_large_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit3_medium_patch16_224
              namezh: deit3_medium_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit3_small_patch16_224
              namezh: deit3_small_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit3_small_patch16_384
              namezh: deit3_small_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit_base_distilled_patch16_224
              namezh: deit_base_distilled_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit_base_distilled_patch16_384
              namezh: deit_base_distilled_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit_base_patch16_224
              namezh: deit_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit_base_patch16_384
              namezh: deit_base_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit_small_distilled_patch16_224
              namezh: deit_small_distilled_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit_small_patch16_224
              namezh: deit_small_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit_tiny_distilled_patch16_224
              namezh: deit_tiny_distilled_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: deit_tiny_patch16_224
              namezh: deit_tiny_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: densenet121
              namezh: densenet121
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: densenet161
              namezh: densenet161
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: densenet169
              namezh: densenet169
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: densenet201
              namezh: densenet201
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: densenet264d
              namezh: densenet264d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: densenetblur121d
              namezh: densenetblur121d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla34
              namezh: dla34
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla46_c
              namezh: dla46_c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla46x_c
              namezh: dla46x_c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla60
              namezh: dla60
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla60_res2net
              namezh: dla60_res2net
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla60_res2next
              namezh: dla60_res2next
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla60x
              namezh: dla60x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla60x_c
              namezh: dla60x_c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla102
              namezh: dla102
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla102x
              namezh: dla102x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla102x2
              namezh: dla102x2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dla169
              namezh: dla169
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dm_nfnet_f0
              namezh: dm_nfnet_f0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dm_nfnet_f1
              namezh: dm_nfnet_f1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dm_nfnet_f2
              namezh: dm_nfnet_f2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dm_nfnet_f3
              namezh: dm_nfnet_f3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dm_nfnet_f4
              namezh: dm_nfnet_f4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dm_nfnet_f5
              namezh: dm_nfnet_f5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dm_nfnet_f6
              namezh: dm_nfnet_f6
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dpn48b
              namezh: dpn48b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dpn68
              namezh: dpn68
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dpn68b
              namezh: dpn68b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dpn92
              namezh: dpn92
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dpn98
              namezh: dpn98
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dpn107
              namezh: dpn107
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: dpn131
              namezh: dpn131
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eca_botnext26ts_256
              namezh: eca_botnext26ts_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eca_halonext26ts
              namezh: eca_halonext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eca_nfnet_l0
              namezh: eca_nfnet_l0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eca_nfnet_l1
              namezh: eca_nfnet_l1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eca_nfnet_l2
              namezh: eca_nfnet_l2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eca_nfnet_l3
              namezh: eca_nfnet_l3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eca_resnet33ts
              namezh: eca_resnet33ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eca_resnext26ts
              namezh: eca_resnext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eca_vovnet39b
              namezh: eca_vovnet39b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnet26t
              namezh: ecaresnet26t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnet50d
              namezh: ecaresnet50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnet50d_pruned
              namezh: ecaresnet50d_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnet50t
              namezh: ecaresnet50t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnet101d
              namezh: ecaresnet101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnet101d_pruned
              namezh: ecaresnet101d_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnet200d
              namezh: ecaresnet200d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnet269d
              namezh: ecaresnet269d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnetlight
              namezh: ecaresnetlight
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnext26t_32x4d
              namezh: ecaresnext26t_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ecaresnext50t_32x4d
              namezh: ecaresnext50t_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: edgenext_base
              namezh: edgenext_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: edgenext_small
              namezh: edgenext_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: edgenext_small_rw
              namezh: edgenext_small_rw
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: edgenext_x_small
              namezh: edgenext_x_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: edgenext_xx_small
              namezh: edgenext_xx_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientformer_l1
              namezh: efficientformer_l1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientformer_l3
              namezh: efficientformer_l3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientformer_l7
              namezh: efficientformer_l7
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientformerv2_l
              namezh: efficientformerv2_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientformerv2_s0
              namezh: efficientformerv2_s0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientformerv2_s1
              namezh: efficientformerv2_s1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientformerv2_s2
              namezh: efficientformerv2_s2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b0
              namezh: efficientnet_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b0_g8_gn
              namezh: efficientnet_b0_g8_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b0_g16_evos
              namezh: efficientnet_b0_g16_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b0_gn
              namezh: efficientnet_b0_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b1
              namezh: efficientnet_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b1_pruned
              namezh: efficientnet_b1_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b2
              namezh: efficientnet_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b2_pruned
              namezh: efficientnet_b2_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b3
              namezh: efficientnet_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b3_g8_gn
              namezh: efficientnet_b3_g8_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b3_gn
              namezh: efficientnet_b3_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b3_pruned
              namezh: efficientnet_b3_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b4
              namezh: efficientnet_b4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b5
              namezh: efficientnet_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b6
              namezh: efficientnet_b6
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b7
              namezh: efficientnet_b7
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_b8
              namezh: efficientnet_b8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_blur_b0
              namezh: efficientnet_blur_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_cc_b0_4e
              namezh: efficientnet_cc_b0_4e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_cc_b0_8e
              namezh: efficientnet_cc_b0_8e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_cc_b1_8e
              namezh: efficientnet_cc_b1_8e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_el
              namezh: efficientnet_el
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_el_pruned
              namezh: efficientnet_el_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_em
              namezh: efficientnet_em
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_es
              namezh: efficientnet_es
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_es_pruned
              namezh: efficientnet_es_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_h_b5
              namezh: efficientnet_h_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_l2
              namezh: efficientnet_l2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_lite0
              namezh: efficientnet_lite0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_lite1
              namezh: efficientnet_lite1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_lite2
              namezh: efficientnet_lite2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_lite3
              namezh: efficientnet_lite3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_lite4
              namezh: efficientnet_lite4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_x_b3
              namezh: efficientnet_x_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnet_x_b5
              namezh: efficientnet_x_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnetv2_l
              namezh: efficientnetv2_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnetv2_m
              namezh: efficientnetv2_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnetv2_rw_m
              namezh: efficientnetv2_rw_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnetv2_rw_s
              namezh: efficientnetv2_rw_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnetv2_rw_t
              namezh: efficientnetv2_rw_t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnetv2_s
              namezh: efficientnetv2_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientnetv2_xl
              namezh: efficientnetv2_xl
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_b0
              namezh: efficientvit_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_b1
              namezh: efficientvit_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_b2
              namezh: efficientvit_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_b3
              namezh: efficientvit_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_l1
              namezh: efficientvit_l1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_l2
              namezh: efficientvit_l2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_l3
              namezh: efficientvit_l3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_m0
              namezh: efficientvit_m0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_m1
              namezh: efficientvit_m1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_m2
              namezh: efficientvit_m2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_m3
              namezh: efficientvit_m3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_m4
              namezh: efficientvit_m4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: efficientvit_m5
              namezh: efficientvit_m5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ese_vovnet19b_dw
              namezh: ese_vovnet19b_dw
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ese_vovnet19b_slim
              namezh: ese_vovnet19b_slim
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ese_vovnet19b_slim_dw
              namezh: ese_vovnet19b_slim_dw
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ese_vovnet39b
              namezh: ese_vovnet39b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ese_vovnet39b_evos
              namezh: ese_vovnet39b_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ese_vovnet57b
              namezh: ese_vovnet57b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ese_vovnet99b
              namezh: ese_vovnet99b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_base_patch14_224
              namezh: eva02_base_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_base_patch14_448
              namezh: eva02_base_patch14_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_base_patch16_clip_224
              namezh: eva02_base_patch16_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_enormous_patch14_clip_224
              namezh: eva02_enormous_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_large_patch14_224
              namezh: eva02_large_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_large_patch14_448
              namezh: eva02_large_patch14_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_large_patch14_clip_224
              namezh: eva02_large_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_large_patch14_clip_336
              namezh: eva02_large_patch14_clip_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_small_patch14_224
              namezh: eva02_small_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_small_patch14_336
              namezh: eva02_small_patch14_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_tiny_patch14_224
              namezh: eva02_tiny_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva02_tiny_patch14_336
              namezh: eva02_tiny_patch14_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva_giant_patch14_224
              namezh: eva_giant_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva_giant_patch14_336
              namezh: eva_giant_patch14_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva_giant_patch14_560
              namezh: eva_giant_patch14_560
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva_giant_patch14_clip_224
              namezh: eva_giant_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva_large_patch14_196
              namezh: eva_large_patch14_196
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: eva_large_patch14_336
              namezh: eva_large_patch14_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastvit_ma36
              namezh: fastvit_ma36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastvit_mci0
              namezh: fastvit_mci0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastvit_mci1
              namezh: fastvit_mci1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastvit_mci2
              namezh: fastvit_mci2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastvit_s12
              namezh: fastvit_s12
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastvit_sa12
              namezh: fastvit_sa12
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastvit_sa24
              namezh: fastvit_sa24
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastvit_sa36
              namezh: fastvit_sa36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastvit_t8
              namezh: fastvit_t8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fastvit_t12
              namezh: fastvit_t12
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fbnetc_100
              namezh: fbnetc_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fbnetv3_b
              namezh: fbnetv3_b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fbnetv3_d
              namezh: fbnetv3_d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: fbnetv3_g
              namezh: fbnetv3_g
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: flexivit_base
              namezh: flexivit_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: flexivit_large
              namezh: flexivit_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: flexivit_small
              namezh: flexivit_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_base_lrf
              namezh: focalnet_base_lrf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_base_srf
              namezh: focalnet_base_srf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_huge_fl3
              namezh: focalnet_huge_fl3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_huge_fl4
              namezh: focalnet_huge_fl4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_large_fl3
              namezh: focalnet_large_fl3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_large_fl4
              namezh: focalnet_large_fl4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_small_lrf
              namezh: focalnet_small_lrf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_small_srf
              namezh: focalnet_small_srf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_tiny_lrf
              namezh: focalnet_tiny_lrf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_tiny_srf
              namezh: focalnet_tiny_srf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_xlarge_fl3
              namezh: focalnet_xlarge_fl3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: focalnet_xlarge_fl4
              namezh: focalnet_xlarge_fl4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gc_efficientnetv2_rw_t
              namezh: gc_efficientnetv2_rw_t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcresnet33ts
              namezh: gcresnet33ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcresnet50t
              namezh: gcresnet50t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcresnext26ts
              namezh: gcresnext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcresnext50ts
              namezh: gcresnext50ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcvit_base
              namezh: gcvit_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcvit_small
              namezh: gcvit_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcvit_tiny
              namezh: gcvit_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcvit_xtiny
              namezh: gcvit_xtiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gcvit_xxtiny
              namezh: gcvit_xxtiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gernet_l
              namezh: gernet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gernet_m
              namezh: gernet_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gernet_s
              namezh: gernet_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ghostnet_050
              namezh: ghostnet_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ghostnet_100
              namezh: ghostnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ghostnet_130
              namezh: ghostnet_130
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ghostnetv2_100
              namezh: ghostnetv2_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ghostnetv2_130
              namezh: ghostnetv2_130
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: ghostnetv2_160
              namezh: ghostnetv2_160
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gmixer_12_224
              namezh: gmixer_12_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gmixer_24_224
              namezh: gmixer_24_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gmlp_b16_224
              namezh: gmlp_b16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gmlp_s16_224
              namezh: gmlp_s16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: gmlp_ti16_224
              namezh: gmlp_ti16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: halo2botnet50ts_256
              namezh: halo2botnet50ts_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: halonet26t
              namezh: halonet26t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: halonet50ts
              namezh: halonet50ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: halonet_h1
              namezh: halonet_h1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: haloregnetz_b
              namezh: haloregnetz_b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hardcorenas_a
              namezh: hardcorenas_a
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hardcorenas_b
              namezh: hardcorenas_b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hardcorenas_c
              namezh: hardcorenas_c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hardcorenas_d
              namezh: hardcorenas_d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hardcorenas_e
              namezh: hardcorenas_e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hardcorenas_f
              namezh: hardcorenas_f
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hgnet_base
              namezh: hgnet_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hgnet_small
              namezh: hgnet_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hgnet_tiny
              namezh: hgnet_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hgnetv2_b0
              namezh: hgnetv2_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hgnetv2_b1
              namezh: hgnetv2_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hgnetv2_b2
              namezh: hgnetv2_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hgnetv2_b3
              namezh: hgnetv2_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hgnetv2_b4
              namezh: hgnetv2_b4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hgnetv2_b5
              namezh: hgnetv2_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hgnetv2_b6
              namezh: hgnetv2_b6
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hiera_base_224
              namezh: hiera_base_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hiera_base_plus_224
              namezh: hiera_base_plus_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hiera_huge_224
              namezh: hiera_huge_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hiera_large_224
              namezh: hiera_large_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hiera_small_224
              namezh: hiera_small_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hiera_tiny_224
              namezh: hiera_tiny_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w18
              namezh: hrnet_w18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w18_small
              namezh: hrnet_w18_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w18_small_v2
              namezh: hrnet_w18_small_v2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w18_ssld
              namezh: hrnet_w18_ssld
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w30
              namezh: hrnet_w30
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w32
              namezh: hrnet_w32
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w40
              namezh: hrnet_w40
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w44
              namezh: hrnet_w44
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w48
              namezh: hrnet_w48
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w48_ssld
              namezh: hrnet_w48_ssld
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: hrnet_w64
              namezh: hrnet_w64
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: inception_next_base
              namezh: inception_next_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: inception_next_small
              namezh: inception_next_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: inception_next_tiny
              namezh: inception_next_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: inception_resnet_v2
              namezh: inception_resnet_v2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: inception_v3
              namezh: inception_v3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: inception_v4
              namezh: inception_v4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lambda_resnet26rpt_256
              namezh: lambda_resnet26rpt_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lambda_resnet26t
              namezh: lambda_resnet26t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lambda_resnet50ts
              namezh: lambda_resnet50ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lamhalobotnet50ts_256
              namezh: lamhalobotnet50ts_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lcnet_035
              namezh: lcnet_035
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lcnet_050
              namezh: lcnet_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lcnet_075
              namezh: lcnet_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lcnet_100
              namezh: lcnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: lcnet_150
              namezh: lcnet_150
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: legacy_senet154
              namezh: legacy_senet154
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: legacy_seresnet18
              namezh: legacy_seresnet18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: legacy_seresnet34
              namezh: legacy_seresnet34
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: legacy_seresnet50
              namezh: legacy_seresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: legacy_seresnet101
              namezh: legacy_seresnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: legacy_seresnet152
              namezh: legacy_seresnet152
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: legacy_seresnext26_32x4d
              namezh: legacy_seresnext26_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: legacy_seresnext50_32x4d
              namezh: legacy_seresnext50_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: legacy_seresnext101_32x4d
              namezh: legacy_seresnext101_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: legacy_xception
              namezh: legacy_xception
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_128
              namezh: levit_128
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_128s
              namezh: levit_128s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_192
              namezh: levit_192
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_256
              namezh: levit_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_256d
              namezh: levit_256d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_384
              namezh: levit_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_384_s8
              namezh: levit_384_s8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_512
              namezh: levit_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_512_s8
              namezh: levit_512_s8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_512d
              namezh: levit_512d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_conv_128
              namezh: levit_conv_128
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_conv_128s
              namezh: levit_conv_128s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_conv_192
              namezh: levit_conv_192
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_conv_256
              namezh: levit_conv_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_conv_256d
              namezh: levit_conv_256d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_conv_384
              namezh: levit_conv_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_conv_384_s8
              namezh: levit_conv_384_s8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_conv_512
              namezh: levit_conv_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_conv_512_s8
              namezh: levit_conv_512_s8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: levit_conv_512d
              namezh: levit_conv_512d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_base_tf_224
              namezh: maxvit_base_tf_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_base_tf_384
              namezh: maxvit_base_tf_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_base_tf_512
              namezh: maxvit_base_tf_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_large_tf_224
              namezh: maxvit_large_tf_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_large_tf_384
              namezh: maxvit_large_tf_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_large_tf_512
              namezh: maxvit_large_tf_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_nano_rw_256
              namezh: maxvit_nano_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_pico_rw_256
              namezh: maxvit_pico_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_rmlp_base_rw_224
              namezh: maxvit_rmlp_base_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_rmlp_base_rw_384
              namezh: maxvit_rmlp_base_rw_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_rmlp_nano_rw_256
              namezh: maxvit_rmlp_nano_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_rmlp_pico_rw_256
              namezh: maxvit_rmlp_pico_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_rmlp_small_rw_224
              namezh: maxvit_rmlp_small_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_rmlp_small_rw_256
              namezh: maxvit_rmlp_small_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_rmlp_tiny_rw_256
              namezh: maxvit_rmlp_tiny_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_small_tf_224
              namezh: maxvit_small_tf_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_small_tf_384
              namezh: maxvit_small_tf_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_small_tf_512
              namezh: maxvit_small_tf_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_tiny_pm_256
              namezh: maxvit_tiny_pm_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_tiny_rw_224
              namezh: maxvit_tiny_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_tiny_rw_256
              namezh: maxvit_tiny_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_tiny_tf_224
              namezh: maxvit_tiny_tf_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_tiny_tf_384
              namezh: maxvit_tiny_tf_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_tiny_tf_512
              namezh: maxvit_tiny_tf_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_xlarge_tf_224
              namezh: maxvit_xlarge_tf_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_xlarge_tf_384
              namezh: maxvit_xlarge_tf_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxvit_xlarge_tf_512
              namezh: maxvit_xlarge_tf_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxxvit_rmlp_nano_rw_256
              namezh: maxxvit_rmlp_nano_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxxvit_rmlp_small_rw_256
              namezh: maxxvit_rmlp_small_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxxvit_rmlp_tiny_rw_256
              namezh: maxxvit_rmlp_tiny_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxxvitv2_nano_rw_256
              namezh: maxxvitv2_nano_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxxvitv2_rmlp_base_rw_224
              namezh: maxxvitv2_rmlp_base_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxxvitv2_rmlp_base_rw_384
              namezh: maxxvitv2_rmlp_base_rw_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: maxxvitv2_rmlp_large_rw_224
              namezh: maxxvitv2_rmlp_large_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixer_b16_224
              namezh: mixer_b16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixer_b32_224
              namezh: mixer_b32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixer_l16_224
              namezh: mixer_l16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixer_l32_224
              namezh: mixer_l32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixer_s16_224
              namezh: mixer_s16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixer_s32_224
              namezh: mixer_s32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixnet_l
              namezh: mixnet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixnet_m
              namezh: mixnet_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixnet_s
              namezh: mixnet_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixnet_xl
              namezh: mixnet_xl
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mixnet_xxl
              namezh: mixnet_xxl
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mnasnet_050
              namezh: mnasnet_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mnasnet_075
              namezh: mnasnet_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mnasnet_100
              namezh: mnasnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mnasnet_140
              namezh: mnasnet_140
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mnasnet_small
              namezh: mnasnet_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet_100
              namezh: mobilenet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet_125
              namezh: mobilenet_125
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet_edgetpu_100
              namezh: mobilenet_edgetpu_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet_edgetpu_v2_l
              namezh: mobilenet_edgetpu_v2_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet_edgetpu_v2_m
              namezh: mobilenet_edgetpu_v2_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet_edgetpu_v2_s
              namezh: mobilenet_edgetpu_v2_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenet_edgetpu_v2_xs
              namezh: mobilenet_edgetpu_v2_xs
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv2_035
              namezh: mobilenetv2_035
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv2_050
              namezh: mobilenetv2_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv2_075
              namezh: mobilenetv2_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv2_100
              namezh: mobilenetv2_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv2_110d
              namezh: mobilenetv2_110d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv2_120d
              namezh: mobilenetv2_120d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv2_140
              namezh: mobilenetv2_140
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv3_large_075
              namezh: mobilenetv3_large_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv3_large_100
              namezh: mobilenetv3_large_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv3_rw
              namezh: mobilenetv3_rw
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv3_small_050
              namezh: mobilenetv3_small_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv3_small_075
              namezh: mobilenetv3_small_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv3_small_100
              namezh: mobilenetv3_small_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv4_conv_aa_medium
              namezh: mobilenetv4_conv_aa_medium
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv4_conv_blur_medium
              namezh: mobilenetv4_conv_blur_medium
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv4_conv_large
              namezh: mobilenetv4_conv_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv4_conv_medium
              namezh: mobilenetv4_conv_medium
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv4_conv_small
              namezh: mobilenetv4_conv_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv4_hybrid_large
              namezh: mobilenetv4_hybrid_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv4_hybrid_large_075
              namezh: mobilenetv4_hybrid_large_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv4_hybrid_medium
              namezh: mobilenetv4_hybrid_medium
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilenetv4_hybrid_medium_075
              namezh: mobilenetv4_hybrid_medium_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobileone_s0
              namezh: mobileone_s0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobileone_s1
              namezh: mobileone_s1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobileone_s2
              namezh: mobileone_s2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobileone_s3
              namezh: mobileone_s3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobileone_s4
              namezh: mobileone_s4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilevit_s
              namezh: mobilevit_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilevit_xs
              namezh: mobilevit_xs
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilevit_xxs
              namezh: mobilevit_xxs
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilevitv2_050
              namezh: mobilevitv2_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilevitv2_075
              namezh: mobilevitv2_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilevitv2_100
              namezh: mobilevitv2_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilevitv2_125
              namezh: mobilevitv2_125
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilevitv2_150
              namezh: mobilevitv2_150
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilevitv2_175
              namezh: mobilevitv2_175
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mobilevitv2_200
              namezh: mobilevitv2_200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mvitv2_base
              namezh: mvitv2_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mvitv2_base_cls
              namezh: mvitv2_base_cls
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mvitv2_huge_cls
              namezh: mvitv2_huge_cls
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mvitv2_large
              namezh: mvitv2_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mvitv2_large_cls
              namezh: mvitv2_large_cls
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mvitv2_small
              namezh: mvitv2_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mvitv2_small_cls
              namezh: mvitv2_small_cls
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: mvitv2_tiny
              namezh: mvitv2_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nasnetalarge
              namezh: nasnetalarge
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nest_base
              namezh: nest_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nest_base_jx
              namezh: nest_base_jx
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nest_small
              namezh: nest_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nest_small_jx
              namezh: nest_small_jx
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nest_tiny
              namezh: nest_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nest_tiny_jx
              namezh: nest_tiny_jx
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nextvit_base
              namezh: nextvit_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nextvit_large
              namezh: nextvit_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nextvit_small
              namezh: nextvit_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_ecaresnet26
              namezh: nf_ecaresnet26
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_ecaresnet50
              namezh: nf_ecaresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_ecaresnet101
              namezh: nf_ecaresnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_regnet_b0
              namezh: nf_regnet_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_regnet_b1
              namezh: nf_regnet_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_regnet_b2
              namezh: nf_regnet_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_regnet_b3
              namezh: nf_regnet_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_regnet_b4
              namezh: nf_regnet_b4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_regnet_b5
              namezh: nf_regnet_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_resnet26
              namezh: nf_resnet26
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_resnet50
              namezh: nf_resnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_resnet101
              namezh: nf_resnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_seresnet26
              namezh: nf_seresnet26
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_seresnet50
              namezh: nf_seresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nf_seresnet101
              namezh: nf_seresnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nfnet_f0
              namezh: nfnet_f0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nfnet_f1
              namezh: nfnet_f1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nfnet_f2
              namezh: nfnet_f2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nfnet_f3
              namezh: nfnet_f3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nfnet_f4
              namezh: nfnet_f4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nfnet_f5
              namezh: nfnet_f5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nfnet_f6
              namezh: nfnet_f6
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nfnet_f7
              namezh: nfnet_f7
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: nfnet_l0
              namezh: nfnet_l0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pit_b_224
              namezh: pit_b_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pit_b_distilled_224
              namezh: pit_b_distilled_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pit_s_224
              namezh: pit_s_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pit_s_distilled_224
              namezh: pit_s_distilled_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pit_ti_224
              namezh: pit_ti_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pit_ti_distilled_224
              namezh: pit_ti_distilled_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pit_xs_224
              namezh: pit_xs_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pit_xs_distilled_224
              namezh: pit_xs_distilled_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pnasnet5large
              namezh: pnasnet5large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: poolformer_m36
              namezh: poolformer_m36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: poolformer_m48
              namezh: poolformer_m48
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: poolformer_s12
              namezh: poolformer_s12
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: poolformer_s24
              namezh: poolformer_s24
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: poolformer_s36
              namezh: poolformer_s36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: poolformerv2_m36
              namezh: poolformerv2_m36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: poolformerv2_m48
              namezh: poolformerv2_m48
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: poolformerv2_s12
              namezh: poolformerv2_s12
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: poolformerv2_s24
              namezh: poolformerv2_s24
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: poolformerv2_s36
              namezh: poolformerv2_s36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pvt_v2_b0
              namezh: pvt_v2_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pvt_v2_b1
              namezh: pvt_v2_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pvt_v2_b2
              namezh: pvt_v2_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pvt_v2_b2_li
              namezh: pvt_v2_b2_li
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pvt_v2_b3
              namezh: pvt_v2_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pvt_v2_b4
              namezh: pvt_v2_b4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: pvt_v2_b5
              namezh: pvt_v2_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetv_040
              namezh: regnetv_040
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetv_064
              namezh: regnetv_064
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_002
              namezh: regnetx_002
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_004
              namezh: regnetx_004
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_004_tv
              namezh: regnetx_004_tv
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_006
              namezh: regnetx_006
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_008
              namezh: regnetx_008
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_016
              namezh: regnetx_016
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_032
              namezh: regnetx_032
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_040
              namezh: regnetx_040
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_064
              namezh: regnetx_064
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_080
              namezh: regnetx_080
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_120
              namezh: regnetx_120
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_160
              namezh: regnetx_160
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetx_320
              namezh: regnetx_320
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_002
              namezh: regnety_002
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_004
              namezh: regnety_004
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_006
              namezh: regnety_006
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_008
              namezh: regnety_008
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_008_tv
              namezh: regnety_008_tv
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_016
              namezh: regnety_016
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_032
              namezh: regnety_032
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_040
              namezh: regnety_040
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_040_sgn
              namezh: regnety_040_sgn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_064
              namezh: regnety_064
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_080
              namezh: regnety_080
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_080_tv
              namezh: regnety_080_tv
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_120
              namezh: regnety_120
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_160
              namezh: regnety_160
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_320
              namezh: regnety_320
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_640
              namezh: regnety_640
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_1280
              namezh: regnety_1280
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnety_2560
              namezh: regnety_2560
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_005
              namezh: regnetz_005
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_040
              namezh: regnetz_040
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_040_h
              namezh: regnetz_040_h
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_b16
              namezh: regnetz_b16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_b16_evos
              namezh: regnetz_b16_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_c16
              namezh: regnetz_c16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_c16_evos
              namezh: regnetz_c16_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_d8
              namezh: regnetz_d8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_d8_evos
              namezh: regnetz_d8_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_d32
              namezh: regnetz_d32
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: regnetz_e8
              namezh: regnetz_e8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repghostnet_050
              namezh: repghostnet_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repghostnet_058
              namezh: repghostnet_058
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repghostnet_080
              namezh: repghostnet_080
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repghostnet_100
              namezh: repghostnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repghostnet_111
              namezh: repghostnet_111
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repghostnet_130
              namezh: repghostnet_130
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repghostnet_150
              namezh: repghostnet_150
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repghostnet_200
              namezh: repghostnet_200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_a0
              namezh: repvgg_a0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_a1
              namezh: repvgg_a1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_a2
              namezh: repvgg_a2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_b0
              namezh: repvgg_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_b1
              namezh: repvgg_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_b1g4
              namezh: repvgg_b1g4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_b2
              namezh: repvgg_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_b2g4
              namezh: repvgg_b2g4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_b3
              namezh: repvgg_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_b3g4
              namezh: repvgg_b3g4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvgg_d2se
              namezh: repvgg_d2se
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvit_m0_9
              namezh: repvit_m0_9
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvit_m1
              namezh: repvit_m1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvit_m1_0
              namezh: repvit_m1_0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvit_m1_1
              namezh: repvit_m1_1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvit_m1_5
              namezh: repvit_m1_5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvit_m2
              namezh: repvit_m2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvit_m2_3
              namezh: repvit_m2_3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: repvit_m3
              namezh: repvit_m3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: res2net50_14w_8s
              namezh: res2net50_14w_8s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: res2net50_26w_4s
              namezh: res2net50_26w_4s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: res2net50_26w_6s
              namezh: res2net50_26w_6s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: res2net50_26w_8s
              namezh: res2net50_26w_8s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: res2net50_48w_2s
              namezh: res2net50_48w_2s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: res2net50d
              namezh: res2net50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: res2net101_26w_4s
              namezh: res2net101_26w_4s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: res2net101d
              namezh: res2net101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: res2next50
              namezh: res2next50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resmlp_12_224
              namezh: resmlp_12_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resmlp_24_224
              namezh: resmlp_24_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resmlp_36_224
              namezh: resmlp_36_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resmlp_big_24_224
              namezh: resmlp_big_24_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest14d
              namezh: resnest14d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest26d
              namezh: resnest26d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest50d
              namezh: resnest50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest50d_1s4x24d
              namezh: resnest50d_1s4x24d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest50d_4s2x40d
              namezh: resnest50d_4s2x40d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest101e
              namezh: resnest101e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest200e
              namezh: resnest200e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnest269e
              namezh: resnest269e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet10t
              namezh: resnet10t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet14t
              namezh: resnet14t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet18
              namezh: resnet18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet18d
              namezh: resnet18d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet26
              namezh: resnet26
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet26d
              namezh: resnet26d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet26t
              namezh: resnet26t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet32ts
              namezh: resnet32ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet33ts
              namezh: resnet33ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet34
              namezh: resnet34
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet34d
              namezh: resnet34d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50
              namezh: resnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50_clip
              namezh: resnet50_clip
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50_clip_gap
              namezh: resnet50_clip_gap
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50_gn
              namezh: resnet50_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50_mlp
              namezh: resnet50_mlp
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50c
              namezh: resnet50c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50d
              namezh: resnet50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50s
              namezh: resnet50s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50t
              namezh: resnet50t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50x4_clip
              namezh: resnet50x4_clip
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50x4_clip_gap
              namezh: resnet50x4_clip_gap
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50x16_clip
              namezh: resnet50x16_clip
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50x16_clip_gap
              namezh: resnet50x16_clip_gap
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50x64_clip
              namezh: resnet50x64_clip
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet50x64_clip_gap
              namezh: resnet50x64_clip_gap
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet51q
              namezh: resnet51q
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet61q
              namezh: resnet61q
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet101
              namezh: resnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet101_clip
              namezh: resnet101_clip
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet101_clip_gap
              namezh: resnet101_clip_gap
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet101c
              namezh: resnet101c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet101d
              namezh: resnet101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet101s
              namezh: resnet101s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet152
              namezh: resnet152
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet152c
              namezh: resnet152c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet152d
              namezh: resnet152d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet152s
              namezh: resnet152s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet200
              namezh: resnet200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnet200d
              namezh: resnet200d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetaa34d
              namezh: resnetaa34d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetaa50
              namezh: resnetaa50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetaa50d
              namezh: resnetaa50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetaa101d
              namezh: resnetaa101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetblur18
              namezh: resnetblur18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetblur50
              namezh: resnetblur50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetblur50d
              namezh: resnetblur50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetblur101d
              namezh: resnetblur101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetrs50
              namezh: resnetrs50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetrs101
              namezh: resnetrs101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetrs152
              namezh: resnetrs152
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetrs200
              namezh: resnetrs200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetrs270
              namezh: resnetrs270
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetrs350
              namezh: resnetrs350
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetrs420
              namezh: resnetrs420
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_50
              namezh: resnetv2_50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_50d
              namezh: resnetv2_50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_50d_evos
              namezh: resnetv2_50d_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_50d_frn
              namezh: resnetv2_50d_frn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_50d_gn
              namezh: resnetv2_50d_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_50t
              namezh: resnetv2_50t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_50x1_bit
              namezh: resnetv2_50x1_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_50x3_bit
              namezh: resnetv2_50x3_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_101
              namezh: resnetv2_101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_101d
              namezh: resnetv2_101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_101x1_bit
              namezh: resnetv2_101x1_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_101x3_bit
              namezh: resnetv2_101x3_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_152
              namezh: resnetv2_152
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_152d
              namezh: resnetv2_152d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_152x2_bit
              namezh: resnetv2_152x2_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnetv2_152x4_bit
              namezh: resnetv2_152x4_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnext26ts
              namezh: resnext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnext50_32x4d
              namezh: resnext50_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnext50d_32x4d
              namezh: resnext50d_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnext101_32x4d
              namezh: resnext101_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnext101_32x8d
              namezh: resnext101_32x8d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnext101_32x16d
              namezh: resnext101_32x16d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnext101_32x32d
              namezh: resnext101_32x32d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: resnext101_64x4d
              namezh: resnext101_64x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rexnet_100
              namezh: rexnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rexnet_130
              namezh: rexnet_130
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rexnet_150
              namezh: rexnet_150
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rexnet_200
              namezh: rexnet_200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rexnet_300
              namezh: rexnet_300
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rexnetr_100
              namezh: rexnetr_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rexnetr_130
              namezh: rexnetr_130
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rexnetr_150
              namezh: rexnetr_150
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rexnetr_200
              namezh: rexnetr_200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: rexnetr_300
              namezh: rexnetr_300
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: samvit_base_patch16
              namezh: samvit_base_patch16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: samvit_base_patch16_224
              namezh: samvit_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: samvit_huge_patch16
              namezh: samvit_huge_patch16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: samvit_large_patch16
              namezh: samvit_large_patch16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sebotnet33ts_256
              namezh: sebotnet33ts_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sedarknet21
              namezh: sedarknet21
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sehalonet33ts
              namezh: sehalonet33ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: selecsls42
              namezh: selecsls42
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: selecsls42b
              namezh: selecsls42b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: selecsls60
              namezh: selecsls60
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: selecsls60b
              namezh: selecsls60b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: selecsls84
              namezh: selecsls84
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: semnasnet_050
              namezh: semnasnet_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: semnasnet_075
              namezh: semnasnet_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: semnasnet_100
              namezh: semnasnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: semnasnet_140
              namezh: semnasnet_140
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: senet154
              namezh: senet154
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sequencer2d_l
              namezh: sequencer2d_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sequencer2d_m
              namezh: sequencer2d_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: sequencer2d_s
              namezh: sequencer2d_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnet18
              namezh: seresnet18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnet33ts
              namezh: seresnet33ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnet34
              namezh: seresnet34
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnet50
              namezh: seresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnet50t
              namezh: seresnet50t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnet101
              namezh: seresnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnet152
              namezh: seresnet152
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnet152d
              namezh: seresnet152d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnet200d
              namezh: seresnet200d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnet269d
              namezh: seresnet269d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnetaa50d
              namezh: seresnetaa50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnext26d_32x4d
              namezh: seresnext26d_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnext26t_32x4d
              namezh: seresnext26t_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnext26ts
              namezh: seresnext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnext50_32x4d
              namezh: seresnext50_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnext101_32x4d
              namezh: seresnext101_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnext101_32x8d
              namezh: seresnext101_32x8d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnext101_64x4d
              namezh: seresnext101_64x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnext101d_32x8d
              namezh: seresnext101d_32x8d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnextaa101d_32x8d
              namezh: seresnextaa101d_32x8d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: seresnextaa201d_32x8d
              namezh: seresnextaa201d_32x8d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: skresnet18
              namezh: skresnet18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: skresnet34
              namezh: skresnet34
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: skresnet50
              namezh: skresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: skresnet50d
              namezh: skresnet50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: skresnext50_32x4d
              namezh: skresnext50_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: spnasnet_100
              namezh: spnasnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin_base_patch4_window7_224
              namezh: swin_base_patch4_window7_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin_base_patch4_window12_384
              namezh: swin_base_patch4_window12_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin_large_patch4_window7_224
              namezh: swin_large_patch4_window7_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin_large_patch4_window12_384
              namezh: swin_large_patch4_window12_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin_s3_base_224
              namezh: swin_s3_base_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin_s3_small_224
              namezh: swin_s3_small_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin_s3_tiny_224
              namezh: swin_s3_tiny_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin_small_patch4_window7_224
              namezh: swin_small_patch4_window7_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swin_tiny_patch4_window7_224
              namezh: swin_tiny_patch4_window7_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_base_window8_256
              namezh: swinv2_base_window8_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_base_window12_192
              namezh: swinv2_base_window12_192
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_base_window12to16_192to256
              namezh: swinv2_base_window12to16_192to256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_base_window12to24_192to384
              namezh: swinv2_base_window12to24_192to384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_base_window16_256
              namezh: swinv2_base_window16_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_base_224
              namezh: swinv2_cr_base_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_base_384
              namezh: swinv2_cr_base_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_base_ns_224
              namezh: swinv2_cr_base_ns_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_giant_224
              namezh: swinv2_cr_giant_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_giant_384
              namezh: swinv2_cr_giant_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_huge_224
              namezh: swinv2_cr_huge_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_huge_384
              namezh: swinv2_cr_huge_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_large_224
              namezh: swinv2_cr_large_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_large_384
              namezh: swinv2_cr_large_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_small_224
              namezh: swinv2_cr_small_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_small_384
              namezh: swinv2_cr_small_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_small_ns_224
              namezh: swinv2_cr_small_ns_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_small_ns_256
              namezh: swinv2_cr_small_ns_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_tiny_224
              namezh: swinv2_cr_tiny_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_tiny_384
              namezh: swinv2_cr_tiny_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_cr_tiny_ns_224
              namezh: swinv2_cr_tiny_ns_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_large_window12_192
              namezh: swinv2_large_window12_192
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_large_window12to16_192to256
              namezh: swinv2_large_window12to16_192to256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_large_window12to24_192to384
              namezh: swinv2_large_window12to24_192to384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_small_window8_256
              namezh: swinv2_small_window8_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_small_window16_256
              namezh: swinv2_small_window16_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_tiny_window8_256
              namezh: swinv2_tiny_window8_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: swinv2_tiny_window16_256
              namezh: swinv2_tiny_window16_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_b0
              namezh: tf_efficientnet_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_b1
              namezh: tf_efficientnet_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_b2
              namezh: tf_efficientnet_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_b3
              namezh: tf_efficientnet_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_b4
              namezh: tf_efficientnet_b4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_b5
              namezh: tf_efficientnet_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_b6
              namezh: tf_efficientnet_b6
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_b7
              namezh: tf_efficientnet_b7
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_b8
              namezh: tf_efficientnet_b8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_cc_b0_4e
              namezh: tf_efficientnet_cc_b0_4e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_cc_b0_8e
              namezh: tf_efficientnet_cc_b0_8e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_cc_b1_8e
              namezh: tf_efficientnet_cc_b1_8e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_el
              namezh: tf_efficientnet_el
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_em
              namezh: tf_efficientnet_em
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_es
              namezh: tf_efficientnet_es
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_l2
              namezh: tf_efficientnet_l2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_lite0
              namezh: tf_efficientnet_lite0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_lite1
              namezh: tf_efficientnet_lite1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_lite2
              namezh: tf_efficientnet_lite2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_lite3
              namezh: tf_efficientnet_lite3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnet_lite4
              namezh: tf_efficientnet_lite4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnetv2_b0
              namezh: tf_efficientnetv2_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnetv2_b1
              namezh: tf_efficientnetv2_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnetv2_b2
              namezh: tf_efficientnetv2_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnetv2_b3
              namezh: tf_efficientnetv2_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnetv2_l
              namezh: tf_efficientnetv2_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnetv2_m
              namezh: tf_efficientnetv2_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnetv2_s
              namezh: tf_efficientnetv2_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_efficientnetv2_xl
              namezh: tf_efficientnetv2_xl
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_mixnet_l
              namezh: tf_mixnet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_mixnet_m
              namezh: tf_mixnet_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_mixnet_s
              namezh: tf_mixnet_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_mobilenetv3_large_075
              namezh: tf_mobilenetv3_large_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_mobilenetv3_large_100
              namezh: tf_mobilenetv3_large_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_mobilenetv3_large_minimal_100
              namezh: tf_mobilenetv3_large_minimal_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_mobilenetv3_small_075
              namezh: tf_mobilenetv3_small_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_mobilenetv3_small_100
              namezh: tf_mobilenetv3_small_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tf_mobilenetv3_small_minimal_100
              namezh: tf_mobilenetv3_small_minimal_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tiny_vit_5m_224
              namezh: tiny_vit_5m_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tiny_vit_11m_224
              namezh: tiny_vit_11m_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tiny_vit_21m_224
              namezh: tiny_vit_21m_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tiny_vit_21m_384
              namezh: tiny_vit_21m_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tiny_vit_21m_512
              namezh: tiny_vit_21m_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tinynet_a
              namezh: tinynet_a
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tinynet_b
              namezh: tinynet_b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tinynet_c
              namezh: tinynet_c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tinynet_d
              namezh: tinynet_d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tinynet_e
              namezh: tinynet_e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tnt_b_patch16_224
              namezh: tnt_b_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tnt_s_patch16_224
              namezh: tnt_s_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tresnet_l
              namezh: tresnet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tresnet_m
              namezh: tresnet_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tresnet_v2_l
              namezh: tresnet_v2_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: tresnet_xl
              namezh: tresnet_xl
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_pcpvt_base
              namezh: twins_pcpvt_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_pcpvt_large
              namezh: twins_pcpvt_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_pcpvt_small
              namezh: twins_pcpvt_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_svt_base
              namezh: twins_svt_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_svt_large
              namezh: twins_svt_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: twins_svt_small
              namezh: twins_svt_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vgg11
              namezh: vgg11
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vgg11_bn
              namezh: vgg11_bn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vgg13
              namezh: vgg13
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vgg13_bn
              namezh: vgg13_bn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vgg16
              namezh: vgg16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vgg16_bn
              namezh: vgg16_bn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vgg19
              namezh: vgg19
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vgg19_bn
              namezh: vgg19_bn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: visformer_small
              namezh: visformer_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: visformer_tiny
              namezh: visformer_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_mci_224
              namezh: vit_base_mci_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch8_224
              namezh: vit_base_patch8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch14_dinov2
              namezh: vit_base_patch14_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch14_reg4_dinov2
              namezh: vit_base_patch14_reg4_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_18x2_224
              namezh: vit_base_patch16_18x2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_224
              namezh: vit_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_224_miil
              namezh: vit_base_patch16_224_miil
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_384
              namezh: vit_base_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_clip_224
              namezh: vit_base_patch16_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_clip_384
              namezh: vit_base_patch16_clip_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_clip_quickgelu_224
              namezh: vit_base_patch16_clip_quickgelu_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_gap_224
              namezh: vit_base_patch16_gap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_plus_240
              namezh: vit_base_patch16_plus_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_reg4_gap_256
              namezh: vit_base_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_rope_reg1_gap_256
              namezh: vit_base_patch16_rope_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_rpn_224
              namezh: vit_base_patch16_rpn_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_siglip_224
              namezh: vit_base_patch16_siglip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_siglip_256
              namezh: vit_base_patch16_siglip_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_siglip_384
              namezh: vit_base_patch16_siglip_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_siglip_512
              namezh: vit_base_patch16_siglip_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_siglip_gap_224
              namezh: vit_base_patch16_siglip_gap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_siglip_gap_256
              namezh: vit_base_patch16_siglip_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_siglip_gap_384
              namezh: vit_base_patch16_siglip_gap_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_siglip_gap_512
              namezh: vit_base_patch16_siglip_gap_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch16_xp_224
              namezh: vit_base_patch16_xp_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch32_224
              namezh: vit_base_patch32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch32_384
              namezh: vit_base_patch32_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch32_clip_224
              namezh: vit_base_patch32_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch32_clip_256
              namezh: vit_base_patch32_clip_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch32_clip_384
              namezh: vit_base_patch32_clip_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch32_clip_448
              namezh: vit_base_patch32_clip_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch32_clip_quickgelu_224
              namezh: vit_base_patch32_clip_quickgelu_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_patch32_plus_256
              namezh: vit_base_patch32_plus_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_r26_s32_224
              namezh: vit_base_r26_s32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_r50_s16_224
              namezh: vit_base_r50_s16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_r50_s16_384
              namezh: vit_base_r50_s16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_resnet26d_224
              namezh: vit_base_resnet26d_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_base_resnet50d_224
              namezh: vit_base_resnet50d_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_betwixt_patch16_gap_256
              namezh: vit_betwixt_patch16_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_betwixt_patch16_reg1_gap_256
              namezh: vit_betwixt_patch16_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_betwixt_patch16_reg4_gap_256
              namezh: vit_betwixt_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_betwixt_patch16_rope_reg4_gap_256
              namezh: vit_betwixt_patch16_rope_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_betwixt_patch32_clip_224
              namezh: vit_betwixt_patch32_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_giant_patch14_224
              namezh: vit_giant_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_giant_patch14_clip_224
              namezh: vit_giant_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_giant_patch14_dinov2
              namezh: vit_giant_patch14_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_giant_patch14_reg4_dinov2
              namezh: vit_giant_patch14_reg4_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_giant_patch16_gap_224
              namezh: vit_giant_patch16_gap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_gigantic_patch14_224
              namezh: vit_gigantic_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_gigantic_patch14_clip_224
              namezh: vit_gigantic_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_huge_patch14_224
              namezh: vit_huge_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_huge_patch14_clip_224
              namezh: vit_huge_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_huge_patch14_clip_336
              namezh: vit_huge_patch14_clip_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_huge_patch14_clip_378
              namezh: vit_huge_patch14_clip_378
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_huge_patch14_clip_quickgelu_224
              namezh: vit_huge_patch14_clip_quickgelu_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_huge_patch14_clip_quickgelu_378
              namezh: vit_huge_patch14_clip_quickgelu_378
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_huge_patch14_gap_224
              namezh: vit_huge_patch14_gap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_huge_patch14_xp_224
              namezh: vit_huge_patch14_xp_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_huge_patch16_gap_448
              namezh: vit_huge_patch16_gap_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch14_224
              namezh: vit_large_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch14_clip_224
              namezh: vit_large_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch14_clip_336
              namezh: vit_large_patch14_clip_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch14_clip_quickgelu_224
              namezh: vit_large_patch14_clip_quickgelu_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch14_clip_quickgelu_336
              namezh: vit_large_patch14_clip_quickgelu_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch14_dinov2
              namezh: vit_large_patch14_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch14_reg4_dinov2
              namezh: vit_large_patch14_reg4_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch14_xp_224
              namezh: vit_large_patch14_xp_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch16_224
              namezh: vit_large_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch16_384
              namezh: vit_large_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch16_siglip_256
              namezh: vit_large_patch16_siglip_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch16_siglip_384
              namezh: vit_large_patch16_siglip_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch16_siglip_gap_256
              namezh: vit_large_patch16_siglip_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch16_siglip_gap_384
              namezh: vit_large_patch16_siglip_gap_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch32_224
              namezh: vit_large_patch32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_patch32_384
              namezh: vit_large_patch32_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_r50_s32_224
              namezh: vit_large_r50_s32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_large_r50_s32_384
              namezh: vit_large_r50_s32_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_little_patch16_reg1_gap_256
              namezh: vit_little_patch16_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_little_patch16_reg4_gap_256
              namezh: vit_little_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_medium_patch16_clip_224
              namezh: vit_medium_patch16_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_medium_patch16_gap_240
              namezh: vit_medium_patch16_gap_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_medium_patch16_gap_256
              namezh: vit_medium_patch16_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_medium_patch16_gap_384
              namezh: vit_medium_patch16_gap_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_medium_patch16_reg1_gap_256
              namezh: vit_medium_patch16_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_medium_patch16_reg4_gap_256
              namezh: vit_medium_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_medium_patch16_rope_reg1_gap_256
              namezh: vit_medium_patch16_rope_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_medium_patch32_clip_224
              namezh: vit_medium_patch32_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_mediumd_patch16_reg4_gap_256
              namezh: vit_mediumd_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_mediumd_patch16_rope_reg1_gap_256
              namezh: vit_mediumd_patch16_rope_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_pwee_patch16_reg1_gap_256
              namezh: vit_pwee_patch16_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_base_patch16_224
              namezh: vit_relpos_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_base_patch16_cls_224
              namezh: vit_relpos_base_patch16_cls_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_base_patch16_clsgap_224
              namezh: vit_relpos_base_patch16_clsgap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_base_patch16_plus_240
              namezh: vit_relpos_base_patch16_plus_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_base_patch16_rpn_224
              namezh: vit_relpos_base_patch16_rpn_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_base_patch32_plus_rpn_256
              namezh: vit_relpos_base_patch32_plus_rpn_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_medium_patch16_224
              namezh: vit_relpos_medium_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_medium_patch16_cls_224
              namezh: vit_relpos_medium_patch16_cls_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_medium_patch16_rpn_224
              namezh: vit_relpos_medium_patch16_rpn_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_small_patch16_224
              namezh: vit_relpos_small_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_relpos_small_patch16_rpn_224
              namezh: vit_relpos_small_patch16_rpn_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_patch8_224
              namezh: vit_small_patch8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_patch14_dinov2
              namezh: vit_small_patch14_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_patch14_reg4_dinov2
              namezh: vit_small_patch14_reg4_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_patch16_18x2_224
              namezh: vit_small_patch16_18x2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_patch16_36x1_224
              namezh: vit_small_patch16_36x1_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_patch16_224
              namezh: vit_small_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_patch16_384
              namezh: vit_small_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_patch32_224
              namezh: vit_small_patch32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_patch32_384
              namezh: vit_small_patch32_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_r26_s32_224
              namezh: vit_small_r26_s32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_r26_s32_384
              namezh: vit_small_r26_s32_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_resnet26d_224
              namezh: vit_small_resnet26d_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_small_resnet50d_s16_224
              namezh: vit_small_resnet50d_s16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_so150m_patch16_reg4_gap_256
              namezh: vit_so150m_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_so150m_patch16_reg4_map_256
              namezh: vit_so150m_patch16_reg4_map_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_so400m_patch14_siglip_224
              namezh: vit_so400m_patch14_siglip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_so400m_patch14_siglip_384
              namezh: vit_so400m_patch14_siglip_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_so400m_patch14_siglip_gap_224
              namezh: vit_so400m_patch14_siglip_gap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_so400m_patch14_siglip_gap_384
              namezh: vit_so400m_patch14_siglip_gap_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_so400m_patch14_siglip_gap_448
              namezh: vit_so400m_patch14_siglip_gap_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_so400m_patch14_siglip_gap_896
              namezh: vit_so400m_patch14_siglip_gap_896
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_srelpos_medium_patch16_224
              namezh: vit_srelpos_medium_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_srelpos_small_patch16_224
              namezh: vit_srelpos_small_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_tiny_patch16_224
              namezh: vit_tiny_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_tiny_patch16_384
              namezh: vit_tiny_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_tiny_r_s16_p8_224
              namezh: vit_tiny_r_s16_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_tiny_r_s16_p8_384
              namezh: vit_tiny_r_s16_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_wee_patch16_reg1_gap_256
              namezh: vit_wee_patch16_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vit_xsmall_patch16_clip_224
              namezh: vit_xsmall_patch16_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_base_224
              namezh: vitamin_base_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_large2_224
              namezh: vitamin_large2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_large2_256
              namezh: vitamin_large2_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_large2_336
              namezh: vitamin_large2_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_large2_384
              namezh: vitamin_large2_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_large_224
              namezh: vitamin_large_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_large_256
              namezh: vitamin_large_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_large_336
              namezh: vitamin_large_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_large_384
              namezh: vitamin_large_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_small_224
              namezh: vitamin_small_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_xlarge_256
              namezh: vitamin_xlarge_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_xlarge_336
              namezh: vitamin_xlarge_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vitamin_xlarge_384
              namezh: vitamin_xlarge_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d1_224
              namezh: volo_d1_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d1_384
              namezh: volo_d1_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d2_224
              namezh: volo_d2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d2_384
              namezh: volo_d2_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d3_224
              namezh: volo_d3_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d3_448
              namezh: volo_d3_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d4_224
              namezh: volo_d4_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d4_448
              namezh: volo_d4_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d5_224
              namezh: volo_d5_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d5_448
              namezh: volo_d5_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: volo_d5_512
              namezh: volo_d5_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vovnet39a
              namezh: vovnet39a
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: vovnet57a
              namezh: vovnet57a
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: wide_resnet50_2
              namezh: wide_resnet50_2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: wide_resnet101_2
              namezh: wide_resnet101_2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xception41
              namezh: xception41
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xception41p
              namezh: xception41p
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xception65
              namezh: xception65
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xception65p
              namezh: xception65p
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xception71
              namezh: xception71
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_large_24_p8_224
              namezh: xcit_large_24_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_large_24_p8_384
              namezh: xcit_large_24_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_large_24_p16_224
              namezh: xcit_large_24_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_large_24_p16_384
              namezh: xcit_large_24_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_medium_24_p8_224
              namezh: xcit_medium_24_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_medium_24_p8_384
              namezh: xcit_medium_24_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_medium_24_p16_224
              namezh: xcit_medium_24_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_medium_24_p16_384
              namezh: xcit_medium_24_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_nano_12_p8_224
              namezh: xcit_nano_12_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_nano_12_p8_384
              namezh: xcit_nano_12_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_nano_12_p16_224
              namezh: xcit_nano_12_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_nano_12_p16_384
              namezh: xcit_nano_12_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_small_12_p8_224
              namezh: xcit_small_12_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_small_12_p8_384
              namezh: xcit_small_12_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_small_12_p16_224
              namezh: xcit_small_12_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_small_12_p16_384
              namezh: xcit_small_12_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_small_24_p8_224
              namezh: xcit_small_24_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_small_24_p8_384
              namezh: xcit_small_24_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_small_24_p16_224
              namezh: xcit_small_24_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_small_24_p16_384
              namezh: xcit_small_24_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_tiny_12_p8_224
              namezh: xcit_tiny_12_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_tiny_12_p8_384
              namezh: xcit_tiny_12_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_tiny_12_p16_224
              namezh: xcit_tiny_12_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_tiny_12_p16_384
              namezh: xcit_tiny_12_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_tiny_24_p8_224
              namezh: xcit_tiny_24_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_tiny_24_p8_384
              namezh: xcit_tiny_24_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_tiny_24_p16_224
              namezh: xcit_tiny_24_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
            - nameen: xcit_tiny_24_p16_38
              namezh: xcit_tiny_24_p16_38
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9999
              datasetA: 99%
              datasetB: 99%
              ranking: '999'
            
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

