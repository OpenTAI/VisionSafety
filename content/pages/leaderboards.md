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
              download: 888
              datasetA: '39.91'
              datasetB: '27.67'
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
              download: 4932
              datasetA: '39.78'
              datasetB: '35.56'
              ranking: '2'
            - nameen: segformer_mit-b4_512x512_160k_ade20k
              namezh: segformer_mit-b4_512x512_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/Segformer?athId=94937aa281ea263f6484a359dfa3ec4b&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/Segformer?athId=94937aa281ea263f6484a359dfa3ec4b&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 5110
              datasetA: '39.75'
              datasetB: '48.01'
              ranking: '3'
            - nameen: setr_mla_512x512_160k_b16_ade20k
              namezh: setr_mla_512x512_160k_b16_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/SETR?athId=a0088b8a1527ee3e20b6241c2b66b496&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/SETR?athId=a0088b8a1527ee3e20b6241c2b66b496&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 5806
              datasetA: '39.62'
              datasetB: '40.79'
              ranking: '4'
            - nameen: twins_svt-b_fpn_fpnhead_8x4_512x512_80k_ade20k
              namezh: twins_svt-b_fpn_fpnhead_8x4_512x512_80k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/FPN?athId=7f617fa591d3dfd31fb2a9a7cc0ae8ba&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/FPN?athId=7f617fa591d3dfd31fb2a9a7cc0ae8ba&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 7033
              datasetA: '39.42'
              datasetB: '22.80'
              ranking: '5'
            - nameen: dpt_vit-b16_512x512_160k_ade20k
              namezh: dpt_vit-b16_512x512_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DPT?athId=b2c699d0fddf59a4e952cecea08b1b8b&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DPT?athId=b2c699d0fddf59a4e952cecea08b1b8b&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 5535
              datasetA: '39.40'
              datasetB: '24.25'
              ranking: '6'
            - nameen: deeplabv3_r101-d8_512x512_160k_ade20k
              namezh: deeplabv3_r101-d8_512x512_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DeepLabV3?athId=6f315fcddecd0407b37cae1346078876&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DeepLabV3?athId=6f315fcddecd0407b37cae1346078876&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 5084
              datasetA: '39.40'
              datasetB: '48.16'
              ranking: '7'
            - nameen: fcn_hr48_512x512_160k_ade20k
              namezh: fcn_hr48_512x512_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/FCN?athId=9cb4ee8cc5fee1e37d4418259aa76d81&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/FCN?athId=9cb4ee8cc5fee1e37d4418259aa76d81&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 1103
              datasetA: '39.39'
              datasetB: '35.78'
              ranking: '8'
            - nameen: dnl_r50-d8_4xb4-160k_ade20k-512x512
              namezh: dnl_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DNLNet?athId=e7a94769be0d3a1b41a6e067db8e0f5d&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/DNLNet?athId=e7a94769be0d3a1b41a6e067db8e0f5d&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 1957
              datasetA: '39.29'
              datasetB: '44.40'
              ranking: '9'
            - nameen: segmenter_vit-b_mask_8x1_512x512_160k_ade20k
              namezh: segmenter_vit-b_mask_8x1_512x512_160k_ade20k
              paper:
                text: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/Segmenter?athId=0a8f2e1dccdce40c26a35ebe5b074f36&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
                link: >-
                  https://platform.openmmlab.com/modelzoo/mmsegmentation/Segmenter?athId=0a8f2e1dccdce40c26a35ebe5b074f36&repo=mmsegmentation&repoNameId=aa8108d30b48600d2dd34b4b6ef93112&task=Semantic%20Segmentation
              download: 9498
              datasetA: '39.27'
              datasetB: '54.76'
              ranking: '10'
              
            - nameen: ann_r50-d8_4xb4-80k_ade20k-512x512
              namezh: ann_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9536
              datasetA: '39.12'
              datasetB: '43.18'
              ranking: '11'
            
            - nameen: ann_r101-d8_4xb4-80k_ade20k-512x512
              namezh: ann_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1759
              datasetA: '39.11'
              datasetB: '53.14'
              ranking: '12'
            
            - nameen: ann_r50-d8_4xb4-160k_ade20k-512x512
              namezh: ann_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6248
              datasetA: '39.10'
              datasetB: '41.13'
              ranking: '13'
            
            - nameen: ann_r101-d8_4xb4-160k_ade20k-512x512
              namezh: ann_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4232
              datasetA: '39.10'
              datasetB: '44.86'
              ranking: '14'
            
            - nameen: apcnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: apcnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4648
              datasetA: '38.95'
              datasetB: '57.64'
              ranking: '15'
            
            - nameen: apcnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: apcnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5377
              datasetA: '38.93'
              datasetB: '44.38'
              ranking: '16'
            
            - nameen: apcnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: apcnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8915
              datasetA: '38.90'
              datasetB: '26.49'
              ranking: '17'
            
            - nameen: apcnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: apcnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9670
              datasetA: '38.89'
              datasetB: '57.73'
              ranking: '18'
            
            - nameen: beit-base_upernet_8xb2-160k_ade20k-640x640
              namezh: beit-base_upernet_8xb2-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8432
              datasetA: '38.85'
              datasetB: '21.74'
              ranking: '19'
            
            - nameen: beit-large_upernet_8xb1-amp-160k_ade20k-640x640
              namezh: beit-large_upernet_8xb1-amp-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2378
              datasetA: '38.76'
              datasetB: '43.92'
              ranking: '20'
            
            - nameen: ccnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: ccnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 484
              datasetA: '38.74'
              datasetB: '21.96'
              ranking: '21'
            
            - nameen: ccnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: ccnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3850
              datasetA: '38.72'
              datasetB: '56.55'
              ranking: '22'
            
            - nameen: ccnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: ccnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5157
              datasetA: '38.60'
              datasetB: '34.95'
              ranking: '23'
            
            - nameen: ccnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: ccnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9032
              datasetA: '38.57'
              datasetB: '21.65'
              ranking: '24'
            
            - nameen: convnext-tiny_upernet_8xb2-amp-160k_ade20k-512x512
              namezh: convnext-tiny_upernet_8xb2-amp-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6839
              datasetA: '38.51'
              datasetB: '32.47'
              ranking: '25'
            
            - nameen: convnext-small_upernet_8xb2-amp-160k_ade20k-512x512
              namezh: convnext-small_upernet_8xb2-amp-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2489
              datasetA: '38.31'
              datasetB: '45.62'
              ranking: '26'
            
            - nameen: convnext-base_upernet_8xb2-amp-160k_ade20k-512x512
              namezh: convnext-base_upernet_8xb2-amp-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2055
              datasetA: '38.21'
              datasetB: '39.88'
              ranking: '27'
            
            - nameen: convnext-base_upernet_8xb2-amp-160k_ade20k-640x640
              namezh: convnext-base_upernet_8xb2-amp-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9670
              datasetA: '38.16'
              datasetB: '36.87'
              ranking: '28'
            
            - nameen: convnext-large_upernet_8xb2-amp-160k_ade20k-640x640
              namezh: convnext-large_upernet_8xb2-amp-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1754
              datasetA: '38.06'
              datasetB: '58.22'
              ranking: '29'
            
            - nameen: convnext-xlarge_upernet_8xb2-amp-160k_ade20k-640x640
              namezh: convnext-xlarge_upernet_8xb2-amp-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5371
              datasetA: '37.89'
              datasetB: '33.68'
              ranking: '30'
            
            - nameen: danet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: danet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 448
              datasetA: '37.83'
              datasetB: '33.08'
              ranking: '31'
            
            - nameen: danet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: danet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2622
              datasetA: '37.78'
              datasetB: '32.74'
              ranking: '32'
            
            - nameen: danet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: danet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1840
              datasetA: '37.62'
              datasetB: '29.84'
              ranking: '33'
            
            - nameen: danet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: danet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1116
              datasetA: '37.43'
              datasetB: '26.24'
              ranking: '34'
            
            - nameen: deeplabv3_r50-d8_4xb4-80k_ade20k-512x512
              namezh: deeplabv3_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6265
              datasetA: '37.38'
              datasetB: '30.64'
              ranking: '35'
            
            - nameen: deeplabv3_r101-d8_4xb4-80k_ade20k-512x512
              namezh: deeplabv3_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1451
              datasetA: '37.13'
              datasetB: '32.04'
              ranking: '36'
            
            - nameen: deeplabv3_r50-d8_4xb4-160k_ade20k-512x512
              namezh: deeplabv3_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1367
              datasetA: '37.06'
              datasetB: '53.90'
              ranking: '37'
            
            - nameen: deeplabv3_r101-d8_4xb4-160k_ade20k-512x512
              namezh: deeplabv3_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5705
              datasetA: '36.96'
              datasetB: '58.75'
              ranking: '38'
            
            - nameen: deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512
              namezh: deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2942
              datasetA: '36.81'
              datasetB: '22.38'
              ranking: '39'
            
            - nameen: deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512
              namezh: deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1670
              datasetA: '36.52'
              datasetB: '55.86'
              ranking: '40'
            
            - nameen: deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512
              namezh: deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1194
              datasetA: '36.43'
              datasetB: '56.89'
              ranking: '41'
            
            - nameen: deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512
              namezh: deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3042
              datasetA: '36.40'
              datasetB: '25.02'
              ranking: '42'
            
            - nameen: dmnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: dmnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4130
              datasetA: '36.23'
              datasetB: '34.50'
              ranking: '43'
            
            - nameen: dmnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: dmnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9327
              datasetA: '36.17'
              datasetB: '46.88'
              ranking: '44'
            
            - nameen: dmnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: dmnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4252
              datasetA: '36.15'
              datasetB: '34.71'
              ranking: '45'
            
            - nameen: dmnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: dmnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7667
              datasetA: '35.99'
              datasetB: '48.05'
              ranking: '46'
            
            - nameen: dnl_r50-d8_4xb4-80k_ade20k-512x512
              namezh: dnl_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9786
              datasetA: '35.93'
              datasetB: '32.24'
              ranking: '47'
            
            - nameen: dnl_r101-d8_4xb4-80k_ade20k-512x512
              namezh: dnl_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9293
              datasetA: '35.78'
              datasetB: '44.22'
              ranking: '48'
            
            - nameen: dnl_r50-d8_4xb4-160k_ade20k-512x512
              namezh: dnl_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5180
              datasetA: '35.74'
              datasetB: '32.93'
              ranking: '49'
            
            - nameen: dnl_r101-d8_4xb4-160k_ade20k-512x512
              namezh: dnl_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1010
              datasetA: '35.74'
              datasetB: '56.91'
              ranking: '50'
            
            - nameen: dpt_vit-b16_8xb2-160k_ade20k-512x512
              namezh: dpt_vit-b16_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8271
              datasetA: '35.69'
              datasetB: '22.07'
              ranking: '51'
            
            - nameen: encnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: encnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5956
              datasetA: '35.65'
              datasetB: '44.95'
              ranking: '52'
            
            - nameen: encnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: encnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9795
              datasetA: '35.33'
              datasetB: '50.23'
              ranking: '53'
            
            - nameen: encnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: encnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7796
              datasetA: '35.15'
              datasetB: '53.70'
              ranking: '54'
            
            - nameen: encnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: encnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3533
              datasetA: '34.72'
              datasetB: '22.20'
              ranking: '55'
            
            - nameen: fastfcn_r50-d32_jpu_aspp_4xb4-80k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_aspp_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3470
              datasetA: '34.54'
              datasetB: '39.82'
              ranking: '56'
            
            - nameen: fastfcn_r50-d32_jpu_aspp_4xb4-160k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_aspp_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1591
              datasetA: '34.53'
              datasetB: '50.05'
              ranking: '57'
            
            - nameen: fastfcn_r50-d32_jpu_psp_4xb4-80k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_psp_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8865
              datasetA: '34.37'
              datasetB: '36.93'
              ranking: '58'
            
            - nameen: fastfcn_r50-d32_jpu_psp_4xb4-160k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_psp_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7112
              datasetA: '34.25'
              datasetB: '46.80'
              ranking: '59'
            
            - nameen: fastfcn_r50-d32_jpu_enc_4xb4-80k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_enc_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3669
              datasetA: '34.11'
              datasetB: '43.87'
              ranking: '60'
            
            - nameen: fastfcn_r50-d32_jpu_enc_4xb4-160k_ade20k-512x512
              namezh: fastfcn_r50-d32_jpu_enc_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6932
              datasetA: '33.96'
              datasetB: '44.91'
              ranking: '61'
            
            - nameen: fcn_r50-d8_4xb4-80k_ade20k-512x512
              namezh: fcn_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6425
              datasetA: '33.92'
              datasetB: '24.13'
              ranking: '62'
            
            - nameen: fcn_r101-d8_4xb4-80k_ade20k-512x512
              namezh: fcn_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9436
              datasetA: '33.81'
              datasetB: '40.36'
              ranking: '63'
            
            - nameen: fcn_r50-d8_4xb4-160k_ade20k-512x512
              namezh: fcn_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2624
              datasetA: '33.58'
              datasetB: '49.62'
              ranking: '64'
            
            - nameen: fcn_r101-d8_4xb4-160k_ade20k-512x512
              namezh: fcn_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3665
              datasetA: '33.58'
              datasetB: '45.01'
              ranking: '65'
            
            - nameen: gcnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: gcnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5500
              datasetA: '33.53'
              datasetB: '48.92'
              ranking: '66'
            
            - nameen: gcnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: gcnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7080
              datasetA: '33.04'
              datasetB: '56.64'
              ranking: '67'
            
            - nameen: gcnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: gcnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1845
              datasetA: '32.88'
              datasetB: '45.55'
              ranking: '68'
            
            - nameen: gcnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: gcnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5528
              datasetA: '32.81'
              datasetB: '34.88'
              ranking: '69'
            
            - nameen: fcn_hr18s_4xb4-80k_ade20k-512x512
              namezh: fcn_hr18s_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9079
              datasetA: '32.75'
              datasetB: '46.89'
              ranking: '70'
            
            - nameen: fcn_hr18_4xb4-80k_ade20k-512x512
              namezh: fcn_hr18_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9100
              datasetA: '32.72'
              datasetB: '21.55'
              ranking: '71'
            
            - nameen: fcn_hr48_4xb4-80k_ade20k-512x512
              namezh: fcn_hr48_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9815
              datasetA: '32.58'
              datasetB: '28.38'
              ranking: '72'
            
            - nameen: fcn_hr18s_4xb4-160k_ade20k-512x512
              namezh: fcn_hr18s_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 226
              datasetA: '32.56'
              datasetB: '39.01'
              ranking: '73'
            
            - nameen: fcn_hr18_4xb4-160k_ade20k-512x512
              namezh: fcn_hr18_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3208
              datasetA: '32.54'
              datasetB: '24.18'
              ranking: '74'
            
            - nameen: fcn_hr48_4xb4-160k_ade20k-512x512
              namezh: fcn_hr48_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7700
              datasetA: '32.35'
              datasetB: '27.77'
              ranking: '75'
            
            - nameen: isanet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: isanet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 313
              datasetA: '32.27'
              datasetB: '48.99'
              ranking: '76'
            
            - nameen: isanet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: isanet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1322
              datasetA: '32.15'
              datasetB: '28.62'
              ranking: '77'
            
            - nameen: isanet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: isanet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5798
              datasetA: '32.06'
              datasetB: '25.30'
              ranking: '78'
            
            - nameen: isanet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: isanet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1343
              datasetA: '32.02'
              datasetB: '27.26'
              ranking: '79'
            
            - nameen: knet-s3_r50-d8_fcn_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_r50-d8_fcn_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3523
              datasetA: '32.01'
              datasetB: '51.19'
              ranking: '80'
            
            - nameen: knet-s3_r50-d8_pspnet_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_r50-d8_pspnet_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7819
              datasetA: '31.99'
              datasetB: '26.08'
              ranking: '81'
            
            - nameen: knet-s3_r50-d8_deeplabv3_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_r50-d8_deeplabv3_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1788
              datasetA: '31.92'
              datasetB: '40.56'
              ranking: '82'
            
            - nameen: knet-s3_r50-d8_upernet_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_r50-d8_upernet_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1117
              datasetA: '31.91'
              datasetB: '24.83'
              ranking: '83'
            
            - nameen: knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_swin-t_upernet_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3303
              datasetA: '31.66'
              datasetB: '32.96'
              ranking: '84'
            
            - nameen: knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512
              namezh: knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9478
              datasetA: '31.25'
              datasetB: '48.30'
              ranking: '85'
            
            - nameen: knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-640x640
              namezh: knet-s3_swin-l_upernet_8xb2-adamw-80k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7142
              datasetA: '31.15'
              datasetB: '54.76'
              ranking: '86'
            
            - nameen: mae-base_upernet_8xb2-amp-160k_ade20k-512x512
              namezh: mae-base_upernet_8xb2-amp-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5791
              datasetA: '31.15'
              datasetB: '58.91'
              ranking: '87'
            
            - nameen: mask2former_r50_8xb2-160k_ade20k-512x512
              namezh: mask2former_r50_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2203
              datasetA: '31.13'
              datasetB: '56.20'
              ranking: '88'
            
            - nameen: mask2former_r101_8xb2-160k_ade20k-512x512
              namezh: mask2former_r101_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4989
              datasetA: '30.95'
              datasetB: '32.84'
              ranking: '89'
            
            - nameen: mask2former_swin-t_8xb2-160k_ade20k-512x512
              namezh: mask2former_swin-t_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5906
              datasetA: '30.92'
              datasetB: '50.36'
              ranking: '90'
            
            - nameen: mask2former_swin-s_8xb2-160k_ade20k-512x512
              namezh: mask2former_swin-s_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2738
              datasetA: '30.79'
              datasetB: '32.26'
              ranking: '91'
            
            - nameen: mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640
              namezh: mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6779
              datasetA: '30.75'
              datasetB: '59.78'
              ranking: '92'
            
            - nameen: mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640
              namezh: mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2904
              datasetA: '30.59'
              datasetB: '34.29'
              ranking: '93'
            
            - nameen: mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640
              namezh: mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8167
              datasetA: '30.42'
              datasetB: '32.11'
              ranking: '94'
            
            - nameen: maskformer_r50-d32_8xb2-160k_ade20k-512x512
              namezh: maskformer_r50-d32_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3840
              datasetA: '30.40'
              datasetB: '33.50'
              ranking: '95'
            
            - nameen: maskformer_r101-d32_8xb2-160k_ade20k-512x512
              namezh: maskformer_r101-d32_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4812
              datasetA: '30.30'
              datasetB: '40.82'
              ranking: '96'
            
            - nameen: maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512
              namezh: maskformer_swin-t_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7426
              datasetA: '30.18'
              datasetB: '22.26'
              ranking: '97'
            
            - nameen: maskformer_swin-s_upernet_8xb2-160k_ade20k-512x512
              namezh: maskformer_swin-s_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 100
              datasetA: '30.04'
              datasetB: '23.20'
              ranking: '98'
            
            - nameen: mobilenet-v2-d8_fcn_4xb4-160k_ade20k-512x512
              namezh: mobilenet-v2-d8_fcn_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8117
              datasetA: '29.99'
              datasetB: '31.15'
              ranking: '99'
            
            - nameen: mobilenet-v2-d8_pspnet_4xb4-160k_ade20k-512x512
              namezh: mobilenet-v2-d8_pspnet_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7332
              datasetA: '29.97'
              datasetB: '26.92'
              ranking: '100'
            
            - nameen: mobilenet-v2-d8_deeplabv3_4xb4-160k_ade20k-512x512
              namezh: mobilenet-v2-d8_deeplabv3_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3987
              datasetA: '29.95'
              datasetB: '29.09'
              ranking: '101'
            
            - nameen: mobilenet-v2-d8_deeplabv3plus_4xb4-160k_ade20k-512x512
              namezh: mobilenet-v2-d8_deeplabv3plus_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7958
              datasetA: '29.88'
              datasetB: '47.61'
              ranking: '102'
            
            - nameen: nonlocal_r50-d8_4xb4-80k_ade20k-512x512
              namezh: nonlocal_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2597
              datasetA: '29.86'
              datasetB: '37.72'
              ranking: '103'
            
            - nameen: nonlocal_r101-d8_4xb4-80k_ade20k-512x512
              namezh: nonlocal_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5464
              datasetA: '29.72'
              datasetB: '47.36'
              ranking: '104'
            
            - nameen: nonlocal_r50-d8_4xb4-160k_ade20k-512x512
              namezh: nonlocal_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4260
              datasetA: '29.62'
              datasetB: '57.50'
              ranking: '105'
            
            - nameen: nonlocal_r101-d8_4xb4-160k_ade20k-512x512
              namezh: nonlocal_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4264
              datasetA: '29.53'
              datasetB: '56.91'
              ranking: '106'
            
            - nameen: ocrnet_hr18s_4xb4-80k_ade20k-512x512
              namezh: ocrnet_hr18s_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8712
              datasetA: '29.51'
              datasetB: '31.71'
              ranking: '107'
            
            - nameen: ocrnet_hr18_4xb4-80k_ade20k-512x512
              namezh: ocrnet_hr18_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3046
              datasetA: '29.50'
              datasetB: '38.51'
              ranking: '108'
            
            - nameen: ocrnet_hr48_4xb4-80k_ade20k-512x512
              namezh: ocrnet_hr48_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7899
              datasetA: '29.41'
              datasetB: '39.37'
              ranking: '109'
            
            - nameen: ocrnet_hr18s_4xb4-80k_ade20k-512x512
              namezh: ocrnet_hr18s_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6857
              datasetA: '29.37'
              datasetB: '22.70'
              ranking: '110'
            
            - nameen: ocrnet_hr18_4xb4-80k_ade20k-512x512
              namezh: ocrnet_hr18_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6008
              datasetA: '29.35'
              datasetB: '58.52'
              ranking: '111'
            
            - nameen: ocrnet_hr48_4xb4-160k_ade20k-512x512
              namezh: ocrnet_hr48_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8255
              datasetA: '29.31'
              datasetB: '37.20'
              ranking: '112'
            
            - nameen: pointrend_r50_4xb4-160k_ade20k-512x512
              namezh: pointrend_r50_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8768
              datasetA: '29.25'
              datasetB: '20.29'
              ranking: '113'
            
            - nameen: pointrend_r101_4xb4-160k_ade20k-512x512
              namezh: pointrend_r101_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3930
              datasetA: '29.16'
              datasetB: '42.36'
              ranking: '114'
            
            - nameen: fpn_poolformer_s12_8xb4-40k_ade20k-512x512
              namezh: fpn_poolformer_s12_8xb4-40k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3331
              datasetA: '28.91'
              datasetB: '37.94'
              ranking: '115'
            
            - nameen: fpn_poolformer_s24_8xb4-40k_ade20k-512x512
              namezh: fpn_poolformer_s24_8xb4-40k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 200
              datasetA: '28.90'
              datasetB: '25.23'
              ranking: '116'
            
            - nameen: fpn_poolformer_s36_8xb4-40k_ade20k-512x512
              namezh: fpn_poolformer_s36_8xb4-40k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3770
              datasetA: '28.54'
              datasetB: '22.69'
              ranking: '117'
            
            - nameen: fpn_poolformer_m36_8xb4-40k_ade20k-512x512
              namezh: fpn_poolformer_m36_8xb4-40k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6086
              datasetA: '28.46'
              datasetB: '59.84'
              ranking: '118'
            
            - nameen: fpn_poolformer_m48_8xb4-40k_ade20k-512x512
              namezh: fpn_poolformer_m48_8xb4-40k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7379
              datasetA: '28.34'
              datasetB: '30.67'
              ranking: '119'
            
            - nameen: psanet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: psanet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9339
              datasetA: '28.18'
              datasetB: '42.42'
              ranking: '120'
            
            - nameen: psanet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: psanet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9930
              datasetA: '27.96'
              datasetB: '41.57'
              ranking: '121'
            
            - nameen: psanet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: psanet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2349
              datasetA: '27.95'
              datasetB: '36.87'
              ranking: '122'
            
            - nameen: psanet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: psanet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9530
              datasetA: '27.60'
              datasetB: '28.22'
              ranking: '123'
            
            - nameen: pspnet_r50-d8_4xb4-80k_ade20k-512x512
              namezh: pspnet_r50-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3275
              datasetA: '27.59'
              datasetB: '29.99'
              ranking: '124'
            
            - nameen: pspnet_r101-d8_4xb4-80k_ade20k-512x512
              namezh: pspnet_r101-d8_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7846
              datasetA: '27.44'
              datasetB: '23.20'
              ranking: '125'
            
            - nameen: pspnet_r50-d8_4xb4-160k_ade20k-512x512
              namezh: pspnet_r50-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6204
              datasetA: '27.36'
              datasetB: '31.27'
              ranking: '126'
            
            - nameen: pspnet_r101-d8_4xb4-160k_ade20k-512x512
              namezh: pspnet_r101-d8_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4471
              datasetA: '27.31'
              datasetB: '48.92'
              ranking: '127'
            
            - nameen: resnest_s101-d8_fcn_4xb4-160k_ade20k-512x512
              namezh: resnest_s101-d8_fcn_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8326
              datasetA: '27.21'
              datasetB: '55.38'
              ranking: '128'
            
            - nameen: resnest_s101-d8_pspnet_4xb4-160k_ade20k-512x512
              namezh: resnest_s101-d8_pspnet_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4284
              datasetA: '27.05'
              datasetB: '40.90'
              ranking: '129'
            
            - nameen: resnest_s101-d8_deeplabv3_4xb4-160k_ade20k-512x512
              namezh: resnest_s101-d8_deeplabv3_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4842
              datasetA: '26.84'
              datasetB: '55.94'
              ranking: '130'
            
            - nameen: resnest_s101-d8_deeplabv3plus_4xb4-160k_ade20k-512x512
              namezh: resnest_s101-d8_deeplabv3plus_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1542
              datasetA: '26.70'
              datasetB: '20.27'
              ranking: '131'
            
            - nameen: segformer_mit-b0_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b0_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4754
              datasetA: '26.67'
              datasetB: '28.21'
              ranking: '132'
            
            - nameen: segformer_mit-b1_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b1_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4587
              datasetA: '26.58'
              datasetB: '41.05'
              ranking: '133'
            
            - nameen: segformer_mit-b2_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b2_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 735
              datasetA: '26.53'
              datasetB: '52.32'
              ranking: '134'
            
            - nameen: segformer_mit-b3_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b3_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 9396
              datasetA: '26.06'
              datasetB: '33.07'
              ranking: '135'
            
            - nameen: segformer_mit-b4_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b4_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8993
              datasetA: '25.90'
              datasetB: '36.44'
              ranking: '136'
            
            - nameen: segformer_mit-b5_8xb2-160k_ade20k-512x512
              namezh: segformer_mit-b5_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4772
              datasetA: '25.70'
              datasetB: '33.30'
              ranking: '137'
            
            - nameen: segformer_mit-b5_8xb2-160k_ade20k-640x640
              namezh: segformer_mit-b5_8xb2-160k_ade20k-640x640
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3937
              datasetA: '25.49'
              datasetB: '32.74'
              ranking: '138'
            
            - nameen: segmenter_vit-t_mask_8xb1-160k_ade20k-512x512
              namezh: segmenter_vit-t_mask_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3259
              datasetA: '25.33'
              datasetB: '55.34'
              ranking: '139'
            
            - nameen: segmenter_vit-s_fcn_8xb1-160k_ade20k-512x512
              namezh: segmenter_vit-s_fcn_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 385
              datasetA: '25.23'
              datasetB: '49.66'
              ranking: '140'
            
            - nameen: segmenter_vit-s_mask_8xb1-160k_ade20k-512x512
              namezh: segmenter_vit-s_mask_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3950
              datasetA: '25.11'
              datasetB: '28.30'
              ranking: '141'
            
            - nameen: segmenter_vit-b_mask_8xb1-160k_ade20k-512x512
              namezh: segmenter_vit-b_mask_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8969
              datasetA: '25.02'
              datasetB: '43.94'
              ranking: '142'
            
            - nameen: segmenter_vit-l_mask_8xb1-160k_ade20k-512x512
              namezh: segmenter_vit-l_mask_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8045
              datasetA: '24.99'
              datasetB: '37.75'
              ranking: '143'
            
            - nameen: segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512
              namezh: segnext_mscan-t_1xb16-adamw-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6660
              datasetA: '24.98'
              datasetB: '31.03'
              ranking: '144'
            
            - nameen: segnext_mscan-s_1xb16-adamw-160k_ade20k-512x512
              namezh: segnext_mscan-s_1xb16-adamw-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1992
              datasetA: '24.93'
              datasetB: '33.09'
              ranking: '145'
            
            - nameen: segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512
              namezh: segnext_mscan-b_1xb16-adamw-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8335
              datasetA: '24.78'
              datasetB: '30.34'
              ranking: '146'
            
            - nameen: segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512
              namezh: segnext_mscan-l_1xb16-adamw-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 354
              datasetA: '24.73'
              datasetB: '46.49'
              ranking: '147'
            
            - nameen: fpn_r50_4xb4-160k_ade20k-512x512
              namezh: fpn_r50_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8143
              datasetA: '24.48'
              datasetB: '21.27'
              ranking: '148'
            
            - nameen: fpn_r101_4xb4-160k_ade20k-512x512
              namezh: fpn_r101_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4603
              datasetA: '24.47'
              datasetB: '34.05'
              ranking: '149'
            
            - nameen: setr_vit-l_naive_8xb2-160k_ade20k-512x512
              namezh: setr_vit-l_naive_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1863
              datasetA: '24.40'
              datasetB: '48.51'
              ranking: '150'
            
            - nameen: setr_vit-l_pup_8xb2-160k_ade20k-512x512
              namezh: setr_vit-l_pup_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6187
              datasetA: '24.38'
              datasetB: '30.98'
              ranking: '151'
            
            - nameen: setr_vit-l-mla_8xb1-160k_ade20k-512x512
              namezh: setr_vit-l-mla_8xb1-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6735
              datasetA: '24.24'
              datasetB: '42.25'
              ranking: '152'
            
            - nameen: setr_vit-l_mla_8xb2-160k_ade20k-512x512
              namezh: setr_vit-l_mla_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1083
              datasetA: '24.01'
              datasetB: '23.76'
              ranking: '153'
            
            - nameen: swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5885
              datasetA: '23.90'
              datasetB: '41.18'
              ranking: '154'
            
            - nameen: swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-small-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5104
              datasetA: '23.65'
              datasetB: '20.37'
              ranking: '155'
            
            - nameen: swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-base-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8781
              datasetA: '23.62'
              datasetB: '47.75'
              ranking: '156'
            
            - nameen: swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-base-patch4-window7-in22k-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3476
              datasetA: '23.54'
              datasetB: '31.75'
              ranking: '157'
            
            - nameen: swin-base-patch4-window12-in1k-384x384-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-base-patch4-window12-in1k-384x384-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7881
              datasetA: '23.48'
              datasetB: '38.84'
              ranking: '158'
            
            - nameen: swin-base-patch4-window12-in22k-384x384-pre_upernet_8xb2-160k_ade20k-512x512
              namezh: swin-base-patch4-window12-in22k-384x384-pre_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2851
              datasetA: '23.41'
              datasetB: '48.99'
              ranking: '159'
            
            - nameen: twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 322
              datasetA: '23.30'
              datasetB: '48.12'
              ranking: '160'
            
            - nameen: twins_pcpvt-s_uperhead_8xb4-160k_ade20k-512x512
              namezh: twins_pcpvt-s_uperhead_8xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1106
              datasetA: '23.21'
              datasetB: '40.53'
              ranking: '161'
            
            - nameen: twins_pcpvt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_pcpvt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8255
              datasetA: '23.07'
              datasetB: '30.62'
              ranking: '162'
            
            - nameen: twins_pcpvt-b_uperhead_8xb2-160k_ade20k-512x512
              namezh: twins_pcpvt-b_uperhead_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 549
              datasetA: '22.96'
              datasetB: '34.95'
              ranking: '163'
            
            - nameen: twins_pcpvt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_pcpvt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8558
              datasetA: '22.75'
              datasetB: '34.30'
              ranking: '164'
            
            - nameen: twins_pcpvt-l_uperhead_8xb2-160k_ade20k-512x512
              namezh: twins_pcpvt-l_uperhead_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4589
              datasetA: '22.62'
              datasetB: '33.82'
              ranking: '165'
            
            - nameen: twins_svt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_svt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8916
              datasetA: '22.58'
              datasetB: '43.95'
              ranking: '166'
            
            - nameen: twins_svt-s_uperhead_8xb2-160k_ade20k-512x512
              namezh: twins_svt-s_uperhead_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4721
              datasetA: '22.50'
              datasetB: '36.09'
              ranking: '167'
            
            - nameen: twins_svt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_svt-b_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8636
              datasetA: '22.21'
              datasetB: '31.85'
              ranking: '168'
            
            - nameen: twins_svt-b_uperhead_8xb2-160k_ade20k-512x512
              namezh: twins_svt-b_uperhead_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 344
              datasetA: '21.84'
              datasetB: '32.64'
              ranking: '169'
            
            - nameen: twins_svt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512
              namezh: twins_svt-l_fpn_fpnhead_8xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3559
              datasetA: '21.65'
              datasetB: '48.65'
              ranking: '170'
            
            - nameen: twins_pcpvt-l_uperhead_8xb2-160k_ade20k-512x512
              namezh: twins_pcpvt-l_uperhead_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2103
              datasetA: '21.60'
              datasetB: '29.17'
              ranking: '171'
            
            - nameen: upernet_r50_4xb4-80k_ade20k-512x512
              namezh: upernet_r50_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3111
              datasetA: '21.54'
              datasetB: '25.67'
              ranking: '172'
            
            - nameen: upernet_r101_4xb4-80k_ade20k-512x512
              namezh: upernet_r101_4xb4-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2391
              datasetA: '21.51'
              datasetB: '54.21'
              ranking: '173'
            
            - nameen: upernet_r50_4xb4-160k_ade20k-512x512
              namezh: upernet_r50_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2430
              datasetA: '21.51'
              datasetB: '26.67'
              ranking: '174'
            
            - nameen: upernet_r101_4xb4-160k_ade20k-512x512
              namezh: upernet_r101_4xb4-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8901
              datasetA: '21.50'
              datasetB: '51.96'
              ranking: '175'
            
            - nameen: vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512
              namezh: vit_vit-b16_mln_upernet_8xb2-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5313
              datasetA: '21.28'
              datasetB: '52.88'
              ranking: '176'
            
            - nameen: vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6854
              datasetA: '20.86'
              datasetB: '30.22'
              ranking: '177'
            
            - nameen: vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_vit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 4166
              datasetA: '20.54'
              datasetB: '48.19'
              ranking: '178'
            
            - nameen: vit_deit-s16_upernet_8xb2-80k_ade20k-512x512
              namezh: vit_deit-s16_upernet_8xb2-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 5716
              datasetA: '20.47'
              datasetB: '57.86'
              ranking: '179'
            
            - nameen: vit_deit-s16_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-s16_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 6261
              datasetA: '20.45'
              datasetB: '34.90'
              ranking: '180'
            
            - nameen: vit_deit-s16_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-s16_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 8326
              datasetA: '20.44'
              datasetB: '40.42'
              ranking: '181'
            
            - nameen: vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-s16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 1447
              datasetA: '20.36'
              datasetB: '27.46'
              ranking: '182'
            
            - nameen: vit_deit-b16_upernet_8xb2-80k_ade20k-512x512
              namezh: vit_deit-b16_upernet_8xb2-80k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 3112
              datasetA: '20.35'
              datasetB: '34.98'
              ranking: '183'
            
            - nameen: vit_deit-b16_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-b16_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 889
              datasetA: '20.33'
              datasetB: '35.19'
              ranking: '184'
            
            - nameen: vit_deit-b16_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-b16_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 2322
              datasetA: '20.23'
              datasetB: '55.81'
              ranking: '185'
            
            - nameen: vit_deit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              namezh: vit_deit-b16-ln_mln_upernet_8xb2-160k_ade20k-512x512
              paper:
                text: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
                link: 'https://github.com/open-mmlab/mmsegmentation/tree/main/configs'
              download: 7448
              datasetA: '20.21'
              datasetB: '49.21'
              ranking: '186'
            
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
              download: 1821
              datasetA: '81.07'
              datasetB: '63.09'
              ranking: '1'
            - nameen: Swin-L
              namezh: Swin-L
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 3730
              datasetA: '81.06'
              datasetB: '60.78'
              ranking: '2'
            - nameen: ConvNeXt-L
              namezh: ConvNeXt-L
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 5378
              datasetA: '81.04'
              datasetB: '40.24'
              ranking: '3'
            - nameen: ConvNeXt-L + ConvStem
              namezh: ConvNeXt-L + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 4509
              datasetA: '81.03'
              datasetB: '61.09'
              ranking: '4'
            - nameen: Swin-B
              namezh: Swin-B
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 2584
              datasetA: '80.98'
              datasetB: '57.85'
              ranking: '5'
            - nameen: ConvNeXt-B
              namezh: ConvNeXt-B
              paper:
                text: >-
                  A Comprehensive Study on Robustness of Image Classification
                  Models: Benchmarking and Rethinking
                link: null
              download: 2790
              datasetA: '80.97'
              datasetB: '53.62'
              ranking: '6'
            - nameen: ConvNeXt-B + ConvStem
              namezh: ConvNeXt-B + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 8049
              datasetA: '80.97'
              datasetB: '61.76'
              ranking: '7'
            - nameen: ViT-B + ConvStem
              namezh: ViT-B + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 3424
              datasetA: '80.97'
              datasetB: '45.14'
              ranking: '8'
            - nameen: ConvNeXt-S + ConvStem
              namezh: ConvNeXt-S + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 9430
              datasetA: '80.92'
              datasetB: '49.71'
              ranking: '9'
            - nameen: RaWideResNet-101-2
              namezh: RaWideResNet-101-2
              paper:
                text: >-
                  Robust Principles: Architectural Design Principles for
                  Adversarially Robust CNNs
                link: null
              download: 7680
              datasetA: '80.9'
              datasetB: '50.66'
              ranking: '10'
            - nameen: ConvNeXt-T + ConvStem
              namezh: ConvNeXt-T + ConvStem
              paper:
                text: >-
                  Revisiting Adversarial Training for ImageNet: Architectures,
                  Training and Generalization across Threat Models
                link: null
              download: 1523
              datasetA: '80.86'
              datasetB: '60.73'
              ranking: '11'
              
            - nameen: bat_resnext26ts
              namezh: bat_resnext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3109
              datasetA: '80.86'
              datasetB: '52.49'
              ranking: '12'
            
            - nameen: beit_base_patch16_224
              namezh: beit_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4773
              datasetA: '80.84'
              datasetB: '57.2'
              ranking: '13'
            
            - nameen: beit_base_patch16_384
              namezh: beit_base_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2682
              datasetA: '80.82'
              datasetB: '54.51'
              ranking: '14'
            
            - nameen: beit_large_patch16_224
              namezh: beit_large_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9623
              datasetA: '80.79'
              datasetB: '53.4'
              ranking: '15'
            
            - nameen: beit_large_patch16_384
              namezh: beit_large_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9502
              datasetA: '80.75'
              datasetB: '45.96'
              ranking: '16'
            
            - nameen: beit_large_patch16_512
              namezh: beit_large_patch16_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9082
              datasetA: '80.75'
              datasetB: '63.24'
              ranking: '17'
            
            - nameen: beitv2_base_patch16_224
              namezh: beitv2_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2859
              datasetA: '80.75'
              datasetB: '56.26'
              ranking: '18'
            
            - nameen: beitv2_large_patch16_224
              namezh: beitv2_large_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9353
              datasetA: '80.69'
              datasetB: '56.01'
              ranking: '19'
            
            - nameen: botnet26t_256
              namezh: botnet26t_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4688
              datasetA: '80.65'
              datasetB: '43.68'
              ranking: '20'
            
            - nameen: botnet50ts_256
              namezh: botnet50ts_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2459
              datasetA: '80.6'
              datasetB: '57.28'
              ranking: '21'
            
            - nameen: caformer_b36
              namezh: caformer_b36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9113
              datasetA: '80.6'
              datasetB: '42.64'
              ranking: '22'
            
            - nameen: caformer_m36
              namezh: caformer_m36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1088
              datasetA: '80.58'
              datasetB: '44.74'
              ranking: '23'
            
            - nameen: caformer_s18
              namezh: caformer_s18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2828
              datasetA: '80.52'
              datasetB: '40.13'
              ranking: '24'
            
            - nameen: caformer_s36
              namezh: caformer_s36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6798
              datasetA: '80.51'
              datasetB: '63.48'
              ranking: '25'
            
            - nameen: cait_m36_384
              namezh: cait_m36_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7105
              datasetA: '80.5'
              datasetB: '57.65'
              ranking: '26'
            
            - nameen: cait_m48_448
              namezh: cait_m48_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5735
              datasetA: '80.49'
              datasetB: '44.81'
              ranking: '27'
            
            - nameen: cait_s24_224
              namezh: cait_s24_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1949
              datasetA: '80.47'
              datasetB: '41.67'
              ranking: '28'
            
            - nameen: cait_s24_384
              namezh: cait_s24_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1179
              datasetA: '80.46'
              datasetB: '48.23'
              ranking: '29'
            
            - nameen: cait_s36_384
              namezh: cait_s36_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5757
              datasetA: '80.45'
              datasetB: '60.13'
              ranking: '30'
            
            - nameen: cait_xs24_384
              namezh: cait_xs24_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7381
              datasetA: '80.42'
              datasetB: '40.74'
              ranking: '31'
            
            - nameen: cait_xxs24_224
              namezh: cait_xxs24_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1763
              datasetA: '80.4'
              datasetB: '41.28'
              ranking: '32'
            
            - nameen: cait_xxs24_384
              namezh: cait_xxs24_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6316
              datasetA: '80.37'
              datasetB: '53.29'
              ranking: '33'
            
            - nameen: cait_xxs36_224
              namezh: cait_xxs36_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5110
              datasetA: '80.36'
              datasetB: '49.51'
              ranking: '34'
            
            - nameen: cait_xxs36_384
              namezh: cait_xxs36_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1007
              datasetA: '80.35'
              datasetB: '55.5'
              ranking: '35'
            
            - nameen: coat_lite_medium
              namezh: coat_lite_medium
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4344
              datasetA: '80.33'
              datasetB: '60.78'
              ranking: '36'
            
            - nameen: coat_lite_medium_384
              namezh: coat_lite_medium_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7463
              datasetA: '80.33'
              datasetB: '50.21'
              ranking: '37'
            
            - nameen: coat_lite_mini
              namezh: coat_lite_mini
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5647
              datasetA: '80.31'
              datasetB: '43.1'
              ranking: '38'
            
            - nameen: coat_lite_small
              namezh: coat_lite_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3625
              datasetA: '80.31'
              datasetB: '56.99'
              ranking: '39'
            
            - nameen: coat_lite_tiny
              namezh: coat_lite_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8453
              datasetA: '80.3'
              datasetB: '44.89'
              ranking: '40'
            
            - nameen: coat_mini
              namezh: coat_mini
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5881
              datasetA: '80.3'
              datasetB: '56.86'
              ranking: '41'
            
            - nameen: coat_small
              namezh: coat_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7889
              datasetA: '80.29'
              datasetB: '62.85'
              ranking: '42'
            
            - nameen: coat_tiny
              namezh: coat_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3648
              datasetA: '80.28'
              datasetB: '56.81'
              ranking: '43'
            
            - nameen: coatnet_0_224
              namezh: coatnet_0_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4412
              datasetA: '80.21'
              datasetB: '52.32'
              ranking: '44'
            
            - nameen: coatnet_0_rw_224
              namezh: coatnet_0_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5535
              datasetA: '80.2'
              datasetB: '52.11'
              ranking: '45'
            
            - nameen: coatnet_1_224
              namezh: coatnet_1_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9268
              datasetA: '80.19'
              datasetB: '44.53'
              ranking: '46'
            
            - nameen: coatnet_1_rw_224
              namezh: coatnet_1_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8987
              datasetA: '80.18'
              datasetB: '61.16'
              ranking: '47'
            
            - nameen: coatnet_2_224
              namezh: coatnet_2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3254
              datasetA: '80.18'
              datasetB: '54.9'
              ranking: '48'
            
            - nameen: coatnet_2_rw_224
              namezh: coatnet_2_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6360
              datasetA: '80.17'
              datasetB: '56.99'
              ranking: '49'
            
            - nameen: coatnet_3_224
              namezh: coatnet_3_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4332
              datasetA: '80.16'
              datasetB: '55.27'
              ranking: '50'
            
            - nameen: coatnet_3_rw_224
              namezh: coatnet_3_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6544
              datasetA: '80.15'
              datasetB: '40.4'
              ranking: '51'
            
            - nameen: coatnet_4_224
              namezh: coatnet_4_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2070
              datasetA: '80.15'
              datasetB: '57.58'
              ranking: '52'
            
            - nameen: coatnet_5_224
              namezh: coatnet_5_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9789
              datasetA: '80.14'
              datasetB: '55.6'
              ranking: '53'
            
            - nameen: coatnet_bn_0_rw_224
              namezh: coatnet_bn_0_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8848
              datasetA: '80.12'
              datasetB: '45.51'
              ranking: '54'
            
            - nameen: coatnet_nano_cc_224
              namezh: coatnet_nano_cc_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3401
              datasetA: '80.12'
              datasetB: '56.43'
              ranking: '55'
            
            - nameen: coatnet_nano_rw_224
              namezh: coatnet_nano_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5604
              datasetA: '80.11'
              datasetB: '50.05'
              ranking: '56'
            
            - nameen: coatnet_pico_rw_224
              namezh: coatnet_pico_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4966
              datasetA: '80.09'
              datasetB: '49.74'
              ranking: '57'
            
            - nameen: coatnet_rmlp_0_rw_224
              namezh: coatnet_rmlp_0_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1014
              datasetA: '80.09'
              datasetB: '49.78'
              ranking: '58'
            
            - nameen: coatnet_rmlp_1_rw2_224
              namezh: coatnet_rmlp_1_rw2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5976
              datasetA: '80.08'
              datasetB: '49.06'
              ranking: '59'
            
            - nameen: coatnet_rmlp_1_rw_224
              namezh: coatnet_rmlp_1_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9623
              datasetA: '80.06'
              datasetB: '47.08'
              ranking: '60'
            
            - nameen: coatnet_rmlp_2_rw_224
              namezh: coatnet_rmlp_2_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7091
              datasetA: '80.04'
              datasetB: '40.34'
              ranking: '61'
            
            - nameen: coatnet_rmlp_2_rw_384
              namezh: coatnet_rmlp_2_rw_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2874
              datasetA: '80.01'
              datasetB: '41.05'
              ranking: '62'
            
            - nameen: coatnet_rmlp_3_rw_224
              namezh: coatnet_rmlp_3_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6197
              datasetA: '80.01'
              datasetB: '49.22'
              ranking: '63'
            
            - nameen: coatnet_rmlp_nano_rw_224
              namezh: coatnet_rmlp_nano_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1899
              datasetA: '79.97'
              datasetB: '56.48'
              ranking: '64'
            
            - nameen: coatnext_nano_rw_224
              namezh: coatnext_nano_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3645
              datasetA: '79.96'
              datasetB: '63.02'
              ranking: '65'
            
            - nameen: convformer_b36
              namezh: convformer_b36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7876
              datasetA: '79.96'
              datasetB: '58.61'
              ranking: '66'
            
            - nameen: convformer_m36
              namezh: convformer_m36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1480
              datasetA: '79.92'
              datasetB: '55.13'
              ranking: '67'
            
            - nameen: convformer_s18
              namezh: convformer_s18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3722
              datasetA: '79.85'
              datasetB: '54.07'
              ranking: '68'
            
            - nameen: convformer_s36
              namezh: convformer_s36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5712
              datasetA: '79.81'
              datasetB: '46.55'
              ranking: '69'
            
            - nameen: convit_base
              namezh: convit_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5806
              datasetA: '79.81'
              datasetB: '60.36'
              ranking: '70'
            
            - nameen: convit_small
              namezh: convit_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4699
              datasetA: '79.8'
              datasetB: '51.94'
              ranking: '71'
            
            - nameen: convit_tiny
              namezh: convit_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1980
              datasetA: '79.77'
              datasetB: '42.57'
              ranking: '72'
            
            - nameen: convmixer_768_32
              namezh: convmixer_768_32
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1680
              datasetA: '79.76'
              datasetB: '50.29'
              ranking: '73'
            
            - nameen: convmixer_1024_20_ks9_p14
              namezh: convmixer_1024_20_ks9_p14
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8708
              datasetA: '79.73'
              datasetB: '58.86'
              ranking: '74'
            
            - nameen: convmixer_1536_20
              namezh: convmixer_1536_20
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7849
              datasetA: '79.72'
              datasetB: '43.41'
              ranking: '75'
            
            - nameen: convnext_atto
              namezh: convnext_atto
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3141
              datasetA: '79.7'
              datasetB: '41.4'
              ranking: '76'
            
            - nameen: convnext_atto_ols
              namezh: convnext_atto_ols
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6673
              datasetA: '79.7'
              datasetB: '47.52'
              ranking: '77'
            
            - nameen: convnext_base
              namezh: convnext_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4035
              datasetA: '79.68'
              datasetB: '41.56'
              ranking: '78'
            
            - nameen: convnext_femto
              namezh: convnext_femto
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1703
              datasetA: '79.68'
              datasetB: '54.65'
              ranking: '79'
            
            - nameen: convnext_femto_ols
              namezh: convnext_femto_ols
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3215
              datasetA: '79.66'
              datasetB: '61.04'
              ranking: '80'
            
            - nameen: convnext_large
              namezh: convnext_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7573
              datasetA: '79.65'
              datasetB: '49.6'
              ranking: '81'
            
            - nameen: convnext_large_mlp
              namezh: convnext_large_mlp
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3207
              datasetA: '79.65'
              datasetB: '53.01'
              ranking: '82'
            
            - nameen: convnext_nano
              namezh: convnext_nano
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7513
              datasetA: '79.65'
              datasetB: '43.22'
              ranking: '83'
            
            - nameen: convnext_nano_ols
              namezh: convnext_nano_ols
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5312
              datasetA: '79.65'
              datasetB: '60.99'
              ranking: '84'
            
            - nameen: convnext_pico
              namezh: convnext_pico
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4837
              datasetA: '79.61'
              datasetB: '46.25'
              ranking: '85'
            
            - nameen: convnext_pico_ols
              namezh: convnext_pico_ols
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3801
              datasetA: '79.61'
              datasetB: '59.29'
              ranking: '86'
            
            - nameen: convnext_small
              namezh: convnext_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5040
              datasetA: '79.6'
              datasetB: '50.95'
              ranking: '87'
            
            - nameen: convnext_tiny
              namezh: convnext_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5246
              datasetA: '79.55'
              datasetB: '42.72'
              ranking: '88'
            
            - nameen: convnext_tiny_hnf
              namezh: convnext_tiny_hnf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1259
              datasetA: '79.55'
              datasetB: '47.67'
              ranking: '89'
            
            - nameen: convnext_xlarge
              namezh: convnext_xlarge
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1744
              datasetA: '79.51'
              datasetB: '50.43'
              ranking: '90'
            
            - nameen: convnext_xxlarge
              namezh: convnext_xxlarge
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7845
              datasetA: '79.51'
              datasetB: '42.67'
              ranking: '91'
            
            - nameen: convnextv2_atto
              namezh: convnextv2_atto
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6538
              datasetA: '79.49'
              datasetB: '45.94'
              ranking: '92'
            
            - nameen: convnextv2_base
              namezh: convnextv2_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2209
              datasetA: '79.49'
              datasetB: '51.9'
              ranking: '93'
            
            - nameen: convnextv2_femto
              namezh: convnextv2_femto
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1540
              datasetA: '79.49'
              datasetB: '47.98'
              ranking: '94'
            
            - nameen: convnextv2_huge
              namezh: convnextv2_huge
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7317
              datasetA: '79.47'
              datasetB: '53.26'
              ranking: '95'
            
            - nameen: convnextv2_large
              namezh: convnextv2_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7729
              datasetA: '79.46'
              datasetB: '60.34'
              ranking: '96'
            
            - nameen: convnextv2_nano
              namezh: convnextv2_nano
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1863
              datasetA: '79.44'
              datasetB: '60.62'
              ranking: '97'
            
            - nameen: convnextv2_pico
              namezh: convnextv2_pico
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9096
              datasetA: '79.4'
              datasetB: '63.47'
              ranking: '98'
            
            - nameen: convnextv2_small
              namezh: convnextv2_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3758
              datasetA: '79.38'
              datasetB: '57.71'
              ranking: '99'
            
            - nameen: convnextv2_tiny
              namezh: convnextv2_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4206
              datasetA: '79.37'
              datasetB: '57.29'
              ranking: '100'
            
            - nameen: crossvit_9_240
              namezh: crossvit_9_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2906
              datasetA: '79.37'
              datasetB: '53.32'
              ranking: '101'
            
            - nameen: crossvit_9_dagger_240
              namezh: crossvit_9_dagger_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9292
              datasetA: '79.36'
              datasetB: '61.97'
              ranking: '102'
            
            - nameen: crossvit_15_240
              namezh: crossvit_15_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4423
              datasetA: '79.35'
              datasetB: '60.99'
              ranking: '103'
            
            - nameen: crossvit_15_dagger_240
              namezh: crossvit_15_dagger_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2052
              datasetA: '79.35'
              datasetB: '48.68'
              ranking: '104'
            
            - nameen: crossvit_15_dagger_408
              namezh: crossvit_15_dagger_408
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1147
              datasetA: '79.35'
              datasetB: '60.82'
              ranking: '105'
            
            - nameen: crossvit_18_240
              namezh: crossvit_18_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7434
              datasetA: '79.32'
              datasetB: '58.62'
              ranking: '106'
            
            - nameen: crossvit_18_dagger_240
              namezh: crossvit_18_dagger_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4884
              datasetA: '79.29'
              datasetB: '42.72'
              ranking: '107'
            
            - nameen: crossvit_18_dagger_408
              namezh: crossvit_18_dagger_408
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3569
              datasetA: '79.29'
              datasetB: '52.7'
              ranking: '108'
            
            - nameen: crossvit_base_240
              namezh: crossvit_base_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2032
              datasetA: '79.28'
              datasetB: '51.34'
              ranking: '109'
            
            - nameen: crossvit_small_240
              namezh: crossvit_small_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8120
              datasetA: '79.26'
              datasetB: '41.17'
              ranking: '110'
            
            - nameen: crossvit_tiny_240
              namezh: crossvit_tiny_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1335
              datasetA: '79.25'
              datasetB: '55.77'
              ranking: '111'
            
            - nameen: cs3darknet_focus_l
              namezh: cs3darknet_focus_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4086
              datasetA: '79.2'
              datasetB: '53.65'
              ranking: '112'
            
            - nameen: cs3darknet_focus_m
              namezh: cs3darknet_focus_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5938
              datasetA: '79.2'
              datasetB: '60.12'
              ranking: '113'
            
            - nameen: cs3darknet_focus_s
              namezh: cs3darknet_focus_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4008
              datasetA: '79.2'
              datasetB: '40.85'
              ranking: '114'
            
            - nameen: cs3darknet_focus_x
              namezh: cs3darknet_focus_x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6021
              datasetA: '79.2'
              datasetB: '44.94'
              ranking: '115'
            
            - nameen: cs3darknet_l
              namezh: cs3darknet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8124
              datasetA: '79.17'
              datasetB: '57.73'
              ranking: '116'
            
            - nameen: cs3darknet_m
              namezh: cs3darknet_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3683
              datasetA: '79.15'
              datasetB: '43.69'
              ranking: '117'
            
            - nameen: cs3darknet_s
              namezh: cs3darknet_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8540
              datasetA: '79.13'
              datasetB: '57.39'
              ranking: '118'
            
            - nameen: cs3darknet_x
              namezh: cs3darknet_x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2183
              datasetA: '79.13'
              datasetB: '55.91'
              ranking: '119'
            
            - nameen: cs3edgenet_x
              namezh: cs3edgenet_x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2546
              datasetA: '79.09'
              datasetB: '61.84'
              ranking: '120'
            
            - nameen: cs3se_edgenet_x
              namezh: cs3se_edgenet_x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2996
              datasetA: '79.02'
              datasetB: '57.68'
              ranking: '121'
            
            - nameen: cs3sedarknet_l
              namezh: cs3sedarknet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7974
              datasetA: '79.02'
              datasetB: '54.02'
              ranking: '122'
            
            - nameen: cs3sedarknet_x
              namezh: cs3sedarknet_x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1824
              datasetA: '79.02'
              datasetB: '44.49'
              ranking: '123'
            
            - nameen: cs3sedarknet_xdw
              namezh: cs3sedarknet_xdw
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6775
              datasetA: '79.02'
              datasetB: '57.19'
              ranking: '124'
            
            - nameen: cspdarknet53
              namezh: cspdarknet53
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4805
              datasetA: '79.01'
              datasetB: '42.16'
              ranking: '125'
            
            - nameen: cspresnet50
              namezh: cspresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8099
              datasetA: '78.99'
              datasetB: '47.16'
              ranking: '126'
            
            - nameen: cspresnet50d
              namezh: cspresnet50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9965
              datasetA: '78.99'
              datasetB: '44.26'
              ranking: '127'
            
            - nameen: cspresnet50w
              namezh: cspresnet50w
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9949
              datasetA: '78.98'
              datasetB: '52.97'
              ranking: '128'
            
            - nameen: cspresnext50
              namezh: cspresnext50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1091
              datasetA: '78.97'
              datasetB: '47.84'
              ranking: '129'
            
            - nameen: darknet17
              namezh: darknet17
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1202
              datasetA: '78.96'
              datasetB: '40.32'
              ranking: '130'
            
            - nameen: darknet21
              namezh: darknet21
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6661
              datasetA: '78.91'
              datasetB: '45.96'
              ranking: '131'
            
            - nameen: darknet53
              namezh: darknet53
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2382
              datasetA: '78.9'
              datasetB: '46.47'
              ranking: '132'
            
            - nameen: darknetaa53
              namezh: darknetaa53
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3388
              datasetA: '78.88'
              datasetB: '45.62'
              ranking: '133'
            
            - nameen: davit_base
              namezh: davit_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6710
              datasetA: '78.86'
              datasetB: '44.19'
              ranking: '134'
            
            - nameen: davit_giant
              namezh: davit_giant
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8772
              datasetA: '78.86'
              datasetB: '63.14'
              ranking: '135'
            
            - nameen: davit_huge
              namezh: davit_huge
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3802
              datasetA: '78.85'
              datasetB: '42.03'
              ranking: '136'
            
            - nameen: davit_large
              namezh: davit_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3482
              datasetA: '78.84'
              datasetB: '53.17'
              ranking: '137'
            
            - nameen: davit_small
              namezh: davit_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1668
              datasetA: '78.82'
              datasetB: '41.79'
              ranking: '138'
            
            - nameen: davit_tiny
              namezh: davit_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8544
              datasetA: '78.81'
              datasetB: '48.21'
              ranking: '139'
            
            - nameen: deit3_base_patch16_224
              namezh: deit3_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3843
              datasetA: '78.78'
              datasetB: '43.9'
              ranking: '140'
            
            - nameen: deit3_base_patch16_384
              namezh: deit3_base_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8295
              datasetA: '78.78'
              datasetB: '59.04'
              ranking: '141'
            
            - nameen: deit3_huge_patch14_224
              namezh: deit3_huge_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8806
              datasetA: '78.76'
              datasetB: '60.14'
              ranking: '142'
            
            - nameen: deit3_large_patch16_224
              namezh: deit3_large_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6579
              datasetA: '78.75'
              datasetB: '62.47'
              ranking: '143'
            
            - nameen: deit3_large_patch16_384
              namezh: deit3_large_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3364
              datasetA: '78.73'
              datasetB: '59.37'
              ranking: '144'
            
            - nameen: deit3_medium_patch16_224
              namezh: deit3_medium_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7047
              datasetA: '78.73'
              datasetB: '44.53'
              ranking: '145'
            
            - nameen: deit3_small_patch16_224
              namezh: deit3_small_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6660
              datasetA: '78.67'
              datasetB: '40.9'
              ranking: '146'
            
            - nameen: deit3_small_patch16_384
              namezh: deit3_small_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5506
              datasetA: '78.66'
              datasetB: '61.04'
              ranking: '147'
            
            - nameen: deit_base_distilled_patch16_224
              namezh: deit_base_distilled_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8893
              datasetA: '78.65'
              datasetB: '54.96'
              ranking: '148'
            
            - nameen: deit_base_distilled_patch16_384
              namezh: deit_base_distilled_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6622
              datasetA: '78.63'
              datasetB: '59.38'
              ranking: '149'
            
            - nameen: deit_base_patch16_224
              namezh: deit_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4706
              datasetA: '78.56'
              datasetB: '51.39'
              ranking: '150'
            
            - nameen: deit_base_patch16_384
              namezh: deit_base_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7470
              datasetA: '78.55'
              datasetB: '54.35'
              ranking: '151'
            
            - nameen: deit_small_distilled_patch16_224
              namezh: deit_small_distilled_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9015
              datasetA: '78.52'
              datasetB: '57.62'
              ranking: '152'
            
            - nameen: deit_small_patch16_224
              namezh: deit_small_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5923
              datasetA: '78.51'
              datasetB: '40.83'
              ranking: '153'
            
            - nameen: deit_tiny_distilled_patch16_224
              namezh: deit_tiny_distilled_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5624
              datasetA: '78.51'
              datasetB: '59.51'
              ranking: '154'
            
            - nameen: deit_tiny_patch16_224
              namezh: deit_tiny_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1382
              datasetA: '78.5'
              datasetB: '52.97'
              ranking: '155'
            
            - nameen: densenet121
              namezh: densenet121
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9049
              datasetA: '78.49'
              datasetB: '43.31'
              ranking: '156'
            
            - nameen: densenet161
              namezh: densenet161
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9296
              datasetA: '78.47'
              datasetB: '47.46'
              ranking: '157'
            
            - nameen: densenet169
              namezh: densenet169
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9204
              datasetA: '78.47'
              datasetB: '46.01'
              ranking: '158'
            
            - nameen: densenet201
              namezh: densenet201
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3916
              datasetA: '78.45'
              datasetB: '45.38'
              ranking: '159'
            
            - nameen: densenet264d
              namezh: densenet264d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4309
              datasetA: '78.45'
              datasetB: '51.17'
              ranking: '160'
            
            - nameen: densenetblur121d
              namezh: densenetblur121d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9073
              datasetA: '78.45'
              datasetB: '59.12'
              ranking: '161'
            
            - nameen: dla34
              namezh: dla34
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8405
              datasetA: '78.45'
              datasetB: '42.67'
              ranking: '162'
            
            - nameen: dla46_c
              namezh: dla46_c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4354
              datasetA: '78.44'
              datasetB: '49.61'
              ranking: '163'
            
            - nameen: dla46x_c
              namezh: dla46x_c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9937
              datasetA: '78.42'
              datasetB: '45.95'
              ranking: '164'
            
            - nameen: dla60
              namezh: dla60
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6191
              datasetA: '78.4'
              datasetB: '52.33'
              ranking: '165'
            
            - nameen: dla60_res2net
              namezh: dla60_res2net
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9055
              datasetA: '78.39'
              datasetB: '52.21'
              ranking: '166'
            
            - nameen: dla60_res2next
              namezh: dla60_res2next
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9350
              datasetA: '78.38'
              datasetB: '46.76'
              ranking: '167'
            
            - nameen: dla60x
              namezh: dla60x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8427
              datasetA: '78.35'
              datasetB: '41.63'
              ranking: '168'
            
            - nameen: dla60x_c
              namezh: dla60x_c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3308
              datasetA: '78.33'
              datasetB: '47.25'
              ranking: '169'
            
            - nameen: dla102
              namezh: dla102
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7190
              datasetA: '78.31'
              datasetB: '40.46'
              ranking: '170'
            
            - nameen: dla102x
              namezh: dla102x
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1556
              datasetA: '78.3'
              datasetB: '60.22'
              ranking: '171'
            
            - nameen: dla102x2
              namezh: dla102x2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2206
              datasetA: '78.26'
              datasetB: '63.15'
              ranking: '172'
            
            - nameen: dla169
              namezh: dla169
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1196
              datasetA: '78.26'
              datasetB: '58.25'
              ranking: '173'
            
            - nameen: dm_nfnet_f0
              namezh: dm_nfnet_f0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6416
              datasetA: '78.26'
              datasetB: '42.27'
              ranking: '174'
            
            - nameen: dm_nfnet_f1
              namezh: dm_nfnet_f1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3659
              datasetA: '78.25'
              datasetB: '53.78'
              ranking: '175'
            
            - nameen: dm_nfnet_f2
              namezh: dm_nfnet_f2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7820
              datasetA: '78.23'
              datasetB: '46.57'
              ranking: '176'
            
            - nameen: dm_nfnet_f3
              namezh: dm_nfnet_f3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3424
              datasetA: '78.22'
              datasetB: '51.11'
              ranking: '177'
            
            - nameen: dm_nfnet_f4
              namezh: dm_nfnet_f4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4448
              datasetA: '78.22'
              datasetB: '56.91'
              ranking: '178'
            
            - nameen: dm_nfnet_f5
              namezh: dm_nfnet_f5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6194
              datasetA: '78.2'
              datasetB: '62.09'
              ranking: '179'
            
            - nameen: dm_nfnet_f6
              namezh: dm_nfnet_f6
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5761
              datasetA: '78.13'
              datasetB: '55.69'
              ranking: '180'
            
            - nameen: dpn48b
              namezh: dpn48b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8553
              datasetA: '78.13'
              datasetB: '60.29'
              ranking: '181'
            
            - nameen: dpn68
              namezh: dpn68
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9731
              datasetA: '78.12'
              datasetB: '53.91'
              ranking: '182'
            
            - nameen: dpn68b
              namezh: dpn68b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5328
              datasetA: '78.11'
              datasetB: '60.2'
              ranking: '183'
            
            - nameen: dpn92
              namezh: dpn92
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7005
              datasetA: '78.09'
              datasetB: '62.72'
              ranking: '184'
            
            - nameen: dpn98
              namezh: dpn98
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8431
              datasetA: '78.08'
              datasetB: '58.42'
              ranking: '185'
            
            - nameen: dpn107
              namezh: dpn107
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7261
              datasetA: '78.06'
              datasetB: '41.95'
              ranking: '186'
            
            - nameen: dpn131
              namezh: dpn131
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1874
              datasetA: '78.04'
              datasetB: '57.13'
              ranking: '187'
            
            - nameen: eca_botnext26ts_256
              namezh: eca_botnext26ts_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8011
              datasetA: '78.03'
              datasetB: '45.86'
              ranking: '188'
            
            - nameen: eca_halonext26ts
              namezh: eca_halonext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8674
              datasetA: '78.03'
              datasetB: '48.76'
              ranking: '189'
            
            - nameen: eca_nfnet_l0
              namezh: eca_nfnet_l0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7541
              datasetA: '78.0'
              datasetB: '62.39'
              ranking: '190'
            
            - nameen: eca_nfnet_l1
              namezh: eca_nfnet_l1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9685
              datasetA: '77.98'
              datasetB: '46.21'
              ranking: '191'
            
            - nameen: eca_nfnet_l2
              namezh: eca_nfnet_l2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5232
              datasetA: '77.98'
              datasetB: '45.35'
              ranking: '192'
            
            - nameen: eca_nfnet_l3
              namezh: eca_nfnet_l3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6681
              datasetA: '77.97'
              datasetB: '42.74'
              ranking: '193'
            
            - nameen: eca_resnet33ts
              namezh: eca_resnet33ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9641
              datasetA: '77.94'
              datasetB: '54.55'
              ranking: '194'
            
            - nameen: eca_resnext26ts
              namezh: eca_resnext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2863
              datasetA: '77.92'
              datasetB: '44.06'
              ranking: '195'
            
            - nameen: eca_vovnet39b
              namezh: eca_vovnet39b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8023
              datasetA: '77.88'
              datasetB: '50.26'
              ranking: '196'
            
            - nameen: ecaresnet26t
              namezh: ecaresnet26t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9200
              datasetA: '77.83'
              datasetB: '57.8'
              ranking: '197'
            
            - nameen: ecaresnet50d
              namezh: ecaresnet50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5788
              datasetA: '77.77'
              datasetB: '40.58'
              ranking: '198'
            
            - nameen: ecaresnet50d_pruned
              namezh: ecaresnet50d_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6791
              datasetA: '77.69'
              datasetB: '43.41'
              ranking: '199'
            
            - nameen: ecaresnet50t
              namezh: ecaresnet50t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9090
              datasetA: '77.68'
              datasetB: '42.47'
              ranking: '200'
            
            - nameen: ecaresnet101d
              namezh: ecaresnet101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9857
              datasetA: '77.67'
              datasetB: '47.95'
              ranking: '201'
            
            - nameen: ecaresnet101d_pruned
              namezh: ecaresnet101d_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2335
              datasetA: '77.62'
              datasetB: '40.73'
              ranking: '202'
            
            - nameen: ecaresnet200d
              namezh: ecaresnet200d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3502
              datasetA: '77.61'
              datasetB: '43.03'
              ranking: '203'
            
            - nameen: ecaresnet269d
              namezh: ecaresnet269d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7619
              datasetA: '77.61'
              datasetB: '54.89'
              ranking: '204'
            
            - nameen: ecaresnetlight
              namezh: ecaresnetlight
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9877
              datasetA: '77.59'
              datasetB: '56.36'
              ranking: '205'
            
            - nameen: ecaresnext26t_32x4d
              namezh: ecaresnext26t_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5064
              datasetA: '77.58'
              datasetB: '45.53'
              ranking: '206'
            
            - nameen: ecaresnext50t_32x4d
              namezh: ecaresnext50t_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8306
              datasetA: '77.57'
              datasetB: '60.94'
              ranking: '207'
            
            - nameen: edgenext_base
              namezh: edgenext_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2046
              datasetA: '77.52'
              datasetB: '53.18'
              ranking: '208'
            
            - nameen: edgenext_small
              namezh: edgenext_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6395
              datasetA: '77.52'
              datasetB: '47.18'
              ranking: '209'
            
            - nameen: edgenext_small_rw
              namezh: edgenext_small_rw
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8385
              datasetA: '77.5'
              datasetB: '45.96'
              ranking: '210'
            
            - nameen: edgenext_x_small
              namezh: edgenext_x_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7040
              datasetA: '77.49'
              datasetB: '50.19'
              ranking: '211'
            
            - nameen: edgenext_xx_small
              namezh: edgenext_xx_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7524
              datasetA: '77.49'
              datasetB: '57.58'
              ranking: '212'
            
            - nameen: efficientformer_l1
              namezh: efficientformer_l1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8509
              datasetA: '77.48'
              datasetB: '53.48'
              ranking: '213'
            
            - nameen: efficientformer_l3
              namezh: efficientformer_l3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5767
              datasetA: '77.43'
              datasetB: '40.24'
              ranking: '214'
            
            - nameen: efficientformer_l7
              namezh: efficientformer_l7
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9124
              datasetA: '77.35'
              datasetB: '61.3'
              ranking: '215'
            
            - nameen: efficientformerv2_l
              namezh: efficientformerv2_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1743
              datasetA: '77.33'
              datasetB: '51.36'
              ranking: '216'
            
            - nameen: efficientformerv2_s0
              namezh: efficientformerv2_s0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6651
              datasetA: '77.31'
              datasetB: '61.02'
              ranking: '217'
            
            - nameen: efficientformerv2_s1
              namezh: efficientformerv2_s1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7612
              datasetA: '77.31'
              datasetB: '41.5'
              ranking: '218'
            
            - nameen: efficientformerv2_s2
              namezh: efficientformerv2_s2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3800
              datasetA: '77.3'
              datasetB: '53.64'
              ranking: '219'
            
            - nameen: efficientnet_b0
              namezh: efficientnet_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1217
              datasetA: '77.25'
              datasetB: '45.46'
              ranking: '220'
            
            - nameen: efficientnet_b0_g8_gn
              namezh: efficientnet_b0_g8_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6360
              datasetA: '77.25'
              datasetB: '50.76'
              ranking: '221'
            
            - nameen: efficientnet_b0_g16_evos
              namezh: efficientnet_b0_g16_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6250
              datasetA: '77.25'
              datasetB: '48.63'
              ranking: '222'
            
            - nameen: efficientnet_b0_gn
              namezh: efficientnet_b0_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9648
              datasetA: '77.22'
              datasetB: '44.74'
              ranking: '223'
            
            - nameen: efficientnet_b1
              namezh: efficientnet_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3555
              datasetA: '77.21'
              datasetB: '58.41'
              ranking: '224'
            
            - nameen: efficientnet_b1_pruned
              namezh: efficientnet_b1_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2460
              datasetA: '77.18'
              datasetB: '48.55'
              ranking: '225'
            
            - nameen: efficientnet_b2
              namezh: efficientnet_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5069
              datasetA: '77.14'
              datasetB: '53.4'
              ranking: '226'
            
            - nameen: efficientnet_b2_pruned
              namezh: efficientnet_b2_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7190
              datasetA: '77.14'
              datasetB: '56.14'
              ranking: '227'
            
            - nameen: efficientnet_b3
              namezh: efficientnet_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2562
              datasetA: '77.04'
              datasetB: '47.57'
              ranking: '228'
            
            - nameen: efficientnet_b3_g8_gn
              namezh: efficientnet_b3_g8_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6348
              datasetA: '77.03'
              datasetB: '60.83'
              ranking: '229'
            
            - nameen: efficientnet_b3_gn
              namezh: efficientnet_b3_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2407
              datasetA: '77.02'
              datasetB: '49.31'
              ranking: '230'
            
            - nameen: efficientnet_b3_pruned
              namezh: efficientnet_b3_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3981
              datasetA: '77.02'
              datasetB: '61.72'
              ranking: '231'
            
            - nameen: efficientnet_b4
              namezh: efficientnet_b4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1038
              datasetA: '77.01'
              datasetB: '63.02'
              ranking: '232'
            
            - nameen: efficientnet_b5
              namezh: efficientnet_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6097
              datasetA: '77.0'
              datasetB: '49.42'
              ranking: '233'
            
            - nameen: efficientnet_b6
              namezh: efficientnet_b6
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4460
              datasetA: '76.96'
              datasetB: '49.32'
              ranking: '234'
            
            - nameen: efficientnet_b7
              namezh: efficientnet_b7
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6875
              datasetA: '76.96'
              datasetB: '52.02'
              ranking: '235'
            
            - nameen: efficientnet_b8
              namezh: efficientnet_b8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6076
              datasetA: '76.93'
              datasetB: '63.45'
              ranking: '236'
            
            - nameen: efficientnet_blur_b0
              namezh: efficientnet_blur_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5061
              datasetA: '76.86'
              datasetB: '45.62'
              ranking: '237'
            
            - nameen: efficientnet_cc_b0_4e
              namezh: efficientnet_cc_b0_4e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4307
              datasetA: '76.85'
              datasetB: '42.09'
              ranking: '238'
            
            - nameen: efficientnet_cc_b0_8e
              namezh: efficientnet_cc_b0_8e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2289
              datasetA: '76.84'
              datasetB: '45.54'
              ranking: '239'
            
            - nameen: efficientnet_cc_b1_8e
              namezh: efficientnet_cc_b1_8e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3684
              datasetA: '76.81'
              datasetB: '53.76'
              ranking: '240'
            
            - nameen: efficientnet_el
              namezh: efficientnet_el
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2802
              datasetA: '76.79'
              datasetB: '41.18'
              ranking: '241'
            
            - nameen: efficientnet_el_pruned
              namezh: efficientnet_el_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1082
              datasetA: '76.76'
              datasetB: '63.42'
              ranking: '242'
            
            - nameen: efficientnet_em
              namezh: efficientnet_em
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8799
              datasetA: '76.75'
              datasetB: '59.21'
              ranking: '243'
            
            - nameen: efficientnet_es
              namezh: efficientnet_es
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6622
              datasetA: '76.73'
              datasetB: '41.74'
              ranking: '244'
            
            - nameen: efficientnet_es_pruned
              namezh: efficientnet_es_pruned
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2625
              datasetA: '76.72'
              datasetB: '45.67'
              ranking: '245'
            
            - nameen: efficientnet_h_b5
              namezh: efficientnet_h_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9094
              datasetA: '76.68'
              datasetB: '50.91'
              ranking: '246'
            
            - nameen: efficientnet_l2
              namezh: efficientnet_l2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5080
              datasetA: '76.67'
              datasetB: '47.65'
              ranking: '247'
            
            - nameen: efficientnet_lite0
              namezh: efficientnet_lite0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7327
              datasetA: '76.64'
              datasetB: '47.11'
              ranking: '248'
            
            - nameen: efficientnet_lite1
              namezh: efficientnet_lite1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6429
              datasetA: '76.61'
              datasetB: '42.48'
              ranking: '249'
            
            - nameen: efficientnet_lite2
              namezh: efficientnet_lite2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8902
              datasetA: '76.6'
              datasetB: '54.16'
              ranking: '250'
            
            - nameen: efficientnet_lite3
              namezh: efficientnet_lite3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4458
              datasetA: '76.59'
              datasetB: '44.52'
              ranking: '251'
            
            - nameen: efficientnet_lite4
              namezh: efficientnet_lite4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3285
              datasetA: '76.59'
              datasetB: '51.72'
              ranking: '252'
            
            - nameen: efficientnet_x_b3
              namezh: efficientnet_x_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1815
              datasetA: '76.57'
              datasetB: '59.23'
              ranking: '253'
            
            - nameen: efficientnet_x_b5
              namezh: efficientnet_x_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6249
              datasetA: '76.57'
              datasetB: '41.1'
              ranking: '254'
            
            - nameen: efficientnetv2_l
              namezh: efficientnetv2_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6458
              datasetA: '76.56'
              datasetB: '54.07'
              ranking: '255'
            
            - nameen: efficientnetv2_m
              namezh: efficientnetv2_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7453
              datasetA: '76.52'
              datasetB: '43.56'
              ranking: '256'
            
            - nameen: efficientnetv2_rw_m
              namezh: efficientnetv2_rw_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2446
              datasetA: '76.51'
              datasetB: '53.32'
              ranking: '257'
            
            - nameen: efficientnetv2_rw_s
              namezh: efficientnetv2_rw_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2662
              datasetA: '76.51'
              datasetB: '45.92'
              ranking: '258'
            
            - nameen: efficientnetv2_rw_t
              namezh: efficientnetv2_rw_t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8628
              datasetA: '76.49'
              datasetB: '49.13'
              ranking: '259'
            
            - nameen: efficientnetv2_s
              namezh: efficientnetv2_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5307
              datasetA: '76.44'
              datasetB: '54.29'
              ranking: '260'
            
            - nameen: efficientnetv2_xl
              namezh: efficientnetv2_xl
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4256
              datasetA: '76.43'
              datasetB: '54.74'
              ranking: '261'
            
            - nameen: efficientvit_b0
              namezh: efficientvit_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2238
              datasetA: '76.41'
              datasetB: '49.56'
              ranking: '262'
            
            - nameen: efficientvit_b1
              namezh: efficientvit_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1616
              datasetA: '76.41'
              datasetB: '54.71'
              ranking: '263'
            
            - nameen: efficientvit_b2
              namezh: efficientvit_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4811
              datasetA: '76.41'
              datasetB: '56.74'
              ranking: '264'
            
            - nameen: efficientvit_b3
              namezh: efficientvit_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2333
              datasetA: '76.4'
              datasetB: '55.15'
              ranking: '265'
            
            - nameen: efficientvit_l1
              namezh: efficientvit_l1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2492
              datasetA: '76.38'
              datasetB: '51.71'
              ranking: '266'
            
            - nameen: efficientvit_l2
              namezh: efficientvit_l2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1910
              datasetA: '76.38'
              datasetB: '62.94'
              ranking: '267'
            
            - nameen: efficientvit_l3
              namezh: efficientvit_l3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8900
              datasetA: '76.35'
              datasetB: '47.73'
              ranking: '268'
            
            - nameen: efficientvit_m0
              namezh: efficientvit_m0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9760
              datasetA: '76.32'
              datasetB: '63.2'
              ranking: '269'
            
            - nameen: efficientvit_m1
              namezh: efficientvit_m1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8155
              datasetA: '76.28'
              datasetB: '40.96'
              ranking: '270'
            
            - nameen: efficientvit_m2
              namezh: efficientvit_m2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2740
              datasetA: '76.28'
              datasetB: '63.1'
              ranking: '271'
            
            - nameen: efficientvit_m3
              namezh: efficientvit_m3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3512
              datasetA: '76.27'
              datasetB: '56.52'
              ranking: '272'
            
            - nameen: efficientvit_m4
              namezh: efficientvit_m4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1008
              datasetA: '76.27'
              datasetB: '56.43'
              ranking: '273'
            
            - nameen: efficientvit_m5
              namezh: efficientvit_m5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2768
              datasetA: '76.26'
              datasetB: '47.32'
              ranking: '274'
            
            - nameen: ese_vovnet19b_dw
              namezh: ese_vovnet19b_dw
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5736
              datasetA: '76.26'
              datasetB: '57.04'
              ranking: '275'
            
            - nameen: ese_vovnet19b_slim
              namezh: ese_vovnet19b_slim
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1912
              datasetA: '76.24'
              datasetB: '40.67'
              ranking: '276'
            
            - nameen: ese_vovnet19b_slim_dw
              namezh: ese_vovnet19b_slim_dw
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6354
              datasetA: '76.22'
              datasetB: '55.59'
              ranking: '277'
            
            - nameen: ese_vovnet39b
              namezh: ese_vovnet39b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6288
              datasetA: '76.22'
              datasetB: '50.74'
              ranking: '278'
            
            - nameen: ese_vovnet39b_evos
              namezh: ese_vovnet39b_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7446
              datasetA: '76.2'
              datasetB: '55.04'
              ranking: '279'
            
            - nameen: ese_vovnet57b
              namezh: ese_vovnet57b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6440
              datasetA: '76.18'
              datasetB: '58.11'
              ranking: '280'
            
            - nameen: ese_vovnet99b
              namezh: ese_vovnet99b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9064
              datasetA: '76.16'
              datasetB: '55.08'
              ranking: '281'
            
            - nameen: eva02_base_patch14_224
              namezh: eva02_base_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1916
              datasetA: '76.12'
              datasetB: '51.26'
              ranking: '282'
            
            - nameen: eva02_base_patch14_448
              namezh: eva02_base_patch14_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8471
              datasetA: '76.11'
              datasetB: '50.79'
              ranking: '283'
            
            - nameen: eva02_base_patch16_clip_224
              namezh: eva02_base_patch16_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7336
              datasetA: '76.09'
              datasetB: '40.4'
              ranking: '284'
            
            - nameen: eva02_enormous_patch14_clip_224
              namezh: eva02_enormous_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5565
              datasetA: '76.09'
              datasetB: '49.05'
              ranking: '285'
            
            - nameen: eva02_large_patch14_224
              namezh: eva02_large_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1200
              datasetA: '75.97'
              datasetB: '43.69'
              ranking: '286'
            
            - nameen: eva02_large_patch14_448
              namezh: eva02_large_patch14_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3869
              datasetA: '75.95'
              datasetB: '46.06'
              ranking: '287'
            
            - nameen: eva02_large_patch14_clip_224
              namezh: eva02_large_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7136
              datasetA: '75.93'
              datasetB: '49.7'
              ranking: '288'
            
            - nameen: eva02_large_patch14_clip_336
              namezh: eva02_large_patch14_clip_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6242
              datasetA: '75.91'
              datasetB: '55.8'
              ranking: '289'
            
            - nameen: eva02_small_patch14_224
              namezh: eva02_small_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7004
              datasetA: '75.88'
              datasetB: '46.62'
              ranking: '290'
            
            - nameen: eva02_small_patch14_336
              namezh: eva02_small_patch14_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6313
              datasetA: '75.77'
              datasetB: '52.74'
              ranking: '291'
            
            - nameen: eva02_tiny_patch14_224
              namezh: eva02_tiny_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5548
              datasetA: '75.74'
              datasetB: '56.93'
              ranking: '292'
            
            - nameen: eva02_tiny_patch14_336
              namezh: eva02_tiny_patch14_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8516
              datasetA: '75.72'
              datasetB: '43.83'
              ranking: '293'
            
            - nameen: eva_giant_patch14_224
              namezh: eva_giant_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8716
              datasetA: '75.68'
              datasetB: '55.92'
              ranking: '294'
            
            - nameen: eva_giant_patch14_336
              namezh: eva_giant_patch14_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1688
              datasetA: '75.67'
              datasetB: '49.75'
              ranking: '295'
            
            - nameen: eva_giant_patch14_560
              namezh: eva_giant_patch14_560
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7551
              datasetA: '75.67'
              datasetB: '58.67'
              ranking: '296'
            
            - nameen: eva_giant_patch14_clip_224
              namezh: eva_giant_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9312
              datasetA: '75.67'
              datasetB: '62.9'
              ranking: '297'
            
            - nameen: eva_large_patch14_196
              namezh: eva_large_patch14_196
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3576
              datasetA: '75.65'
              datasetB: '59.67'
              ranking: '298'
            
            - nameen: eva_large_patch14_336
              namezh: eva_large_patch14_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3491
              datasetA: '75.62'
              datasetB: '50.27'
              ranking: '299'
            
            - nameen: fastvit_ma36
              namezh: fastvit_ma36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1143
              datasetA: '75.62'
              datasetB: '57.53'
              ranking: '300'
            
            - nameen: fastvit_mci0
              namezh: fastvit_mci0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1772
              datasetA: '75.62'
              datasetB: '42.59'
              ranking: '301'
            
            - nameen: fastvit_mci1
              namezh: fastvit_mci1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4641
              datasetA: '75.58'
              datasetB: '61.88'
              ranking: '302'
            
            - nameen: fastvit_mci2
              namezh: fastvit_mci2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9723
              datasetA: '75.57'
              datasetB: '47.6'
              ranking: '303'
            
            - nameen: fastvit_s12
              namezh: fastvit_s12
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5585
              datasetA: '75.56'
              datasetB: '43.4'
              ranking: '304'
            
            - nameen: fastvit_sa12
              namezh: fastvit_sa12
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2208
              datasetA: '75.55'
              datasetB: '56.49'
              ranking: '305'
            
            - nameen: fastvit_sa24
              namezh: fastvit_sa24
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5070
              datasetA: '75.53'
              datasetB: '57.03'
              ranking: '306'
            
            - nameen: fastvit_sa36
              namezh: fastvit_sa36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6389
              datasetA: '75.53'
              datasetB: '40.09'
              ranking: '307'
            
            - nameen: fastvit_t8
              namezh: fastvit_t8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9884
              datasetA: '75.53'
              datasetB: '61.77'
              ranking: '308'
            
            - nameen: fastvit_t12
              namezh: fastvit_t12
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5864
              datasetA: '75.49'
              datasetB: '50.57'
              ranking: '309'
            
            - nameen: fbnetc_100
              namezh: fbnetc_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6281
              datasetA: '75.49'
              datasetB: '57.92'
              ranking: '310'
            
            - nameen: fbnetv3_b
              namezh: fbnetv3_b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8421
              datasetA: '75.46'
              datasetB: '43.9'
              ranking: '311'
            
            - nameen: fbnetv3_d
              namezh: fbnetv3_d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4333
              datasetA: '75.44'
              datasetB: '47.34'
              ranking: '312'
            
            - nameen: fbnetv3_g
              namezh: fbnetv3_g
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9916
              datasetA: '75.42'
              datasetB: '44.44'
              ranking: '313'
            
            - nameen: flexivit_base
              namezh: flexivit_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3549
              datasetA: '75.41'
              datasetB: '43.49'
              ranking: '314'
            
            - nameen: flexivit_large
              namezh: flexivit_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9424
              datasetA: '75.4'
              datasetB: '51.76'
              ranking: '315'
            
            - nameen: flexivit_small
              namezh: flexivit_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2884
              datasetA: '75.38'
              datasetB: '45.54'
              ranking: '316'
            
            - nameen: focalnet_base_lrf
              namezh: focalnet_base_lrf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1666
              datasetA: '75.38'
              datasetB: '51.04'
              ranking: '317'
            
            - nameen: focalnet_base_srf
              namezh: focalnet_base_srf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9309
              datasetA: '75.35'
              datasetB: '50.69'
              ranking: '318'
            
            - nameen: focalnet_huge_fl3
              namezh: focalnet_huge_fl3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2009
              datasetA: '75.33'
              datasetB: '63.01'
              ranking: '319'
            
            - nameen: focalnet_huge_fl4
              namezh: focalnet_huge_fl4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1391
              datasetA: '75.31'
              datasetB: '59.44'
              ranking: '320'
            
            - nameen: focalnet_large_fl3
              namezh: focalnet_large_fl3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9779
              datasetA: '75.29'
              datasetB: '59.81'
              ranking: '321'
            
            - nameen: focalnet_large_fl4
              namezh: focalnet_large_fl4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3029
              datasetA: '75.28'
              datasetB: '57.63'
              ranking: '322'
            
            - nameen: focalnet_small_lrf
              namezh: focalnet_small_lrf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8372
              datasetA: '75.27'
              datasetB: '57.04'
              ranking: '323'
            
            - nameen: focalnet_small_srf
              namezh: focalnet_small_srf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5168
              datasetA: '75.22'
              datasetB: '42.94'
              ranking: '324'
            
            - nameen: focalnet_tiny_lrf
              namezh: focalnet_tiny_lrf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6532
              datasetA: '75.2'
              datasetB: '45.68'
              ranking: '325'
            
            - nameen: focalnet_tiny_srf
              namezh: focalnet_tiny_srf
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4981
              datasetA: '75.2'
              datasetB: '55.47'
              ranking: '326'
            
            - nameen: focalnet_xlarge_fl3
              namezh: focalnet_xlarge_fl3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2425
              datasetA: '75.16'
              datasetB: '62.59'
              ranking: '327'
            
            - nameen: focalnet_xlarge_fl4
              namezh: focalnet_xlarge_fl4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9048
              datasetA: '75.14'
              datasetB: '59.17'
              ranking: '328'
            
            - nameen: gc_efficientnetv2_rw_t
              namezh: gc_efficientnetv2_rw_t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9000
              datasetA: '75.13'
              datasetB: '62.75'
              ranking: '329'
            
            - nameen: gcresnet33ts
              namezh: gcresnet33ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5003
              datasetA: '75.12'
              datasetB: '57.0'
              ranking: '330'
            
            - nameen: gcresnet50t
              namezh: gcresnet50t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5388
              datasetA: '75.11'
              datasetB: '60.93'
              ranking: '331'
            
            - nameen: gcresnext26ts
              namezh: gcresnext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6990
              datasetA: '75.1'
              datasetB: '50.04'
              ranking: '332'
            
            - nameen: gcresnext50ts
              namezh: gcresnext50ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7137
              datasetA: '75.1'
              datasetB: '57.71'
              ranking: '333'
            
            - nameen: gcvit_base
              namezh: gcvit_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1413
              datasetA: '75.1'
              datasetB: '57.77'
              ranking: '334'
            
            - nameen: gcvit_small
              namezh: gcvit_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9097
              datasetA: '75.08'
              datasetB: '40.67'
              ranking: '335'
            
            - nameen: gcvit_tiny
              namezh: gcvit_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9286
              datasetA: '75.05'
              datasetB: '40.41'
              ranking: '336'
            
            - nameen: gcvit_xtiny
              namezh: gcvit_xtiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1692
              datasetA: '74.97'
              datasetB: '52.74'
              ranking: '337'
            
            - nameen: gcvit_xxtiny
              namezh: gcvit_xxtiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9057
              datasetA: '74.95'
              datasetB: '55.41'
              ranking: '338'
            
            - nameen: gernet_l
              namezh: gernet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5974
              datasetA: '74.95'
              datasetB: '53.97'
              ranking: '339'
            
            - nameen: gernet_m
              namezh: gernet_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7630
              datasetA: '74.93'
              datasetB: '52.73'
              ranking: '340'
            
            - nameen: gernet_s
              namezh: gernet_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8365
              datasetA: '74.93'
              datasetB: '49.64'
              ranking: '341'
            
            - nameen: ghostnet_050
              namezh: ghostnet_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6214
              datasetA: '74.91'
              datasetB: '46.5'
              ranking: '342'
            
            - nameen: ghostnet_100
              namezh: ghostnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8575
              datasetA: '74.86'
              datasetB: '61.07'
              ranking: '343'
            
            - nameen: ghostnet_130
              namezh: ghostnet_130
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5466
              datasetA: '74.85'
              datasetB: '61.79'
              ranking: '344'
            
            - nameen: ghostnetv2_100
              namezh: ghostnetv2_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4569
              datasetA: '74.84'
              datasetB: '40.09'
              ranking: '345'
            
            - nameen: ghostnetv2_130
              namezh: ghostnetv2_130
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7874
              datasetA: '74.82'
              datasetB: '59.97'
              ranking: '346'
            
            - nameen: ghostnetv2_160
              namezh: ghostnetv2_160
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7306
              datasetA: '74.81'
              datasetB: '50.97'
              ranking: '347'
            
            - nameen: gmixer_12_224
              namezh: gmixer_12_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6358
              datasetA: '74.77'
              datasetB: '49.31'
              ranking: '348'
            
            - nameen: gmixer_24_224
              namezh: gmixer_24_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1121
              datasetA: '74.71'
              datasetB: '57.01'
              ranking: '349'
            
            - nameen: gmlp_b16_224
              namezh: gmlp_b16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7330
              datasetA: '74.67'
              datasetB: '41.02'
              ranking: '350'
            
            - nameen: gmlp_s16_224
              namezh: gmlp_s16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5605
              datasetA: '74.61'
              datasetB: '44.32'
              ranking: '351'
            
            - nameen: gmlp_ti16_224
              namezh: gmlp_ti16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8586
              datasetA: '74.6'
              datasetB: '45.83'
              ranking: '352'
            
            - nameen: halo2botnet50ts_256
              namezh: halo2botnet50ts_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9402
              datasetA: '74.57'
              datasetB: '63.44'
              ranking: '353'
            
            - nameen: halonet26t
              namezh: halonet26t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1156
              datasetA: '74.56'
              datasetB: '48.56'
              ranking: '354'
            
            - nameen: halonet50ts
              namezh: halonet50ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9594
              datasetA: '74.45'
              datasetB: '59.24'
              ranking: '355'
            
            - nameen: halonet_h1
              namezh: halonet_h1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3751
              datasetA: '74.44'
              datasetB: '43.51'
              ranking: '356'
            
            - nameen: haloregnetz_b
              namezh: haloregnetz_b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1109
              datasetA: '74.43'
              datasetB: '44.28'
              ranking: '357'
            
            - nameen: hardcorenas_a
              namezh: hardcorenas_a
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2134
              datasetA: '74.43'
              datasetB: '47.63'
              ranking: '358'
            
            - nameen: hardcorenas_b
              namezh: hardcorenas_b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6580
              datasetA: '74.37'
              datasetB: '45.73'
              ranking: '359'
            
            - nameen: hardcorenas_c
              namezh: hardcorenas_c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4430
              datasetA: '74.36'
              datasetB: '62.46'
              ranking: '360'
            
            - nameen: hardcorenas_d
              namezh: hardcorenas_d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7354
              datasetA: '74.34'
              datasetB: '41.02'
              ranking: '361'
            
            - nameen: hardcorenas_e
              namezh: hardcorenas_e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6945
              datasetA: '74.33'
              datasetB: '58.52'
              ranking: '362'
            
            - nameen: hardcorenas_f
              namezh: hardcorenas_f
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9672
              datasetA: '74.32'
              datasetB: '41.71'
              ranking: '363'
            
            - nameen: hgnet_base
              namezh: hgnet_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5306
              datasetA: '74.29'
              datasetB: '43.14'
              ranking: '364'
            
            - nameen: hgnet_small
              namezh: hgnet_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5639
              datasetA: '74.28'
              datasetB: '54.65'
              ranking: '365'
            
            - nameen: hgnet_tiny
              namezh: hgnet_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4552
              datasetA: '74.24'
              datasetB: '49.29'
              ranking: '366'
            
            - nameen: hgnetv2_b0
              namezh: hgnetv2_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3625
              datasetA: '74.22'
              datasetB: '57.9'
              ranking: '367'
            
            - nameen: hgnetv2_b1
              namezh: hgnetv2_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6283
              datasetA: '74.22'
              datasetB: '48.16'
              ranking: '368'
            
            - nameen: hgnetv2_b2
              namezh: hgnetv2_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5994
              datasetA: '74.22'
              datasetB: '46.57'
              ranking: '369'
            
            - nameen: hgnetv2_b3
              namezh: hgnetv2_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6332
              datasetA: '74.2'
              datasetB: '49.61'
              ranking: '370'
            
            - nameen: hgnetv2_b4
              namezh: hgnetv2_b4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5644
              datasetA: '74.18'
              datasetB: '55.15'
              ranking: '371'
            
            - nameen: hgnetv2_b5
              namezh: hgnetv2_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1400
              datasetA: '74.17'
              datasetB: '54.8'
              ranking: '372'
            
            - nameen: hgnetv2_b6
              namezh: hgnetv2_b6
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6721
              datasetA: '74.15'
              datasetB: '43.0'
              ranking: '373'
            
            - nameen: hiera_base_224
              namezh: hiera_base_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7631
              datasetA: '74.15'
              datasetB: '61.01'
              ranking: '374'
            
            - nameen: hiera_base_plus_224
              namezh: hiera_base_plus_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2236
              datasetA: '74.08'
              datasetB: '56.3'
              ranking: '375'
            
            - nameen: hiera_huge_224
              namezh: hiera_huge_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6068
              datasetA: '74.06'
              datasetB: '47.84'
              ranking: '376'
            
            - nameen: hiera_large_224
              namezh: hiera_large_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7719
              datasetA: '74.04'
              datasetB: '40.79'
              ranking: '377'
            
            - nameen: hiera_small_224
              namezh: hiera_small_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4519
              datasetA: '74.01'
              datasetB: '44.91'
              ranking: '378'
            
            - nameen: hiera_tiny_224
              namezh: hiera_tiny_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8725
              datasetA: '74.01'
              datasetB: '50.67'
              ranking: '379'
            
            - nameen: hrnet_w18
              namezh: hrnet_w18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9283
              datasetA: '73.97'
              datasetB: '52.32'
              ranking: '380'
            
            - nameen: hrnet_w18_small
              namezh: hrnet_w18_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8471
              datasetA: '73.96'
              datasetB: '51.05'
              ranking: '381'
            
            - nameen: hrnet_w18_small_v2
              namezh: hrnet_w18_small_v2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5639
              datasetA: '73.96'
              datasetB: '53.0'
              ranking: '382'
            
            - nameen: hrnet_w18_ssld
              namezh: hrnet_w18_ssld
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1673
              datasetA: '73.95'
              datasetB: '42.95'
              ranking: '383'
            
            - nameen: hrnet_w30
              namezh: hrnet_w30
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5329
              datasetA: '73.94'
              datasetB: '51.6'
              ranking: '384'
            
            - nameen: hrnet_w32
              namezh: hrnet_w32
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1411
              datasetA: '73.92'
              datasetB: '58.59'
              ranking: '385'
            
            - nameen: hrnet_w40
              namezh: hrnet_w40
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4757
              datasetA: '73.9'
              datasetB: '54.59'
              ranking: '386'
            
            - nameen: hrnet_w44
              namezh: hrnet_w44
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8982
              datasetA: '73.89'
              datasetB: '48.39'
              ranking: '387'
            
            - nameen: hrnet_w48
              namezh: hrnet_w48
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5183
              datasetA: '73.88'
              datasetB: '52.49'
              ranking: '388'
            
            - nameen: hrnet_w48_ssld
              namezh: hrnet_w48_ssld
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3068
              datasetA: '73.84'
              datasetB: '45.66'
              ranking: '389'
            
            - nameen: hrnet_w64
              namezh: hrnet_w64
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4112
              datasetA: '73.81'
              datasetB: '56.3'
              ranking: '390'
            
            - nameen: inception_next_base
              namezh: inception_next_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6589
              datasetA: '73.79'
              datasetB: '56.87'
              ranking: '391'
            
            - nameen: inception_next_small
              namezh: inception_next_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1662
              datasetA: '73.78'
              datasetB: '49.02'
              ranking: '392'
            
            - nameen: inception_next_tiny
              namezh: inception_next_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6772
              datasetA: '73.69'
              datasetB: '41.42'
              ranking: '393'
            
            - nameen: inception_resnet_v2
              namezh: inception_resnet_v2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3376
              datasetA: '73.68'
              datasetB: '61.93'
              ranking: '394'
            
            - nameen: inception_v3
              namezh: inception_v3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5047
              datasetA: '73.67'
              datasetB: '52.17'
              ranking: '395'
            
            - nameen: inception_v4
              namezh: inception_v4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6492
              datasetA: '73.64'
              datasetB: '59.03'
              ranking: '396'
            
            - nameen: lambda_resnet26rpt_256
              namezh: lambda_resnet26rpt_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7558
              datasetA: '73.61'
              datasetB: '61.09'
              ranking: '397'
            
            - nameen: lambda_resnet26t
              namezh: lambda_resnet26t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4221
              datasetA: '73.58'
              datasetB: '54.74'
              ranking: '398'
            
            - nameen: lambda_resnet50ts
              namezh: lambda_resnet50ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3197
              datasetA: '73.55'
              datasetB: '61.73'
              ranking: '399'
            
            - nameen: lamhalobotnet50ts_256
              namezh: lamhalobotnet50ts_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4771
              datasetA: '73.53'
              datasetB: '44.51'
              ranking: '400'
            
            - nameen: lcnet_035
              namezh: lcnet_035
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6601
              datasetA: '73.53'
              datasetB: '57.29'
              ranking: '401'
            
            - nameen: lcnet_050
              namezh: lcnet_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2244
              datasetA: '73.53'
              datasetB: '59.93'
              ranking: '402'
            
            - nameen: lcnet_075
              namezh: lcnet_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1793
              datasetA: '73.51'
              datasetB: '46.31'
              ranking: '403'
            
            - nameen: lcnet_100
              namezh: lcnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5536
              datasetA: '73.51'
              datasetB: '51.72'
              ranking: '404'
            
            - nameen: lcnet_150
              namezh: lcnet_150
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7839
              datasetA: '73.5'
              datasetB: '52.71'
              ranking: '405'
            
            - nameen: legacy_senet154
              namezh: legacy_senet154
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7108
              datasetA: '73.48'
              datasetB: '42.7'
              ranking: '406'
            
            - nameen: legacy_seresnet18
              namezh: legacy_seresnet18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4150
              datasetA: '73.48'
              datasetB: '54.03'
              ranking: '407'
            
            - nameen: legacy_seresnet34
              namezh: legacy_seresnet34
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4973
              datasetA: '73.43'
              datasetB: '46.71'
              ranking: '408'
            
            - nameen: legacy_seresnet50
              namezh: legacy_seresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1992
              datasetA: '73.37'
              datasetB: '40.68'
              ranking: '409'
            
            - nameen: legacy_seresnet101
              namezh: legacy_seresnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1975
              datasetA: '73.37'
              datasetB: '60.72'
              ranking: '410'
            
            - nameen: legacy_seresnet152
              namezh: legacy_seresnet152
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5554
              datasetA: '73.34'
              datasetB: '52.02'
              ranking: '411'
            
            - nameen: legacy_seresnext26_32x4d
              namezh: legacy_seresnext26_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3566
              datasetA: '73.31'
              datasetB: '40.31'
              ranking: '412'
            
            - nameen: legacy_seresnext50_32x4d
              namezh: legacy_seresnext50_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5837
              datasetA: '73.29'
              datasetB: '50.57'
              ranking: '413'
            
            - nameen: legacy_seresnext101_32x4d
              namezh: legacy_seresnext101_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3024
              datasetA: '73.29'
              datasetB: '53.46'
              ranking: '414'
            
            - nameen: legacy_xception
              namezh: legacy_xception
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3313
              datasetA: '73.29'
              datasetB: '58.14'
              ranking: '415'
            
            - nameen: levit_128
              namezh: levit_128
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3580
              datasetA: '73.27'
              datasetB: '42.45'
              ranking: '416'
            
            - nameen: levit_128s
              namezh: levit_128s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2377
              datasetA: '73.26'
              datasetB: '43.94'
              ranking: '417'
            
            - nameen: levit_192
              namezh: levit_192
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4370
              datasetA: '73.26'
              datasetB: '60.83'
              ranking: '418'
            
            - nameen: levit_256
              namezh: levit_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3728
              datasetA: '73.19'
              datasetB: '41.91'
              ranking: '419'
            
            - nameen: levit_256d
              namezh: levit_256d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5647
              datasetA: '73.18'
              datasetB: '51.39'
              ranking: '420'
            
            - nameen: levit_384
              namezh: levit_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3596
              datasetA: '73.16'
              datasetB: '46.34'
              ranking: '421'
            
            - nameen: levit_384_s8
              namezh: levit_384_s8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4343
              datasetA: '73.16'
              datasetB: '50.29'
              ranking: '422'
            
            - nameen: levit_512
              namezh: levit_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1464
              datasetA: '73.13'
              datasetB: '42.27'
              ranking: '423'
            
            - nameen: levit_512_s8
              namezh: levit_512_s8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4854
              datasetA: '73.11'
              datasetB: '49.48'
              ranking: '424'
            
            - nameen: levit_512d
              namezh: levit_512d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9124
              datasetA: '73.11'
              datasetB: '60.94'
              ranking: '425'
            
            - nameen: levit_conv_128
              namezh: levit_conv_128
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7414
              datasetA: '73.11'
              datasetB: '44.9'
              ranking: '426'
            
            - nameen: levit_conv_128s
              namezh: levit_conv_128s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3597
              datasetA: '73.08'
              datasetB: '57.12'
              ranking: '427'
            
            - nameen: levit_conv_192
              namezh: levit_conv_192
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9317
              datasetA: '73.06'
              datasetB: '55.47'
              ranking: '428'
            
            - nameen: levit_conv_256
              namezh: levit_conv_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6068
              datasetA: '73.02'
              datasetB: '47.07'
              ranking: '429'
            
            - nameen: levit_conv_256d
              namezh: levit_conv_256d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5139
              datasetA: '72.99'
              datasetB: '52.29'
              ranking: '430'
            
            - nameen: levit_conv_384
              namezh: levit_conv_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3949
              datasetA: '72.97'
              datasetB: '42.59'
              ranking: '431'
            
            - nameen: levit_conv_384_s8
              namezh: levit_conv_384_s8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2856
              datasetA: '72.95'
              datasetB: '62.2'
              ranking: '432'
            
            - nameen: levit_conv_512
              namezh: levit_conv_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5939
              datasetA: '72.93'
              datasetB: '43.0'
              ranking: '433'
            
            - nameen: levit_conv_512_s8
              namezh: levit_conv_512_s8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8309
              datasetA: '72.92'
              datasetB: '51.93'
              ranking: '434'
            
            - nameen: levit_conv_512d
              namezh: levit_conv_512d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5227
              datasetA: '72.91'
              datasetB: '50.38'
              ranking: '435'
            
            - nameen: maxvit_base_tf_224
              namezh: maxvit_base_tf_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5704
              datasetA: '72.9'
              datasetB: '44.47'
              ranking: '436'
            
            - nameen: maxvit_base_tf_384
              namezh: maxvit_base_tf_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3277
              datasetA: '72.9'
              datasetB: '46.88'
              ranking: '437'
            
            - nameen: maxvit_base_tf_512
              namezh: maxvit_base_tf_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1843
              datasetA: '72.9'
              datasetB: '57.6'
              ranking: '438'
            
            - nameen: maxvit_large_tf_224
              namezh: maxvit_large_tf_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1432
              datasetA: '72.88'
              datasetB: '50.34'
              ranking: '439'
            
            - nameen: maxvit_large_tf_384
              namezh: maxvit_large_tf_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6291
              datasetA: '72.87'
              datasetB: '55.57'
              ranking: '440'
            
            - nameen: maxvit_large_tf_512
              namezh: maxvit_large_tf_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5818
              datasetA: '72.87'
              datasetB: '60.17'
              ranking: '441'
            
            - nameen: maxvit_nano_rw_256
              namezh: maxvit_nano_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2608
              datasetA: '72.85'
              datasetB: '62.87'
              ranking: '442'
            
            - nameen: maxvit_pico_rw_256
              namezh: maxvit_pico_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1571
              datasetA: '72.83'
              datasetB: '61.44'
              ranking: '443'
            
            - nameen: maxvit_rmlp_base_rw_224
              namezh: maxvit_rmlp_base_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6224
              datasetA: '72.83'
              datasetB: '51.92'
              ranking: '444'
            
            - nameen: maxvit_rmlp_base_rw_384
              namezh: maxvit_rmlp_base_rw_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6234
              datasetA: '72.76'
              datasetB: '47.56'
              ranking: '445'
            
            - nameen: maxvit_rmlp_nano_rw_256
              namezh: maxvit_rmlp_nano_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6330
              datasetA: '72.72'
              datasetB: '62.01'
              ranking: '446'
            
            - nameen: maxvit_rmlp_pico_rw_256
              namezh: maxvit_rmlp_pico_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5904
              datasetA: '72.7'
              datasetB: '57.74'
              ranking: '447'
            
            - nameen: maxvit_rmlp_small_rw_224
              namezh: maxvit_rmlp_small_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3776
              datasetA: '72.68'
              datasetB: '58.99'
              ranking: '448'
            
            - nameen: maxvit_rmlp_small_rw_256
              namezh: maxvit_rmlp_small_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5582
              datasetA: '72.67'
              datasetB: '45.7'
              ranking: '449'
            
            - nameen: maxvit_rmlp_tiny_rw_256
              namezh: maxvit_rmlp_tiny_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1172
              datasetA: '72.66'
              datasetB: '47.19'
              ranking: '450'
            
            - nameen: maxvit_small_tf_224
              namezh: maxvit_small_tf_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5240
              datasetA: '72.6'
              datasetB: '51.6'
              ranking: '451'
            
            - nameen: maxvit_small_tf_384
              namezh: maxvit_small_tf_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5409
              datasetA: '72.59'
              datasetB: '42.93'
              ranking: '452'
            
            - nameen: maxvit_small_tf_512
              namezh: maxvit_small_tf_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7970
              datasetA: '72.58'
              datasetB: '57.32'
              ranking: '453'
            
            - nameen: maxvit_tiny_pm_256
              namezh: maxvit_tiny_pm_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1809
              datasetA: '72.54'
              datasetB: '63.43'
              ranking: '454'
            
            - nameen: maxvit_tiny_rw_224
              namezh: maxvit_tiny_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6967
              datasetA: '72.53'
              datasetB: '41.14'
              ranking: '455'
            
            - nameen: maxvit_tiny_rw_256
              namezh: maxvit_tiny_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9812
              datasetA: '72.51'
              datasetB: '52.5'
              ranking: '456'
            
            - nameen: maxvit_tiny_tf_224
              namezh: maxvit_tiny_tf_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7662
              datasetA: '72.47'
              datasetB: '53.94'
              ranking: '457'
            
            - nameen: maxvit_tiny_tf_384
              namezh: maxvit_tiny_tf_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7088
              datasetA: '72.47'
              datasetB: '51.35'
              ranking: '458'
            
            - nameen: maxvit_tiny_tf_512
              namezh: maxvit_tiny_tf_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7349
              datasetA: '72.47'
              datasetB: '47.57'
              ranking: '459'
            
            - nameen: maxvit_xlarge_tf_224
              namezh: maxvit_xlarge_tf_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7687
              datasetA: '72.46'
              datasetB: '42.62'
              ranking: '460'
            
            - nameen: maxvit_xlarge_tf_384
              namezh: maxvit_xlarge_tf_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9906
              datasetA: '72.44'
              datasetB: '54.47'
              ranking: '461'
            
            - nameen: maxvit_xlarge_tf_512
              namezh: maxvit_xlarge_tf_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7875
              datasetA: '72.43'
              datasetB: '45.31'
              ranking: '462'
            
            - nameen: maxxvit_rmlp_nano_rw_256
              namezh: maxxvit_rmlp_nano_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2584
              datasetA: '72.43'
              datasetB: '60.47'
              ranking: '463'
            
            - nameen: maxxvit_rmlp_small_rw_256
              namezh: maxxvit_rmlp_small_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3852
              datasetA: '72.42'
              datasetB: '62.53'
              ranking: '464'
            
            - nameen: maxxvit_rmlp_tiny_rw_256
              namezh: maxxvit_rmlp_tiny_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6555
              datasetA: '72.38'
              datasetB: '60.3'
              ranking: '465'
            
            - nameen: maxxvitv2_nano_rw_256
              namezh: maxxvitv2_nano_rw_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1151
              datasetA: '72.38'
              datasetB: '62.8'
              ranking: '466'
            
            - nameen: maxxvitv2_rmlp_base_rw_224
              namezh: maxxvitv2_rmlp_base_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7202
              datasetA: '72.37'
              datasetB: '45.09'
              ranking: '467'
            
            - nameen: maxxvitv2_rmlp_base_rw_384
              namezh: maxxvitv2_rmlp_base_rw_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2223
              datasetA: '72.3'
              datasetB: '49.55'
              ranking: '468'
            
            - nameen: maxxvitv2_rmlp_large_rw_224
              namezh: maxxvitv2_rmlp_large_rw_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2079
              datasetA: '72.29'
              datasetB: '41.39'
              ranking: '469'
            
            - nameen: mixer_b16_224
              namezh: mixer_b16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1141
              datasetA: '72.28'
              datasetB: '44.25'
              ranking: '470'
            
            - nameen: mixer_b32_224
              namezh: mixer_b32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2212
              datasetA: '72.26'
              datasetB: '52.24'
              ranking: '471'
            
            - nameen: mixer_l16_224
              namezh: mixer_l16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7270
              datasetA: '72.25'
              datasetB: '50.18'
              ranking: '472'
            
            - nameen: mixer_l32_224
              namezh: mixer_l32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3845
              datasetA: '72.22'
              datasetB: '54.88'
              ranking: '473'
            
            - nameen: mixer_s16_224
              namezh: mixer_s16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1992
              datasetA: '72.19'
              datasetB: '40.87'
              ranking: '474'
            
            - nameen: mixer_s32_224
              namezh: mixer_s32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1581
              datasetA: '72.18'
              datasetB: '48.66'
              ranking: '475'
            
            - nameen: mixnet_l
              namezh: mixnet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2548
              datasetA: '72.18'
              datasetB: '54.62'
              ranking: '476'
            
            - nameen: mixnet_m
              namezh: mixnet_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8705
              datasetA: '72.17'
              datasetB: '57.5'
              ranking: '477'
            
            - nameen: mixnet_s
              namezh: mixnet_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6424
              datasetA: '72.16'
              datasetB: '49.64'
              ranking: '478'
            
            - nameen: mixnet_xl
              namezh: mixnet_xl
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4048
              datasetA: '72.16'
              datasetB: '52.86'
              ranking: '479'
            
            - nameen: mixnet_xxl
              namezh: mixnet_xxl
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8382
              datasetA: '72.13'
              datasetB: '54.88'
              ranking: '480'
            
            - nameen: mnasnet_050
              namezh: mnasnet_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2751
              datasetA: '72.13'
              datasetB: '42.13'
              ranking: '481'
            
            - nameen: mnasnet_075
              namezh: mnasnet_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9271
              datasetA: '72.11'
              datasetB: '45.83'
              ranking: '482'
            
            - nameen: mnasnet_100
              namezh: mnasnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6873
              datasetA: '72.09'
              datasetB: '54.3'
              ranking: '483'
            
            - nameen: mnasnet_140
              namezh: mnasnet_140
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8281
              datasetA: '72.09'
              datasetB: '60.98'
              ranking: '484'
            
            - nameen: mnasnet_small
              namezh: mnasnet_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9340
              datasetA: '72.08'
              datasetB: '47.89'
              ranking: '485'
            
            - nameen: mobilenet_100
              namezh: mobilenet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9716
              datasetA: '72.08'
              datasetB: '59.5'
              ranking: '486'
            
            - nameen: mobilenet_125
              namezh: mobilenet_125
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7811
              datasetA: '72.07'
              datasetB: '61.95'
              ranking: '487'
            
            - nameen: mobilenet_edgetpu_100
              namezh: mobilenet_edgetpu_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5252
              datasetA: '72.06'
              datasetB: '43.81'
              ranking: '488'
            
            - nameen: mobilenet_edgetpu_v2_l
              namezh: mobilenet_edgetpu_v2_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1733
              datasetA: '72.03'
              datasetB: '58.4'
              ranking: '489'
            
            - nameen: mobilenet_edgetpu_v2_m
              namezh: mobilenet_edgetpu_v2_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5676
              datasetA: '72.02'
              datasetB: '60.01'
              ranking: '490'
            
            - nameen: mobilenet_edgetpu_v2_s
              namezh: mobilenet_edgetpu_v2_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6013
              datasetA: '72.01'
              datasetB: '60.9'
              ranking: '491'
            
            - nameen: mobilenet_edgetpu_v2_xs
              namezh: mobilenet_edgetpu_v2_xs
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5121
              datasetA: '72.0'
              datasetB: '56.6'
              ranking: '492'
            
            - nameen: mobilenetv2_035
              namezh: mobilenetv2_035
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8140
              datasetA: '71.99'
              datasetB: '56.68'
              ranking: '493'
            
            - nameen: mobilenetv2_050
              namezh: mobilenetv2_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8129
              datasetA: '71.98'
              datasetB: '57.12'
              ranking: '494'
            
            - nameen: mobilenetv2_075
              namezh: mobilenetv2_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2627
              datasetA: '71.97'
              datasetB: '40.42'
              ranking: '495'
            
            - nameen: mobilenetv2_100
              namezh: mobilenetv2_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9383
              datasetA: '71.96'
              datasetB: '46.95'
              ranking: '496'
            
            - nameen: mobilenetv2_110d
              namezh: mobilenetv2_110d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1191
              datasetA: '71.95'
              datasetB: '63.38'
              ranking: '497'
            
            - nameen: mobilenetv2_120d
              namezh: mobilenetv2_120d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6082
              datasetA: '71.95'
              datasetB: '48.01'
              ranking: '498'
            
            - nameen: mobilenetv2_140
              namezh: mobilenetv2_140
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5350
              datasetA: '71.94'
              datasetB: '45.05'
              ranking: '499'
            
            - nameen: mobilenetv3_large_075
              namezh: mobilenetv3_large_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8527
              datasetA: '71.93'
              datasetB: '53.65'
              ranking: '500'
            
            - nameen: mobilenetv3_large_100
              namezh: mobilenetv3_large_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6593
              datasetA: '71.93'
              datasetB: '59.61'
              ranking: '501'
            
            - nameen: mobilenetv3_rw
              namezh: mobilenetv3_rw
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8358
              datasetA: '71.92'
              datasetB: '60.28'
              ranking: '502'
            
            - nameen: mobilenetv3_small_050
              namezh: mobilenetv3_small_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7343
              datasetA: '71.91'
              datasetB: '40.06'
              ranking: '503'
            
            - nameen: mobilenetv3_small_075
              namezh: mobilenetv3_small_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9797
              datasetA: '71.9'
              datasetB: '40.59'
              ranking: '504'
            
            - nameen: mobilenetv3_small_100
              namezh: mobilenetv3_small_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7209
              datasetA: '71.89'
              datasetB: '55.91'
              ranking: '505'
            
            - nameen: mobilenetv4_conv_aa_medium
              namezh: mobilenetv4_conv_aa_medium
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6921
              datasetA: '71.88'
              datasetB: '44.84'
              ranking: '506'
            
            - nameen: mobilenetv4_conv_blur_medium
              namezh: mobilenetv4_conv_blur_medium
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1433
              datasetA: '71.87'
              datasetB: '54.11'
              ranking: '507'
            
            - nameen: mobilenetv4_conv_large
              namezh: mobilenetv4_conv_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8358
              datasetA: '71.86'
              datasetB: '48.5'
              ranking: '508'
            
            - nameen: mobilenetv4_conv_medium
              namezh: mobilenetv4_conv_medium
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4487
              datasetA: '71.79'
              datasetB: '56.51'
              ranking: '509'
            
            - nameen: mobilenetv4_conv_small
              namezh: mobilenetv4_conv_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6224
              datasetA: '71.78'
              datasetB: '40.67'
              ranking: '510'
            
            - nameen: mobilenetv4_hybrid_large
              namezh: mobilenetv4_hybrid_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8003
              datasetA: '71.74'
              datasetB: '57.94'
              ranking: '511'
            
            - nameen: mobilenetv4_hybrid_large_075
              namezh: mobilenetv4_hybrid_large_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1019
              datasetA: '71.72'
              datasetB: '58.24'
              ranking: '512'
            
            - nameen: mobilenetv4_hybrid_medium
              namezh: mobilenetv4_hybrid_medium
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8098
              datasetA: '71.72'
              datasetB: '62.29'
              ranking: '513'
            
            - nameen: mobilenetv4_hybrid_medium_075
              namezh: mobilenetv4_hybrid_medium_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8576
              datasetA: '71.71'
              datasetB: '40.48'
              ranking: '514'
            
            - nameen: mobileone_s0
              namezh: mobileone_s0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9920
              datasetA: '71.7'
              datasetB: '61.04'
              ranking: '515'
            
            - nameen: mobileone_s1
              namezh: mobileone_s1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1771
              datasetA: '71.66'
              datasetB: '61.63'
              ranking: '516'
            
            - nameen: mobileone_s2
              namezh: mobileone_s2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3813
              datasetA: '71.58'
              datasetB: '59.04'
              ranking: '517'
            
            - nameen: mobileone_s3
              namezh: mobileone_s3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1671
              datasetA: '71.57'
              datasetB: '47.61'
              ranking: '518'
            
            - nameen: mobileone_s4
              namezh: mobileone_s4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9777
              datasetA: '71.56'
              datasetB: '50.31'
              ranking: '519'
            
            - nameen: mobilevit_s
              namezh: mobilevit_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3690
              datasetA: '71.51'
              datasetB: '55.61'
              ranking: '520'
            
            - nameen: mobilevit_xs
              namezh: mobilevit_xs
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5608
              datasetA: '71.49'
              datasetB: '50.3'
              ranking: '521'
            
            - nameen: mobilevit_xxs
              namezh: mobilevit_xxs
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5680
              datasetA: '71.48'
              datasetB: '57.06'
              ranking: '522'
            
            - nameen: mobilevitv2_050
              namezh: mobilevitv2_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7462
              datasetA: '71.45'
              datasetB: '59.99'
              ranking: '523'
            
            - nameen: mobilevitv2_075
              namezh: mobilevitv2_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8309
              datasetA: '71.44'
              datasetB: '46.33'
              ranking: '524'
            
            - nameen: mobilevitv2_100
              namezh: mobilevitv2_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9182
              datasetA: '71.39'
              datasetB: '43.42'
              ranking: '525'
            
            - nameen: mobilevitv2_125
              namezh: mobilevitv2_125
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4221
              datasetA: '71.38'
              datasetB: '46.44'
              ranking: '526'
            
            - nameen: mobilevitv2_150
              namezh: mobilevitv2_150
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9712
              datasetA: '71.33'
              datasetB: '53.87'
              ranking: '527'
            
            - nameen: mobilevitv2_175
              namezh: mobilevitv2_175
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2432
              datasetA: '71.33'
              datasetB: '58.57'
              ranking: '528'
            
            - nameen: mobilevitv2_200
              namezh: mobilevitv2_200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9883
              datasetA: '71.33'
              datasetB: '50.91'
              ranking: '529'
            
            - nameen: mvitv2_base
              namezh: mvitv2_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1228
              datasetA: '71.31'
              datasetB: '47.33'
              ranking: '530'
            
            - nameen: mvitv2_base_cls
              namezh: mvitv2_base_cls
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9186
              datasetA: '71.29'
              datasetB: '46.58'
              ranking: '531'
            
            - nameen: mvitv2_huge_cls
              namezh: mvitv2_huge_cls
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2772
              datasetA: '71.23'
              datasetB: '49.57'
              ranking: '532'
            
            - nameen: mvitv2_large
              namezh: mvitv2_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1268
              datasetA: '71.21'
              datasetB: '42.76'
              ranking: '533'
            
            - nameen: mvitv2_large_cls
              namezh: mvitv2_large_cls
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4266
              datasetA: '71.19'
              datasetB: '42.95'
              ranking: '534'
            
            - nameen: mvitv2_small
              namezh: mvitv2_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2628
              datasetA: '71.16'
              datasetB: '47.6'
              ranking: '535'
            
            - nameen: mvitv2_small_cls
              namezh: mvitv2_small_cls
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3703
              datasetA: '71.12'
              datasetB: '60.36'
              ranking: '536'
            
            - nameen: mvitv2_tiny
              namezh: mvitv2_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8629
              datasetA: '71.12'
              datasetB: '42.08'
              ranking: '537'
            
            - nameen: nasnetalarge
              namezh: nasnetalarge
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8819
              datasetA: '71.1'
              datasetB: '48.3'
              ranking: '538'
            
            - nameen: nest_base
              namezh: nest_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7054
              datasetA: '71.08'
              datasetB: '50.99'
              ranking: '539'
            
            - nameen: nest_base_jx
              namezh: nest_base_jx
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7825
              datasetA: '71.07'
              datasetB: '53.54'
              ranking: '540'
            
            - nameen: nest_small
              namezh: nest_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8095
              datasetA: '71.06'
              datasetB: '60.19'
              ranking: '541'
            
            - nameen: nest_small_jx
              namezh: nest_small_jx
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3710
              datasetA: '71.04'
              datasetB: '58.42'
              ranking: '542'
            
            - nameen: nest_tiny
              namezh: nest_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1085
              datasetA: '70.95'
              datasetB: '55.74'
              ranking: '543'
            
            - nameen: nest_tiny_jx
              namezh: nest_tiny_jx
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5286
              datasetA: '70.95'
              datasetB: '46.9'
              ranking: '544'
            
            - nameen: nextvit_base
              namezh: nextvit_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6618
              datasetA: '70.92'
              datasetB: '63.09'
              ranking: '545'
            
            - nameen: nextvit_large
              namezh: nextvit_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5326
              datasetA: '70.9'
              datasetB: '43.27'
              ranking: '546'
            
            - nameen: nextvit_small
              namezh: nextvit_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3003
              datasetA: '70.9'
              datasetB: '47.51'
              ranking: '547'
            
            - nameen: nf_ecaresnet26
              namezh: nf_ecaresnet26
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2744
              datasetA: '70.82'
              datasetB: '49.38'
              ranking: '548'
            
            - nameen: nf_ecaresnet50
              namezh: nf_ecaresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4806
              datasetA: '70.79'
              datasetB: '40.83'
              ranking: '549'
            
            - nameen: nf_ecaresnet101
              namezh: nf_ecaresnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1603
              datasetA: '70.78'
              datasetB: '50.75'
              ranking: '550'
            
            - nameen: nf_regnet_b0
              namezh: nf_regnet_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9512
              datasetA: '70.77'
              datasetB: '57.12'
              ranking: '551'
            
            - nameen: nf_regnet_b1
              namezh: nf_regnet_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3509
              datasetA: '70.75'
              datasetB: '44.29'
              ranking: '552'
            
            - nameen: nf_regnet_b2
              namezh: nf_regnet_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1763
              datasetA: '70.74'
              datasetB: '51.18'
              ranking: '553'
            
            - nameen: nf_regnet_b3
              namezh: nf_regnet_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9792
              datasetA: '70.72'
              datasetB: '57.53'
              ranking: '554'
            
            - nameen: nf_regnet_b4
              namezh: nf_regnet_b4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9739
              datasetA: '70.71'
              datasetB: '51.1'
              ranking: '555'
            
            - nameen: nf_regnet_b5
              namezh: nf_regnet_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3900
              datasetA: '70.7'
              datasetB: '58.32'
              ranking: '556'
            
            - nameen: nf_resnet26
              namezh: nf_resnet26
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2201
              datasetA: '70.63'
              datasetB: '56.54'
              ranking: '557'
            
            - nameen: nf_resnet50
              namezh: nf_resnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5492
              datasetA: '70.61'
              datasetB: '53.13'
              ranking: '558'
            
            - nameen: nf_resnet101
              namezh: nf_resnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4862
              datasetA: '70.59'
              datasetB: '46.75'
              ranking: '559'
            
            - nameen: nf_seresnet26
              namezh: nf_seresnet26
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3835
              datasetA: '70.59'
              datasetB: '52.13'
              ranking: '560'
            
            - nameen: nf_seresnet50
              namezh: nf_seresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7903
              datasetA: '70.55'
              datasetB: '43.21'
              ranking: '561'
            
            - nameen: nf_seresnet101
              namezh: nf_seresnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5791
              datasetA: '70.55'
              datasetB: '42.58'
              ranking: '562'
            
            - nameen: nfnet_f0
              namezh: nfnet_f0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5243
              datasetA: '70.53'
              datasetB: '48.32'
              ranking: '563'
            
            - nameen: nfnet_f1
              namezh: nfnet_f1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3235
              datasetA: '70.53'
              datasetB: '52.88'
              ranking: '564'
            
            - nameen: nfnet_f2
              namezh: nfnet_f2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2225
              datasetA: '70.52'
              datasetB: '54.6'
              ranking: '565'
            
            - nameen: nfnet_f3
              namezh: nfnet_f3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4161
              datasetA: '70.52'
              datasetB: '58.5'
              ranking: '566'
            
            - nameen: nfnet_f4
              namezh: nfnet_f4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4667
              datasetA: '70.52'
              datasetB: '54.83'
              ranking: '567'
            
            - nameen: nfnet_f5
              namezh: nfnet_f5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6168
              datasetA: '70.47'
              datasetB: '50.87'
              ranking: '568'
            
            - nameen: nfnet_f6
              namezh: nfnet_f6
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6623
              datasetA: '70.47'
              datasetB: '42.17'
              ranking: '569'
            
            - nameen: nfnet_f7
              namezh: nfnet_f7
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4085
              datasetA: '70.47'
              datasetB: '56.8'
              ranking: '570'
            
            - nameen: nfnet_l0
              namezh: nfnet_l0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7558
              datasetA: '70.44'
              datasetB: '48.32'
              ranking: '571'
            
            - nameen: pit_b_224
              namezh: pit_b_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7600
              datasetA: '70.43'
              datasetB: '45.39'
              ranking: '572'
            
            - nameen: pit_b_distilled_224
              namezh: pit_b_distilled_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2253
              datasetA: '70.42'
              datasetB: '54.6'
              ranking: '573'
            
            - nameen: pit_s_224
              namezh: pit_s_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4631
              datasetA: '70.42'
              datasetB: '62.74'
              ranking: '574'
            
            - nameen: pit_s_distilled_224
              namezh: pit_s_distilled_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2106
              datasetA: '70.4'
              datasetB: '57.36'
              ranking: '575'
            
            - nameen: pit_ti_224
              namezh: pit_ti_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6303
              datasetA: '70.33'
              datasetB: '54.18'
              ranking: '576'
            
            - nameen: pit_ti_distilled_224
              namezh: pit_ti_distilled_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3489
              datasetA: '70.31'
              datasetB: '61.99'
              ranking: '577'
            
            - nameen: pit_xs_224
              namezh: pit_xs_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8485
              datasetA: '70.29'
              datasetB: '51.74'
              ranking: '578'
            
            - nameen: pit_xs_distilled_224
              namezh: pit_xs_distilled_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5652
              datasetA: '70.27'
              datasetB: '47.99'
              ranking: '579'
            
            - nameen: pnasnet5large
              namezh: pnasnet5large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2539
              datasetA: '70.24'
              datasetB: '61.86'
              ranking: '580'
            
            - nameen: poolformer_m36
              namezh: poolformer_m36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4658
              datasetA: '70.22'
              datasetB: '53.97'
              ranking: '581'
            
            - nameen: poolformer_m48
              namezh: poolformer_m48
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2203
              datasetA: '70.21'
              datasetB: '58.49'
              ranking: '582'
            
            - nameen: poolformer_s12
              namezh: poolformer_s12
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4464
              datasetA: '70.21'
              datasetB: '41.34'
              ranking: '583'
            
            - nameen: poolformer_s24
              namezh: poolformer_s24
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3951
              datasetA: '70.2'
              datasetB: '61.84'
              ranking: '584'
            
            - nameen: poolformer_s36
              namezh: poolformer_s36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8382
              datasetA: '70.12'
              datasetB: '59.7'
              ranking: '585'
            
            - nameen: poolformerv2_m36
              namezh: poolformerv2_m36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1484
              datasetA: '70.07'
              datasetB: '62.08'
              ranking: '586'
            
            - nameen: poolformerv2_m48
              namezh: poolformerv2_m48
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9232
              datasetA: '70.06'
              datasetB: '52.01'
              ranking: '587'
            
            - nameen: poolformerv2_s12
              namezh: poolformerv2_s12
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7075
              datasetA: '70.03'
              datasetB: '47.6'
              ranking: '588'
            
            - nameen: poolformerv2_s24
              namezh: poolformerv2_s24
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5177
              datasetA: '70.02'
              datasetB: '48.45'
              ranking: '589'
            
            - nameen: poolformerv2_s36
              namezh: poolformerv2_s36
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2530
              datasetA: '69.98'
              datasetB: '49.36'
              ranking: '590'
            
            - nameen: pvt_v2_b0
              namezh: pvt_v2_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3729
              datasetA: '69.98'
              datasetB: '48.19'
              ranking: '591'
            
            - nameen: pvt_v2_b1
              namezh: pvt_v2_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5221
              datasetA: '69.96'
              datasetB: '48.4'
              ranking: '592'
            
            - nameen: pvt_v2_b2
              namezh: pvt_v2_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8045
              datasetA: '69.89'
              datasetB: '58.55'
              ranking: '593'
            
            - nameen: pvt_v2_b2_li
              namezh: pvt_v2_b2_li
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6834
              datasetA: '69.84'
              datasetB: '62.81'
              ranking: '594'
            
            - nameen: pvt_v2_b3
              namezh: pvt_v2_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1765
              datasetA: '69.78'
              datasetB: '48.11'
              ranking: '595'
            
            - nameen: pvt_v2_b4
              namezh: pvt_v2_b4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1130
              datasetA: '69.78'
              datasetB: '49.55'
              ranking: '596'
            
            - nameen: pvt_v2_b5
              namezh: pvt_v2_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2295
              datasetA: '69.74'
              datasetB: '58.65'
              ranking: '597'
            
            - nameen: regnetv_040
              namezh: regnetv_040
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4692
              datasetA: '69.74'
              datasetB: '53.93'
              ranking: '598'
            
            - nameen: regnetv_064
              namezh: regnetv_064
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7907
              datasetA: '69.74'
              datasetB: '50.1'
              ranking: '599'
            
            - nameen: regnetx_002
              namezh: regnetx_002
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4401
              datasetA: '69.73'
              datasetB: '54.9'
              ranking: '600'
            
            - nameen: regnetx_004
              namezh: regnetx_004
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9231
              datasetA: '69.71'
              datasetB: '52.55'
              ranking: '601'
            
            - nameen: regnetx_004_tv
              namezh: regnetx_004_tv
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6802
              datasetA: '69.7'
              datasetB: '47.39'
              ranking: '602'
            
            - nameen: regnetx_006
              namezh: regnetx_006
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4774
              datasetA: '69.68'
              datasetB: '46.59'
              ranking: '603'
            
            - nameen: regnetx_008
              namezh: regnetx_008
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7644
              datasetA: '69.67'
              datasetB: '45.31'
              ranking: '604'
            
            - nameen: regnetx_016
              namezh: regnetx_016
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2266
              datasetA: '69.6'
              datasetB: '45.04'
              ranking: '605'
            
            - nameen: regnetx_032
              namezh: regnetx_032
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1489
              datasetA: '69.6'
              datasetB: '61.51'
              ranking: '606'
            
            - nameen: regnetx_040
              namezh: regnetx_040
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5283
              datasetA: '69.57'
              datasetB: '55.15'
              ranking: '607'
            
            - nameen: regnetx_064
              namezh: regnetx_064
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6429
              datasetA: '69.56'
              datasetB: '48.99'
              ranking: '608'
            
            - nameen: regnetx_080
              namezh: regnetx_080
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3265
              datasetA: '69.55'
              datasetB: '58.16'
              ranking: '609'
            
            - nameen: regnetx_120
              namezh: regnetx_120
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1078
              datasetA: '69.53'
              datasetB: '47.83'
              ranking: '610'
            
            - nameen: regnetx_160
              namezh: regnetx_160
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3772
              datasetA: '69.52'
              datasetB: '48.99'
              ranking: '611'
            
            - nameen: regnetx_320
              namezh: regnetx_320
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8223
              datasetA: '69.5'
              datasetB: '59.28'
              ranking: '612'
            
            - nameen: regnety_002
              namezh: regnety_002
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1823
              datasetA: '69.47'
              datasetB: '52.48'
              ranking: '613'
            
            - nameen: regnety_004
              namezh: regnety_004
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9341
              datasetA: '69.45'
              datasetB: '49.75'
              ranking: '614'
            
            - nameen: regnety_006
              namezh: regnety_006
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8899
              datasetA: '69.43'
              datasetB: '48.08'
              ranking: '615'
            
            - nameen: regnety_008
              namezh: regnety_008
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3471
              datasetA: '69.43'
              datasetB: '43.16'
              ranking: '616'
            
            - nameen: regnety_008_tv
              namezh: regnety_008_tv
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3546
              datasetA: '69.42'
              datasetB: '42.49'
              ranking: '617'
            
            - nameen: regnety_016
              namezh: regnety_016
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1200
              datasetA: '69.41'
              datasetB: '61.17'
              ranking: '618'
            
            - nameen: regnety_032
              namezh: regnety_032
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2240
              datasetA: '69.41'
              datasetB: '58.23'
              ranking: '619'
            
            - nameen: regnety_040
              namezh: regnety_040
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5905
              datasetA: '69.4'
              datasetB: '44.23'
              ranking: '620'
            
            - nameen: regnety_040_sgn
              namezh: regnety_040_sgn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4351
              datasetA: '69.38'
              datasetB: '52.77'
              ranking: '621'
            
            - nameen: regnety_064
              namezh: regnety_064
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7922
              datasetA: '69.35'
              datasetB: '51.61'
              ranking: '622'
            
            - nameen: regnety_080
              namezh: regnety_080
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9641
              datasetA: '69.34'
              datasetB: '52.64'
              ranking: '623'
            
            - nameen: regnety_080_tv
              namezh: regnety_080_tv
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4850
              datasetA: '69.32'
              datasetB: '49.41'
              ranking: '624'
            
            - nameen: regnety_120
              namezh: regnety_120
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2259
              datasetA: '69.31'
              datasetB: '59.94'
              ranking: '625'
            
            - nameen: regnety_160
              namezh: regnety_160
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8301
              datasetA: '69.26'
              datasetB: '61.3'
              ranking: '626'
            
            - nameen: regnety_320
              namezh: regnety_320
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9497
              datasetA: '69.25'
              datasetB: '62.2'
              ranking: '627'
            
            - nameen: regnety_640
              namezh: regnety_640
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8147
              datasetA: '69.25'
              datasetB: '61.51'
              ranking: '628'
            
            - nameen: regnety_1280
              namezh: regnety_1280
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6017
              datasetA: '69.21'
              datasetB: '42.81'
              ranking: '629'
            
            - nameen: regnety_2560
              namezh: regnety_2560
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9901
              datasetA: '69.21'
              datasetB: '55.17'
              ranking: '630'
            
            - nameen: regnetz_005
              namezh: regnetz_005
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2385
              datasetA: '69.2'
              datasetB: '58.59'
              ranking: '631'
            
            - nameen: regnetz_040
              namezh: regnetz_040
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4898
              datasetA: '69.19'
              datasetB: '51.86'
              ranking: '632'
            
            - nameen: regnetz_040_h
              namezh: regnetz_040_h
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8650
              datasetA: '69.16'
              datasetB: '62.21'
              ranking: '633'
            
            - nameen: regnetz_b16
              namezh: regnetz_b16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9363
              datasetA: '69.15'
              datasetB: '57.76'
              ranking: '634'
            
            - nameen: regnetz_b16_evos
              namezh: regnetz_b16_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2978
              datasetA: '69.14'
              datasetB: '54.7'
              ranking: '635'
            
            - nameen: regnetz_c16
              namezh: regnetz_c16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8305
              datasetA: '69.12'
              datasetB: '61.76'
              ranking: '636'
            
            - nameen: regnetz_c16_evos
              namezh: regnetz_c16_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8891
              datasetA: '69.12'
              datasetB: '48.43'
              ranking: '637'
            
            - nameen: regnetz_d8
              namezh: regnetz_d8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8533
              datasetA: '69.1'
              datasetB: '49.82'
              ranking: '638'
            
            - nameen: regnetz_d8_evos
              namezh: regnetz_d8_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7239
              datasetA: '69.1'
              datasetB: '46.0'
              ranking: '639'
            
            - nameen: regnetz_d32
              namezh: regnetz_d32
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2259
              datasetA: '69.07'
              datasetB: '40.69'
              ranking: '640'
            
            - nameen: regnetz_e8
              namezh: regnetz_e8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9144
              datasetA: '69.07'
              datasetB: '58.44'
              ranking: '641'
            
            - nameen: repghostnet_050
              namezh: repghostnet_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7095
              datasetA: '69.05'
              datasetB: '49.1'
              ranking: '642'
            
            - nameen: repghostnet_058
              namezh: repghostnet_058
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1865
              datasetA: '69.02'
              datasetB: '47.96'
              ranking: '643'
            
            - nameen: repghostnet_080
              namezh: repghostnet_080
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1853
              datasetA: '69.01'
              datasetB: '59.77'
              ranking: '644'
            
            - nameen: repghostnet_100
              namezh: repghostnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9590
              datasetA: '69.0'
              datasetB: '42.56'
              ranking: '645'
            
            - nameen: repghostnet_111
              namezh: repghostnet_111
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7551
              datasetA: '68.97'
              datasetB: '56.46'
              ranking: '646'
            
            - nameen: repghostnet_130
              namezh: repghostnet_130
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5622
              datasetA: '68.97'
              datasetB: '59.67'
              ranking: '647'
            
            - nameen: repghostnet_150
              namezh: repghostnet_150
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6225
              datasetA: '68.97'
              datasetB: '58.16'
              ranking: '648'
            
            - nameen: repghostnet_200
              namezh: repghostnet_200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4964
              datasetA: '68.97'
              datasetB: '55.5'
              ranking: '649'
            
            - nameen: repvgg_a0
              namezh: repvgg_a0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5079
              datasetA: '68.95'
              datasetB: '57.96'
              ranking: '650'
            
            - nameen: repvgg_a1
              namezh: repvgg_a1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6004
              datasetA: '68.89'
              datasetB: '57.21'
              ranking: '651'
            
            - nameen: repvgg_a2
              namezh: repvgg_a2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3674
              datasetA: '68.88'
              datasetB: '62.51'
              ranking: '652'
            
            - nameen: repvgg_b0
              namezh: repvgg_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7019
              datasetA: '68.87'
              datasetB: '55.26'
              ranking: '653'
            
            - nameen: repvgg_b1
              namezh: repvgg_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5232
              datasetA: '68.86'
              datasetB: '43.97'
              ranking: '654'
            
            - nameen: repvgg_b1g4
              namezh: repvgg_b1g4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4667
              datasetA: '68.85'
              datasetB: '52.48'
              ranking: '655'
            
            - nameen: repvgg_b2
              namezh: repvgg_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2305
              datasetA: '68.85'
              datasetB: '46.47'
              ranking: '656'
            
            - nameen: repvgg_b2g4
              namezh: repvgg_b2g4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3349
              datasetA: '68.8'
              datasetB: '49.86'
              ranking: '657'
            
            - nameen: repvgg_b3
              namezh: repvgg_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1913
              datasetA: '68.79'
              datasetB: '43.5'
              ranking: '658'
            
            - nameen: repvgg_b3g4
              namezh: repvgg_b3g4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2377
              datasetA: '68.77'
              datasetB: '52.21'
              ranking: '659'
            
            - nameen: repvgg_d2se
              namezh: repvgg_d2se
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5576
              datasetA: '68.76'
              datasetB: '49.83'
              ranking: '660'
            
            - nameen: repvit_m0_9
              namezh: repvit_m0_9
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3935
              datasetA: '68.73'
              datasetB: '49.99'
              ranking: '661'
            
            - nameen: repvit_m1
              namezh: repvit_m1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6526
              datasetA: '68.72'
              datasetB: '43.56'
              ranking: '662'
            
            - nameen: repvit_m1_0
              namezh: repvit_m1_0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3747
              datasetA: '68.72'
              datasetB: '62.89'
              ranking: '663'
            
            - nameen: repvit_m1_1
              namezh: repvit_m1_1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2558
              datasetA: '68.7'
              datasetB: '49.9'
              ranking: '664'
            
            - nameen: repvit_m1_5
              namezh: repvit_m1_5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7003
              datasetA: '68.69'
              datasetB: '61.57'
              ranking: '665'
            
            - nameen: repvit_m2
              namezh: repvit_m2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8824
              datasetA: '68.67'
              datasetB: '54.94'
              ranking: '666'
            
            - nameen: repvit_m2_3
              namezh: repvit_m2_3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3979
              datasetA: '68.66'
              datasetB: '60.23'
              ranking: '667'
            
            - nameen: repvit_m3
              namezh: repvit_m3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1867
              datasetA: '68.64'
              datasetB: '42.17'
              ranking: '668'
            
            - nameen: res2net50_14w_8s
              namezh: res2net50_14w_8s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7600
              datasetA: '68.6'
              datasetB: '44.47'
              ranking: '669'
            
            - nameen: res2net50_26w_4s
              namezh: res2net50_26w_4s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6992
              datasetA: '68.56'
              datasetB: '54.66'
              ranking: '670'
            
            - nameen: res2net50_26w_6s
              namezh: res2net50_26w_6s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3356
              datasetA: '68.56'
              datasetB: '40.55'
              ranking: '671'
            
            - nameen: res2net50_26w_8s
              namezh: res2net50_26w_8s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7439
              datasetA: '68.53'
              datasetB: '51.36'
              ranking: '672'
            
            - nameen: res2net50_48w_2s
              namezh: res2net50_48w_2s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1132
              datasetA: '68.5'
              datasetB: '42.67'
              ranking: '673'
            
            - nameen: res2net50d
              namezh: res2net50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7501
              datasetA: '68.5'
              datasetB: '47.92'
              ranking: '674'
            
            - nameen: res2net101_26w_4s
              namezh: res2net101_26w_4s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4342
              datasetA: '68.49'
              datasetB: '48.93'
              ranking: '675'
            
            - nameen: res2net101d
              namezh: res2net101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8893
              datasetA: '68.48'
              datasetB: '45.71'
              ranking: '676'
            
            - nameen: res2next50
              namezh: res2next50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5210
              datasetA: '68.48'
              datasetB: '49.4'
              ranking: '677'
            
            - nameen: resmlp_12_224
              namezh: resmlp_12_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2802
              datasetA: '68.46'
              datasetB: '59.11'
              ranking: '678'
            
            - nameen: resmlp_24_224
              namezh: resmlp_24_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7035
              datasetA: '68.45'
              datasetB: '40.02'
              ranking: '679'
            
            - nameen: resmlp_36_224
              namezh: resmlp_36_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2889
              datasetA: '68.42'
              datasetB: '50.34'
              ranking: '680'
            
            - nameen: resmlp_big_24_224
              namezh: resmlp_big_24_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8983
              datasetA: '68.41'
              datasetB: '60.31'
              ranking: '681'
            
            - nameen: resnest14d
              namezh: resnest14d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2288
              datasetA: '68.4'
              datasetB: '52.62'
              ranking: '682'
            
            - nameen: resnest26d
              namezh: resnest26d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7264
              datasetA: '68.4'
              datasetB: '60.78'
              ranking: '683'
            
            - nameen: resnest50d
              namezh: resnest50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5384
              datasetA: '68.4'
              datasetB: '43.92'
              ranking: '684'
            
            - nameen: resnest50d_1s4x24d
              namezh: resnest50d_1s4x24d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8287
              datasetA: '68.36'
              datasetB: '58.54'
              ranking: '685'
            
            - nameen: resnest50d_4s2x40d
              namezh: resnest50d_4s2x40d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6975
              datasetA: '68.35'
              datasetB: '56.18'
              ranking: '686'
            
            - nameen: resnest101e
              namezh: resnest101e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7552
              datasetA: '68.34'
              datasetB: '51.78'
              ranking: '687'
            
            - nameen: resnest200e
              namezh: resnest200e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7738
              datasetA: '68.34'
              datasetB: '49.57'
              ranking: '688'
            
            - nameen: resnest269e
              namezh: resnest269e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5096
              datasetA: '68.33'
              datasetB: '50.77'
              ranking: '689'
            
            - nameen: resnet10t
              namezh: resnet10t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8604
              datasetA: '68.19'
              datasetB: '55.29'
              ranking: '690'
            
            - nameen: resnet14t
              namezh: resnet14t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8802
              datasetA: '68.19'
              datasetB: '46.25'
              ranking: '691'
            
            - nameen: resnet18
              namezh: resnet18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7862
              datasetA: '68.14'
              datasetB: '58.39'
              ranking: '692'
            
            - nameen: resnet18d
              namezh: resnet18d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4596
              datasetA: '68.13'
              datasetB: '50.69'
              ranking: '693'
            
            - nameen: resnet26
              namezh: resnet26
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1329
              datasetA: '68.12'
              datasetB: '55.81'
              ranking: '694'
            
            - nameen: resnet26d
              namezh: resnet26d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1731
              datasetA: '68.12'
              datasetB: '48.24'
              ranking: '695'
            
            - nameen: resnet26t
              namezh: resnet26t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3201
              datasetA: '68.09'
              datasetB: '44.37'
              ranking: '696'
            
            - nameen: resnet32ts
              namezh: resnet32ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7391
              datasetA: '68.07'
              datasetB: '46.05'
              ranking: '697'
            
            - nameen: resnet33ts
              namezh: resnet33ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5279
              datasetA: '68.06'
              datasetB: '44.03'
              ranking: '698'
            
            - nameen: resnet34
              namezh: resnet34
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3111
              datasetA: '68.01'
              datasetB: '42.04'
              ranking: '699'
            
            - nameen: resnet34d
              namezh: resnet34d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9131
              datasetA: '68.0'
              datasetB: '61.58'
              ranking: '700'
            
            - nameen: resnet50
              namezh: resnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7092
              datasetA: '68.0'
              datasetB: '56.51'
              ranking: '701'
            
            - nameen: resnet50_clip
              namezh: resnet50_clip
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3920
              datasetA: '67.96'
              datasetB: '44.68'
              ranking: '702'
            
            - nameen: resnet50_clip_gap
              namezh: resnet50_clip_gap
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2266
              datasetA: '67.93'
              datasetB: '54.36'
              ranking: '703'
            
            - nameen: resnet50_gn
              namezh: resnet50_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5419
              datasetA: '67.92'
              datasetB: '42.87'
              ranking: '704'
            
            - nameen: resnet50_mlp
              namezh: resnet50_mlp
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8749
              datasetA: '67.91'
              datasetB: '47.47'
              ranking: '705'
            
            - nameen: resnet50c
              namezh: resnet50c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2565
              datasetA: '67.89'
              datasetB: '59.3'
              ranking: '706'
            
            - nameen: resnet50d
              namezh: resnet50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2827
              datasetA: '67.83'
              datasetB: '59.96'
              ranking: '707'
            
            - nameen: resnet50s
              namezh: resnet50s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7979
              datasetA: '67.75'
              datasetB: '43.67'
              ranking: '708'
            
            - nameen: resnet50t
              namezh: resnet50t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7718
              datasetA: '67.73'
              datasetB: '41.92'
              ranking: '709'
            
            - nameen: resnet50x4_clip
              namezh: resnet50x4_clip
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5983
              datasetA: '67.7'
              datasetB: '59.34'
              ranking: '710'
            
            - nameen: resnet50x4_clip_gap
              namezh: resnet50x4_clip_gap
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5736
              datasetA: '67.7'
              datasetB: '42.02'
              ranking: '711'
            
            - nameen: resnet50x16_clip
              namezh: resnet50x16_clip
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7639
              datasetA: '67.67'
              datasetB: '46.02'
              ranking: '712'
            
            - nameen: resnet50x16_clip_gap
              namezh: resnet50x16_clip_gap
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7105
              datasetA: '67.61'
              datasetB: '47.6'
              ranking: '713'
            
            - nameen: resnet50x64_clip
              namezh: resnet50x64_clip
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3030
              datasetA: '67.61'
              datasetB: '57.78'
              ranking: '714'
            
            - nameen: resnet50x64_clip_gap
              namezh: resnet50x64_clip_gap
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3240
              datasetA: '67.6'
              datasetB: '48.46'
              ranking: '715'
            
            - nameen: resnet51q
              namezh: resnet51q
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6270
              datasetA: '67.58'
              datasetB: '56.01'
              ranking: '716'
            
            - nameen: resnet61q
              namezh: resnet61q
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3229
              datasetA: '67.54'
              datasetB: '62.28'
              ranking: '717'
            
            - nameen: resnet101
              namezh: resnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4483
              datasetA: '67.49'
              datasetB: '63.36'
              ranking: '718'
            
            - nameen: resnet101_clip
              namezh: resnet101_clip
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9962
              datasetA: '67.48'
              datasetB: '55.93'
              ranking: '719'
            
            - nameen: resnet101_clip_gap
              namezh: resnet101_clip_gap
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8165
              datasetA: '67.47'
              datasetB: '43.96'
              ranking: '720'
            
            - nameen: resnet101c
              namezh: resnet101c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4496
              datasetA: '67.45'
              datasetB: '50.7'
              ranking: '721'
            
            - nameen: resnet101d
              namezh: resnet101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2604
              datasetA: '67.44'
              datasetB: '49.17'
              ranking: '722'
            
            - nameen: resnet101s
              namezh: resnet101s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2444
              datasetA: '67.38'
              datasetB: '56.28'
              ranking: '723'
            
            - nameen: resnet152
              namezh: resnet152
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2680
              datasetA: '67.35'
              datasetB: '45.01'
              ranking: '724'
            
            - nameen: resnet152c
              namezh: resnet152c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8219
              datasetA: '67.35'
              datasetB: '43.7'
              ranking: '725'
            
            - nameen: resnet152d
              namezh: resnet152d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6612
              datasetA: '67.34'
              datasetB: '46.02'
              ranking: '726'
            
            - nameen: resnet152s
              namezh: resnet152s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4184
              datasetA: '67.33'
              datasetB: '49.21'
              ranking: '727'
            
            - nameen: resnet200
              namezh: resnet200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4531
              datasetA: '67.29'
              datasetB: '54.44'
              ranking: '728'
            
            - nameen: resnet200d
              namezh: resnet200d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5454
              datasetA: '67.26'
              datasetB: '55.73'
              ranking: '729'
            
            - nameen: resnetaa34d
              namezh: resnetaa34d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7475
              datasetA: '67.25'
              datasetB: '43.86'
              ranking: '730'
            
            - nameen: resnetaa50
              namezh: resnetaa50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4252
              datasetA: '67.23'
              datasetB: '61.29'
              ranking: '731'
            
            - nameen: resnetaa50d
              namezh: resnetaa50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4617
              datasetA: '67.22'
              datasetB: '41.57'
              ranking: '732'
            
            - nameen: resnetaa101d
              namezh: resnetaa101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5058
              datasetA: '67.21'
              datasetB: '52.7'
              ranking: '733'
            
            - nameen: resnetblur18
              namezh: resnetblur18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5984
              datasetA: '67.21'
              datasetB: '40.08'
              ranking: '734'
            
            - nameen: resnetblur50
              namezh: resnetblur50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5985
              datasetA: '67.2'
              datasetB: '43.66'
              ranking: '735'
            
            - nameen: resnetblur50d
              namezh: resnetblur50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3723
              datasetA: '67.19'
              datasetB: '59.62'
              ranking: '736'
            
            - nameen: resnetblur101d
              namezh: resnetblur101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1469
              datasetA: '67.18'
              datasetB: '52.42'
              ranking: '737'
            
            - nameen: resnetrs50
              namezh: resnetrs50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8304
              datasetA: '67.18'
              datasetB: '57.86'
              ranking: '738'
            
            - nameen: resnetrs101
              namezh: resnetrs101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9087
              datasetA: '67.17'
              datasetB: '58.34'
              ranking: '739'
            
            - nameen: resnetrs152
              namezh: resnetrs152
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6249
              datasetA: '67.15'
              datasetB: '47.23'
              ranking: '740'
            
            - nameen: resnetrs200
              namezh: resnetrs200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5077
              datasetA: '67.13'
              datasetB: '55.08'
              ranking: '741'
            
            - nameen: resnetrs270
              namezh: resnetrs270
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5926
              datasetA: '67.1'
              datasetB: '60.31'
              ranking: '742'
            
            - nameen: resnetrs350
              namezh: resnetrs350
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8298
              datasetA: '67.02'
              datasetB: '58.45'
              ranking: '743'
            
            - nameen: resnetrs420
              namezh: resnetrs420
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1908
              datasetA: '67.02'
              datasetB: '42.51'
              ranking: '744'
            
            - nameen: resnetv2_50
              namezh: resnetv2_50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9102
              datasetA: '67.01'
              datasetB: '52.94'
              ranking: '745'
            
            - nameen: resnetv2_50d
              namezh: resnetv2_50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2976
              datasetA: '67.01'
              datasetB: '53.95'
              ranking: '746'
            
            - nameen: resnetv2_50d_evos
              namezh: resnetv2_50d_evos
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9524
              datasetA: '66.98'
              datasetB: '49.79'
              ranking: '747'
            
            - nameen: resnetv2_50d_frn
              namezh: resnetv2_50d_frn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1665
              datasetA: '66.97'
              datasetB: '49.84'
              ranking: '748'
            
            - nameen: resnetv2_50d_gn
              namezh: resnetv2_50d_gn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7911
              datasetA: '66.96'
              datasetB: '42.66'
              ranking: '749'
            
            - nameen: resnetv2_50t
              namezh: resnetv2_50t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3500
              datasetA: '66.94'
              datasetB: '50.43'
              ranking: '750'
            
            - nameen: resnetv2_50x1_bit
              namezh: resnetv2_50x1_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8751
              datasetA: '66.94'
              datasetB: '51.38'
              ranking: '751'
            
            - nameen: resnetv2_50x3_bit
              namezh: resnetv2_50x3_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3710
              datasetA: '66.91'
              datasetB: '48.69'
              ranking: '752'
            
            - nameen: resnetv2_101
              namezh: resnetv2_101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3557
              datasetA: '66.85'
              datasetB: '60.16'
              ranking: '753'
            
            - nameen: resnetv2_101d
              namezh: resnetv2_101d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5287
              datasetA: '66.83'
              datasetB: '53.5'
              ranking: '754'
            
            - nameen: resnetv2_101x1_bit
              namezh: resnetv2_101x1_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2966
              datasetA: '66.8'
              datasetB: '57.72'
              ranking: '755'
            
            - nameen: resnetv2_101x3_bit
              namezh: resnetv2_101x3_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8294
              datasetA: '66.78'
              datasetB: '60.73'
              ranking: '756'
            
            - nameen: resnetv2_152
              namezh: resnetv2_152
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7162
              datasetA: '66.78'
              datasetB: '47.52'
              ranking: '757'
            
            - nameen: resnetv2_152d
              namezh: resnetv2_152d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4837
              datasetA: '66.77'
              datasetB: '61.5'
              ranking: '758'
            
            - nameen: resnetv2_152x2_bit
              namezh: resnetv2_152x2_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2054
              datasetA: '66.75'
              datasetB: '56.22'
              ranking: '759'
            
            - nameen: resnetv2_152x4_bit
              namezh: resnetv2_152x4_bit
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8656
              datasetA: '66.75'
              datasetB: '47.7'
              ranking: '760'
            
            - nameen: resnext26ts
              namezh: resnext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7197
              datasetA: '66.75'
              datasetB: '40.79'
              ranking: '761'
            
            - nameen: resnext50_32x4d
              namezh: resnext50_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4188
              datasetA: '66.74'
              datasetB: '44.02'
              ranking: '762'
            
            - nameen: resnext50d_32x4d
              namezh: resnext50d_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1995
              datasetA: '66.72'
              datasetB: '54.95'
              ranking: '763'
            
            - nameen: resnext101_32x4d
              namezh: resnext101_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1762
              datasetA: '66.72'
              datasetB: '53.54'
              ranking: '764'
            
            - nameen: resnext101_32x8d
              namezh: resnext101_32x8d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5668
              datasetA: '66.72'
              datasetB: '42.91'
              ranking: '765'
            
            - nameen: resnext101_32x16d
              namezh: resnext101_32x16d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1083
              datasetA: '66.67'
              datasetB: '52.62'
              ranking: '766'
            
            - nameen: resnext101_32x32d
              namezh: resnext101_32x32d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1769
              datasetA: '66.66'
              datasetB: '58.89'
              ranking: '767'
            
            - nameen: resnext101_64x4d
              namezh: resnext101_64x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7771
              datasetA: '66.66'
              datasetB: '45.77'
              ranking: '768'
            
            - nameen: rexnet_100
              namezh: rexnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7157
              datasetA: '66.65'
              datasetB: '53.46'
              ranking: '769'
            
            - nameen: rexnet_130
              namezh: rexnet_130
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6073
              datasetA: '66.64'
              datasetB: '47.75'
              ranking: '770'
            
            - nameen: rexnet_150
              namezh: rexnet_150
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1368
              datasetA: '66.62'
              datasetB: '61.96'
              ranking: '771'
            
            - nameen: rexnet_200
              namezh: rexnet_200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1308
              datasetA: '66.6'
              datasetB: '45.05'
              ranking: '772'
            
            - nameen: rexnet_300
              namezh: rexnet_300
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4318
              datasetA: '66.59'
              datasetB: '44.34'
              ranking: '773'
            
            - nameen: rexnetr_100
              namezh: rexnetr_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3038
              datasetA: '66.53'
              datasetB: '56.52'
              ranking: '774'
            
            - nameen: rexnetr_130
              namezh: rexnetr_130
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1334
              datasetA: '66.53'
              datasetB: '44.97'
              ranking: '775'
            
            - nameen: rexnetr_150
              namezh: rexnetr_150
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1351
              datasetA: '66.52'
              datasetB: '51.42'
              ranking: '776'
            
            - nameen: rexnetr_200
              namezh: rexnetr_200
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4058
              datasetA: '66.51'
              datasetB: '55.66'
              ranking: '777'
            
            - nameen: rexnetr_300
              namezh: rexnetr_300
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7238
              datasetA: '66.47'
              datasetB: '57.16'
              ranking: '778'
            
            - nameen: samvit_base_patch16
              namezh: samvit_base_patch16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3071
              datasetA: '66.46'
              datasetB: '43.87'
              ranking: '779'
            
            - nameen: samvit_base_patch16_224
              namezh: samvit_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1323
              datasetA: '66.39'
              datasetB: '51.99'
              ranking: '780'
            
            - nameen: samvit_huge_patch16
              namezh: samvit_huge_patch16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1694
              datasetA: '66.37'
              datasetB: '60.38'
              ranking: '781'
            
            - nameen: samvit_large_patch16
              namezh: samvit_large_patch16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7124
              datasetA: '66.36'
              datasetB: '62.79'
              ranking: '782'
            
            - nameen: sebotnet33ts_256
              namezh: sebotnet33ts_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5718
              datasetA: '66.34'
              datasetB: '57.34'
              ranking: '783'
            
            - nameen: sedarknet21
              namezh: sedarknet21
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8377
              datasetA: '66.29'
              datasetB: '54.66'
              ranking: '784'
            
            - nameen: sehalonet33ts
              namezh: sehalonet33ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2144
              datasetA: '66.28'
              datasetB: '44.11'
              ranking: '785'
            
            - nameen: selecsls42
              namezh: selecsls42
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4159
              datasetA: '66.26'
              datasetB: '50.24'
              ranking: '786'
            
            - nameen: selecsls42b
              namezh: selecsls42b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6079
              datasetA: '66.24'
              datasetB: '53.06'
              ranking: '787'
            
            - nameen: selecsls60
              namezh: selecsls60
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2882
              datasetA: '66.22'
              datasetB: '48.36'
              ranking: '788'
            
            - nameen: selecsls60b
              namezh: selecsls60b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2049
              datasetA: '66.21'
              datasetB: '49.46'
              ranking: '789'
            
            - nameen: selecsls84
              namezh: selecsls84
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5057
              datasetA: '66.21'
              datasetB: '54.66'
              ranking: '790'
            
            - nameen: semnasnet_050
              namezh: semnasnet_050
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5685
              datasetA: '66.19'
              datasetB: '47.08'
              ranking: '791'
            
            - nameen: semnasnet_075
              namezh: semnasnet_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9032
              datasetA: '66.18'
              datasetB: '50.79'
              ranking: '792'
            
            - nameen: semnasnet_100
              namezh: semnasnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4575
              datasetA: '66.15'
              datasetB: '41.41'
              ranking: '793'
            
            - nameen: semnasnet_140
              namezh: semnasnet_140
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4674
              datasetA: '66.14'
              datasetB: '50.39'
              ranking: '794'
            
            - nameen: senet154
              namezh: senet154
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1598
              datasetA: '66.13'
              datasetB: '55.95'
              ranking: '795'
            
            - nameen: sequencer2d_l
              namezh: sequencer2d_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5724
              datasetA: '66.12'
              datasetB: '59.64'
              ranking: '796'
            
            - nameen: sequencer2d_m
              namezh: sequencer2d_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9690
              datasetA: '66.1'
              datasetB: '41.62'
              ranking: '797'
            
            - nameen: sequencer2d_s
              namezh: sequencer2d_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7932
              datasetA: '66.02'
              datasetB: '57.26'
              ranking: '798'
            
            - nameen: seresnet18
              namezh: seresnet18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3918
              datasetA: '66.0'
              datasetB: '61.11'
              ranking: '799'
            
            - nameen: seresnet33ts
              namezh: seresnet33ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4241
              datasetA: '65.99'
              datasetB: '49.05'
              ranking: '800'
            
            - nameen: seresnet34
              namezh: seresnet34
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7075
              datasetA: '65.99'
              datasetB: '63.11'
              ranking: '801'
            
            - nameen: seresnet50
              namezh: seresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5660
              datasetA: '65.95'
              datasetB: '51.24'
              ranking: '802'
            
            - nameen: seresnet50t
              namezh: seresnet50t
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5759
              datasetA: '65.94'
              datasetB: '57.29'
              ranking: '803'
            
            - nameen: seresnet101
              namezh: seresnet101
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4587
              datasetA: '65.93'
              datasetB: '63.04'
              ranking: '804'
            
            - nameen: seresnet152
              namezh: seresnet152
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4717
              datasetA: '65.91'
              datasetB: '47.28'
              ranking: '805'
            
            - nameen: seresnet152d
              namezh: seresnet152d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5055
              datasetA: '65.88'
              datasetB: '51.11'
              ranking: '806'
            
            - nameen: seresnet200d
              namezh: seresnet200d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3746
              datasetA: '65.86'
              datasetB: '41.22'
              ranking: '807'
            
            - nameen: seresnet269d
              namezh: seresnet269d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3495
              datasetA: '65.85'
              datasetB: '52.64'
              ranking: '808'
            
            - nameen: seresnetaa50d
              namezh: seresnetaa50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6995
              datasetA: '65.85'
              datasetB: '45.28'
              ranking: '809'
            
            - nameen: seresnext26d_32x4d
              namezh: seresnext26d_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7636
              datasetA: '65.83'
              datasetB: '50.24'
              ranking: '810'
            
            - nameen: seresnext26t_32x4d
              namezh: seresnext26t_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2692
              datasetA: '65.81'
              datasetB: '40.84'
              ranking: '811'
            
            - nameen: seresnext26ts
              namezh: seresnext26ts
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1455
              datasetA: '65.8'
              datasetB: '59.85'
              ranking: '812'
            
            - nameen: seresnext50_32x4d
              namezh: seresnext50_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5030
              datasetA: '65.8'
              datasetB: '62.01'
              ranking: '813'
            
            - nameen: seresnext101_32x4d
              namezh: seresnext101_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4489
              datasetA: '65.8'
              datasetB: '62.79'
              ranking: '814'
            
            - nameen: seresnext101_32x8d
              namezh: seresnext101_32x8d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8648
              datasetA: '65.79'
              datasetB: '49.17'
              ranking: '815'
            
            - nameen: seresnext101_64x4d
              namezh: seresnext101_64x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2120
              datasetA: '65.78'
              datasetB: '62.54'
              ranking: '816'
            
            - nameen: seresnext101d_32x8d
              namezh: seresnext101d_32x8d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6843
              datasetA: '65.73'
              datasetB: '41.9'
              ranking: '817'
            
            - nameen: seresnextaa101d_32x8d
              namezh: seresnextaa101d_32x8d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9167
              datasetA: '65.72'
              datasetB: '49.0'
              ranking: '818'
            
            - nameen: seresnextaa201d_32x8d
              namezh: seresnextaa201d_32x8d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2771
              datasetA: '65.72'
              datasetB: '50.72'
              ranking: '819'
            
            - nameen: skresnet18
              namezh: skresnet18
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5204
              datasetA: '65.71'
              datasetB: '63.02'
              ranking: '820'
            
            - nameen: skresnet34
              namezh: skresnet34
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8721
              datasetA: '65.69'
              datasetB: '40.33'
              ranking: '821'
            
            - nameen: skresnet50
              namezh: skresnet50
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3213
              datasetA: '65.69'
              datasetB: '56.06'
              ranking: '822'
            
            - nameen: skresnet50d
              namezh: skresnet50d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4243
              datasetA: '65.67'
              datasetB: '45.45'
              ranking: '823'
            
            - nameen: skresnext50_32x4d
              namezh: skresnext50_32x4d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7316
              datasetA: '65.65'
              datasetB: '54.42'
              ranking: '824'
            
            - nameen: spnasnet_100
              namezh: spnasnet_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4973
              datasetA: '65.63'
              datasetB: '50.1'
              ranking: '825'
            
            - nameen: swin_base_patch4_window7_224
              namezh: swin_base_patch4_window7_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8420
              datasetA: '65.61'
              datasetB: '57.73'
              ranking: '826'
            
            - nameen: swin_base_patch4_window12_384
              namezh: swin_base_patch4_window12_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7456
              datasetA: '65.61'
              datasetB: '41.53'
              ranking: '827'
            
            - nameen: swin_large_patch4_window7_224
              namezh: swin_large_patch4_window7_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3363
              datasetA: '65.59'
              datasetB: '43.08'
              ranking: '828'
            
            - nameen: swin_large_patch4_window12_384
              namezh: swin_large_patch4_window12_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9463
              datasetA: '65.58'
              datasetB: '55.11'
              ranking: '829'
            
            - nameen: swin_s3_base_224
              namezh: swin_s3_base_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1151
              datasetA: '65.57'
              datasetB: '52.91'
              ranking: '830'
            
            - nameen: swin_s3_small_224
              namezh: swin_s3_small_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2182
              datasetA: '65.57'
              datasetB: '41.46'
              ranking: '831'
            
            - nameen: swin_s3_tiny_224
              namezh: swin_s3_tiny_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3147
              datasetA: '65.56'
              datasetB: '53.14'
              ranking: '832'
            
            - nameen: swin_small_patch4_window7_224
              namezh: swin_small_patch4_window7_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4346
              datasetA: '65.55'
              datasetB: '42.05'
              ranking: '833'
            
            - nameen: swin_tiny_patch4_window7_224
              namezh: swin_tiny_patch4_window7_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8831
              datasetA: '65.53'
              datasetB: '52.65'
              ranking: '834'
            
            - nameen: swinv2_base_window8_256
              namezh: swinv2_base_window8_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3335
              datasetA: '65.53'
              datasetB: '41.04'
              ranking: '835'
            
            - nameen: swinv2_base_window12_192
              namezh: swinv2_base_window12_192
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7518
              datasetA: '65.52'
              datasetB: '61.15'
              ranking: '836'
            
            - nameen: swinv2_base_window12to16_192to256
              namezh: swinv2_base_window12to16_192to256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9424
              datasetA: '65.52'
              datasetB: '46.9'
              ranking: '837'
            
            - nameen: swinv2_base_window12to24_192to384
              namezh: swinv2_base_window12to24_192to384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5648
              datasetA: '65.49'
              datasetB: '41.91'
              ranking: '838'
            
            - nameen: swinv2_base_window16_256
              namezh: swinv2_base_window16_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7746
              datasetA: '65.49'
              datasetB: '47.57'
              ranking: '839'
            
            - nameen: swinv2_cr_base_224
              namezh: swinv2_cr_base_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7327
              datasetA: '65.49'
              datasetB: '42.09'
              ranking: '840'
            
            - nameen: swinv2_cr_base_384
              namezh: swinv2_cr_base_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8949
              datasetA: '65.48'
              datasetB: '62.83'
              ranking: '841'
            
            - nameen: swinv2_cr_base_ns_224
              namezh: swinv2_cr_base_ns_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2365
              datasetA: '65.47'
              datasetB: '47.47'
              ranking: '842'
            
            - nameen: swinv2_cr_giant_224
              namezh: swinv2_cr_giant_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2220
              datasetA: '65.46'
              datasetB: '42.87'
              ranking: '843'
            
            - nameen: swinv2_cr_giant_384
              namezh: swinv2_cr_giant_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2514
              datasetA: '65.45'
              datasetB: '61.42'
              ranking: '844'
            
            - nameen: swinv2_cr_huge_224
              namezh: swinv2_cr_huge_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5896
              datasetA: '65.44'
              datasetB: '57.78'
              ranking: '845'
            
            - nameen: swinv2_cr_huge_384
              namezh: swinv2_cr_huge_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5648
              datasetA: '65.44'
              datasetB: '49.51'
              ranking: '846'
            
            - nameen: swinv2_cr_large_224
              namezh: swinv2_cr_large_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2935
              datasetA: '65.44'
              datasetB: '59.08'
              ranking: '847'
            
            - nameen: swinv2_cr_large_384
              namezh: swinv2_cr_large_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9891
              datasetA: '65.43'
              datasetB: '45.11'
              ranking: '848'
            
            - nameen: swinv2_cr_small_224
              namezh: swinv2_cr_small_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5867
              datasetA: '65.42'
              datasetB: '56.88'
              ranking: '849'
            
            - nameen: swinv2_cr_small_384
              namezh: swinv2_cr_small_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5935
              datasetA: '65.41'
              datasetB: '44.17'
              ranking: '850'
            
            - nameen: swinv2_cr_small_ns_224
              namezh: swinv2_cr_small_ns_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9557
              datasetA: '65.4'
              datasetB: '61.06'
              ranking: '851'
            
            - nameen: swinv2_cr_small_ns_256
              namezh: swinv2_cr_small_ns_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7879
              datasetA: '65.4'
              datasetB: '52.24'
              ranking: '852'
            
            - nameen: swinv2_cr_tiny_224
              namezh: swinv2_cr_tiny_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9221
              datasetA: '65.36'
              datasetB: '44.26'
              ranking: '853'
            
            - nameen: swinv2_cr_tiny_384
              namezh: swinv2_cr_tiny_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1351
              datasetA: '65.35'
              datasetB: '55.11'
              ranking: '854'
            
            - nameen: swinv2_cr_tiny_ns_224
              namezh: swinv2_cr_tiny_ns_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7644
              datasetA: '65.33'
              datasetB: '40.92'
              ranking: '855'
            
            - nameen: swinv2_large_window12_192
              namezh: swinv2_large_window12_192
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7422
              datasetA: '65.29'
              datasetB: '44.55'
              ranking: '856'
            
            - nameen: swinv2_large_window12to16_192to256
              namezh: swinv2_large_window12to16_192to256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7108
              datasetA: '65.29'
              datasetB: '40.11'
              ranking: '857'
            
            - nameen: swinv2_large_window12to24_192to384
              namezh: swinv2_large_window12to24_192to384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6635
              datasetA: '65.27'
              datasetB: '59.73'
              ranking: '858'
            
            - nameen: swinv2_small_window8_256
              namezh: swinv2_small_window8_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4731
              datasetA: '65.23'
              datasetB: '62.73'
              ranking: '859'
            
            - nameen: swinv2_small_window16_256
              namezh: swinv2_small_window16_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2324
              datasetA: '65.22'
              datasetB: '56.11'
              ranking: '860'
            
            - nameen: swinv2_tiny_window8_256
              namezh: swinv2_tiny_window8_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1564
              datasetA: '65.21'
              datasetB: '44.18'
              ranking: '861'
            
            - nameen: swinv2_tiny_window16_256
              namezh: swinv2_tiny_window16_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1986
              datasetA: '65.18'
              datasetB: '43.54'
              ranking: '862'
            
            - nameen: tf_efficientnet_b0
              namezh: tf_efficientnet_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2672
              datasetA: '65.14'
              datasetB: '54.56'
              ranking: '863'
            
            - nameen: tf_efficientnet_b1
              namezh: tf_efficientnet_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1290
              datasetA: '65.13'
              datasetB: '43.83'
              ranking: '864'
            
            - nameen: tf_efficientnet_b2
              namezh: tf_efficientnet_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8918
              datasetA: '65.05'
              datasetB: '57.79'
              ranking: '865'
            
            - nameen: tf_efficientnet_b3
              namezh: tf_efficientnet_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5331
              datasetA: '65.03'
              datasetB: '61.49'
              ranking: '866'
            
            - nameen: tf_efficientnet_b4
              namezh: tf_efficientnet_b4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1423
              datasetA: '65.03'
              datasetB: '48.32'
              ranking: '867'
            
            - nameen: tf_efficientnet_b5
              namezh: tf_efficientnet_b5
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3827
              datasetA: '65.03'
              datasetB: '56.61'
              ranking: '868'
            
            - nameen: tf_efficientnet_b6
              namezh: tf_efficientnet_b6
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3913
              datasetA: '65.0'
              datasetB: '58.26'
              ranking: '869'
            
            - nameen: tf_efficientnet_b7
              namezh: tf_efficientnet_b7
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9523
              datasetA: '65.0'
              datasetB: '45.57'
              ranking: '870'
            
            - nameen: tf_efficientnet_b8
              namezh: tf_efficientnet_b8
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2290
              datasetA: '64.99'
              datasetB: '60.38'
              ranking: '871'
            
            - nameen: tf_efficientnet_cc_b0_4e
              namezh: tf_efficientnet_cc_b0_4e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9419
              datasetA: '64.99'
              datasetB: '56.8'
              ranking: '872'
            
            - nameen: tf_efficientnet_cc_b0_8e
              namezh: tf_efficientnet_cc_b0_8e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6344
              datasetA: '64.98'
              datasetB: '51.73'
              ranking: '873'
            
            - nameen: tf_efficientnet_cc_b1_8e
              namezh: tf_efficientnet_cc_b1_8e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3109
              datasetA: '64.97'
              datasetB: '62.42'
              ranking: '874'
            
            - nameen: tf_efficientnet_el
              namezh: tf_efficientnet_el
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3392
              datasetA: '64.97'
              datasetB: '45.56'
              ranking: '875'
            
            - nameen: tf_efficientnet_em
              namezh: tf_efficientnet_em
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1065
              datasetA: '64.95'
              datasetB: '42.72'
              ranking: '876'
            
            - nameen: tf_efficientnet_es
              namezh: tf_efficientnet_es
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1797
              datasetA: '64.94'
              datasetB: '46.0'
              ranking: '877'
            
            - nameen: tf_efficientnet_l2
              namezh: tf_efficientnet_l2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7956
              datasetA: '64.93'
              datasetB: '40.2'
              ranking: '878'
            
            - nameen: tf_efficientnet_lite0
              namezh: tf_efficientnet_lite0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4886
              datasetA: '64.92'
              datasetB: '59.0'
              ranking: '879'
            
            - nameen: tf_efficientnet_lite1
              namezh: tf_efficientnet_lite1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7902
              datasetA: '64.88'
              datasetB: '62.16'
              ranking: '880'
            
            - nameen: tf_efficientnet_lite2
              namezh: tf_efficientnet_lite2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9678
              datasetA: '64.88'
              datasetB: '48.67'
              ranking: '881'
            
            - nameen: tf_efficientnet_lite3
              namezh: tf_efficientnet_lite3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5836
              datasetA: '64.87'
              datasetB: '59.86'
              ranking: '882'
            
            - nameen: tf_efficientnet_lite4
              namezh: tf_efficientnet_lite4
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6300
              datasetA: '64.84'
              datasetB: '60.61'
              ranking: '883'
            
            - nameen: tf_efficientnetv2_b0
              namezh: tf_efficientnetv2_b0
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8825
              datasetA: '64.83'
              datasetB: '53.09'
              ranking: '884'
            
            - nameen: tf_efficientnetv2_b1
              namezh: tf_efficientnetv2_b1
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3492
              datasetA: '64.83'
              datasetB: '49.15'
              ranking: '885'
            
            - nameen: tf_efficientnetv2_b2
              namezh: tf_efficientnetv2_b2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8656
              datasetA: '64.79'
              datasetB: '61.16'
              ranking: '886'
            
            - nameen: tf_efficientnetv2_b3
              namezh: tf_efficientnetv2_b3
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9510
              datasetA: '64.75'
              datasetB: '50.95'
              ranking: '887'
            
            - nameen: tf_efficientnetv2_l
              namezh: tf_efficientnetv2_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7083
              datasetA: '64.72'
              datasetB: '55.39'
              ranking: '888'
            
            - nameen: tf_efficientnetv2_m
              namezh: tf_efficientnetv2_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6334
              datasetA: '64.68'
              datasetB: '58.3'
              ranking: '889'
            
            - nameen: tf_efficientnetv2_s
              namezh: tf_efficientnetv2_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8809
              datasetA: '64.61'
              datasetB: '57.57'
              ranking: '890'
            
            - nameen: tf_efficientnetv2_xl
              namezh: tf_efficientnetv2_xl
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1947
              datasetA: '64.6'
              datasetB: '42.62'
              ranking: '891'
            
            - nameen: tf_mixnet_l
              namezh: tf_mixnet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7529
              datasetA: '64.57'
              datasetB: '61.64'
              ranking: '892'
            
            - nameen: tf_mixnet_m
              namezh: tf_mixnet_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1177
              datasetA: '64.56'
              datasetB: '52.65'
              ranking: '893'
            
            - nameen: tf_mixnet_s
              namezh: tf_mixnet_s
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5836
              datasetA: '64.55'
              datasetB: '45.62'
              ranking: '894'
            
            - nameen: tf_mobilenetv3_large_075
              namezh: tf_mobilenetv3_large_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8570
              datasetA: '64.5'
              datasetB: '50.87'
              ranking: '895'
            
            - nameen: tf_mobilenetv3_large_100
              namezh: tf_mobilenetv3_large_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2069
              datasetA: '64.48'
              datasetB: '51.16'
              ranking: '896'
            
            - nameen: tf_mobilenetv3_large_minimal_100
              namezh: tf_mobilenetv3_large_minimal_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1375
              datasetA: '64.46'
              datasetB: '52.04'
              ranking: '897'
            
            - nameen: tf_mobilenetv3_small_075
              namezh: tf_mobilenetv3_small_075
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1293
              datasetA: '64.45'
              datasetB: '54.51'
              ranking: '898'
            
            - nameen: tf_mobilenetv3_small_100
              namezh: tf_mobilenetv3_small_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8379
              datasetA: '64.44'
              datasetB: '47.03'
              ranking: '899'
            
            - nameen: tf_mobilenetv3_small_minimal_100
              namezh: tf_mobilenetv3_small_minimal_100
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1104
              datasetA: '64.4'
              datasetB: '44.56'
              ranking: '900'
            
            - nameen: tiny_vit_5m_224
              namezh: tiny_vit_5m_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3882
              datasetA: '64.38'
              datasetB: '60.59'
              ranking: '901'
            
            - nameen: tiny_vit_11m_224
              namezh: tiny_vit_11m_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2010
              datasetA: '64.36'
              datasetB: '41.46'
              ranking: '902'
            
            - nameen: tiny_vit_21m_224
              namezh: tiny_vit_21m_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1574
              datasetA: '64.36'
              datasetB: '57.02'
              ranking: '903'
            
            - nameen: tiny_vit_21m_384
              namezh: tiny_vit_21m_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3958
              datasetA: '64.36'
              datasetB: '54.53'
              ranking: '904'
            
            - nameen: tiny_vit_21m_512
              namezh: tiny_vit_21m_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6703
              datasetA: '64.34'
              datasetB: '58.04'
              ranking: '905'
            
            - nameen: tinynet_a
              namezh: tinynet_a
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4506
              datasetA: '64.31'
              datasetB: '51.91'
              ranking: '906'
            
            - nameen: tinynet_b
              namezh: tinynet_b
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3634
              datasetA: '64.29'
              datasetB: '55.25'
              ranking: '907'
            
            - nameen: tinynet_c
              namezh: tinynet_c
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2156
              datasetA: '64.29'
              datasetB: '51.72'
              ranking: '908'
            
            - nameen: tinynet_d
              namezh: tinynet_d
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9422
              datasetA: '64.28'
              datasetB: '44.51'
              ranking: '909'
            
            - nameen: tinynet_e
              namezh: tinynet_e
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5869
              datasetA: '64.27'
              datasetB: '41.72'
              ranking: '910'
            
            - nameen: tnt_b_patch16_224
              namezh: tnt_b_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8163
              datasetA: '64.24'
              datasetB: '46.99'
              ranking: '911'
            
            - nameen: tnt_s_patch16_224
              namezh: tnt_s_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9217
              datasetA: '64.24'
              datasetB: '46.84'
              ranking: '912'
            
            - nameen: tresnet_l
              namezh: tresnet_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7666
              datasetA: '64.19'
              datasetB: '44.66'
              ranking: '913'
            
            - nameen: tresnet_m
              namezh: tresnet_m
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5570
              datasetA: '64.15'
              datasetB: '42.03'
              ranking: '914'
            
            - nameen: tresnet_v2_l
              namezh: tresnet_v2_l
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4425
              datasetA: '64.15'
              datasetB: '60.96'
              ranking: '915'
            
            - nameen: tresnet_xl
              namezh: tresnet_xl
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1754
              datasetA: '64.14'
              datasetB: '54.6'
              ranking: '916'
            
            - nameen: twins_pcpvt_base
              namezh: twins_pcpvt_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6027
              datasetA: '64.1'
              datasetB: '49.57'
              ranking: '917'
            
            - nameen: twins_pcpvt_large
              namezh: twins_pcpvt_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3440
              datasetA: '64.08'
              datasetB: '44.52'
              ranking: '918'
            
            - nameen: twins_pcpvt_small
              namezh: twins_pcpvt_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2746
              datasetA: '64.08'
              datasetB: '61.99'
              ranking: '919'
            
            - nameen: twins_svt_base
              namezh: twins_svt_base
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2192
              datasetA: '64.08'
              datasetB: '47.4'
              ranking: '920'
            
            - nameen: twins_svt_large
              namezh: twins_svt_large
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4546
              datasetA: '64.07'
              datasetB: '42.61'
              ranking: '921'
            
            - nameen: twins_svt_small
              namezh: twins_svt_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1229
              datasetA: '64.05'
              datasetB: '49.89'
              ranking: '922'
            
            - nameen: vgg11
              namezh: vgg11
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8337
              datasetA: '63.99'
              datasetB: '60.49'
              ranking: '923'
            
            - nameen: vgg11_bn
              namezh: vgg11_bn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6540
              datasetA: '63.97'
              datasetB: '54.46'
              ranking: '924'
            
            - nameen: vgg13
              namezh: vgg13
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4780
              datasetA: '63.97'
              datasetB: '47.78'
              ranking: '925'
            
            - nameen: vgg13_bn
              namezh: vgg13_bn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1204
              datasetA: '63.95'
              datasetB: '60.88'
              ranking: '926'
            
            - nameen: vgg16
              namezh: vgg16
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4138
              datasetA: '63.94'
              datasetB: '47.96'
              ranking: '927'
            
            - nameen: vgg16_bn
              namezh: vgg16_bn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4836
              datasetA: '63.93'
              datasetB: '41.37'
              ranking: '928'
            
            - nameen: vgg19
              namezh: vgg19
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7448
              datasetA: '63.91'
              datasetB: '55.52'
              ranking: '929'
            
            - nameen: vgg19_bn
              namezh: vgg19_bn
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1842
              datasetA: '63.9'
              datasetB: '62.67'
              ranking: '930'
            
            - nameen: visformer_small
              namezh: visformer_small
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1409
              datasetA: '63.9'
              datasetB: '43.12'
              ranking: '931'
            
            - nameen: visformer_tiny
              namezh: visformer_tiny
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7765
              datasetA: '63.86'
              datasetB: '59.51'
              ranking: '932'
            
            - nameen: vit_base_mci_224
              namezh: vit_base_mci_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8967
              datasetA: '63.85'
              datasetB: '63.43'
              ranking: '933'
            
            - nameen: vit_base_patch8_224
              namezh: vit_base_patch8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3474
              datasetA: '63.84'
              datasetB: '42.92'
              ranking: '934'
            
            - nameen: vit_base_patch14_dinov2
              namezh: vit_base_patch14_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3196
              datasetA: '63.8'
              datasetB: '45.98'
              ranking: '935'
            
            - nameen: vit_base_patch14_reg4_dinov2
              namezh: vit_base_patch14_reg4_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3694
              datasetA: '63.69'
              datasetB: '42.19'
              ranking: '936'
            
            - nameen: vit_base_patch16_18x2_224
              namezh: vit_base_patch16_18x2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5709
              datasetA: '63.69'
              datasetB: '63.45'
              ranking: '937'
            
            - nameen: vit_base_patch16_224
              namezh: vit_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5508
              datasetA: '63.68'
              datasetB: '45.74'
              ranking: '938'
            
            - nameen: vit_base_patch16_224_miil
              namezh: vit_base_patch16_224_miil
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9927
              datasetA: '63.68'
              datasetB: '46.6'
              ranking: '939'
            
            - nameen: vit_base_patch16_384
              namezh: vit_base_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7425
              datasetA: '63.64'
              datasetB: '58.9'
              ranking: '940'
            
            - nameen: vit_base_patch16_clip_224
              namezh: vit_base_patch16_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9284
              datasetA: '63.63'
              datasetB: '57.11'
              ranking: '941'
            
            - nameen: vit_base_patch16_clip_384
              namezh: vit_base_patch16_clip_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7579
              datasetA: '63.61'
              datasetB: '40.53'
              ranking: '942'
            
            - nameen: vit_base_patch16_clip_quickgelu_224
              namezh: vit_base_patch16_clip_quickgelu_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8047
              datasetA: '63.61'
              datasetB: '59.01'
              ranking: '943'
            
            - nameen: vit_base_patch16_gap_224
              namezh: vit_base_patch16_gap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2317
              datasetA: '63.6'
              datasetB: '55.64'
              ranking: '944'
            
            - nameen: vit_base_patch16_plus_240
              namezh: vit_base_patch16_plus_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3142
              datasetA: '63.59'
              datasetB: '42.59'
              ranking: '945'
            
            - nameen: vit_base_patch16_reg4_gap_256
              namezh: vit_base_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4411
              datasetA: '63.59'
              datasetB: '61.55'
              ranking: '946'
            
            - nameen: vit_base_patch16_rope_reg1_gap_256
              namezh: vit_base_patch16_rope_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2062
              datasetA: '63.54'
              datasetB: '59.62'
              ranking: '947'
            
            - nameen: vit_base_patch16_rpn_224
              namezh: vit_base_patch16_rpn_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7567
              datasetA: '63.52'
              datasetB: '52.6'
              ranking: '948'
            
            - nameen: vit_base_patch16_siglip_224
              namezh: vit_base_patch16_siglip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5851
              datasetA: '63.51'
              datasetB: '55.04'
              ranking: '949'
            
            - nameen: vit_base_patch16_siglip_256
              namezh: vit_base_patch16_siglip_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8329
              datasetA: '63.51'
              datasetB: '41.38'
              ranking: '950'
            
            - nameen: vit_base_patch16_siglip_384
              namezh: vit_base_patch16_siglip_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8043
              datasetA: '63.49'
              datasetB: '57.51'
              ranking: '951'
            
            - nameen: vit_base_patch16_siglip_512
              namezh: vit_base_patch16_siglip_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7435
              datasetA: '63.49'
              datasetB: '62.28'
              ranking: '952'
            
            - nameen: vit_base_patch16_siglip_gap_224
              namezh: vit_base_patch16_siglip_gap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6700
              datasetA: '63.45'
              datasetB: '41.21'
              ranking: '953'
            
            - nameen: vit_base_patch16_siglip_gap_256
              namezh: vit_base_patch16_siglip_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6627
              datasetA: '63.42'
              datasetB: '62.16'
              ranking: '954'
            
            - nameen: vit_base_patch16_siglip_gap_384
              namezh: vit_base_patch16_siglip_gap_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3572
              datasetA: '63.41'
              datasetB: '43.16'
              ranking: '955'
            
            - nameen: vit_base_patch16_siglip_gap_512
              namezh: vit_base_patch16_siglip_gap_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6114
              datasetA: '63.39'
              datasetB: '55.47'
              ranking: '956'
            
            - nameen: vit_base_patch16_xp_224
              namezh: vit_base_patch16_xp_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7913
              datasetA: '63.35'
              datasetB: '47.94'
              ranking: '957'
            
            - nameen: vit_base_patch32_224
              namezh: vit_base_patch32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1449
              datasetA: '63.3'
              datasetB: '57.06'
              ranking: '958'
            
            - nameen: vit_base_patch32_384
              namezh: vit_base_patch32_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2631
              datasetA: '63.3'
              datasetB: '44.69'
              ranking: '959'
            
            - nameen: vit_base_patch32_clip_224
              namezh: vit_base_patch32_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7589
              datasetA: '63.25'
              datasetB: '53.68'
              ranking: '960'
            
            - nameen: vit_base_patch32_clip_256
              namezh: vit_base_patch32_clip_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1876
              datasetA: '63.24'
              datasetB: '48.3'
              ranking: '961'
            
            - nameen: vit_base_patch32_clip_384
              namezh: vit_base_patch32_clip_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1918
              datasetA: '63.2'
              datasetB: '58.61'
              ranking: '962'
            
            - nameen: vit_base_patch32_clip_448
              namezh: vit_base_patch32_clip_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2468
              datasetA: '63.19'
              datasetB: '56.35'
              ranking: '963'
            
            - nameen: vit_base_patch32_clip_quickgelu_224
              namezh: vit_base_patch32_clip_quickgelu_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6475
              datasetA: '63.18'
              datasetB: '42.11'
              ranking: '964'
            
            - nameen: vit_base_patch32_plus_256
              namezh: vit_base_patch32_plus_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9485
              datasetA: '63.15'
              datasetB: '41.44'
              ranking: '965'
            
            - nameen: vit_base_r26_s32_224
              namezh: vit_base_r26_s32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8466
              datasetA: '63.12'
              datasetB: '43.79'
              ranking: '966'
            
            - nameen: vit_base_r50_s16_224
              namezh: vit_base_r50_s16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4651
              datasetA: '63.1'
              datasetB: '60.02'
              ranking: '967'
            
            - nameen: vit_base_r50_s16_384
              namezh: vit_base_r50_s16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2506
              datasetA: '63.1'
              datasetB: '53.23'
              ranking: '968'
            
            - nameen: vit_base_resnet26d_224
              namezh: vit_base_resnet26d_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7607
              datasetA: '63.08'
              datasetB: '40.39'
              ranking: '969'
            
            - nameen: vit_base_resnet50d_224
              namezh: vit_base_resnet50d_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5022
              datasetA: '63.08'
              datasetB: '59.58'
              ranking: '970'
            
            - nameen: vit_betwixt_patch16_gap_256
              namezh: vit_betwixt_patch16_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7524
              datasetA: '63.05'
              datasetB: '54.22'
              ranking: '971'
            
            - nameen: vit_betwixt_patch16_reg1_gap_256
              namezh: vit_betwixt_patch16_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2135
              datasetA: '63.04'
              datasetB: '59.27'
              ranking: '972'
            
            - nameen: vit_betwixt_patch16_reg4_gap_256
              namezh: vit_betwixt_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3267
              datasetA: '63.02'
              datasetB: '60.12'
              ranking: '973'
            
            - nameen: vit_betwixt_patch16_rope_reg4_gap_256
              namezh: vit_betwixt_patch16_rope_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6964
              datasetA: '62.99'
              datasetB: '43.58'
              ranking: '974'
            
            - nameen: vit_betwixt_patch32_clip_224
              namezh: vit_betwixt_patch32_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4856
              datasetA: '62.98'
              datasetB: '59.35'
              ranking: '975'
            
            - nameen: vit_giant_patch14_224
              namezh: vit_giant_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7280
              datasetA: '62.96'
              datasetB: '46.36'
              ranking: '976'
            
            - nameen: vit_giant_patch14_clip_224
              namezh: vit_giant_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5702
              datasetA: '62.96'
              datasetB: '63.37'
              ranking: '977'
            
            - nameen: vit_giant_patch14_dinov2
              namezh: vit_giant_patch14_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3471
              datasetA: '62.95'
              datasetB: '51.49'
              ranking: '978'
            
            - nameen: vit_giant_patch14_reg4_dinov2
              namezh: vit_giant_patch14_reg4_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1857
              datasetA: '62.92'
              datasetB: '62.14'
              ranking: '979'
            
            - nameen: vit_giant_patch16_gap_224
              namezh: vit_giant_patch16_gap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5510
              datasetA: '62.89'
              datasetB: '40.14'
              ranking: '980'
            
            - nameen: vit_gigantic_patch14_224
              namezh: vit_gigantic_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9208
              datasetA: '62.87'
              datasetB: '55.31'
              ranking: '981'
            
            - nameen: vit_gigantic_patch14_clip_224
              namezh: vit_gigantic_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7896
              datasetA: '62.85'
              datasetB: '59.32'
              ranking: '982'
            
            - nameen: vit_huge_patch14_224
              namezh: vit_huge_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1640
              datasetA: '62.81'
              datasetB: '41.87'
              ranking: '983'
            
            - nameen: vit_huge_patch14_clip_224
              namezh: vit_huge_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3101
              datasetA: '62.8'
              datasetB: '43.9'
              ranking: '984'
            
            - nameen: vit_huge_patch14_clip_336
              namezh: vit_huge_patch14_clip_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4638
              datasetA: '62.78'
              datasetB: '53.26'
              ranking: '985'
            
            - nameen: vit_huge_patch14_clip_378
              namezh: vit_huge_patch14_clip_378
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7122
              datasetA: '62.77'
              datasetB: '47.69'
              ranking: '986'
            
            - nameen: vit_huge_patch14_clip_quickgelu_224
              namezh: vit_huge_patch14_clip_quickgelu_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8038
              datasetA: '62.75'
              datasetB: '47.14'
              ranking: '987'
            
            - nameen: vit_huge_patch14_clip_quickgelu_378
              namezh: vit_huge_patch14_clip_quickgelu_378
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5861
              datasetA: '62.75'
              datasetB: '49.64'
              ranking: '988'
            
            - nameen: vit_huge_patch14_gap_224
              namezh: vit_huge_patch14_gap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1229
              datasetA: '62.71'
              datasetB: '52.07'
              ranking: '989'
            
            - nameen: vit_huge_patch14_xp_224
              namezh: vit_huge_patch14_xp_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5839
              datasetA: '62.63'
              datasetB: '54.11'
              ranking: '990'
            
            - nameen: vit_huge_patch16_gap_448
              namezh: vit_huge_patch16_gap_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7472
              datasetA: '62.62'
              datasetB: '49.46'
              ranking: '991'
            
            - nameen: vit_large_patch14_224
              namezh: vit_large_patch14_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9844
              datasetA: '62.62'
              datasetB: '57.88'
              ranking: '992'
            
            - nameen: vit_large_patch14_clip_224
              namezh: vit_large_patch14_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8497
              datasetA: '62.58'
              datasetB: '58.8'
              ranking: '993'
            
            - nameen: vit_large_patch14_clip_336
              namezh: vit_large_patch14_clip_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1884
              datasetA: '62.55'
              datasetB: '58.07'
              ranking: '994'
            
            - nameen: vit_large_patch14_clip_quickgelu_224
              namezh: vit_large_patch14_clip_quickgelu_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8752
              datasetA: '62.53'
              datasetB: '52.15'
              ranking: '995'
            
            - nameen: vit_large_patch14_clip_quickgelu_336
              namezh: vit_large_patch14_clip_quickgelu_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4100
              datasetA: '62.51'
              datasetB: '62.87'
              ranking: '996'
            
            - nameen: vit_large_patch14_dinov2
              namezh: vit_large_patch14_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8001
              datasetA: '62.51'
              datasetB: '51.15'
              ranking: '997'
            
            - nameen: vit_large_patch14_reg4_dinov2
              namezh: vit_large_patch14_reg4_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8875
              datasetA: '62.5'
              datasetB: '45.92'
              ranking: '998'
            
            - nameen: vit_large_patch14_xp_224
              namezh: vit_large_patch14_xp_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5436
              datasetA: '62.48'
              datasetB: '58.84'
              ranking: '999'
            
            - nameen: vit_large_patch16_224
              namezh: vit_large_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9190
              datasetA: '62.48'
              datasetB: '55.45'
              ranking: '1000'
            
            - nameen: vit_large_patch16_384
              namezh: vit_large_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9946
              datasetA: '62.45'
              datasetB: '62.04'
              ranking: '1001'
            
            - nameen: vit_large_patch16_siglip_256
              namezh: vit_large_patch16_siglip_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1002
              datasetA: '62.43'
              datasetB: '47.3'
              ranking: '1002'
            
            - nameen: vit_large_patch16_siglip_384
              namezh: vit_large_patch16_siglip_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7741
              datasetA: '62.41'
              datasetB: '44.77'
              ranking: '1003'
            
            - nameen: vit_large_patch16_siglip_gap_256
              namezh: vit_large_patch16_siglip_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8020
              datasetA: '62.41'
              datasetB: '60.23'
              ranking: '1004'
            
            - nameen: vit_large_patch16_siglip_gap_384
              namezh: vit_large_patch16_siglip_gap_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8723
              datasetA: '62.41'
              datasetB: '46.57'
              ranking: '1005'
            
            - nameen: vit_large_patch32_224
              namezh: vit_large_patch32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2043
              datasetA: '62.34'
              datasetB: '46.73'
              ranking: '1006'
            
            - nameen: vit_large_patch32_384
              namezh: vit_large_patch32_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6800
              datasetA: '62.31'
              datasetB: '49.92'
              ranking: '1007'
            
            - nameen: vit_large_r50_s32_224
              namezh: vit_large_r50_s32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8540
              datasetA: '62.25'
              datasetB: '59.96'
              ranking: '1008'
            
            - nameen: vit_large_r50_s32_384
              namezh: vit_large_r50_s32_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7814
              datasetA: '62.24'
              datasetB: '43.52'
              ranking: '1009'
            
            - nameen: vit_little_patch16_reg1_gap_256
              namezh: vit_little_patch16_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9187
              datasetA: '62.24'
              datasetB: '41.71'
              ranking: '1010'
            
            - nameen: vit_little_patch16_reg4_gap_256
              namezh: vit_little_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5824
              datasetA: '62.24'
              datasetB: '57.9'
              ranking: '1011'
            
            - nameen: vit_medium_patch16_clip_224
              namezh: vit_medium_patch16_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1394
              datasetA: '62.2'
              datasetB: '49.28'
              ranking: '1012'
            
            - nameen: vit_medium_patch16_gap_240
              namezh: vit_medium_patch16_gap_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1378
              datasetA: '62.16'
              datasetB: '58.98'
              ranking: '1013'
            
            - nameen: vit_medium_patch16_gap_256
              namezh: vit_medium_patch16_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8768
              datasetA: '62.06'
              datasetB: '52.44'
              ranking: '1014'
            
            - nameen: vit_medium_patch16_gap_384
              namezh: vit_medium_patch16_gap_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9230
              datasetA: '62.04'
              datasetB: '45.7'
              ranking: '1015'
            
            - nameen: vit_medium_patch16_reg1_gap_256
              namezh: vit_medium_patch16_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1441
              datasetA: '62.03'
              datasetB: '54.44'
              ranking: '1016'
            
            - nameen: vit_medium_patch16_reg4_gap_256
              namezh: vit_medium_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6106
              datasetA: '62.02'
              datasetB: '43.11'
              ranking: '1017'
            
            - nameen: vit_medium_patch16_rope_reg1_gap_256
              namezh: vit_medium_patch16_rope_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6305
              datasetA: '62.01'
              datasetB: '60.96'
              ranking: '1018'
            
            - nameen: vit_medium_patch32_clip_224
              namezh: vit_medium_patch32_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4856
              datasetA: '62.0'
              datasetB: '63.35'
              ranking: '1019'
            
            - nameen: vit_mediumd_patch16_reg4_gap_256
              namezh: vit_mediumd_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9049
              datasetA: '61.99'
              datasetB: '42.12'
              ranking: '1020'
            
            - nameen: vit_mediumd_patch16_rope_reg1_gap_256
              namezh: vit_mediumd_patch16_rope_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6663
              datasetA: '61.98'
              datasetB: '58.91'
              ranking: '1021'
            
            - nameen: vit_pwee_patch16_reg1_gap_256
              namezh: vit_pwee_patch16_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4566
              datasetA: '61.96'
              datasetB: '44.1'
              ranking: '1022'
            
            - nameen: vit_relpos_base_patch16_224
              namezh: vit_relpos_base_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7197
              datasetA: '61.96'
              datasetB: '50.66'
              ranking: '1023'
            
            - nameen: vit_relpos_base_patch16_cls_224
              namezh: vit_relpos_base_patch16_cls_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8702
              datasetA: '61.95'
              datasetB: '56.73'
              ranking: '1024'
            
            - nameen: vit_relpos_base_patch16_clsgap_224
              namezh: vit_relpos_base_patch16_clsgap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1878
              datasetA: '61.9'
              datasetB: '47.48'
              ranking: '1025'
            
            - nameen: vit_relpos_base_patch16_plus_240
              namezh: vit_relpos_base_patch16_plus_240
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7065
              datasetA: '61.83'
              datasetB: '44.03'
              ranking: '1026'
            
            - nameen: vit_relpos_base_patch16_rpn_224
              namezh: vit_relpos_base_patch16_rpn_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1087
              datasetA: '61.81'
              datasetB: '49.52'
              ranking: '1027'
            
            - nameen: vit_relpos_base_patch32_plus_rpn_256
              namezh: vit_relpos_base_patch32_plus_rpn_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2330
              datasetA: '61.78'
              datasetB: '63.31'
              ranking: '1028'
            
            - nameen: vit_relpos_medium_patch16_224
              namezh: vit_relpos_medium_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1122
              datasetA: '61.76'
              datasetB: '62.3'
              ranking: '1029'
            
            - nameen: vit_relpos_medium_patch16_cls_224
              namezh: vit_relpos_medium_patch16_cls_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1548
              datasetA: '61.69'
              datasetB: '51.81'
              ranking: '1030'
            
            - nameen: vit_relpos_medium_patch16_rpn_224
              namezh: vit_relpos_medium_patch16_rpn_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8704
              datasetA: '61.69'
              datasetB: '54.27'
              ranking: '1031'
            
            - nameen: vit_relpos_small_patch16_224
              namezh: vit_relpos_small_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1115
              datasetA: '61.66'
              datasetB: '56.69'
              ranking: '1032'
            
            - nameen: vit_relpos_small_patch16_rpn_224
              namezh: vit_relpos_small_patch16_rpn_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6906
              datasetA: '61.65'
              datasetB: '46.61'
              ranking: '1033'
            
            - nameen: vit_small_patch8_224
              namezh: vit_small_patch8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5227
              datasetA: '61.63'
              datasetB: '50.02'
              ranking: '1034'
            
            - nameen: vit_small_patch14_dinov2
              namezh: vit_small_patch14_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7220
              datasetA: '61.59'
              datasetB: '41.57'
              ranking: '1035'
            
            - nameen: vit_small_patch14_reg4_dinov2
              namezh: vit_small_patch14_reg4_dinov2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5734
              datasetA: '61.57'
              datasetB: '49.03'
              ranking: '1036'
            
            - nameen: vit_small_patch16_18x2_224
              namezh: vit_small_patch16_18x2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4061
              datasetA: '61.56'
              datasetB: '52.61'
              ranking: '1037'
            
            - nameen: vit_small_patch16_36x1_224
              namezh: vit_small_patch16_36x1_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8396
              datasetA: '61.55'
              datasetB: '50.32'
              ranking: '1038'
            
            - nameen: vit_small_patch16_224
              namezh: vit_small_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4943
              datasetA: '61.53'
              datasetB: '60.86'
              ranking: '1039'
            
            - nameen: vit_small_patch16_384
              namezh: vit_small_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5308
              datasetA: '61.53'
              datasetB: '52.18'
              ranking: '1040'
            
            - nameen: vit_small_patch32_224
              namezh: vit_small_patch32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5092
              datasetA: '61.53'
              datasetB: '48.73'
              ranking: '1041'
            
            - nameen: vit_small_patch32_384
              namezh: vit_small_patch32_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9981
              datasetA: '61.51'
              datasetB: '41.96'
              ranking: '1042'
            
            - nameen: vit_small_r26_s32_224
              namezh: vit_small_r26_s32_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3380
              datasetA: '61.43'
              datasetB: '44.88'
              ranking: '1043'
            
            - nameen: vit_small_r26_s32_384
              namezh: vit_small_r26_s32_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5775
              datasetA: '61.39'
              datasetB: '62.32'
              ranking: '1044'
            
            - nameen: vit_small_resnet26d_224
              namezh: vit_small_resnet26d_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2396
              datasetA: '61.39'
              datasetB: '60.22'
              ranking: '1045'
            
            - nameen: vit_small_resnet50d_s16_224
              namezh: vit_small_resnet50d_s16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9383
              datasetA: '61.37'
              datasetB: '61.14'
              ranking: '1046'
            
            - nameen: vit_so150m_patch16_reg4_gap_256
              namezh: vit_so150m_patch16_reg4_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8501
              datasetA: '61.36'
              datasetB: '44.5'
              ranking: '1047'
            
            - nameen: vit_so150m_patch16_reg4_map_256
              namezh: vit_so150m_patch16_reg4_map_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2635
              datasetA: '61.35'
              datasetB: '54.83'
              ranking: '1048'
            
            - nameen: vit_so400m_patch14_siglip_224
              namezh: vit_so400m_patch14_siglip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4983
              datasetA: '61.35'
              datasetB: '51.81'
              ranking: '1049'
            
            - nameen: vit_so400m_patch14_siglip_384
              namezh: vit_so400m_patch14_siglip_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9870
              datasetA: '61.31'
              datasetB: '56.22'
              ranking: '1050'
            
            - nameen: vit_so400m_patch14_siglip_gap_224
              namezh: vit_so400m_patch14_siglip_gap_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4950
              datasetA: '61.31'
              datasetB: '61.41'
              ranking: '1051'
            
            - nameen: vit_so400m_patch14_siglip_gap_384
              namezh: vit_so400m_patch14_siglip_gap_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9788
              datasetA: '61.3'
              datasetB: '60.34'
              ranking: '1052'
            
            - nameen: vit_so400m_patch14_siglip_gap_448
              namezh: vit_so400m_patch14_siglip_gap_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2827
              datasetA: '61.27'
              datasetB: '52.24'
              ranking: '1053'
            
            - nameen: vit_so400m_patch14_siglip_gap_896
              namezh: vit_so400m_patch14_siglip_gap_896
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1838
              datasetA: '61.22'
              datasetB: '48.46'
              ranking: '1054'
            
            - nameen: vit_srelpos_medium_patch16_224
              namezh: vit_srelpos_medium_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7407
              datasetA: '61.19'
              datasetB: '61.42'
              ranking: '1055'
            
            - nameen: vit_srelpos_small_patch16_224
              namezh: vit_srelpos_small_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7958
              datasetA: '61.19'
              datasetB: '50.91'
              ranking: '1056'
            
            - nameen: vit_tiny_patch16_224
              namezh: vit_tiny_patch16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1284
              datasetA: '61.19'
              datasetB: '62.78'
              ranking: '1057'
            
            - nameen: vit_tiny_patch16_384
              namezh: vit_tiny_patch16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3001
              datasetA: '61.1'
              datasetB: '58.52'
              ranking: '1058'
            
            - nameen: vit_tiny_r_s16_p8_224
              namezh: vit_tiny_r_s16_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1825
              datasetA: '61.06'
              datasetB: '63.43'
              ranking: '1059'
            
            - nameen: vit_tiny_r_s16_p8_384
              namezh: vit_tiny_r_s16_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2183
              datasetA: '61.05'
              datasetB: '46.06'
              ranking: '1060'
            
            - nameen: vit_wee_patch16_reg1_gap_256
              namezh: vit_wee_patch16_reg1_gap_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8016
              datasetA: '61.05'
              datasetB: '58.68'
              ranking: '1061'
            
            - nameen: vit_xsmall_patch16_clip_224
              namezh: vit_xsmall_patch16_clip_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1284
              datasetA: '61.02'
              datasetB: '58.34'
              ranking: '1062'
            
            - nameen: vitamin_base_224
              namezh: vitamin_base_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5464
              datasetA: '60.99'
              datasetB: '54.2'
              ranking: '1063'
            
            - nameen: vitamin_large2_224
              namezh: vitamin_large2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7553
              datasetA: '60.93'
              datasetB: '56.66'
              ranking: '1064'
            
            - nameen: vitamin_large2_256
              namezh: vitamin_large2_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3794
              datasetA: '60.9'
              datasetB: '60.22'
              ranking: '1065'
            
            - nameen: vitamin_large2_336
              namezh: vitamin_large2_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1964
              datasetA: '60.89'
              datasetB: '52.09'
              ranking: '1066'
            
            - nameen: vitamin_large2_384
              namezh: vitamin_large2_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9431
              datasetA: '60.84'
              datasetB: '55.24'
              ranking: '1067'
            
            - nameen: vitamin_large_224
              namezh: vitamin_large_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8819
              datasetA: '60.82'
              datasetB: '56.75'
              ranking: '1068'
            
            - nameen: vitamin_large_256
              namezh: vitamin_large_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6919
              datasetA: '60.79'
              datasetB: '58.8'
              ranking: '1069'
            
            - nameen: vitamin_large_336
              namezh: vitamin_large_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3886
              datasetA: '60.79'
              datasetB: '44.36'
              ranking: '1070'
            
            - nameen: vitamin_large_384
              namezh: vitamin_large_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8698
              datasetA: '60.75'
              datasetB: '48.05'
              ranking: '1071'
            
            - nameen: vitamin_small_224
              namezh: vitamin_small_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5770
              datasetA: '60.71'
              datasetB: '41.12'
              ranking: '1072'
            
            - nameen: vitamin_xlarge_256
              namezh: vitamin_xlarge_256
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5312
              datasetA: '60.71'
              datasetB: '62.63'
              ranking: '1073'
            
            - nameen: vitamin_xlarge_336
              namezh: vitamin_xlarge_336
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2917
              datasetA: '60.71'
              datasetB: '46.07'
              ranking: '1074'
            
            - nameen: vitamin_xlarge_384
              namezh: vitamin_xlarge_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1608
              datasetA: '60.68'
              datasetB: '56.65'
              ranking: '1075'
            
            - nameen: volo_d1_224
              namezh: volo_d1_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2457
              datasetA: '60.66'
              datasetB: '57.16'
              ranking: '1076'
            
            - nameen: volo_d1_384
              namezh: volo_d1_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7332
              datasetA: '60.63'
              datasetB: '45.33'
              ranking: '1077'
            
            - nameen: volo_d2_224
              namezh: volo_d2_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7304
              datasetA: '60.62'
              datasetB: '62.87'
              ranking: '1078'
            
            - nameen: volo_d2_384
              namezh: volo_d2_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9810
              datasetA: '60.62'
              datasetB: '40.54'
              ranking: '1079'
            
            - nameen: volo_d3_224
              namezh: volo_d3_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8952
              datasetA: '60.61'
              datasetB: '57.6'
              ranking: '1080'
            
            - nameen: volo_d3_448
              namezh: volo_d3_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7970
              datasetA: '60.58'
              datasetB: '54.01'
              ranking: '1081'
            
            - nameen: volo_d4_224
              namezh: volo_d4_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3067
              datasetA: '60.56'
              datasetB: '54.05'
              ranking: '1082'
            
            - nameen: volo_d4_448
              namezh: volo_d4_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 6758
              datasetA: '60.56'
              datasetB: '40.75'
              ranking: '1083'
            
            - nameen: volo_d5_224
              namezh: volo_d5_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5606
              datasetA: '60.54'
              datasetB: '51.52'
              ranking: '1084'
            
            - nameen: volo_d5_448
              namezh: volo_d5_448
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5083
              datasetA: '60.53'
              datasetB: '63.25'
              ranking: '1085'
            
            - nameen: volo_d5_512
              namezh: volo_d5_512
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8665
              datasetA: '60.52'
              datasetB: '49.09'
              ranking: '1086'
            
            - nameen: vovnet39a
              namezh: vovnet39a
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1222
              datasetA: '60.51'
              datasetB: '59.0'
              ranking: '1087'
            
            - nameen: vovnet57a
              namezh: vovnet57a
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1439
              datasetA: '60.5'
              datasetB: '43.95'
              ranking: '1088'
            
            - nameen: wide_resnet50_2
              namezh: wide_resnet50_2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9600
              datasetA: '60.47'
              datasetB: '51.44'
              ranking: '1089'
            
            - nameen: wide_resnet101_2
              namezh: wide_resnet101_2
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8821
              datasetA: '60.47'
              datasetB: '47.74'
              ranking: '1090'
            
            - nameen: xception41
              namezh: xception41
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 7226
              datasetA: '60.45'
              datasetB: '53.37'
              ranking: '1091'
            
            - nameen: xception41p
              namezh: xception41p
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9441
              datasetA: '60.45'
              datasetB: '42.91'
              ranking: '1092'
            
            - nameen: xception65
              namezh: xception65
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5158
              datasetA: '60.43'
              datasetB: '44.97'
              ranking: '1093'
            
            - nameen: xception65p
              namezh: xception65p
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9944
              datasetA: '60.4'
              datasetB: '50.13'
              ranking: '1094'
            
            - nameen: xception71
              namezh: xception71
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1689
              datasetA: '60.39'
              datasetB: '47.59'
              ranking: '1095'
            
            - nameen: xcit_large_24_p8_224
              namezh: xcit_large_24_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2243
              datasetA: '60.37'
              datasetB: '44.55'
              ranking: '1096'
            
            - nameen: xcit_large_24_p8_384
              namezh: xcit_large_24_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4908
              datasetA: '60.37'
              datasetB: '54.55'
              ranking: '1097'
            
            - nameen: xcit_large_24_p16_224
              namezh: xcit_large_24_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9131
              datasetA: '60.37'
              datasetB: '49.47'
              ranking: '1098'
            
            - nameen: xcit_large_24_p16_384
              namezh: xcit_large_24_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4547
              datasetA: '60.33'
              datasetB: '55.77'
              ranking: '1099'
            
            - nameen: xcit_medium_24_p8_224
              namezh: xcit_medium_24_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8169
              datasetA: '60.32'
              datasetB: '41.61'
              ranking: '1100'
            
            - nameen: xcit_medium_24_p8_384
              namezh: xcit_medium_24_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3938
              datasetA: '60.31'
              datasetB: '42.12'
              ranking: '1101'
            
            - nameen: xcit_medium_24_p16_224
              namezh: xcit_medium_24_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1142
              datasetA: '60.29'
              datasetB: '59.67'
              ranking: '1102'
            
            - nameen: xcit_medium_24_p16_384
              namezh: xcit_medium_24_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4683
              datasetA: '60.29'
              datasetB: '41.72'
              ranking: '1103'
            
            - nameen: xcit_nano_12_p8_224
              namezh: xcit_nano_12_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2393
              datasetA: '60.28'
              datasetB: '50.91'
              ranking: '1104'
            
            - nameen: xcit_nano_12_p8_384
              namezh: xcit_nano_12_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5773
              datasetA: '60.27'
              datasetB: '59.59'
              ranking: '1105'
            
            - nameen: xcit_nano_12_p16_224
              namezh: xcit_nano_12_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2735
              datasetA: '60.26'
              datasetB: '42.46'
              ranking: '1106'
            
            - nameen: xcit_nano_12_p16_384
              namezh: xcit_nano_12_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9264
              datasetA: '60.25'
              datasetB: '55.45'
              ranking: '1107'
            
            - nameen: xcit_small_12_p8_224
              namezh: xcit_small_12_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2896
              datasetA: '60.25'
              datasetB: '53.95'
              ranking: '1108'
            
            - nameen: xcit_small_12_p8_384
              namezh: xcit_small_12_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4442
              datasetA: '60.2'
              datasetB: '60.7'
              ranking: '1109'
            
            - nameen: xcit_small_12_p16_224
              namezh: xcit_small_12_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2527
              datasetA: '60.2'
              datasetB: '44.56'
              ranking: '1110'
            
            - nameen: xcit_small_12_p16_384
              namezh: xcit_small_12_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4955
              datasetA: '60.19'
              datasetB: '44.76'
              ranking: '1111'
            
            - nameen: xcit_small_24_p8_224
              namezh: xcit_small_24_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8247
              datasetA: '60.17'
              datasetB: '59.86'
              ranking: '1112'
            
            - nameen: xcit_small_24_p8_384
              namezh: xcit_small_24_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4931
              datasetA: '60.17'
              datasetB: '44.86'
              ranking: '1113'
            
            - nameen: xcit_small_24_p16_224
              namezh: xcit_small_24_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1052
              datasetA: '60.15'
              datasetB: '51.69'
              ranking: '1114'
            
            - nameen: xcit_small_24_p16_384
              namezh: xcit_small_24_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 2809
              datasetA: '60.14'
              datasetB: '58.63'
              ranking: '1115'
            
            - nameen: xcit_tiny_12_p8_224
              namezh: xcit_tiny_12_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 9937
              datasetA: '60.14'
              datasetB: '44.11'
              ranking: '1116'
            
            - nameen: xcit_tiny_12_p8_384
              namezh: xcit_tiny_12_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 8863
              datasetA: '60.11'
              datasetB: '55.23'
              ranking: '1117'
            
            - nameen: xcit_tiny_12_p16_224
              namezh: xcit_tiny_12_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 4873
              datasetA: '60.07'
              datasetB: '57.65'
              ranking: '1118'
            
            - nameen: xcit_tiny_12_p16_384
              namezh: xcit_tiny_12_p16_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3996
              datasetA: '60.07'
              datasetB: '49.36'
              ranking: '1119'
            
            - nameen: xcit_tiny_24_p8_224
              namezh: xcit_tiny_24_p8_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1346
              datasetA: '60.07'
              datasetB: '61.83'
              ranking: '1120'
            
            - nameen: xcit_tiny_24_p8_384
              namezh: xcit_tiny_24_p8_384
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 5616
              datasetA: '60.06'
              datasetB: '42.61'
              ranking: '1121'
            
            - nameen: xcit_tiny_24_p16_224
              namezh: xcit_tiny_24_p16_224
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 3418
              datasetA: '60.03'
              datasetB: '52.22'
              ranking: '1122'
            
            - nameen: xcit_tiny_24_p16_38
              namezh: xcit_tiny_24_p16_38
              paper:
                text: 'https://huggingface.co/docs/timm/index'
                link: 'https://huggingface.co/docs/timm/index'
              download: 1520
              datasetA: '60.01'
              datasetB: '56.19'
              ranking: '1123'
            
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

