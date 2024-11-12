---
title: 对抗评测平台
blocks:
  - headline1en: Adversarial Robustness Evaluation Platform
    headline1zh: 对抗鲁棒性评测平台
    headline2en: Ensuring AI's Robust Perception of Reality
    headline2zh: 确保人工智能对现实世界的鲁棒感知
    buttonTexten: Tools
    buttonTextzh: 工具集
    buttonText1en: Datasets
    buttonText1zh: 数据集
    buttonText2en: Leaderboards
    buttonText2zh: 排行榜
    subtitle1en: ''
    subtitle1zh: ''
    subtitle2en: ''
    subtitle2zh: ''
    text1en: ''
    text1zh: ''
    text2en: ''
    text2zh: ''
    image:
      src: /uploads/transferbasedAttackIcon2.png
      alt: >-
        Photo of palm trees at sunset by Adam Birkett -
        unsplash.com/photos/75EFpyXu3Wg
    _template: hero
  - title1en: Million-scale Evaluation
    title1zh: 白盒攻击评测White
    items1:
      - image:
          src: /uploads/whiteBoxIcon1.png
        titleen: 'Strongest Individual Adversarial Attack and Improved Ensemble '
        titlezh: '白盒攻击评测亮点 #111'
        texten: >-
          A new indivisual attack method PMA (Probability Margin Attack) is
          introduced to achieve the strongest indivisual attack performance
          using a probability margin loss. An improved ensemble PMA+ is then
          developped, which is stronger than AutoAttack.
        textzh: >-
          Our white-box penetration testing model offers unparalleled insights
          into system vulnerabilities by providing testers with complete access
          to internal architecture
      - image:
          src: /uploads/whiteBoxIcon2.png
        titleen: A Comprehensive Toolbox for Adverarial Robustnes Evaluation
        titlezh: '白盒攻击评测亮点 #222'
        texten: >-
          Our toolbox provides a comrehsneive implementation of different attack
          methods and supports combinations of diverse attack strategies, loss
          functions, and ensembale strategies.
        textzh: >-
          Our white-box penetration testing model offers unparalleled insights
          into system vulnerabilities by providing testers with complete access
          to internal architecture
      - image:
          src: /uploads/whiteBoxIcon3.png
        titleen: A Million-Scale Adversarial Dataset for Image Classification Models
        titlezh: '白盒攻击评测亮点 #333'
        texten: >-
          We release a million-scale adversarial dataset CC1M-Adv-C(lass) for
          large-scale adversarial robustness evaluation of image classification
          models.  CC1M-Adv-C consists of 1M images sampled from CC3M and
          adversarially perturbed using PMA.
        textzh: >-
          Our white-box penetration testing model offers unparalleled insights
          into system vulnerabilities by providing testers with complete access
          to internal architecture
    title2en: Transfer-based Evaluation
    title2zh: 迁移攻击评测attack
    items2:
      - image:
          src: /uploads/transferbasedAttackIcon1.png
        titleen: A New Super-Transfer Attack Based on Mulitiple Surrogate Models
        titlezh: '迁移攻击评测亮点 #1'
        texten: >-
          A new attack method is designed to generate adversarial examples that
          transfer across vision tasks and architectures. It achieves super
          transfarability by exploiting multiple surrogate models up to the
          number of avaliable GPUs.
        textzh: >-
          Our white-box penetration testing model offers unparalleled insights
          into system vulnerabilities by providing testers with complete access
          to internal architecture
      - image:
          src: /uploads/transferbasedAttackIcon2.png
        titleen: A Tool for Convinient Attack Generation and Testing
        titlezh: '迁移攻击评测亮点 #2'
        texten: >-
          We provide a tool for the super-transfer attack method to allow easy
          generation of highly transferable adversarial images across different
          vision tasks and backbones. 
        textzh: >-
          Our white-box penetration testing model offers unparalleled insights
          into system vulnerabilities by providing testers with complete access
          to internal architecture
      - icon:
          name: Tina
        image:
          src: /uploads/transferbasedAttackIcon3.png
        titleen: A Million-Scale Adversarial Dataset for Any Vision Models
        titlezh: '迁移攻击评测亮点 #3'
        texten: >-
          We release CC1M-Adv-F(eature), a large-scale dataset of 1M adversarial
          images crafted using the super-transfer attack. The dataset can be
          used to test any vision models, including image classifiers, object
          detectors, segmentation models, and feature extractors. 
        textzh: >-
          Our white-box penetration testing model offers unparalleled insights
          into system vulnerabilities by providing testers with complete access
          to internal architecture
    _template: features
  - title1en: Physical-world Evaluation
    items1:
      - image:
          src: /uploads/whiteBoxIcon2.png
        titleen: 'DiffPatch: Diffusion-based Adversarial Patch Generation'
        titlezh: ''
        texten: >-
          A diffusion model-based method for generating naturalistic and
          customizable adversarial patches. This method allows users to specify
          a reference image, using the object within it to generate a stylized
          adversarial patch.
        textzh: 中文
      - image:
          src: /uploads/transferbasedAttackIcon3.png
        titleen: A Tool for Generating Digital and Physical Adversarial Patches
        texten: >-
          We open-source the DiffPatch tool for easy generation of customized
          digital and physical adversarial patches. Users can adapt the training
          process of DiffPatch with different diffusion models for different
          application scenarios.
      - image:
          src: /uploads/transferbasedAttackIcon1.png
        titleen: 'AdvPatch-1K: The First Physical-world Adversarial T-Shirt Dataset'
        texten: >-
          AdvPatch-1K consists of 1,131 images, including 5+ scenarios (lab,
          outdoor, cafeteria, subway, shopping mall), 1-10+ persons in the
          images, and 9 adversarial t-shirts. The dataset is available on
          Huggingface.
    _template: features
  - titleen: Tools and Datasets
    titlezh: Models Chinese
    subtitle1en: taiadv.vision
    subtitle1zh: ''
    text1en: >-
      The taiadv.vision toolbox integrates all the methods used to create the
      adversarial image datasets and benchmarks on this platform.
    text1zh: ''
    image1:
      src: /uploads/GitHubButton.png
      href: 'https://github.com/OpenTAI/taiadv'
    bgImage1:
      src: /uploads/GitHubBackground1.jpg
    subtitle2en: CC1M-Adv-C/F
    subtitle2zh: CC1M
    text2en: >-
      Two million-scale adversarial images datasets.  CC1M-Adv-C was generated
      to evaluate classification models, while CC1M-Adv-F can be applied to any
      vision models. 
    text2zh: >-
      This comprehensive visibility allows for a meticulous examination of
      security loopholes, enabling us to simulate real-world attack scenarios
      pinpoint accuracy.
    image2:
      src: /uploads/GitHubButton.png
      href: 'https://huggingface.co/datasets/xingjunm/cc1m-adv-F'
    bgImage2:
      src: /uploads/GitHubBackground2.jpg
    subtitle3en: AdvPatch-1K
    subtitle3zh: TAI.adv
    text3en: >-
      The first physical-world adversarial T-shirt dataset released to evaluate
      the robustness of object detection models and support defense research.
    text3zh: >-
      This comprehensive visibility allows for a meticulous examination of
      security loopholes, enabling us to simulate real-world attack scenarios
      pinpoint accuracy.
    image3:
      src: /uploads/GitHubButton.png
      href: 'https://huggingface.co/datasets/xingjunm/AdvPatch-1K'
    bgImage3:
      src: /uploads/GitHubBackground3.jpg
      alt: ''
    _template: repositories
  - titleen: LeaderBoards
    titlezh: 排行榜
    leftListTitleen: Black-box Board
    leftListTitlezh: 黑盒榜单
    items1:
      - titleen: ImageNet Models
        titlezh: 'ImageNet '
        subtitleen: ImageNet
        subtitlezh: Classification
        modelSum: 10
        score: '4.8'
        detailen: Reviewed by more than 200K Users
      - titleen: Object Detection
        titlezh: 物体检测模型
        subtitleen: COCO
        subtitlezh: COCO
        modelSum: 10
        score: '4.8'
        detailen: Reviewed by more than 200K Users
        detailzh: Reviewed by more than 200K Users
      - titleen: Instance Segmentation
        titlezh: 实例分割
        subtitleen: COCO
        subtitlezh: COCO
        modelSum: 10
        score: '4.8'
        detailen: Reviewed by more than 200K Users
        detailzh: Reviewed by more than 200K Users
      - titleen: Semantic Segmentation
        titlezh: 语义分割
        subtitleen: ADE20K
        subtitlezh: ADE20K
        modelSum: 10
        score: '4.8'
        detailen: Reviewed by more than 200K Users
        detailzh: Reviewed by more than 200K Users
      - titleen: Medical Image Classification
        titlezh: 医学图像分类
        subtitleen: CheXpert
        subtitlezh: CheXpert
        modelSum: 3
        score: '4.8'
        detailen: Reviewed by more than 200K Users
        detailzh: Reviewed by more than 200K Users
    rightListTitleen: ''
    rightListTitlezh: ''
    _template: testimonial
---

