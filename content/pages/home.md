---
title: 对抗评测平台
blocks:
  - headline1en: Vision Safety Platform
    headline1zh: 视觉安全平台
    headline2en: 'Adversarial Robustness: Ensuring AI''s Robust Perception of Reality'
    headline2zh: 确保人工智能对现实世界的鲁棒感知
    buttonTexten: Tools
    buttonTextzh: 工具集
    buttonLink: /#datasets
    buttonText1en: Datasets
    buttonText1zh: 数据集
    button1Link: /#datasets
    buttonText2en: Leaderboards
    buttonText2zh: 排行榜
    button2Link: /#leaderboards
    subtitle1en: ''
    subtitle1zh: ''
    subtitle2en: ''
    subtitle2zh: ''
    text1en: ''
    text1zh: ''
    text2en: ''
    text2zh: ''
    image: /uploads/Bg.jpg
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
        titleen: A Comprehensive Toolbox for Adversarial Robustness Evaluation
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
        titleen: 'AdvT-shirt-1k: The First Physical-world Adversarial T-Shirt Dataset'
        texten: >-
          AdvT-shirt-1k consists of 1,131 images, including 5+ scenarios (lab,
          outdoor, cafeteria, subway, shopping mall), 1-10+ persons in the
          images, and 9 adversarial t-shirts. The dataset is available on
          Huggingface.
    _template: features
  - titleen: Tools and Datasets
    titlezh: Models Chinese
    repositories:
      - repoNameen: taiadv.vision
        repoNamezh: taiadv.vision
        repoTexten: >-
          The taiadv.vision toolbox integrates all the methods used to create
          the adversarial image datasets and benchmarks on this platform.
        repoTextzh: >-
          The taiadv.vision toolbox integrates all the methods used to create
          the adversarial image datasets and benchmarks on this platform.
        linkImage:
          src: /uploads/GitHubButton.png
          alt: taiadv.vision
          href: 'https://github.com/OpenTAI/taiadv'
        bgImage: /uploads/GitHubBackground1.jpg
      - repoNameen: CC1M-Adv-C/F
        repoNamezh: CC1M-Adv-C/F
        repoTexten: >-
          Two million-scale adversarial images datasets.  CC1M-Adv-C was generated
          to evaluate classification models, while CC1M-Adv-F can be applied to any
          vision models. 
        repoTextzh: >-
          This comprehensive visibility allows for a meticulous examination of
          security loopholes, enabling us to simulate real-world attack scenarios
          pinpoint accuracy.
        linkImage:
          src: /uploads/GitHubButton.png
          alt: CC1M-Adv-C/F
          href: 'https://github.com/treeman2000/CC1M-Adv-CF'
        bgImage: /uploads/GitHubBackground2.jpg
      - repoNameen: AdvT-shirt-1k
        repoNamezh: AdvT-shirt-1k
        repoTexten: >-
          The first physical-world adversarial T-shirt dataset released to evaluate
          the robustness of object detection models and support defense research.
        repoTextzh: >-
          This comprehensive visibility allows for a meticulous examination of
          security loopholes, enabling us to simulate real-world attack scenarios
          pinpoint accuracy.
        linkImage:
          src: /uploads/GitHubButton.png
          alt: AdvT-shirt-1k
          href: 'https://github.com/Wwangb/AdvT-shirt-1k'
        bgImage: /uploads/GitHubBackground3.jpg
    _template: repositories
  - titleen: Leaderboards
    titlezh: 排行榜
    leftListTitleen: ''
    leftListTitlezh: 黑盒榜单
    items1:
      - titleen: ImageNet - Image Classification
        titlezh: ImageNet
        subtitleen: Feature Attack
        subtitlezh: 特征攻击
        modelSum: 1123
        score: '4.8'
        detailen: ''
      - titleen: ADE20K - Semantic Segmentation
        titlezh: ADE20K - 语义分割
        subtitleen: Feature Attack
        subtitlezh: 特征攻击
        modelSum: 186
        score: '4.8'
      - titleen: CIFAR100 - Image Classification
        titlezh: CIFAR100 -  图像分类
        subtitleen: Label Attack
        subtitlezh: 类别攻击
        modelSum: 8
      - titleen: ImageNet - Image Classification
        titlezh: ImageNet
        subtitleen: Label Attack
        subtitlezh: 类别攻击
        modelSum: 1123
        score: '4.8'
        detailen: ''
        detailzh: Reviewed by more than 200K Users
      - titleen: COCO - Object Detection
        titlezh: COCO - OD
        subtitleen: Feature Attack
        subtitlezh: 特征攻击
        modelSum: 394
        score: '4.8'
        detailen: ''
        detailzh: Reviewed by more than 200K Users
      - titleen: COCO - Instance Segmentation
        titlezh: 语义分割
        subtitleen: Feature Attack
        subtitlezh: 特征攻击
        modelSum: 152
        score: '4.8'
        detailen: ''
        detailzh: Reviewed by more than 200K Users
      - titleen: CIFAR10 - Image Classification
        titlezh: CIFAR10 - 图像分类
        subtitleen: Label Attack
        subtitlezh: 类别攻击
        modelSum: 10
      - titleen: CC1M - Image Classification
        titlezh: CC1M - 图像分类
        subtitleen: Label Attack
        subtitlezh: 类别攻击
        modelSum: 5
      - titleen: CheXpert - Medical Image Classification
        titlezh: CheXpert - 医学图像分类
        subtitleen: Feature Attack
        subtitlezh: 特征攻击
        modelSum: 3
        score: '4.8'
        detailen: ''
        detailzh: ''
    rightListTitleen: ''
    rightListTitlezh: ''
    _template: testimonial
---
