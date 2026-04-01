# Hallucinated Adversarial Learning for Robust Visual Tracking

Qiangqiang $\mathbf { W } \mathbf { u } ^ { 1 }$ , Zhihui Chen1, Lin Cheng1, Yan $\mathbf { Y a n } ^ { 1 }$ , Bo Li2, Hanzi Wang1∗

1 Department of Computer Science, Xiamen University, Xiamen, China

2 Department of Computer Science and Engineering, Beihang University, Beijing, China qiangwu $@$ stu.xmu.edu.cn, {yanyan, hanzi.wang}@xmu.edu.cn, zhihui.qz.chen@gmail.com, cheng.charm.lin $@$ hotmail.com, libo $@$ buaa.edu.cn

# Abstract

Humans can easily learn new concepts from just a single exemplar, mainly due to their remarkable ability to imagine or hallucinate what the unseen exemplar may look like in different settings. Incorporating such an ability to hallucinate diverse new samples of the tracked instance can help the trackers alleviate the over-fitting problem in the low-data tracking regime. To achieve this, we propose an effective adversarial approach, denoted as adversarial “hallucinator” (AH), for robust visual tracking. The proposed AH is designed to firstly learn transferable non-linear deformations between a pair of same-identity instances, and then apply these deformations to an unseen tracked instance in order to generate diverse positive training samples. By incorporating AH into an online tracking-bydetection framework, we propose the hallucinated adversarial tracker (HAT), which jointly optimizes AH with an online classifier (e.g., MDNet) in an end-to-end manner. In addition, a novel selective deformation transfer (SDT) method is presented to better select the deformations which are more suitable for transfer. Extensive experiments on 3 popular benchmarks demonstrate that our HAT achieves the state-of-the-art performance.

# 1 Introduction

Given the initial state of a target at the first frame, generic visual tracking aims to estimate the trajectory of the target at subsequent frames in a video. Despite the outstanding success achieved by deep convolutional neural networks (CNNs) in a variety of computer vision tasks [He et al., 2016; Wang et al., 2018c], their impact in visual tracking is still limited. The main reason is that deep CNNs greatly rely on the large-scale annotated training data. For online tracking, it is impossible to gather enough training data since the tracker is required to track arbitrary objects. Thus the problem of learning an effective CNN model for visual tracking is particularly challenging, mainly due to limited online training data.

To alleviate this problem, one strategy is to treat visual tracking as a more general similarity learning problem, thus enabling deep CNNs (e.g., Siamese networks) to be trained

![](images/ed770ca9bc82d3486b290c7c909a1c671752adbbb2262ce8e35ce9fa87ab2688.jpg)  
Figure 1: Given an unseen exemplar (a bear), humans can effectively hallucinate what the bear may look like in different views or poses based on their previous learning experience. We propose an adversarial hallucinator, which mimics such hallucination or imagination for robust visual tracking.

with large-scale annotated datasets in an offline manner. However, these Siamese network based trackers [Bertinetto et al., 2016b] still cannot achieve high accuracies on the benchmarks, since they inherently lack the online adaptability. Another strategy is to effectively leverage few online training samples and adopt the online learning based tracking-bydetection schema. This schema based trackers maintain an online CNN classifier, which models the temporal variations of the target appearance by updating the network parameters. Compared with the Siamese network based trackers, they gain a large accuracy improvement. However, they may easily suffer from the over-fitting problem due to the limited online training samples (especially for the positive training samples), thus leading to suboptimal tracking performance.

Compared with state-of-the-art CNN-based trackers, visual tracking is a relatively simple task for humans. Although how human brain works is far from being fully understood, one can conjecture that humans have a remarkable imaginary mechanism derived from their previous learning experience. That is, as illustrated in Fig. 1, humans can firstly learn many shared modes of variation (e.g., rotation, illumination change and translation) from different pairs of same-identity instances. Then they are able to hallucinate what novel instances look like in different surroundings or poses, by ap-

plying their previous learned modes of variation to novel instances. For example, we can learn the motion of rotation from a windmill. Based on it, we can easily imagine how a completely different windmill or even an electric fan rotates. Interestingly, it seems that we build a visual classifier in the brain and then hallucinate novel deformable samples of the exemplar to train the classifier, which is particularly similar to the data augmentation technique in machine learning.

In this paper, our main motivation is to help CNN-based trackers do such “imagination” or “hallucination”, so that they can achieve robust tracking in the low-data tracking regime. To achieve this, we propose a novel adversarial “hallucinator” (AH), which is based on an encoder-decoder generative adversarial network. The proposed AH learns nonlinear deformations using different instance pairs collected from large-scale datasets, and then AH can effectively generate new deformable samples of an unseen instance $x$ by applying the learned deformations to $x$ . For the offline training of AH, we present a deformation reconstruction loss, which enables AH to be trained in a self-supervised manner without the need of mining additional samples. Based on AH, we further propose HAT (hallucinated adversarial tracker), which incorporates AH into a tracking-by-detection framework. Specifically, our AH is jointly optimized with the online CNN classifier (MDNet [Nam and Han, 2016]) in an endto-end manner, which can effectively help the CNN classifier alleviate the over-fitting problem, thus leading to better generalization of the tracker. In addition, we present a novel selective deformation transfer (SDT) method to effectively select the suitable deformations for better transfer. To sum up, this paper has the following contributions:

• We propose an adversarial hallucination method, namely adversarial “hallucinator” (AH), which mimics the human imaginary mechanism for data augmentation. To effectively train AH in a self-supervised manner, a novel deformation reconstruction loss is presented.   
• Based on AH, we present a hallucinated adversarial tracker (HAT), which jointly optimizes our AH and the online classifier (MDNet) in an end-to-end manner. The joint learning schema can effectively help the online classifier alleviate the over-fitting problem.   
• We propose a novel selective deformation transfer (SDT) method to better select the deformations which are more suitable for transfer.

We perform comprehensive experiments on three popular benchmarks: OTB-2013 [Wu et al., 2013], OTB-2015 [Wu et al., 2015], and VOT-2016 [Kristan et al., 2016]. Experimental results demonstrate that our HAT performs favorably against state-of-the-art trackers. In particularly, HAT achieves the leading accuracy $( 9 5 . 1 \% )$ on OTB-2013.

# 2 Related Work

CNN-based tracking. The CNN-based trackers typically employ an one-stage template matching framework or a twostage classification framework. The representative one-stage framework is based on deep Siamese networks. Starting from SiamFC [Bertinetto et al., 2016b], many efforts have been

made, including target template modeling [Yang and Chan, 2018], proposal generation [Li et al., 2018], network architecture design [Wang et al., 2018a], and loss function design [Li et al., 2017]. In comparison, the two-stage framework consists of two steps: 1) Sample target candidates around the estimated location in the previous frame. 2) Classify all the candidates to obtain the best one in terms of the classification score. The pioneering work of the CNN-based two-stage framework is MDNet [Nam and Han, 2016]. Based on MD-Net, extensions include meta learning [Park and Berg, 2018], adversarial learning [Song et al., 2018], online regularization [Han et al., 2017], and reciprocative learning [Pu et al., 2018]. Although much progress has been made, these trackers still suffer from the over-fitting problem due to limited online samples, which severely impedes the great potential of CNNs for visual tracking. In this paper, we extend the twostage framework by incorporating our AH, which effectively mimics the human imaginary mechanism to hallucinate novel positive samples for training a more robust CNN classifier.

Visual tracking by data augmentation. To facilitate the learning of few online training samples in visual tracking, several data augmentation based trackers have been proposed, including UPDT [Bhat et al., 2018], $\mathrm { S I N T + + }$ [Wang et al., 2018b] and VITAL [Song et al., 2018]. More specifically, UPDT uses several simple data augmentation techniques (e.g., shift, blur and rotation) for correlation filter tracking. $\mathrm { S I N T + + }$ and VITAL respectively employ deep reinforcement learning and adversarial learning to generate hard positive samples that are occluded by the randomly generated rectangular masks. However, they only use well-designed geometric transforms or fixed shapes of occlusion masks to generate plausible samples. There exists a big gap between the generated samples and real deformable samples. In comparison, the proposed AH learns various non-linear deformations from real instance pairs for transfer.

Augmentation-based few-shot learning. Recently, several augmentation-based few-shot learning methods have been proposed. In [Antoniou et al., 2017] and [Wang et al., 2018c], a data augmentation generative adversarial network and a multilayer perceptron are respectively proposed to randomly generate additional samples. Based on the assumption that the deformation between two same-class instances is linear, the authors in [Hariharan and Girshick, 2017] train their generator in a supervised manner. [Schwartz et al., 2018] improves the above assumption and proposes an encoder-decoder network to learn deformations in a selfreconstruction manner. However, such a way may lead to the domain shift problem. In this work, we overcome this problem by applying the learned deformations between a pair of same-identity instances to instances with other identities via adversarial learning, which makes AH generalize better to unseen instances, even for low-quality online tracked instances.

# 3 Proposed Method

In this section, we firstly introduce the proposed adversarial hallucinator (AH), and then propose the selective deformation transfer (SDT) method for better hallucination. Finally, we detail our hallucinated adversarial tracker (HAT).

# 3.1 The Adversarial Hallucinator

The goal of this paper is to enable the CNN-based trackers to have the capability of “imagination” or “hallucination” like humans. Inspired by several recent works [Schwartz et al., 2018; Wang et al., 2018c] in few-shot learning, we propose a novel adversarial approach, namely adversarial “hallucinator” (AH), which consists of an encoder and a decoder, to hallucinate diverse positive samples for visual tracking. The encoder in AH learns to extract non-linear transferable deformations between pairs of same-identity instances, while the decoder applies these deformations to an unseen instance in order to generate diverse reasonable deformable samples of the instance.

To effectively train AH, we collect a large number of pairs of same-identity instances from the snippets in the ImageNet-VID [Russakovsky et al., 2015] dataset. We call this dataset as $\mathbb { D } _ { T }$ . For each pair of instances $( x _ { 1 } ^ { a } , \ x _ { 2 } ^ { a } )$ with the same identity $a$ in $\mathbb { D } _ { T }$ , we randomly select another pair of instances $( x _ { 1 } ^ { b } , ~ x _ { 2 } ^ { \bar { b } } )$ with different identity $b$ to constitute a quadruplet training sample $( x _ { 1 } ^ { a } , x _ { 2 } ^ { a } , x _ { 1 } ^ { b } , x _ { 2 } ^ { \bar { b } } )$ ). We collect a large number of such quadruplet training samples to generate a new dataset $\mathbb { D } _ { Q }$ . Then we use the pre-trained feature extractor $\phi ( \cdot )$ to extract deep features of all the instances in $\mathbb { D } _ { Q }$ . Therefore, the instances in $\mathbb { D } _ { Q }$ are represented by these pre-computed feature vectors, which are more suitable for training.

We now use the dataset $\mathbb { D } _ { Q }$ to train the proposed AH (i.e., the generator $G$ ) comprised of the encoder $E _ { n }$ and the decoder $D _ { e }$ . For each quadruplet training sample $( x _ { 1 } ^ { a } , x _ { 2 } ^ { a } , x _ { 1 } ^ { b }$ $x _ { 2 } ^ { b } )$ , we feed the concatenated feature vector $[ \phi ( x _ { 1 } ^ { a } ) , \phi ( x _ { 2 } ^ { a } ) ]$ to the encoder $E _ { n }$ . Let $z ^ { a } = E _ { n } ( [ \phi ( x _ { 1 } ^ { a } ) , \phi ( x _ { 2 } ^ { a } ) ] )$ be the encoder output, which is a low-dimensional vector and represents a plausible transformation. Then, we apply this transformation to a novel instance $x _ { 1 } ^ { b }$ in order to generate a reasonable deformable sample $\hat { x } ^ { b } \overset { ^ { \bullet } } { = } D _ { e } ( [ z ^ { a } , \phi ( x _ { 1 } ^ { \overline { { b } } } ) ] )$ .

To guarantee that the generated sample ${ \hat { x } } ^ { b }$ has the same identity as the input $\phi ( x _ { 1 } ^ { b } )$ , i.e., their data distributions should be similar, we employ a discriminator $D$ , which is jointly optimized with our AH in an adversarial manner. The whole optimization process can be formulated as:

$$
\begin{array}{l} \mathcal {L} _ {a d v} = \min  _ {G} \max  _ {D} \mathbb {E} _ {x _ {1} ^ {b}, x _ {2} ^ {b} \sim P _ {d a t a} (x _ {1} ^ {b})} \left[ \log D ([ \phi (x _ {1} ^ {b}), \phi (x _ {2} ^ {b}) ]) \right] \\ + \mathbb {E} _ {x _ {1} ^ {b} \sim P _ {d a t a} \left(x _ {1} ^ {b}\right), \hat {x} ^ {b} \sim P _ {d a t a} \left(\hat {x} ^ {b}\right)} \left[ \log \left(1 - D \left(\left[ \phi \left(x _ {1} ^ {b}\right), \hat {x} ^ {b} \right]\right)\right) \right], \tag {1} \\ \end{array}
$$

where $G = \{ E _ { n } , D _ { e } \}$ . The proposed AH tries to minimize the adversarial loss $\mathcal { L } _ { a d v }$ while $D$ aims to maximize it. That is, AH aims to generate samples that fit $P _ { d a t a } ( x ^ { b } )$ so that the discriminator cannot discriminate the real pair $( x _ { 1 } ^ { b } , x _ { 2 } ^ { b } )$ from the fake pair $( x _ { 1 } ^ { b } , D _ { e } ( [ z ^ { a } , \phi ( x _ { 1 } ^ { b } ) ] ) )$ . Thus, by optimizing the adversarial loss, the proposed AH can effectively generate samples having the same identity as the input $x _ { 1 } ^ { b }$ .

Deformation Reconstruction Loss. By solely minimizing the adversarial loss to train AH, we can only get a random generated sample $D _ { e } ( [ z ^ { a } , \phi ( x _ { 1 } ^ { b } ) ] )$ that has the same identity as the input $x _ { 1 } ^ { \hat { b } }$ . However, we cannot ensure that the generated ${ \hat { x } } ^ { b }$ effectively learns the non-linear deformation between $x _ { 1 } ^ { a }$ and $x _ { 2 } ^ { a }$ . In [Hariharan and Girshick, 2017], based on the linear deformation assumption, the authors mine various resulting samples and use these samples to train the network in

![](images/cd21a1fb3a341701566a2c5a7c2dfb84bfca3b8a1f9441ee442e18df04072185.jpg)  
Figure 2: Illustration of our deformation reconstruction loss.

a standard supervised fashion. But since the actual deformations considered in this paper are non-linear, their method is limited and it is even impossible to mine accurate non-linear transformed samples.

To solve this problem, we present a deformation reconstruction (DR) loss, which can be used to train the proposed AH in a self-supervised manner. Assume that the generated sample ${ \hat { x } } ^ { b }$ correctly encodes the transformation $z ^ { a }$ between $x _ { 1 } ^ { a }$ and $x _ { 2 } ^ { a }$ . Then, the original sample $x _ { 2 } ^ { a }$ should also be reconstructed by applying the transformation between $x _ { 1 } ^ { b }$ and the generated ${ \dot { \boldsymbol { x } } } ^ { b }$ to $x _ { 1 } ^ { a }$ . Therefore, we define a DR loss, which can be described as:

$$
\mathcal {L} _ {d e f} = \left\| D _ {e} \left(\left[ z ^ {b}, \phi \left(x _ {1} ^ {a}\right) \right]\right) - \phi \left(x _ {2} ^ {a}\right) \right\| _ {2}, \tag {2}
$$

where $z ^ { b } = E _ { n } ( [ \phi ( x _ { 1 } ^ { b } ) , { \hat { x } } ^ { b } ] )$ . As illustrated in Fig. 2, our AH can learn how to correctly perform deformation transfer by itself, and we do not need to use additional samples. The overall loss function of the proposed adversarial hallucinator is written as:

$$
\mathcal {L} _ {\text {o v e r a l l}} = \mathcal {L} _ {\text {a d v}} + \lambda \mathcal {L} _ {\text {d e f}}, \tag {3}
$$

where $\lambda$ is a hyper-parameter.

# 3.2 Selective Deformation Transfer

For humans, we can easily learn deformations from different classes of objects, and applying these deformations to a similar or the same class of objects is easier to be done than applying them to a totally different class of objects. For example, we learn different pose variations from one person, and then those variations are more reasonable and easier to be transferred to another person rather than a car. Based on this observation, we present our selective deformation transfer (SDT) method to better select deformations which are more suitable for transfer.

Let we denote the training dataset $\mathbb { D } _ { T }$ and the online tracking videos as the source domain and the target domain, respectively. In our SDT method, we do not use all the pairs of instances $( x _ { 1 } ^ { a } , \ x _ { 2 } ^ { a } )$ from the source domain to perform data augmentation on the initial target exemplar $x ^ { e }$ of a given video in the target domain. Instead, for each $x ^ { e }$ , we search a certain number of snippets with similar high-level or semantic characteristics from the source domain. We only use the instances in the snippets, according to the searching results, to perform data augmentation in the online tracking process.

Snippet descriptor. Let $N _ { s }$ be the number of snippets in the source domain $\mathbb { D } _ { T }$ . We describe each snippet descriptor in the source domain as $\{ \psi ( s _ { i } ) \} _ { i = 1 } ^ { N _ { s } }$ , where $\psi ( \cdot )$ represents a descriptor calculation function and $s _ { i }$ is the snippet with the identity $i$ . Moreover, $s _ { i }$ can be further described as

![](images/ad796ae0c3dcd02cdc2b5380f8f01518085ee7a2479370589891e16e43a551ff.jpg)  
Exemplars

![](images/a220f95936214b19e30d4df884a6f19c7739b3b17a9fe9dd5290d0b572686d97.jpg)

![](images/337b4de24daf63e63eed2f2fe9284e562b69667a64b210aace5925a13fd703cb.jpg)

![](images/003f06eb9598dd19c8985111eaee993e26ab9534e37344f6dd3419e29e6935e7.jpg)  
Snippets   
Figure 3: Snippets in the source domain that have similar semantic characteristics with target exemplars. The 1st column shows target exemplars from online test videos. The 2nd to 4th columns indicate the corresponding 1st, 2nd and 3rd nearest snippets in the source domain. Each snippet is visualized using its first instance.

$s _ { i } = \{ x _ { j } ^ { i } \} _ { j = 1 } ^ { N _ { s _ { i } } }$ Ns , where order to $N _ { s _ { i } }$ is the number of instances in thech for suitable snippets, an essen-$s _ { i }$ tial problem is how to effectively calculate snippet descriptors. Since our goal is to search instances in the snippets that have similar or the same classes with the target exemplar $x ^ { e }$ , we use the deep features extracted from the very deep convolutional layers, which provide rich high-level or semantic information, to calculate each snippet descriptor $\psi ( s _ { i } )$ :

$$
\psi \left(s _ {i}\right) = \frac {1}{N _ {s _ {i}}} \sum_ {j = 1} ^ {N _ {s _ {i}}} \varphi \left(x _ {j} ^ {i}\right), \tag {4}
$$

where $\varphi ( \cdot )$ denotes the convolutional feature extractor. Note that almost any existing deep CNNs can be used as our feature extractor $\varphi ( \cdot )$ . In this work, we use the pre-trained ResNet34 [He et al., 2016] model and remove the last fully-connected layer for feature extraction.

Nearest neighbor ranking. After obtaining snippet descriptors $\{ \psi ( s _ { i } ) \} _ { i = 1 } ^ { N _ { v } }$ , for the target exemplar descriptor $\varphi ( x ^ { e } )$ in a test video, we search for its nearest-neighbor snippets in the source domain. Specifically, we calculate the Euclidean distance between $\varphi ( x ^ { e } )$ and each snippet descriptor, and rank the snippets in the ascending order. We select the top $T$ snippets $\stackrel { \cdot } { \{ s _ { j } \} } _ { j = 1 } ^ { T }$ , and collect pairs of same-identity instances from the selected snippets into a dataset $\mathbb { D } _ { S }$ for transfer. In order to reduce the retrieval time, we perform feature extraction for all the instances in the source domain in an offline manner, and calculate the snippet descriptors in advance. Note that there are about 6, 500 snippets in the source domain, and the whole retrieval step can be implemented within 3 seconds.

Fig. 3 shows that the proposed SDT method can effectively select the snippets that have similar semantic characteristics with the target exemplar into $\mathbb { D } _ { S }$ for better hallucination.

# 3.3 Proposed Tracking Algorithm

After training the proposed AH in an end-to-end offline manner and obtaining the selective dataset $\mathbb { D } _ { S }$ , we illustrate how we perform hallucinated tracking in an online tracking-bydetection framework. The details of three main components of HAT are given as follows:

Joint model initialization. Given the initial target exemplar $x ^ { e }$ in the first frame, we randomly draw 32 positive samples and 96 negative samples around it as in [Nam and Han, 2016]

![](images/462a3ca5106092f2cc6e29a53640cb1cc1702abb694937ad4b1a14ec495b9e3d.jpg)  
Figure 4: Online tracking with hallucination. Given the target exemplar of a test video, we use AH to learn non-linear deformations from pairs of instances in $D _ { S }$ and hallucinate diverse positive samples based on the exemplar. We create an augmented positive set by adding the hallucinated positive samples. The online classifier is jointly optimized with AH. The black and dotted red arrows respectively indicate the forward and back-propagation steps.

in each iteration. Since these samples are highly spatially overlapped, they cannot capture rich appearance variations. Thus directly using these samples to train the network may lead to the over-fitting problem. To alleviate this problem, we randomly select various pairs of instances in the selective dataset $\mathbb { D } _ { S }$ , and use the proposed AH to learn reasonable deformations in the pairs, and then apply those deformations to the target exemplar $x ^ { e }$ in order to generate diverse deformable target samples, which are labeled as positive. As illustrated in Fig. 4, we use both the augmented positive samples and negative samples to jointly update AH and the fully-connected layers in the classifier for $N _ { 1 }$ iterations.

Online detection. Given an input frame, we first randomly draw samples around the target location estimated in the previous frame. Then, we feed all these samples to the classifier in order to select the best candidate with the highest classification score. Finally, we refine the final target location using the bounding box regression as in [Nam and Han, 2016].

Joint model update. The joint model update step is similar to the joint model initialization step. First, in each frame, we randomly draw positive and negative samples around the estimated target location. Second, we perform data augmentation to the target exemplar $x ^ { e }$ using the proposed AH as described in the joint model initialization step. Finally, we use all the samples to jointly update the fully-connected layers in the classifier and the proposed AH for $N _ { 2 }$ iterations.

Note that our AH learns generic deformations during the offline learning step. Based on the well-learned offline model, AH can effectively adapt to the online specific deformations of the tracked instance by jointly updating with the classifier.

Jointly training AH and the online classifier has two main benefits. First, there still exists the domain gap between the offline training data and the online tracking data, our joint learning schema makes allowances for errors made by AH due to the domain gap. Second, the joint learning facilitates AH to generate diverse complement positive samples that are more useful for classification, which helps the classifier generalize well in the low-data tracking regime.

Table 1: DPRs $( \% )$ and AUCs $( \% )$ obtained by the variations of the proposed HAT and the baseline tracker on the OTB-2013, OTB-2015 and OTB-50 datasets. $r$ represents the ratio of positive and negative training samples. The red bold fonts and blue italic fonts respectively indicate the best and the second best results.   

<table><tr><td rowspan="2">Trackers</td><td colspan="2">OTB-2013</td><td colspan="2">OTB-2015</td><td colspan="2">OTB-50</td></tr><tr><td>DPR</td><td>AUC</td><td>DPR</td><td>AUC</td><td>DPR</td><td>AUC</td></tr><tr><td>Base-MDNet (r = 1/3)</td><td>90.9</td><td>66.8</td><td>87.3</td><td>64.3</td><td>82.2</td><td>58.6</td></tr><tr><td>HAT (r = 2/3)</td><td>91.4</td><td>67.4</td><td>90.2</td><td>66.1</td><td>86.7</td><td>61.6</td></tr><tr><td>SDT-HAT (r = 2/3)</td><td>92.6</td><td>68.6</td><td>90.3</td><td>66.5</td><td>87.2</td><td>62.0</td></tr><tr><td>HAT (r = 1/1)</td><td>93.2</td><td>68.7</td><td>90.8</td><td>66.3</td><td>87.9</td><td>62.3</td></tr><tr><td>SDT-HAT (r = 1/1)</td><td>95.1</td><td>69.6</td><td>91.6</td><td>66.9</td><td>89.4</td><td>63.2</td></tr></table>

Table 2: DPRs $( \% )$ and AUCs $( \% )$ obtained by our HAT equipped with an AH with (HAT-w-Up) and without (HAT-w/o-Up) online update on the OTB-2013, OTB-2015 and OTB-50 datasets.   

<table><tr><td rowspan="2">Trackers</td><td colspan="2">OTB-2013</td><td colspan="2">OTB-2015</td><td colspan="2">OTB-50</td></tr><tr><td>DPR</td><td>AUC</td><td>DPR</td><td>AUC</td><td>DPR</td><td>AUC</td></tr><tr><td>HAT-w/o-Up</td><td>91.8</td><td>68.0</td><td>90.0</td><td>66.4</td><td>86.6</td><td>62.0</td></tr><tr><td>HAT-w-Up</td><td>95.1</td><td>69.6</td><td>91.6</td><td>66.9</td><td>89.4</td><td>63.2</td></tr></table>

# 4 Experiments

# 4.1 Implementation Details

Network architecture and training. We use the original network architecture in MDNet [Nam and Han, 2016] as our backbone network, which consists of three convolutional layers and three randomly initialized fully-connected layers. Since HAT does not need to perform the multi-domain learning, the three convolutional layers in HAT share the same weights with the first three convolutional layers in VGG-M [Simonyan and Zisserman, 2015]. The encoder and decoder sub-networks in AH and the discriminator are all designed as three-layer perceptrons with a single hidden layer of 2048 units. The output of the encoder is a 64-dimensional vector. The discriminator output is a probability score. Each layer in these networks is followed by a ReLU activation. Each pair of instances in $\mathbb { D } _ { T }$ is collected from two frames (within the nearest 20 frames) in a snippet from ImageNet-VID. For the offline training of AH, the hyper-parameter $\lambda$ in Eqn. (3) is experimentally set to 0.5. Since the original adversarial loss may lead to unstable training, we use the gradient penalty as in [Gulrajani et al., 2017] for stable training. We use the Adam solver to optimize both the AH and the discriminator with a learning rate of $2 \times 1 0 ^ { - 4 }$ for $5 \times 1 0 ^ { 5 }$ iterations. For the joint model initialization step in online tracking, AH is optimized by using the Adam solver with a learning rate of $1 . 2 \times 1 0 ^ { - 4 }$ for 35 iterations. In the joint model update step, we train our AH for 15 iterations. For the SDT method, $T$ is set to 2, 000.

We implement our method using PyTorch [Paszke et al., 2017] on a computer with an i7-4.0 GHz CPU and a GeForce GTX 1080 GPU. The average tracking speed is 1.6 FPS.

Instance representation. In all the experiments, the in-

![](images/6347ad48bb9dbde5c22667c860078d1b90177e92ae3833a99854ea4c58cd7ee4.jpg)  
(a)

![](images/2ff2453976c9e5a76bdd55a96c9917d86c010994bab9af8fa2e89ec003bc71b8.jpg)  
(b)

![](images/c7449b7d087ffe06afccef4f6bb1280c5e6d48b62cb269f18362f3b755b13c8c.jpg)

![](images/d56bc69a10a0e3736c6d57e93746e679a47973e29926fbd3ed09c10a379091ab.jpg)  
(c)   
Figure 5: (a, b) t-SNE visualizations of the samples hallucinated by the offline learned AHs trained (a) with and $\mathbf { ( b ) }$ without the use of the DR loss. A pair of two instances $( x _ { 1 } ^ { a } , x _ { 2 } ^ { a } )$ used for deformation extraction are shown as a circle and a polygon. By applying the deformation to a novel instance $x _ { 1 } ^ { b }$ (cross), AH can effectively hallucinate a deformable sample ${ \hat { x } } ^ { b }$ (plus). Each quadruplet set $( x _ { 1 } ^ { a }$ , $x _ { 2 } ^ { a } , x _ { 1 } ^ { b } , { \hat { x } } ^ { b } )$ is colored uniquely. (c) Hallucinated sample visualization. The hallucinated samples (plus) in (a) are visualized using their nearest real-instance neighbors in the feature space.

stances are represented by pre-computed feature vectors. We use the fixed three convolution layers in the backbone network of the proposed HAT as the feature extractor $\phi ( \cdot )$ . We first resize all the instances to a fixed size of $1 0 7 \times 1 0 7 \times 3$ , and then feed them to the feature extractor in order to extract 4608-dimensional feature vectors.

Evaluation methodology. For the OTB datasets, we adopt the distance precision (DP) and overlap success (OS) plots for evaluation. We report both the DP rates at the threshold of 20 pixels (DPR) and the Area Under the Curve (AUC). For VOT2016, the expected average overlap (EAO), accuracy, failures and robustness are adopted to evaluate each tracker.

# 4.2 Ablation Study

First, we analyze the factors that may affect the performance of HAT in Table 1, including the SDT method and the ratio $r$ of positive and negative training samples. Then, we analyze the impact of the online update of AH as shown in Table 2. Finally, we analyze the influence of the proposed DR loss and visualize the learned hallucinations in Fig. 5. Note that our baseline tracker is MDNet, which does not perform the multidomain learning for fair comparison. Since MDNet uses 32 positive samples and 96 negative samples in each iteration for training, we simply call it as Base-MDNet $( r = 1 / 3$ ).

SDT method. In Table 1, by employing the proposed SDT method, HAT gains the improvements on all the three datasets in terms of both DPR and AUC. Specifically, SDT-HAT $( r ~ = ~ 1 / 1 )$ achieves the best DPR $( 9 5 . 1 \% )$ by improving $1 . 9 \%$ of HAT $( r = 1 / 1 )$ ) on OTB-2013, which can be explained that the instances selected by our SDT method share more semantic characteristics with the exemplar, thus providing more reasonable transformations for hallucination.

Ratios of positive and negative training samples. Since the original ratio of positive and negative samples used in MDNet is $1 / 3$ (or 32/96), there still exists the data imbalance problem. We use our AH to hallucinate diverse positive samples such that the ratios of augmented positive samples and negative samples are respectively 2/3 (or 64/96) and

![](images/9d2a0327232b3ec6f8cb0f0fc299ef64e48b93a4c23e4faf9d1fc54c44c8acf6.jpg)

![](images/8684e268ad28168118ffa8fb85ccaca0f810df4754b0abdef7332c3af1a240d6.jpg)  
Figure 6: Precision and success plots on OTB-2013 using one pass evaluation.

$1 / 1$ (or 96/96). In Table 1, we can find that the tracking performance is significantly improved by adding more hallucinated positive samples for learning. The promising results (DPR: $9 5 . 1 \%$ , AUC: $6 9 . 6 \%$ can be achieved by HAT on OTB-2013 when the ratio is set to $1 / 1$ , which demonstrates that the balanced data can lead to better results.

Impact of online update. As shown in Table 2, we can find that HAT-w-Up significantly outperforms HAT-w/o-Up in all the three datasets in terms of both DPR and AUC metrics. This is because that online update of AH can effective alleviate the domain gap between the offline training data and the online tracking data. In addition, even without online updating AH, HAT-w/o-Up still achieves much better results than the baseline tracker as shown in Table 1, which demonstrates the effectiveness of the proposed AH.

Influence of DR loss. We apply t-SNE to visualize the samples hallucinated by the offline learned AHs trained by using our DR loss in Fig. 5(a) or without using it in Fig. 5(b). Note that the transformed instances (crosses) are unseen during the offline training. As can be seen, the AH trained by using the DR loss generates the samples that keep more information of the original learned deformations. For example, since the deformation between the two instances (red circle and polygon) in Fig. 5 is relatively large, the generated sample (red plus) in Fig. 5(a) also lies relatively far away from its exemplar (red cross), which indicates similar large deformation. In comparison, in Fig. 5(b), without using the DR loss, the learned AH tends to generate random samples (lie close to their exemplars (crosses)), which are irrelevant to the applied deformations.

Visualizing the hallucinated samples. We visualize the hallucinated samples in Fig. 5(c) using their nearest realinstance neighbors in the validation set (including 83, 996 instances), which demonstrates that AH can effectively hallucinate reasonable deformable samples.

# 4.3 Evaluations on OTB-2013 and OTB-2015

We compare HAT with 13 state-of-the-art trackers including MDNet [Nam and Han, 2016], VITAL [Song et al., 2018], MetaSDNet [Park and Berg, 2018], ADNet [Yun et al., 2017], SiamRPN [Li et al., 2018], CCOT [Danelljan et al., 2016], SiamFC [Bertinetto et al., 2016b], TRACA [Choi et al., 2018], MCPF [Zhang et al., 2017], CREST [Song et al., 2017], HDT [Qi et al., 2016], HCFT [Ma et al., 2015] and DeepSRDCF [Danelljan et al., 2015]. For fair compari-

![](images/a6fe2864fb67f8d5a13f22f22e9611f7ff001c054ab795f2052d868db5ce910c.jpg)

![](images/231cb203edb4790e93f72e19f6ec1edce7b1b5a06b0f60838f74a7e269b8003d.jpg)  
Figure 7: Precision and success plots on OTB-2015 using one pass evaluation.

Table 3: The EAO, accuracy (Acc.), failures (Fai.) and robustness (Rob.) obtained by HAT and five state-of-the-art trackers on VOT2016. The best and the second best results are highlighted by the red bold and blue italic fonts, respectively.   

<table><tr><td></td><td>HAT</td><td>MetaSDNet</td><td>CCOT</td><td>VITAL</td><td>MDNet</td><td>Staple</td></tr><tr><td>EAO</td><td>0.32</td><td>0.31</td><td>0.33</td><td>0.32</td><td>0.26</td><td>0.30</td></tr><tr><td>Acc.</td><td>0.58</td><td>0.54</td><td>0.54</td><td>0.56</td><td>0.54</td><td>0.55</td></tr><tr><td>Fai.</td><td>16.52</td><td>17.36</td><td>16.58</td><td>18.37</td><td>21.08</td><td>23.90</td></tr><tr><td>Rob.</td><td>0.27</td><td>0.26</td><td>0.24</td><td>0.27</td><td>0.34</td><td>0.38</td></tr></table>

son, we do not apply the multi-domain learning for MDNet. We only report the top 10 trackers for presentation clarity.

Fig. 6 shows the results achieved by all the trackers on the OTB-2013 dataset. More particularly, the DPR obtained by HAT is $9 5 . 1 \%$ , which is the leading accuracy on the OTB-2013 dataset. Furthermore, HAT achieves the best AUC $( 6 9 . 6 \% )$ among all the compared trackers, outperforming its baseline tracker MDNet with a large margin of $2 . 8 \%$ . Compared with the VITAL tracker, the DPR and AUC obtained by our HAT are both higher than those obtained by VITAL on OTB-2013, which empirically shows the superiority of the proposed hallucination method. In Fig. 7, HAT achieves the best accuracy $( 9 1 . 6 \% )$ on OTB-2015 followed by VITAL $( 8 9 . 9 \% )$ , CCOT $( 8 9 . 6 \% )$ and ADNet $( 8 8 . 0 \% )$ . This comparison shows the highest accuracy achieved by HAT among the state-of-the-art deep trackers. In addition, HAT obtains the comparable AUC $( 6 6 . 9 \% )$ to VITAL $( 6 7 . 0 \% )$ on OTB-2015, and it is better than the others. Overall, compared with the 13 state-of-the-art trackers, HAT achieves the better overall performance on the OTB-2013 and OTB-2015 datasets.

# 4.4 Evaluation on VOT-2016

The proposed HAT is evaluated on VOT-2016 [Kristan et al., 2016] with the comparison to the state-of-the-art trackers including VITAL [Song et al., 2018], MetaSDNet [Park and Berg, 2018], CCOT [Danelljan et al., 2016], MDNet [Nam and Han, 2016] and Staple [Bertinetto et al., 2016a].

As can be seen from Table 3, the proposed HAT and VI-TAL achieve the comparable EAO (0.32) to CCOT (0.33), and HAT significantly outperforms the baseline tracker MD-Net (0.26) with a relative gain of $23 \%$ , which demonstrates the overall effectiveness of the proposed tracker. In terms of accuracy, HAT obtains a relative gain of $4 \%$ compared to VITAL, meanwhile outperforming the other trackers with

large margins. Furthermore, our tracker achieves the best results (16.52) in terms of failures, which indicates its stability in visual tracking. In terms of robustness, HAT and VI-TAL achieves the comparable robustness (0.27) to MetaSD-Net (0.26). Meanwhile, HAT outperforms MetaSDNet on the other metrics, including EAO, accuracy and failures.

# 5 Conclusions and Future Work

Inspired by the human imaginary mechanism, we propose an adversarial hallucinator (AH) for data augmentation. A novel deformation reconstruction loss is introduced to train AH in a self-supervised manner. By incorporating AH into a trackingby-detection framework, the hallucinated adversarial tracker (HAT) is proposed. HAT jointly optimizes AH with the classifier (MDNet) in an end-to-end manner, which facilitates AH to generate diverse positive samples for reducing tracking failures. In addition, we present a novel selective deformation transfer method to further improve the tracking performance. Experiments on three popular datasets demonstrate that HAT achieves the state-of-the-art performance. Except for visual tracking, we believe that our generic AH can be utilized in a variety of tasks, e.g., few-shot and semi-supervised learning. We leave these directions for future work.

# References

[Antoniou et al., 2017] A. Antoniou, A. Storkey, and H. Edwards. Data augmentation generative adversarial networks. In arXiv:1711.04340, 2017.   
[Bertinetto et al., 2016a] L. Bertinetto, J. Valmadre, S. Golodetz, O. Miksik, and P. Torr. Staple: Complementary learners for real-time tracking. In CVPR, 2016.   
[Bertinetto et al., 2016b] L. Bertinetto, J. Valmadre, J.F. Henriques, A. Vedaldi, and P.H.S. Vedaldi. Fullyconvolutional siamese networks for object tracking. In ECCV Workshop, 2016.   
[Bhat et al., 2018] G. Bhat, J. Johnander, M. Danelljan, F. S. Khan, and M. Felsberg. Unveiling the power of deep tracking. In ECCV, 2018.   
[Choi et al., 2018] J. Choi, H. J. Chang, T. Fischer, S. Yun, K. Lee, J. Jeong, Y. Demiris, and J. Y. Choi. Context-aware deep feature compression for high-speed visual tracking. In CVPR, 2018.   
[Danelljan et al., 2015] M. Danelljan, G. Hager, F. S. Khan, and M. Felsberg. Convolutional features for correlation filter based visual tracking. In ICCV Workshop, 2015.   
[Danelljan et al., 2016] M. Danelljan, A. Robinson, F. S. Khan, and M. Felsberg. Beyond correlation filters: learning continuous convolution operators for visual tracking. In ECCV, 2016.   
[Gulrajani et al., 2017] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville. Improved training of wasserstein gans. In NIPS, 2017.   
[Han et al., 2017] B. Han, H. Adam, and J. Sim. Branchout: Regularization for online ensemble tracking with cnns. In CVPR, 2017.

[Hariharan and Girshick, 2017] B. Hariharan and R. Girshick. Low-shot visual recognition by shrinking and hallucinating features. In ICCV, 2017.   
[He et al., 2016] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.   
[Kristan et al., 2016] M. Kristan, A. Leonardis, J. Metas, M. Felsberg, R. Pflugfelder, and L. Cehovin. The visual object tracking vot2016 challenge results. In ECCV Workshop, 2016.   
[Li et al., 2017] K. Li, Y. Kong, and Y. Fu. Multi-stream deep similarity learning networks for visual tracking. In IJCAI, 2017.   
[Li et al., 2018] B. Li, W. Wu, Z. Zhu, and J. Yan. High performance visual tracking with siamese region proposal network. In CVPR, 2018.   
[Ma et al., 2015] C. Ma, J.-B. Huang, X. Yang, and M.-H. Yang. Hierarchical convolutional features for visual tracking. In ICCV, 2015.   
[Nam and Han, 2016] H. Nam and B. Han. Learning multidomain convolutional neural networks for visual tracking. In CVPR, 2016.   
[Park and Berg, 2018] E. Park and A. C. Berg. Meta-tracker: Fast and robust online adaptation for visual object trackers. In ECCV, 2018.   
[Paszke et al., 2017] A. Paszke, S. Gross, S. Chintala, G. Chanan, E. Yang, Z. DeVito, Z. Lin, A. Desmaison, L. Antiga, and A. Lerer. Automatic differentiation in pytorch. In NIPS Workshop, 2017.   
[Pu et al., 2018] S. Pu, Y. Song, C. Ma, H. Zhang, and M.-H. Yang. Deep attentive tracking via reciprocative learning. In NIPS, 2018.   
[Qi et al., 2016] Y. Qi, S. Zhang, L. Qin, H. Yao, Q. Huang, J. Lim, and M.-H. Yang. Hedged deep tracking. In CVPR, 2016.   
[Russakovsky et al., 2015] O. Russakovsky, J. Deng, and et al. Imagenet large scale visual recognition challenge. IJCV, 115(3):211–252, 2015.   
[Schwartz et al., 2018] L. Schwartz, L. Karlinsky, J. Shtok, and A. M. Bronstein. Delta-encoder: an effective sample synthesis method for few-shot object recognition. In NIPS, 2018.   
[Simonyan and Zisserman, 2015] K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image. In ICLR, 2015.   
[Song et al., 2017] Y. Song, C. Ma, L. Gong, J. Zhang, R. W.H. Lau, and M.-H. Yang. Crest: Convolutional residual learning for visual tracking. In ICCV, 2017.   
[Song et al., 2018] Y. Song, C. Ma, X. Wu, L. Gong, L. Bao, W. Zuo, C. Shen, R. Lau, and M.-H. Yang. Vital: Visual tracking via adversarial learning. In CVPR, 2018.   
[Wang et al., 2018a] Q. Wang, M. Zhang, J. Xing, J. Gao, W. Hu, and S. Maybank. Do not lose the details: Reinforced representation learning for high performance visual tracking. In IJCAI, 2018.

[Wang et al., 2018b] X. Wang, C. Li, B. Luo, and J. Tang. Sint++: Robust visual tracking via adversarial positive instance generation. In CVPR, 2018.   
[Wang et al., 2018c] Y. Wang, R. Girshick, M. Hebert, and B. Hariharan. Low-shot learning from imaginary data. In CVPR, 2018.   
[Wu et al., 2013] Y. Wu, J. Lim, and M.-H. Yang. Online object tracking: A benchmark. In CVPR, 2013.   
[Wu et al., 2015] Y. Wu, J. Lim, and M.-H. Yang. Object tracking benchmark. TPAMI, 37(9):1834–1848, 2015.   
[Yang and Chan, 2018] T. Yang and A. B. Chan. Learning dynamic memory networks for object tracking. In ECCV, 2018.   
[Yun et al., 2017] S. Yun, J. Choi, Y. Yoo, K. Yun, and J. Y. Choi. Action-decision networks for visual tracking with deep reinforcement learning. In CVPR, 2017.   
[Zhang et al., 2017] T. Zhang, C. Xu, and M.-H. Yang. Multi-task correlation particle filter for robust object tracking. In CVPR, 2017.