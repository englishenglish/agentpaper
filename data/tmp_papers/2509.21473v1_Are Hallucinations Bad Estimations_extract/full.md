# Are Hallucinations Bad Estimations?

Hude Liu∗1 Jerry Yao-Chieh $\mathrm { H u } ^ { \dag \ddag * 2 }$ Jennifer Yuntong Zhang♯3 Zhao Song§4 Han Liu†♮5

† Center for Foundation Models and Generative AI, Northwestern University, Evanston, IL 60208, USA Department of Computer Science, Northwestern University, Evanston, IL 60208, USA   
‡ Ensemble AI, San Francisco, CA 94133, USA   
♯ Engineering Science, University of Toronto, Toronto, ON M5S 1A4, CA   
§ University of California, Berkeley, Berkeley, CA 94720, USA   
♮ Department of Statistics and Data Science, Northwestern University, Evanston, IL 60208, USA

We formalize hallucinations in generative models as failures to link an estimate to any plausible cause. Under this interpretation, we show that even loss-minimizing optimal estimators still hallucinate. We confirm this with a general high probability lower bound on hallucinate rate for generic data distributions. This reframes hallucination as structural misalignment between loss minimization and human-acceptable outputs, and hence estimation errors induced by miscalibration. Experiments on coin aggregation, open-ended QA, and text-to-image support our theory.

Keywords: Hallucination in Generative Models, Foundation Model, Generative Model, Large Language Models (LLMs), Text-to-Image Generation, Trustworthy AI, Calibration

# Contents

1 Introduction 2   
2 Related Work 4   
3 Preliminaries 5   
4 $\delta$ -Hallucination 6   
5 Optimal Estimator Still Hallucinates 8   
6 Hallucination Probability Lower Bound 9   
7 Experiments 11

7.1 Synthetic Coin Flipping Problem . 11   
7.2 Open-Ended Text Questions 12   
7.3 Open-Ended Text-to-Image . . . 13

8 Conclusion 14

A Highest Conditional Density Regions 17   
B Proofs of Main Text 18

B.1 Proof of Theorem 5.1 18   
B.2 Proof of Corollary 5.2.1 23   
B.3 Proof of Section B.3 23   
B.4 Proof of Theorem 5.3 25   
B.5 Proof of Theorem 6.1 25

C Derivation to Cross-Entropy Loss 30

# 1 Introduction

Hallucination in generative model refers to a model generating confident yet unsupported or nonfactual outputs. This failure undermines user trust, safety, and the practical utility of AI systems. It becomes a critical concern in modern machine learning with the widespread deployment of large-scale generative models across language, vision, and multimodal domains [Ji et al., 2023, Liu et al., 2024, Bai et al., 2024, Kalai et al., 2025]. To address it, we must understand why models hallucinate at a fundamental level. In this work, we formalize hallucination as an attribution failure: the estimated prediction does not align with any plausible input cause under standard loss-minimizing training. From this perspective, we prove hallucination persists even for Bayesoptimal estimators.

Prior theory attributes hallucination to resource limits, sparse data, or computational hardness. Xu et al. [2024] study hallucination as the mismatch between a model’s computed function and the ground-truth function. They prove that any polynomial-time language model hallucinates on some tasks due to computational limits. Kalai and Vempala [2024] show that even a calibrated model hallucinates on rare “singleton” facts. They lower bound the hallucination rate by the frequency (redundancy) of these facts in the training data. Banerjee et al. [2024] study hallucination through Gödel’s first incompleteness theorem. They argue that no finite dataset captures all valid inferences, so hallucination persists regardless of model or data scale. Taken together, these results

frame hallucination as a byproduct of constraints rather than a structural feature of estimation.

In contrast, we posit that hallucination is not only a symptom of modeling limitations but also a structural phenomenon of estimation itself. Our key insight is that hallucinations may still persist even for Bayes-optimal estimators with unlimited capacity that minimize the true training loss. In other words, a model with infinite power, trained without resource constraints, still outputs implausible content. The crux is a misalignment between the model’s objective and human expectations. A loss-minimizing model is optimized to produce the average outcome, whereas a human evaluator expects a specific plausible outcome (typically, one of the modes of the true distribution).

This reframes hallucination as structural misalignment. Hallucination is a manifestation of estimation errors induced by miscalibration. To be concrete, under expected standard loss, the Bayesoptimal predictor for a target distribution $A ( X )$ given the input $X$ is the conditional expectation

$$
A ^ {*} (X) = \mathbb {E} [ A (X) ],
$$

which minimizes the expected error by construction. If the true conditional distribution $\operatorname* { P r } [ A ( X ) ] \ = \ \operatorname* { P r } [ A ( x ) \mid X = x ]$ is multimodel1, then $A ^ { \star } ( X )$ average across all those possible outcomes and may fall in a low-probability region. It matches none of the plausible modes. The estimate minimizes error yet fails to align with any realistic ground-truth outcome. Thus even an optimal estimator may produce outputs that no human would recognize as valid or plausible. We deem this is a fundamental source of hallucination in generative models. To this end, we formalize this into $\delta$ -hallucination: an estimator’s output that lies outside a $\delta$ -neighborhood of every plausible outcome (please see Section 4 for precise definitions.) This reframing shows hallucination as a consequence of the objective misalignment, rather than just a lack of model capacity or data.

Contributions. Our contributions are as follows.

• New Formulation for Hallucination Fundamental Source. We characterize hallucination phenomena in generative models by introducing $\delta$ -hallucination. This interprets hallucination as outputs that fail to match any plausible human-acceptable outcome. The formulation provides a rigorous and measurable way to analyze hallucination in generative models.   
• Hallucination of Optimal Estimators. We prove that loss-minimizing optimal estimators still $\delta$ -hallucination. We extend the result to near-optimal estimators, to multiple inputs, and to inputs with hinted latent variables. These results confirm hallucination as a fundamental source rooted in the estimation process itself.   
• Fundamental Limits of Hallucination. We derive a general lower bound on the probability of $\delta$ -hallucination under mild distribution assumptions. This bound reaffirms that hallucinations persist at a non-zero rate. This establishes a fundamental limit that prevents eliminating the source of hallucinations through larger models or datasets.   
• Experiment Validation. We validate our theory through controlled experiments on coinflipping aggregation, open-ended QA, and text-to-image generation. The results demon-

strate that minimizing loss does not remove hallucination. The persistence across both synthetic and real-world settings confirms hallucination as a structural feature of estimation and a fundamental source of model misalignment.

Organization. Section 4 defines hallucination as $\delta$ -hallucination. Section 5 demonstrates hallucination of optimal estimators. Section 6 provides a lower bound on the probability of hallucination. Section 7 details experiment results.

# 2 Related Work

Hallucinations in generative models have been studied from both theoretical and empirical perspectives. Prior theory frames them as inevitable outcomes of practical limits: finite parameters, sparse data, or computational hardness. [Xu et al., 2024] prove that any polynomial-time language model hallucinates on certain tasks. Kalai and Vempala [2024] show that even a calibrated model hallucinates at a rate tied to the fraction of “singleton” facts that appear only once in the training set. Banerjee et al. [2024] argue that no finite dataset or architecture covers all valid inferences, ensuring a nonzero hallucination rate regardless of scale. These works treat hallucination not as a flaw in estimation itself, but as an artifact of underfitting caused by resource and computational limits. More recently, Kalai et al. [2025] propose that hallucination stems from mismatches between predictive likelihood training, incomplete coverage, and reinforcement learning, suggesting hallucinations persist even with scale and motivating deeper foundational study.

Recent empirical research has delivered taxonomies, benchmarks, and mitigation techniques for hallucinations in generative models. Huang et al. [2025] survey intrinsic and extrinsic hallucinations, and review detection and mitigation methods. Ji et al. [2023] provide a broad overview of metrics and task-specific phenomena across summarization, dialogue, and machine translation. Zhang et al. [2023] analyze detection and explanation methods. Li et al. [2024] conduct a factuality study, introducing a new benchmark and evaluating detection, sources, and mitigation. Farquhar et al. [2024] propose entropy-based uncertainty estimators to detect confabulations. In contrast to viewing hallucinations only as limitations, Jiang et al. [2024] explore their creative potential. A notable work by Aithal et al. [2024] analyzes hallucinations in diffusion models and attributes them to mode interpolation, where samples fall into regions not supported by training data. Their empirical observations support our theoretical findings by linking artifacts beyond data support to interpolation between nearby modes (corresponding to regions with low conditional probability density under any latent state in our work).

Building on prior work, we propose a new interpretation of hallucination: it arises from a gap between model training objectives and human criteria. Estimation fails when outputs do not align with any plausible human-perceptive category. We formalize this gap as $\delta$ -hallucination and prove that even loss-minimizing optimal estimators produce outputs with low conditional probability under every category. We derive a general lower bound on the probability of $\delta$ -hallucination and validate our claims with empirical studies. These results establish hallucination as a structural feature of estimation itself, not a flaw of model size, data coverage, or specific queries.

# 3 Preliminaries

Notations. In this work, $f _ { Y } ( \cdot )$ denotes the probability density function over the randomness of $Y$ . $\mathbb { E } _ { Y } [ T ]$ denotes the expectation of a random variable $T$ over $Y$ . $[ N ]$ denotes the set: $\{ 1 , 2 , \cdots , N \}$ . $\| \cdot \| _ { 2 }$ denotes 2-norm. We use $\| \cdot \| _ { 2 }$ as the square root of the square sum of all entries. For a column vector $v$ , we use $v _ { i }$ to denote its $i$ -th entry from the top. For a matrix $M$ , we use $M _ { r , c }$ to denote its entry at $r$ -th row and $c$ -th column. We write $M _ { : , c }$ and $M _ { r , \ l }$ : to denote its $c$ -th column and $r$ -th row, respectively. We use $1 _ { a }$ to denote an indicator that is 1 when $a$ happens and 0 otherwise.

Expected Quadratic Loss. We define expected quadratic loss as follows.

Definition 3.1 (Expected Quadratic Loss). Let $X$ be an input, let $A ( X )$ be a random target output associated with $X$ , and let $A ^ { ( } X )$ be an estimator for $A ( X )$ . Define the expected quadratic loss of the estimator $A ^ { ( } X )$ with respect to the true output $A ( X )$ as:

$$
\ell_ {A} \left(A ^ {*} (X)\right) := \mathbb {E} \left[ \| A ^ {*} (X) - A (X) \| _ {2} ^ {2} \right].
$$

In other words, $\ell _ { A } ( A ^ { * } ( X ) )$ is the expected squared $\ell _ { 2 }$ error between the estimate and the actual outcome. This quantity serves as the objective that an optimal estimator would minimize (e.g., the Bayes-optimal estimator minimizes the expected quadratic loss by construction).

Remark 3.1. We use the $\ell _ { 2 }$ loss in the main text for clarity of exposition. In Section C, we show that all results remain valid under the cross-entropy loss, which is the standard training objective for generative models in self-supervised learning. This extension is natural because cross-entropy is a proper scoring rule: its Bayes-optimal solution is the true conditional distribution $P ( \boldsymbol { Y } | \boldsymbol { X } )$ , so the same structural arguments for $\delta$ -hallucination continue to apply.

We use the expected quadratic loss to formalize the objective minimized by an optimal estimator.

Lipschitzness. We define Lipschitzness in 2-norm as follows.

Definition 3.2 (Lipschitzness). We say a function $g$ is $L$ -Lipschitz (with respect to the $\ell _ { 2 }$ -norm) if there exists a constant $L > 0$ such that for all inputs $x$ and $y$ in its domain

$$
\| g (x) - g (y) \| _ {2} \leq L \| x - y \| _ {2}.
$$

We use Lipschitzness to impose a regularity condition on the estimator. This condition ensures that small changes in the input lead to at most $L$ -scaled changes in the output. In our analyses, we assume Lipschitzness as a smoothness property that rules out estimators with abrupt or unstable behavior.

Latent Variable $Z$ . In the context of self-supervised learning, we represent the output of the model as a probability distribution [Devlin et al., 2019, Radford et al., 2021]. Specifically, when an estimator outputs contextual factors such as speaker attitude or intended audience, we may

categorize the possible outputs based on the specific factors they exhibit. Then, we see different categories (which are sub-distributions in the original target distribution) as conditional distributions under different states of a latent variable $Z$ . We illustrate the concept of this latent variable $Z$ in Figure 1.

![](images/e93c3b90e334ed3f457c59c3aeee487c3c251ed853337c0461a01597f3eac85c.jpg)

![](images/fc8e7390b862221dae07ed029c3785cbcfd9ada6992c4db862f429b1f914a9d8.jpg)  
Figure 1: Examples of Latent Variable $Z$ . For an open-ended question or prompt $X$ , the latent variable $Z$ may be the emotional attitude or categories in the target distribution.

# 4 $\delta$ -Hallucination

We present our definition of $\delta$ -hallucination as the gap between objective optimized by the model and the underlying causes of variation $( Z )$ . That is, conditioning on the state of $Z$ changes the distribution of the output. We begin by defining the relation between input $X$ and latent variable $Z$ as follows.

Definition 4.1 (Data Distribution and Latent Variable). Let $X \in \mathbb { R } ^ { d _ { x } }$ denote the input, and let $A ( X ) \in \mathbb { R } ^ { d _ { a } }$ denote a random variable representing the target output associated with $X$ , where $d _ { x }$ and $d _ { a }$ are the input and output dimensions. Let $Z$ be a latent variable associated with $X$ , and let $\{ Z _ { i } \} _ { i \in [ N ] }$ denote its possible states. The conditional output random variable given $Z _ { i }$ is

$$
A (X; Z _ {i}) := A (X) \mid \{Z = Z _ {i} \},
$$

which represents the target output distribution of $X$ under latent state $Z _ { i }$ . If probability densities exist, the conditional density is

$$
f _ {A (X; Z _ {i})} (a) := \frac {f _ {A (X) , Z} (a , Z _ {i})}{\operatorname * {P r} [ Z = Z _ {i} ]},
$$

where $f _ { A ( X ) , Z }$ is the joint density of $( A ( X ) , Z )$ .

Remark 4.1. $A ( X )$ in Definition 4.1 defines the data distribution, but we also view it as the real distribution in this paper. Intuitively, $Z$ indexes hidden causes that resolve ambiguity in the output. $A ( X ; Z _ { i } )$ isolates the distribution of valid outputs when the hidden cause equals $Z _ { i }$ . The marginal $A ( X )$ mixes these conditional laws with weights $\mathrm { P r } [ Z = Z _ { i } ]$ , so multi-modality in $A ( X )$ arises from variation over $Z$ .

![](images/e8bc5280582abfcd4789ce1307e51e878bec61af06b70ee37e5d360d5d97cf10.jpg)  
Figure 2: An Example of Our Key Insight. Suppose the open-ended question is to generate a picture of an animal. Then the output with $9 0 \%$ of conditional probability under the category of cat and a $1 0 \%$ of conditional probability under the category of dogs is considered better than the output which has a $7 0 \%$ of probability density under the category of cat and $7 0 \%$ under the category of dog.

Key Insight. While minimizing the loss on the whole data distribution is critical for model estimations, it is also important to

$$
\max  _ {i \in [ N ]} \left\{f _ {A (X; Z _ {i})} \left(A ^ {*} (X)\right) \right\},
$$

which is the maximum probability density of the estimate $A ^ { * } ( X )$ under $Z = Z _ { i }$ . This reflects that a good estimate aligns with at least one plausible underlying state rather than consistent with all. We give an example in Figure 2 to illustrate the interpretation.

Formally, we present the above insight as $\delta$ -hallucination.

Definition 4.2 ( $\delta$ -Hallucination). Let $X$ be an input and $Z$ a latent variable associated with $X$ taking values in $\{ Z _ { i } \} _ { i \in [ N ] }$ . Fix a tolerance parameter $\delta \in ( 0 , 1 ]$ , and let $A ^ { * }$ be an estimator of $X$ . We say that $A ^ { * } ~ \delta$ -hallucinates at $X$ if, for every $i \in [ N ]$ ,

$$
f \big (A (X; Z _ {i}) = A ^ {*} (X) \big) \leq \delta , \quad i \in [ N ],
$$

where $f _ { A ( X ; Z _ { i } ) }$ denotes the probability mass function (in the discrete case) or probability density function (in the continuous case) of $A ( X ; Z _ { i } )$ .

That is, for every possible latent state, the probability of producing the estimated output $A ^ { * } ( X )$ does not exceed $\delta$ . In other words, Definition 4.2 implies that $\delta$ -hallucination is a generated answer that has low calculated loss but is unlikely to belong to any state or class of possible outputs.

Remark 4.2. Intuitively, $\delta$ -hallucination occurs when the estimator $A ^ { * } ( X )$ outputs a value that has low likelihood under every plausible latent state of $Z$ . In such a case, the prediction fail to be attributed to any genuine cause consistent with the data distribution. This captures the idea that hallucination arises not merely from error, but from producing an output that fails to align with any valid mode of the underlying conditional distributions.

# 5 Optimal Estimator Still Hallucinates

We establish the existence of $\delta$ -hallucination. We begin with the single-input case, showing that even an optimal estimator minimizing loss may $\delta$ -hallucinate, and that this extends to semioptimal estimators within $\epsilon$ of the optimum. We then extend the result to the multi-input setting. Finally, we consider the practical case where the model receives hints about hidden influences in the input, and show that hallucination exists under standard regularity conditions.

$\delta$ -Hallucination Under a Single Input. We show that even an loss-minimizing optimal estimator may output an answer that $\delta$ -hallucinates by Definition 4.2.

Theorem 5.1 (Existence of $\delta$ -Hallucination Under Single Input). For an input $X$ , there exists infinitely many distributions of $A ( X )$ and $Z$ such that for an estimator $A ^ { * }$ that minimizes the expected quadratic loss defined in Definition 3.1 over $A ( X )$ , it is bound to $\delta$ -hallucinate at $X$ .

Proof. See Section B.1 for detailed proof.

![](images/2bc99459bbdc18fec2b09d40a794bc6513fa65ce6334d028edc7cc21be96b8cf.jpg)

We further demonstrate the existence of $\delta$ -hallucination on semi-optimal estimators.

Theorem 5.2 (Existence of $\delta$ -Hallucination on Semi-Optimal Estimators under Single Input). For an input $X$ , there exists infinitely many distributions of $A ( X )$ and $Z$ such that if an estimator $A ^ { \prime }$ is within a distance of $\epsilon$ to the optimal estimator $A ^ { * }$ , which writes as

$$
\| A ^ {\prime} (X) - A ^ {*} (X) \| _ {2} \leq \epsilon ,
$$

then $A ^ { \prime } ( X )$ is bound to $\delta$ -hallucinate.

Proof. See Section B.3 for detailed proof.

![](images/2d98ee312ed535c126eb6ac39c889fe26e5621d3410ce2be9ea6d02f818d3340.jpg)

$\delta$ -Hallucination under Multiple Inputs. When considering a collection of inputs, our definition applies to each input individually. We describe the $\delta$ -hallucination under multiple inputs as follows.

Corollary 5.2.1 (Existence of $\delta$ -Hallucination under Multiple Inputs). For a set of input $X _ { j } , j \in$ $[ S ]$ , there exists infinitely many distributions of $A ( X _ { j } )$ and $Z$ such that any estimator minimizing the expected quadratic loss defined in Definition 3.1 is bound to $\delta$ -hallucinate at $X$ .

Proof. See Corollary B.1.1 for detailed proof.

![](images/def3053c185d8a71d0542f691388787cc03cba9da0a9a95f4dcb25ea6ede033e.jpg)

$\delta$ -Hallucination with Hinted Latent Variables. In practical situations, the model receives hints about hidden influences in the input. We define this hint as a tilt upon the input $X$ as follows.

Definition 5.1 (Effect of Latent Variable on Input). For an input $X$ , let $A ( X )$ be its target distribution. For a latent variable $Z$ associated with $X$ , let $Z _ { i }$ denote the states of this latent variable,

and let $\delta _ { i }$ denote a hint for the state $Z _ { i }$ for all $i \in [ N ]$ , which satisfies

$$
A (X + \delta_ {i}) = A (X; Z = Z _ {i}), \quad i \in [ N ].
$$

This means the target distribution of the tilted input is the posterior distribution when knowing $Z = Z _ { i }$ .

Based on Definition 5.1, we show $\delta$ -hallucination exists for tilted input under Lipschitzness regularity condition as follows.

Theorem 5.3 (Existence of $\delta$ -Hallucination at Tilted Input). Let $B _ { \delta }$ denote the bound of all hints $\delta _ { i } , i \in [ N ]$ , defined as

$$
B _ {\delta} := \sup  _ {i \in [ N ]} \| \delta_ {i} \| _ {2}.
$$

For an $L$ -Lipschitz estimator $A ^ { * }$ satisfying Definition 3.2, there exists infinitely many distributions of $A ( X ; Z )$ such that $\delta$ -Hallucination happens on all $X + \delta _ { i }$ . That is, $A ^ { * } ( X + \delta _ { i } )$ does not fall into the region where $f _ { A \left( X ; Z _ { i } \right) } \geq \delta$ for any $i \in [ N ]$ by Definition 4.1.

Proof. See Section B.4 for detailed proof.

![](images/ee02f6401e8cd1be819ee629bfe91fa8df21ce580a7dea650bc59cca4fc90f20.jpg)

Thus, we show that hallucination is intrinsic to the probabilistic structure of estimation, across optimal and near-optimal estimators, multiple inputs, and even when the answers’ directions are hinted.

# 6 6 Hallucination Probability Lower Bound

We extend our result beyond existence of $\delta$ -hallucination in Section 6 and provide a lower bound on the probability of hallucination for optimal estimators satisfying certain conditions.

We begin with the definition of means and variances for the variables of interest.

Definition 6.1 (Means and Variances). Let $\{ Z _ { i } \} _ { i \in [ N ] }$ denote the possible states of the latent variable $Z$ , with probabilities $p _ { i } : = \mathrm { P r } [ Z = Z _ { i } ]$ . For each $i \in [ N ]$ , define the conditional mean

$$
\mu_ {i} := \mathbb {E} \left[ A \left(X; Z _ {i}\right) \right].
$$

We regard $\mu _ { i }$ as a realization of a random variable distributed according to $d _ { i } ^ { \mu }$ . Let $\mu _ { i } ^ { d } : = \mathbb { E } _ { d _ { i } ^ { \mu } } [ \mu _ { i } ]$ and $\sigma _ { i } ^ { d } : = \operatorname { V a r } _ { d _ { i } ^ { \mu } } [ \mu _ { i } ]$ denote the mean and variance of this distribution, respectively. Let $d ^ { \mu }$ denote the joint distribution of $( \mu _ { 1 } , \ldots , \mu _ { N } )$ . We write $\mu ^ { d } : = \mathbb { E } _ { d ^ { \mu } } [ \mu _ { 1 } , \dots , \mu _ { N } ]$ for its mean vector and $\begin{array} { r } { \sigma ^ { d } : = \mathbb { E } [ \sum _ { i = 1 } ^ { N } ( \mu _ { i } - \mu _ { i } ^ { d } ) ^ { 2 } ] } \end{array}$ as sum of variance.

We then provide the following assumptions applied to $\mu _ { i }$ and $d _ { i } ^ { \mu }$ in Definition 6.1. In particular, we assume that the conditional means align around a common value and that the joint distributions of these conditional means are mutually independent.

Assumption 6.1. We impose the following conditions on the distributions defined in Definition 6.1:

1. Identical means: There exists a constant $\mu _ { 0 } \in \mathbb { R }$ such that $\mu _ { i } ^ { d } = \mu _ { 0 }$ , for all $i \in [ N ]$ .   
2. Independence: The distributions $\{ d _ { i } ^ { \mu } \} _ { i = 1 } ^ { N }$ are mutually independent.

We now characterize hallucination events in terms of output regions that correspond to high $( > \delta )$ conditional probability under each latent state.

Definition 6.2 (High Conditional Density Regions). We define $U _ { i } ^ { \delta }$ to be

$$
U _ {i} ^ {\delta} := \left\{a \mid f (a; Z _ {i}) > \delta \right\},
$$

which is the region with posterior probability of $Z = Z _ { i }$ larger than $\delta$

Remark 6.1. By Definition 6.2, $\delta$ -hallucination of $A ^ { * } ( X )$ is equivalent to

$$
A ^ {*} (X) \notin U _ {i} ^ {\delta}, \quad i \in [ N ].
$$

Remark 6.2. We highlight the relationship between Highest Conditional Density Regions (HC-DRs) and the classical Highest Density Regions (HDRs) [Caprio et al., 2024, Dahl et al., 2024]. When the latent variable $Z$ has only a single state, $\delta$ -hallucination reduces to the event that the target distribution falls outside the HDR of a given mass, where the mass corresponds to a density threshold $\delta$ . When $Z$ has multiple states, we generalize this idea by introducing HCDRs, which capture high-density regions conditioned on each latent state. See Section A for definitions and a detailed discussion.

We then define the following spheres covering $U _ { i } ^ { \delta }$ in Definition 6.2. Specifically, we enclose each $U _ { i } ^ { \delta }$ within the smallest possible sphere centered at the corresponding mean $\mu _ { i }$ .

Definition 6.3 (Minimal Covering Spheres). For each $i \in [ N ]$ , let $U _ { i } ^ { \delta } \subset \mathbb { R } ^ { d _ { a } }$ denote the $\delta$ -high density region associated with state $Z _ { i }$ . Define $B _ { i } ^ { \delta } ( \boldsymbol { r } )$ as the closed Euclidean ball of radius $r$ centered at $\mu _ { i }$ . The minimal covering radius is

$$
r _ {i} := \inf  _ {r _ {i} \in \mathbb {R} ^ {+}} \left\{U _ {i} ^ {\delta} \subset B _ {i} ^ {\delta} (r _ {i}) \right\}.
$$

Thus $B _ { i } ^ { \delta } ( r _ { i } )$ is the smallest sphere centered at $\mu _ { i }$ that contains $U _ { i } ^ { \delta }$ . Finally, define the uniform covering radius

$$
r = \max  _ {i \in [ N ]} \left\{r _ {i} \right\}.
$$

Remark 6.3. Geometrically, $r _ { i }$ measures the worst-case deviation of the $\delta$ -high density region $U _ { i } ^ { \delta }$ from its center $\mu _ { i }$ . In other words, it is the maximum distance one must travel from $\mu _ { i }$ to reach any point in $U _ { i } ^ { \delta }$ . The uniform covering radius $r$ then gives a single bound that applies across all latent states, capturing the largest such deviation. This interpretation is useful for intuition: $r _ { i }$ quantifies how “spread out” the high-density region is around its mean, while $r$ aggregates the

largest of these spreads across all $i$

With definitions and assumptions established, we now derive a lower bound on the probability of hallucination for any optimal estimator.

Theorem 6.1 (Hallucination Probability Lower Bound). Let $( A ( X ) , Z )$ satisfy Assumption 6.1. For each $i \in [ N ]$ , let $\mu _ { i } , \sigma _ { i } ^ { d }$ be as in Definition 6.1, let $\mu _ { 0 }$ be as in Assumption 6.1, and let $r _ { x }$ be as in Definition 6.3. Define

$$
d := (\sum_ {j = 1} ^ {N} p _ {j} ^ {2} \sigma_ {j} ^ {d}) ^ {1 / 2}, \quad \theta_ {i} (\alpha) := \frac {(\alpha d + r _ {x}) ^ {2}}{\sigma_ {i} ^ {d}}, \quad \alpha > 1, \quad \text {a n d} \quad K _ {i} ^ {\mu} := \frac {(\mathbb {E} [ (\mu_ {i} - \mu_ {0}) ^ {2} ]) ^ {2}}{\mathbb {E} [ (\mu_ {i} - \mu_ {0}) ^ {4} ]}.
$$

If for every $i \in [ N ]$ there exists $\alpha _ { i } > 1$ such that $\theta _ { i } ( \alpha _ { i } ) \leq 1$ , then

$$
P _ {H} ^ {\delta} > \prod_ {i = 1} ^ {N} (P _ {i} K _ {i} ^ {\mu}),
$$

where $P _ { H } ^ { \delta }$ denotes the probability that the optimal estimator $A ^ { * } ~ \delta$ -hallucinates at $X$ (equivalently, $A ^ { * } ( X ) \notin U _ { i } ^ { \delta }$ for all $i \in [ N ]$ , with $U _ { i } ^ { \delta }$ as in Definition 6.3).

Proof. See Section B.5 for detailed proof.

![](images/a167413e870791cc918c0ec15a66927f2c838729a4f31fba5bfd3ba205ef0b86.jpg)

# 7 Experiments

We validate our interpretations and claims with three complementary experiments. In particular, we first provide a synthetic coin-flipping problem (Section 7.1) where it demonstrates that models trained purely with likelihood objectives shows persistent $\delta$ -hallucination. We then extend these insights to large-scale LLM (Section 7.2) and text-to-image generation (Section 7.2) settings. Both experiments validate our claim that a loss-minimizing optimal estimator $\delta$ -hallucinates.

# 7.1 Synthetic Coin Flipping Problem

Objective. We evaluate our claim that minimizing loss may not increase the conditional probability of estimated output with respect to input labels as in Theorem 5.1.

Experiment Design. We design a controlled experiment based on the classical coin-flipping problem. We choose a subset of coins from a collection of coins (each with a distinct probability of landing heads), flip them, and record the total number of heads observed. The model receives the labels of the chosen coins as input. We then train the model to predict the recorded total. These labels do not explicitly reveal the head probabilities, and thus act as latent hints rather than explicit supervision.

Data. We generate $2 N$ coins, each with a unique head probability, and perform $M$ flips to construct the dataset. We consider $N = 2 , 3$ , and 5, with $M$ ranging from 20000 to 40000.

Model Architecture. We adopt an 8-layer transformer with 64 hidden dimensions and 256 feedforward dimensions for this experiment.

![](images/ce26f23889fe2a57783f932fd84a5afc9ee5a027a2c11cbfda2635017158518a.jpg)  
(a) N = 2

![](images/6140bbfc6bc44a112bb4cd353762f6138acae1b0a75c9e228508fd49a016f4e1.jpg)  
(b) N = 3

![](images/99eb5fff588e2270a684f1401dc8c7620adfc3b34bb1d478706f53b6e2328381.jpg)  
(c) N = 5   
Figure 3: We conducted 5 rounds of experiments on each of $N = 2 , 3$ and 5. The results show that training loss does not correlate with the conditional probability of the model estimation with respect to input labels. This aligns with our theoretical result in Theorem 5.1.

Results. As shown in Figure 3, we observe that the descent of training losses does not correlate with the rise or drop of the conditional probability of the estimations generated on the validation set. This result aligns with our theoretical claim that minimizing the loss does not necessarily maximize the conditional probability (of a latent state) of the estimate.

# 7.2 Open-Ended Text Questions

Objective. We evaluate hallucination in the LLM models by measruing the resemblance of model output to the commonly incorrect answers in TruthfulQA [Lin et al., 2021].

Experiment Design. We fine-tune pretrained language models on a dataset of open-ended questions and compare their

outputs to those of the original models. We measure the the model’s tendency to resemble the commonly incorrect answers in TruthfulQA [Lin et al., 2021]. We use Gestalt Pattern Matching (difflib in Python) to measure resemblance.

Data. We use GPT5, Gemini 2.5 Flash, and DeepSeek R1 to generate a dataset of 300 openended questions with 2 possible answers. This forms a dataset of 600 question-answer pairs.

![](images/6677200727bef223111c75f04375da862dc85fad9a8a32e9b4dd16c1a489a8bb.jpg)  
Figure 4: Resemblance vs. Epochs. We fine-tune Qwen1.5-1.8B-Chat and Qwen2.5-7B-Instruct for 2, 3, and 4 epochs and test the answers’ resemblance to commonly incorrect answers in TruthfulQA. We repeat this process for 2 random seeds. Results validate that hallucination persists even as the model minimizes its predictive objective.

Model Architecture. We fine-tune Qwen1.5- 1.8B-Chat and Qwen2.5-7B-Instruct on our open-ended question dataset using LLaMA-Factory with LoRA adapters.

Results. As shown in Figure 4 and Table 1, both models show a consistent increase in resemblance over additional fine-tuning epochs. The results reveal that, though we fine-tune the models to obtain low predictive loss, both models become more aligned with commonly incorrect answers. This pattern is consistent across all seeds as shown in Table 1. The finding supports our theoretical claim that loss

minimization alone is insufficient to eliminate $\delta$ -hallucination.

Table 1: Resemblance of Fine-Tuned Models’ Answers to Commonly Incorrect Answers in TruthfulQA. Each model is fine-tuned for 2, 3, and 4 epochs with 2 random seeds. The resemblance does not decrease with training, validating that hallucination persists in loss-minimizing optimal models.

<table><tr><td rowspan="2">Epochs</td><td colspan="2">Qwen1.5-1.8B-Chat</td><td colspan="2">Qwen2.5-7B-Instruct</td></tr><tr><td>Seed 1</td><td>Seed 2</td><td>Seed 1</td><td>Seed 2</td></tr><tr><td>Original</td><td>0.1975</td><td>-</td><td>0.1868</td><td>-</td></tr><tr><td>2</td><td>0.2338</td><td>0.2431</td><td>0.2043</td><td>0.1997</td></tr><tr><td>3</td><td>0.2338</td><td>0.2486</td><td>0.2123</td><td>0.2028</td></tr><tr><td>4</td><td>0.2450</td><td>0.2539</td><td>0.2173</td><td>0.2099</td></tr></table>

# 7.3 Open-Ended Text-to-Image

Objective. We evaluate hallucination in a text-to-image setting where we detect generated samples falling outside a calibrated HCDR as in Definition B.2 and Remark 6.2.

Experiment Design. We first construct HCDR from real AFHQ cat and dog images. We begin by extracting fixed CLIP embeddings from the images, which are then normalized, reduced in dimension via PCA, and standardized through z-scoring. For each class (cats, dogs), we fit a Gaussian Mixture Model (GMM) on an $8 0 \%$ training split of the preprocessed embeddings to learn what cat or dog features look like. We then use the remaining $2 0 \%$ testing data to obtain logdensities and compute a class-specific threshold at the $1 0 \%$ percentile. This threshold corresponds to a cutoff such that the top $9 0 \%$ of the testing images are included in the HDR for each class (See Figure 7 of Section A for a visualization of HDR for cats and dogs). In other words, a new embedding is considered to lie outside of HDR or a specific class if its log-likelihood under that class’s GMM exceeds the threshold. Finally, to form HCDR, we take the union of the per-class HDRs: a generated embedding is inside the HCDR if it lies in at least one class HDR, and outside otherwise.

We then fine-tune a text-to-image generative model, with the text encoder frozen, on the training dataset for the model to mainly learn the image distribution (target). We evaluate the portion of generated images outside of HCDR for given prompts.

Data. We use Animal Faces-HQ (AFHQ) [Choi et al., 2020]. We extract 5558 cat images and 5139 dog images. Each is 512 by 512 pixels. We construct 3 prompts for evaluation: "a realistic photo of a friendly dog", "a fluffy cat sitting on a sofa", and "a cute pet animal".

Model Architecture. We use CLIP ViT-B/32 model model to extract image CLIP embeddings. For generation, we fine-tune the UNet component of Stable Diffusion v1.5, while keeping the text encoder and VAE frozen.

Results. As shown in Figure 5, as we fine-tune the model, the training loss decreases, indicating that the model captures the distribution of the dataset, yet hallucination rate do not converge. It supports our theoretical claim that loss minimization alone is insufficient to eliminate $\delta$ -hallucination.

Ablation Study on Prompts. We further conduct studies on 3 types of prompts for the text-to-image generative model: one targeting the cat category ("a fluffy cat sitting on a soft"), one targeting the dog category ("a realistic photo of a friendly dog"), and one mixed prompt (“a cute pet animal”). We evaluate the hallucination rate for each prompt across training epochs. As shown in Figure 6, our results

consistently show that, even under a loss-minimizing estimator, hallucinations persist and do not converge to zero. This indicates that even when prompts hint information about target category, hallucinations may still occur.

![](images/f6345b330d299f184f3838b472da1a6b60d54651d0f37db8ffba8879395b42a5.jpg)  
Hallucination Rate and Training Loss   
Figure 5: Hallucination Rate and Training Loss. We plot hallucination rate (green, left axis) and training loss (blue, right axis) over epochs. While the training loss decreases, the hallucination rate does not converge and often fluctuates, showing that hallucination persists even as the model minimizes its predictive objective.

![](images/fe5939024c033cf1b7db732b190a257cee09a40f139f6b8b15c5c7795b273786.jpg)

![](images/0c0dabfb64783b6e07b97614ab3d095e5212d77fbeed4fb702f76fb47661c621.jpg)

![](images/918f53b716f7ce16fb6fcdc10b59be3970c4b8e146c114ccba9584269ef2c68f.jpg)  
Figure 6: Prompts Analysis. We create 3 types of prompts and evaluate their hallucination rate respectively. All plots show even a loss-minimizing estimator hallucinates.

# 8 Conclusion

In this work, we reframed hallucination in generative models as a fundamental misalignment between standard loss-based training objectives and human expectations. Under this view, we formalized $\delta$ -hallucination to capture when an estimator’s output fails to match any plausible real-world outcome (Section 4). Crucially, we showed that no amount of model capacity or data can eliminate hallucinations: even an ideal Bayes-optimal estimator (one minimizing the true expected loss) may still generate implausible predictions on inputs with inherently diverse correct answers (Section 5). We derived general lower bounds on how frequently such hallucinations must occur for broad classes of target distributions (Section 6), and validated these predictions with both synthetic and real-world experiments (Section 6). Taken together, our findings establish that hallucination is a structural property of the estimation process itself rather than just a symptom

of limited models or datasets.

Limitations. While our theory offers a new perspective on hallucinations, it has a few limitations. The current lower bound for $\delta$ -hallucination is relatively loose and relies on certain assumptions, leaving room for tighter bounds under more relaxed conditions. Additionally, our analysis focused on a general estimator. Examining specific model families or tasks might yield stronger guarantees or further insight into when and how hallucinations arise.

Implications and Future Work. By identifying hallucination as arising from the core training objective, our results imply that simply scaling up model size or dataset coverage is insufficient to eliminate the problem. Effective mitigation may require rethinking generative model training, with objectives explicitly aligned to human standards of correctness. In practice, this could mean favoring more mode-seeking behavior —generating high-probability, consistent outputs — rather than minimizing average error across all possible outcomes. Future training methods may need to incorporate constraints or decision-theoretic criteria that push models to commit to a single plausible answer instead of blending incompatible modes. Several concrete directions follow from our findings:

• Alternative Loss Functions. Extend our theoretical framework to other loss functions to investigate how the choice of training objective influences hallucination rates.   
• Alignment-Oriented Training Schemes. Design practical strategies that scale our insights, such as HDR-guided sampling or mixed-objective fine-tuning that explicitly penalizes implausible outputs.   
• Multimodal and Structured Outputs. Generalize the analysis to multimodal and structured tasks, where the space of valid outputs is richer, to uncover new alignment strategies tailored to complex domains.

In summary, treating hallucination as a structural phenomenon calls for a shift away from naive average-case error minimization and toward objectives that explicitly prefer outputs aligned with one of the true modes, thereby better matching human standards of reliability.

# Acknowledgments

JH would like to thank Mehak Kawatra, Maojaing Su, Dino Feng and Andrew Chen for enlightening discussions on related topics, the Red Maple Family for support, and Jiayi Wang for facilitating experimental deployments.

JH is partially supported by Ensemble AI and Northwestern University. Han Liu is partially supported by NIH R01LM1372201, NSF AST-2421845, Simons Foundation MPS-AI-00010513, AbbVie , Dolby and Chan Zuckerberg Biohub Chicago Spoke Award. This research was supported in part through the computational resources and staff contributions provided for the Quest high performance computing facility at Northwestern University which is jointly supported by the Office of the Provost, the Office for Research, and Northwestern University Information Technology. The content is solely the responsibility of the authors and does not necessarily represent the official views of the funding agencies.

Typeset with a modified LaTeX template of 1712.09542 [hep-th] by Yuji Tachikawa [Tachikawa, 2020].

# Appendix

A Highest Conditional Density Regions 17   
B Proofs of Main Text 18

B.1 Proof of Theorem 5.1 18   
B.2 Proof of Corollary 5.2.1 23   
B.3 Proof of Section B.3 23   
B.4 Proof of Theorem 5.3 25   
B.5 Proof of Theorem 6.1 25

C Derivation to Cross-Entropy Loss 30

# A Highest Conditional Density Regions

Highest Density Regions. [Hyndman, 1996] popularize the concept of Highest Density Regions (HDRs) as the smallest-volume set containing a given probability mass He provided practical algorithms for computing and visualizing HDRs for univariate and multivariate densities, showing their advantages over equal-tailed intervals in revealing multi-modal structure. [Samworth and Wand, 2010] developed a rigorous asymptotic theory for kernel-based HDR estimation, deriving uniform-in-bandwidth risk approximations and proposing optimal bandwidth selectors that minimize HDR estimation error. [Haselsteiner et al., 2017] introduced the idea of using HDRs to define environmental-contours—termed highest-density contours—in engineering design, demonstrating that HDR-based contours yield more compact, interpretable regions for multimodal environmental distributions.

In a concrete example, we build calibration datasets for the categories of cats and dogs in AFHQ dataset [Choi et al., 2020] and estimate their log-densities under GMM model as shown in Figure 7.

Highest Conditional Density Regions. We emphasize a connection between Highest Conditional Density Regions and HDRs. Specifically, when the latent variable $Z$ only has one latent state, the $\delta$ -hallucination in this special occasion is the expectation of the target distribution falling out of the HDRs of a certain mass that induces a density bound of $\delta$ . We then extend this concept to the distributions correlated with a latent variable with more than one states. Namely, we introduce the concept of Highest Conditional Density Regions (HCDRs) and define it as follows.

Definition A.1 (Highest Conditional Density Regions). Let $d$ be a distribution and $Z$ a latent variable correlated with $d$ . Let $d _ { i }$ denote the conditional probability of $d$ when knowing $Z = Z _ { i }$ , here $Z _ { i }$ , $i \in [ N ]$ is one of the $N$ states of $Z$ . This explicitly writes as

$$
d _ {i} = d \mid \{Z = Z _ {i} \}.
$$

We define the Highest Conditional Density Regions $S _ { M }$ as the smallest region on which the integral of $d _ { i }$ is $M$ .

![](images/9318ca6d28034a86b016ce94ada97db55c556bdfe29ad1ab0542807602ea078b.jpg)  
Figure 7: An Example of HDR. We shown an example of HDR for the class of cats and dogs. Dashed vertical lines mark the HDR thresholds at the $1 0 \%$ quantile. Samples to the right of the threshold belong to the most probable $1 0 \%$ of the calibration distribution for that class. Samples to the left of the threshold are deemed outside the HDR and treated as potential hallucinations.   
Figure 8 shows the difference of HCDR and HDR.

# B Proofs of Main Text

# B.1 Proof of Theorem 5.1

To prove the existence of $\delta$ -hallucination, we state the following lemma.

Lemma B.1. The estimator $A ^ { * } ( X )$ that minimizes the expected quadratic loss over $A ( X )$ is

$$
A ^ {*} (X) = \underset {A (X)} {\mathbb {E}} [ A (X) ].
$$

Proof. As defined in Definition 3.1, for $A ^ { * } ( X )$ , the loss over $A ( X )$ is

$$
\begin{array}{l} \ell_ {A (X)} \left(A ^ {*} (X)\right) = \underset {A (X)} {\mathbb {E}} \left[ \| A ^ {*} (X) - a \| _ {2} ^ {2} \right] \\ = \int_ {a \in \mathcal {A}} \| A ^ {*} (X) - a \| _ {2} ^ {2} \cdot f _ {A (X)} (a) \mathrm {d} a, \tag {B.1} \\ \end{array}
$$

where $A$ is the output domain of $A ( X )$ (the set of all possible outputs). By our notation defined in Section 3, $f _ { A ( X ) }$ is the probability density function of $A ( X )$ .

![](images/d2f35f10a93c41f44f47fddae4dbab86bbbe4ecc141f8a35b5732644d874c7ef.jpg)  
Figure 8: An Example of HCDR vs. HDR. We show the difference between HCDR and HDR for a mixture of two normal distributions. The blue region denotes HDR, whereas the green and blue region together denote HCDR. δ denotes the bound of the HDR $( 1 0 \% )$ , and $\delta _ { 1 }$ $( 5 \% )$ denotes the bound for the conditional probabilities. Though HDR is encapsulated in HCDR in this example, HDR might contain regions outside HCDR in other cases, meaning HCDR is not simply an expansion of HDR.

Now, for an $A ^ { * }$ that minimizes the loss at $X$ . We have its gradient at $A ( X )$ to be $0 _ { d _ { a } }$ $\scriptstyle { \dot { d } } _ { a }$ is the output dimension as in Definition 4.1).

$$
\nabla \ell_ {A (X)} (A ^ {*} (X)) = 0.
$$

Combine the above equation with (B.1) we have

$$
\nabla \left(\int_ {a \in \mathcal {A}} \| A ^ {*} (X) - a \| _ {2} ^ {2} \cdot f _ {A (X)} (a) \mathrm {d} a\right) = 0. \tag {B.2}
$$

Since the $\nabla$ here denotes the gradient of $A ^ { * } ( X )$ , we have

$$
\begin{array}{l} \nabla \left(\int_ {a \in \mathcal {A}} \| A ^ {*} (X) - a \| _ {2} ^ {2} \cdot f _ {A (X)} (a) \mathrm {d} a\right) \\ = \int_ {a \in \mathcal {A}} \nabla \| A ^ {*} (X) - a \| _ {2} ^ {2} \cdot f _ {A (X)} (a) \mathrm {d} a \\ = \int_ {a \in \mathcal {A}} \nabla \left(\left\| A ^ {*} (X) \right\| _ {2} ^ {2} - 2 A ^ {*} (X) ^ {\top} a\right) \cdot f _ {A (X)} (a) d a \quad \left(\left\| A (X) \right\| _ {2} ^ {2} \text {i s e r a s e d w h e n t a k i n g t h e g r a d i e n t}\right) \\ \end{array}
$$

$$
\begin{array}{l} = \int_ {a \in \mathcal {A}} (2 A ^ {*} (X) - 2 a) \cdot f _ {A (X)} (a) d a \\ = 2 \int_ {a \in \mathcal {A}} A ^ {*} (X) \cdot f _ {A (X)} (a) \mathrm {d} a - 2 \int_ {a \in \mathcal {A}} A (X) \cdot f _ {A (X)} (a) \mathrm {d} a \\ = 2 A ^ {*} (X) - 2 \int_ {a \in \mathcal {A}} a \cdot f _ {A (X)} (a) \mathrm {d} a. \quad \left(\text {B y} \int_ {\mathcal {A}} f _ {A (X)} (a) d a = 1\right) \\ \end{array}
$$

Combine the above result with (B.2), we have

$$
2 A ^ {*} (X) - 2 \int_ {a \in \mathcal {A}} A (X) \cdot f _ {A (X)} (a) \mathrm {d} a = 0.
$$

Thus $A ^ { * }$ is

$$
A ^ {*} (X) = \int_ {a \in A} a \cdot f _ {A (X)} (a) \mathrm {d} a = \mathbb {E} [ A (X) ]. \tag {B.3}
$$

This completes the proof.

Theorem B.1 (Existence of $\delta$ -Hallucination under Single Input; Theorem 5.1 Restate). For an input $X$ , there exists infinitely many distributions of $A ( X )$ and $Z$ such that for an estimator $A ^ { * }$ that minimizes the expected quadratic loss defined in Definition 3.1 over $A ( X )$ , it is bound to $\delta$ -hallucinate at $X$ .

Proof. By Lemma B.1, we have

$$
A ^ {*} (X) = \underset {A (X)} {\mathbb {E}} [ A (X) ].
$$

We now construct a wide range of distribution of $A ( X )$ and $Z$ that satisfies

$$
f (A ^ {*} (X); Z) \leq \delta .
$$

Let $N$ (number of latent states) be any positive number. Then, let $A ( X ; Z _ { i } ) , i \in [ N - 1 ]$ be a normal distribution of the form

$$
f _ {A (X; Z _ {i})} (a) := (2 \pi) ^ {\frac {- d _ {a}}{2}} \det (\Sigma_ {i}) ^ {- \frac {1}{2}} \exp \biggl (- \frac {1}{2} (X - \mu_ {i}) ^ {\top} \Sigma_ {i} ^ {- 1} (X - \mu_ {i}) \biggr).
$$

By the requirements of normal distributions, $\Sigma _ { i }$ are positive-definite matrices in $\mathbb { R } ^ { d _ { a } \times d _ { a } }$ , and $\mu _ { i }$ are $d _ { a }$ -dimensional vectors.

This is also denoted as

$$
A (X; Z _ {i}) \sim \mathcal {N} (\mu_ {i}, \Sigma_ {i}),
$$

where $\textstyle { \mathcal { N } } ( \mu _ { i } , \Sigma _ { i } )$ denotes a normal distribution of mean $\mu _ { i }$ and covariance matrix $\Sigma _ { i }$ by convention.

Then, define $\mu _ { i }$ to satisfy

$$
f _ {A (X; Z _ {i})} (0 _ {d _ {a}}) = (2 \pi) ^ {\frac {- d _ {a}}{2}} \det (\Sigma_ {i}) ^ {- \frac {1}{2}} \exp \left(- \frac {1}{2} \mu_ {i} ^ {\top} \Sigma_ {i} ^ {- 1} \mu_ {i}\right) \leq \delta
$$

For any $\delta > 0$ , this $\mu _ { i }$ always exists. We give the following example.

$$
\mu_ {i} = m _ {i} 1 _ {d _ {a}},
$$

where $m _ { i }$ is

$$
\sqrt {\frac {- 2 \ln (\delta) - \ln (\det (\Sigma_ {i}))}{1 _ {d _ {a}} ^ {\top} \Sigma_ {i} 1 _ {d _ {a}}}}. \qquad \qquad \left(\delta \in (0, 1 ]\right)
$$

The probability density is

$$
\begin{array}{l} f _ {A (X; Z _ {i})} \left(0 _ {d _ {a}}\right) = \left(2 \pi\right) ^ {\frac {- d _ {a}}{2}} \det  \left(\Sigma_ {i}\right) ^ {- \frac {1}{2}} \exp \left(- \frac {1}{2} \mu_ {i} ^ {\top} \Sigma_ {i} ^ {- 1} \mu_ {i}\right) \\ = (2 \pi) ^ {\frac {- d _ {a}}{2}} \det (\Sigma_ {i}) ^ {- \frac {1}{2}} \exp \left(- \frac {1}{2} m _ {i} ^ {2} 1 _ {d _ {a}} ^ {\top} \Sigma_ {i} 1 _ {d _ {a}}\right) \\ = (2 \pi) ^ {\frac {- d _ {a}}{2}} \det (\Sigma_ {i}) ^ {- \frac {1}{2}} \exp \left(- \frac {1}{2} \frac {- 2 \ln (\delta) - \ln (\det (\Sigma_ {i}))}{1 _ {d _ {a}} ^ {\top} \Sigma_ {i} 1 _ {d _ {a}}} \cdot 1 _ {d _ {a}} ^ {\top} \Sigma_ {i} 1 _ {d _ {a}}\right) \\ = (2 \pi) ^ {\frac {- d _ {a}}{2}} \det (\Sigma_ {i}) ^ {- \frac {1}{2}} \exp \left(\ln (\delta) + \frac {1}{2} \ln (\det (\Sigma_ {i}))\right) \\ = (2 \pi) ^ {- \frac {d _ {a}}{2}} \det  (\Sigma_ {i}) ^ {- \frac {1}{2}} \cdot \det  (\Sigma_ {i}) ^ {\frac {1}{2}} \delta \\ = (2 \pi) ^ {\frac {- d _ {a}}{2}} \delta \\ \leq \delta . \tag {B.4} \\ \end{array}
$$

This means our definition of $\mu _ { i }$ is valid.

For simplicity, let $p _ { i }$ denote $\mathrm { P r } [ Z = Z _ { i } ]$

$$
p _ {i} := \Pr [ Z = Z _ {i} ].
$$

Now, define $A ( X ; Z _ { N } )$ to be

$$
A (X; Z _ {N}) \sim \mathcal {N} \left(- \sum_ {i \in [ N - 1 ]} \frac {p _ {i}}{p _ {n}} \mu_ {i}, \Sigma_ {N}\right). \tag {B.5}
$$

Let $\mu _ { N }$ denote $\begin{array} { r } { - \sum _ { i \in [ N - 1 ] } p _ { i } / p _ { n } \cdot \mu _ { i } . } \end{array}$

Let $m _ { N } \in \mathbb { R }$ be

$$
m _ {N} := \delta^ {- \frac {2}{d _ {a}}}.
$$

Then let $\Sigma _ { N }$ be defined as

$$
\Sigma_ {N} := \frac {1}{m _ {N}} \cdot I _ {d _ {a}},
$$

which is positive definite.

This means

$$
\Sigma_ {N} ^ {- 1} = m _ {N} \cdot I _ {d _ {a}}
$$

is also positive definite.

Thus we have

$$
\exp \left(- \frac {1}{2} \mu_ {N} ^ {\top} \Sigma_ {N} ^ {- 1} \mu_ {N}\right) \leq \exp (0) = 1.
$$

Then along with (B.5) we have

$$
\begin{array}{l} f _ {A (X; Z _ {N})} \left(0 _ {d _ {a}}\right) = (2 \pi) ^ {\frac {- d _ {a}}{2}} \det  \left(\Sigma_ {N}\right) ^ {- \frac {1}{2}} \exp \left(- \frac {1}{2} \mu_ {N} ^ {\top} \Sigma_ {N} ^ {- 1} \mu_ {N}\right) \\ \leq \det  (\Sigma_ {N}) ^ {- \frac {1}{2}} \\ = \left(m _ {N} ^ {d _ {a}}\right) ^ {- \frac {1}{2}} \\ = \delta^ {- \frac {2}{d _ {a}} \cdot \frac {- d _ {a}}{2}} \\ = \delta . \tag {B.6} \\ \end{array}
$$

Recall in (B.3) we have proven $A ^ { * }$ to be the expectation of $A$ . This means for the distribution $A ( X )$ we’ve constructed here, we have

$$
\begin{array}{l} A ^ {*} (X) = \mathbb {E} [ A (X) ] \\ = \underset {Z} {\mathbb {E}} [ \underset {A} {\mathbb {E}} [ A (X; Z) ] ] \\ = \sum_ {i = 1} ^ {N} \Pr [ Z = Z _ {i} ] \mathbb {E} [ A (X; Z _ {i}) ] \\ \end{array}
$$

$$
\begin{array}{l} = \sum_ {i = 1} ^ {N} p _ {i} \mu_ {i} \\ = \sum_ {i = 1} ^ {N - 1} p _ {i} \mu_ {i} + p _ {N} \mu_ {N} \\ = \sum_ {i = 1} ^ {N - 1} p _ {i} \mu_ {i} + p _ {N} \left(- \sum_ {i = 1} ^ {N - 1} \frac {p _ {i}}{p _ {N}} \mu_ {i}\right) \\ = 0. \\ \end{array}
$$

Combining the fact of $A ^ { * } ( X ) = 0$ with (B.4) and (B.6) satisfies the condition of $\delta$ -hallucination defined in Definition 4.2. This completes the proof.

# B.2 Proof of Corollary 5.2.1

Corollary B.1.1 (Existence of $\delta$ -Hallucination under Multiple Inputs; Corollary 5.2.1 Restate). For a set of input $X _ { j } , j \in [ S ]$ , there exists infinitely many distributions of $A ( X _ { j } )$ and $Z$ such that any estimator minimizing the expected quadratic loss defined in Definition 3.1 is bound to $\delta$ -hallucinate at $X$ .

Proof. Construct every $A ( x _ { j } )$ according to the construction of Section B.1. This makes every $A ^ { * } ( X _ { j } ) , j \in [ S ]$ to fall out of the non-hallucinating region. This completes the proof. □

# B.3 Proof of Section B.3

Theorem B.2 (Existence of $\delta$ -Hallucination on Semi-Optimal Estimators Under Single Input; Theorem 5.2 Restate). For an input $X$ , there exists infinitely many distributions of $A ( X )$ and $Z$ such that if an estimator $A ^ { \prime }$ is within a distance of $\epsilon$ to the optimal estimator $A ^ { * }$ , which writes as

$$
\left\| A ^ {\prime} (X) - A ^ {*} (X) \right\| _ {2} \leq \epsilon ,
$$

then $A ^ { \prime } ( X )$ is bound to $\delta$ -hallucinate.

Proof. By Lemma B.1, we have

$$
A ^ {*} (X) = \underset {A (X)} {\mathbb {E}} [ A (X) ].
$$

Thus we have

$$
\left\| A ^ {\prime} (X) - \mathbb {E} [ A (X) ] \right\| _ {2} \leq \epsilon . \tag {B.7}
$$

Let $N$ be any even number in $N ^ { + }$ .

Construct

$$
A (X; Z _ {i}) \sim \mathcal {N} (\mu_ {i}, I _ {d _ {a}}).
$$

Let $\begin{array} { r } { \mathbb { E } [ A ( X ) ] = \sum _ { i = 1 } ^ { N } p _ { i } \mu _ { i } } \end{array}$ be 0. Here $p _ { i } = \mathrm { P r } [ Z = Z _ { i } ]$ . Then by (B.7), we have

$$
\left\| A ^ {\prime} (X) - 0 \right\| _ {2} \leq \epsilon .
$$

Let $v _ { 0 }$ denote $A ^ { \prime }$ . The probability of $v _ { 0 }$ in $A ( X ; Z _ { i } )$ is

$$
(2 \pi) ^ {\frac {- d _ {a}}{2}} \exp \left(- \frac {1}{2} (v _ {0} - \mu_ {i}) ^ {\top} (v _ {0} - \mu_ {i})\right) = (2 \pi) ^ {\frac {- d _ {a}}{2}} \exp \left(- \frac {1}{2} \| v _ {0} - \mu_ {i} \| _ {2} ^ {2}\right).
$$

Set $\| \mu _ { i } \| _ { 2 } \geq \sqrt { - 2 \ln \delta } + \epsilon$ , we have

$$
\begin{array}{l} (2 \pi) ^ {\frac {- d _ {a}}{2}} \exp \left(- \frac {1}{2} \| v _ {0} - \mu_ {i} \| _ {2} ^ {2}\right) \leq \exp \left(- \frac {1}{2} \| v _ {0} - \mu_ {i} \| _ {2} ^ {2}\right) \\ \leq \exp \left(- \frac {1}{2} \left(\| v _ {0} - \mu_ {i} \| _ {2} - \| v _ {0} \| _ {2}\right) ^ {2}\right) \\ \leq \exp \biggl (- \frac {1}{2} (\sqrt {- 2 \ln \delta} + \epsilon - \epsilon) ^ {2} \biggr) \\ \leq \delta . \\ \end{array}
$$

Finally, let

$$
\mu_ {i} = - \frac {p _ {N - i}}{p _ {i}} \mu_ {N - i}. \quad (N \text {h a s b e e n s e t t o b e e v e n})
$$

This ensures PNi=1 piµi to be 0. $\textstyle \sum _ { i = 1 } ^ { N } p _ { i } \mu _ { i }$

The last constraint can coexist with √ $\| \mu _ { i } \| _ { 2 } \geq \sqrt { - 2 \ln \delta } + \epsilon$ in infinitely many constructions of $\mu _ { i } , i \in [ N ]$ (e.g., $\mu _ { i } = C \cdot i ( N - i ) ( \sqrt { - 2 \ln \delta } + \epsilon ) / p _ { N - i } \cdot 1 _ { d _ { a } }$ for any $C > 1 \AA$ ). This completes the proof.

![](images/1e311b8094d49c44c452c53538abceb4b04b644435de6c08dc3b49ae8ec05d89.jpg)

# B.4 Proof of Theorem 5.3

Theorem B.3 (Existence of $\delta$ -Hallucination at Tilted Input; Theorem 5.3 Restate). Let $B _ { \delta }$ denote the bound of all hints $\delta _ { i } , i \in [ N ]$ , defined as

$$
B _ {\delta} := \sup  _ {i \in [ N ]} \| \delta_ {i} \| _ {2}.
$$

For an $L$ -Lipschitz estimator $A ^ { * }$ satisfying Definition 3.2, there exists infinitely many distributions of $A ( X ; Z )$ such that $\delta$ -Hallucination happens on all $X + \delta _ { i }$ . That is, $A ^ { * } ( X + \delta _ { i } )$ does not fall into the region where $f _ { A \left( X ; Z _ { i } \right) } \geq \delta$ for any $i \in [ N ]$ by Definition 4.1.

Construct Proof. Let $\textstyle \sum _ { i = 1 } ^ { N } p _ { i } \mu _ { i } = 0 _ { d _ { a } }$ $A ( X ; Z _ { i } )$ be a normal distribution with a mean of , where $p _ { i } = \mathrm { P r } [ Z = Z _ { i } ]$ . $\mu _ { i }$ and a covariance matrix of $\Sigma _ { i }$

Because $A ^ { * }$ is $L$ -Lipschitz, we have

$$
\left\| A ^ {*} \left(X + \delta_ {i}\right) - A ^ {*} (X) \right\| _ {2} \leq L \| X + \delta_ {i} - X \| _ {2} = L \| \delta_ {i} \| _ {2} \leq L B _ {\delta}. \tag {B.8}
$$

See $L B _ { \delta }$ as $\epsilon$ , and $A ^ { * } ( X + \delta _ { i } )$ as different $A ^ { \prime }$ in Theorem 5.2. Apply Theorem 5.2 to every $A ( X + \delta _ { i } )$ . Thus, there are infinitely many distributions for $A ^ { * } ( X + \delta _ { i } )$ to $\delta$ -hallucinate over $A ( X )$ . This completes the proof.

![](images/6349e769147df49a08481503064aee2afe53d77361730e2eb7080831db28f3bb.jpg)

# B.5 Proof of Theorem 6.1

To prove Theorem 6.1, we state the following definitions amd assumptions.

We begin with the definition of means and variances for the variables of interest.

Definition B.1 (Means and Variances; Definition 6.1 Restate). Let $\{ Z _ { i } \} _ { i \in [ N ] }$ denote the possible states of the latent variable $Z$ , with probabilities $p _ { i } : = \mathrm { P r } [ Z = Z _ { i } ]$ . For each $i \in [ N ]$ , define the conditional mean

$$
\mu_ {i} := \mathbb {E} \left[ A \left(X; Z _ {i}\right) \right].
$$

We regard $\mu _ { i }$ as a realization of a random variable distributed according to $d _ { i } ^ { \mu }$ . Let $\mu _ { i } ^ { d } : = \mathbb { E } _ { d _ { i } ^ { \mu } } [ \mu _ { i } ]$ and $\sigma _ { i } ^ { d } : = \operatorname { V a r } _ { d _ { i } ^ { \mu } } [ \mu _ { i } ]$ denote the mean and variance of this distribution, respectively. Let $d ^ { \mu }$ denote the joint distribution of $( \mu _ { 1 } , \ldots , \mu _ { N } )$ . We write $\mu ^ { d } : = \mathbb { E } _ { d ^ { \mu } } [ \mu _ { 1 } , \dots , \mu _ { N } ]$ for its mean vector and $\begin{array} { r } { \sigma ^ { d } : = \mathbb { E } [ \sum _ { i = 1 } ^ { N } ( \mu _ { i } - \mu _ { i } ^ { d } ) ^ { 2 } ] } \end{array}$ as sum of variance.

We then provide the following assumptions applied to $\mu _ { i }$ and $d _ { i } ^ { \mu }$ in Definition B.1. In particular, we assume that the conditional means align around a common value and that the joint distributions of these conditional means are mutually independent.

Assumption B.1. We impose the following conditions on the distributions defined in Definition B.1:

1. Identical means: There exists a constant $\mu _ { 0 } \in \mathbb { R }$ such that $\mu _ { i } ^ { d } = \mu _ { 0 }$ , for all $i \in [ N ]$ .   
2. Independence: The distributions $\{ d _ { i } ^ { \mu } \} _ { i = 1 } ^ { N }$ are mutually independent.

We now characterize hallucination events in terms of output regions that correspond to high $( > \delta )$ conditional probability under each latent state.

Definition B.2 (High Conditional Density Regions; Definition 6.2 Restate). We define $U _ { i } ^ { \delta }$ to be

$$
U _ {i} ^ {\delta} := \left\{a \mid f (a; Z _ {i}) > \delta \right\},
$$

which is the region with posterior probability of $Z = Z _ { i }$ larger than $\delta$

Remark B.1 (Remark 6.2 Restate). By Definition B.2, $\delta$ -hallucination of $A ^ { * } ( X )$ is equivalent to

$$
A ^ {*} (X) \notin U _ {i} ^ {\delta}, \quad i \in [ N ].
$$

We then define the following spheres covering $U _ { i } ^ { \delta }$ in Definition B.2. Specifically, we enclose each $U _ { i } ^ { \delta }$ within the smallest possible sphere centered at the corresponding mean $\mu _ { i }$ .

Definition B.3 (Minimal Covering Spheres; Definition 6.3 Restate). For each $i \in [ N ]$ , let $U _ { i } ^ { \delta } \subset \mathbb { R } ^ { d _ { a } }$ denote the $\delta$ -high density region associated with state $Z _ { i }$ . Define $B _ { i } ^ { \delta } ( \boldsymbol { r } )$ as the closed Euclidean ball of radius $r$ centered at $\mu _ { i }$ . The minimal covering radius is

$$
r _ {i} := \inf  _ {r _ {i} \in \mathbb {R} ^ {+}} \left\{U _ {i} ^ {\delta} \subset B _ {i} ^ {\delta} (r _ {i}) \right\}.
$$

Thus $B _ { i } ^ { \delta } ( r _ { i } )$ is the smallest sphere centered at $\mu _ { i }$ that contains $U _ { i } ^ { \delta }$ . Finally, define the uniform covering radius

$$
r = \max  _ {i \in [ N ]} \{r _ {i} \}.
$$

Next, we state the following axillary lemmas.

Lemma B.2 (Paley-Zygmund Inequality). For any non-negative random variable $T$ and any $\theta \in [ 0 , 1 ]$ , we have

$$
\Pr [ T > \theta \cdot \mathbb {E} [ T ] ] \geq (1 - \theta) ^ {2} \frac {(\mathbb {E} [ Z ]) ^ {2}}{\mathbb {E} [ Z ^ {2} ]}.
$$

Lemma B.3 (Chebyshev Inequality). For any random variable $T$ , we have

$$
\Pr \left[ | T - \mathbb {E} [ T ] | \geq a \right] \leq \frac {\operatorname {V a r} [ T ]}{a ^ {2}}, \quad \text {f o r a l l c o n s t a n t} \quad a,
$$

where $\mathrm { V a r } [ T ]$ is the variance of $T$ .

Lemma B.4 (Cauchy Ineqaulity). For any $ { n _ { \mathrm { ~ \tiny ~  ~ } } } \in  { \mathbb { N } } ^ { + }$ along with two sets of variables $x _ { 1 } , x _ { 2 } , \cdots , x _ { n }$ and $y _ { 1 } , y _ { 2 } , \cdots , y _ { n }$ , they satisfy

$$
\left(\sum_ {i = 1} ^ {n} x _ {i} y _ {i}\right) ^ {2} \leq \left(\sum_ {i = 1} ^ {n} x _ {i} ^ {2}\right) \left(\sum_ {i = 1} ^ {n} y _ {i} ^ {2}\right).
$$

By Lemma B.3 and Lemma B.4, we derive a bound for the probability of distances between the loss minimizing estimator and the mean of $d ^ { \mu }$ defined in Definition B.1 which is $\mu _ { 0 }$ by Assumption B.1 as follows.

Lemma B.5 (Probability Upper Bound of Distance between $A ^ { * } ( X )$ and $\mu _ { 0 }$ in Assumption B.1). Let $A ^ { * }$ be the optimal estimator over $A$ . Then for any $d _ { 1 } > 0$ we have

$$
\Pr \left[ \| \mu_ {0} - A ^ {*} (X) \| _ {2} ^ {2} \geq d _ {1} ^ {2} \right] \leq \frac {\left(\sum_ {i = 1} ^ {N} p _ {i} ^ {2}\right) \sigma^ {d}}{d _ {1} ^ {2}}.
$$

Proof. By Lemma B.3, we have

$$
\begin{array}{l} \Pr \left[ (A ^ {*} (X) - \mu_ {0}) ^ {2} \geq d _ {1} ^ {2} \right] \leq \frac {\mathbb {E} [ (A ^ {*} (X) - \mu_ {0}) ^ {2} ]}{d _ {1} ^ {2}} \\ = \frac {\mathbb {E} \left[ \left(\sum_ {i = 1} ^ {N} p _ {i} \mu_ {i} - \mu_ {0}\right) ^ {2} \right]}{d _ {1} ^ {2}} \\ = \frac {\mathbb {E} \left[ \left[ \sum_ {i = 1} ^ {N} p _ {i} \left(\mu_ {i} - \mu_ {0}\right) \right] ^ {2} \right]}{d _ {1} ^ {2}} \\ \leq \frac {\mathbb {E} \left[ \left(\sum_ {i = 1} ^ {N} p _ {i} ^ {2}\right) \left[ \sum_ {i = 1} ^ {N} \left(\mu_ {i} - \mu_ {0}\right) ^ {2} \right] \right]}{d _ {1} ^ {2}} \quad (\text {B y}) \\ = \frac {\left(\sum_ {i = 1} ^ {N} p _ {i} ^ {2}\right) \mathbb {E} \left[ \sum_ {i = 1} ^ {N} \left(\mu_ {i} - \mu_ {0}\right) ^ {2} \right]}{d _ {1} ^ {2}} \\ = \frac {(\sum_ {i = 1} ^ {N} p _ {i} ^ {2}) \sigma^ {d}}{d _ {1} ^ {2}}. \\ \end{array}
$$

This completes the proof.

In addition, by Lemma B.2, we derive a lower bound of the probability of distances between $\mu _ { i }$

defined in Definition B.1 and $\mu _ { 0 }$ defined in Assumption B.1.

Lemma B.6 (Lower Bound on the Probability of Distance between $\mu _ { i }$ in Definition B.1 and $\mu _ { 0 }$ in Assumption B.1). For $i \in [ N ]$ , let $\mu _ { i }$ and $\mu _ { 0 }$ be as defined in Definition B.1 and Assumption B.1. We have, for any $\theta \in [ 0 , 1 ]$ ,

$$
\operatorname * {P r} \big [ \| \mu_ {i} - \mu_ {0} \| _ {2} ^ {2} \geq \theta \sigma_ {i} ^ {d} \big ] \geq (1 - \theta) ^ {2} K _ {i} ^ {\mu}.
$$

Proof. Because $\| \mu _ { i } - \mu _ { 0 } \| _ { 2 } ^ { 2 } \ge 0$ , by Lemma B.2, set $T$ in Lemma B.2 to be $\| \mu _ { i } - \mu _ { 0 } \| _ { 2 } ^ { 2 }$ , and we have

$$
\operatorname * {P r} \big [ \| \mu_ {i} - \mu_ {0} \| _ {2} ^ {2} \geq \theta \mathbb {E} [ \| \mu_ {i} - \mu_ {0} \| _ {2} ^ {2} ] \big ] \geq (1 - \theta) ^ {2} \frac {\mathbb {E} [ (\mu_ {i} - \mu_ {0}) ^ {2} ] ^ {2}}{\mathbb {E} [ \| \mu_ {i} - \mu_ {0} \| _ {2} ^ {4} ]} = (1 - \theta) ^ {2} K _ {i} ^ {\mu}.
$$

Combining with

$$
\mathbb {E} [ \| \mu_ {i} - \mu_ {0} \| _ {2} ^ {2} ] = \sigma_ {i} ^ {d},
$$

we have

$$
\operatorname * {P r} \left[ \| \mu_ {i} - \mu_ {0} \| _ {2} ^ {2} \geq \theta \sigma_ {i} ^ {d} \right] \geq (1 - \theta) ^ {2} K _ {i} ^ {\mu}.
$$

This completes the proof.

Therefore, by Lemma B.5 and Lemma B.6, combined with Definition B.3, we prove the lower bound of the probability of hallucination.

Theorem B.4 (Hallucination Probability Lower Bound; Theorem 6.1 Restate). Let $( A ( X ) , Z )$ satisfy Assumption 6.1. For each $i \in [ N ]$ , let $\mu _ { i } , \sigma _ { i } ^ { d }$ be as in Definition 6.1, let $\mu _ { 0 }$ be as in Assumption 6.1, and let $r _ { x }$ be as in Definition 6.3. Define

$$
d := (\sum_ {j = 1} ^ {N} p _ {j} ^ {2} \sigma_ {j} ^ {d}) ^ {1 / 2}, \quad \theta_ {i} (\alpha) := \frac {(\alpha d + r _ {x}) ^ {2}}{\sigma_ {i} ^ {d}}, \quad \alpha > 1, \quad \mathrm {a n d} \quad K _ {i} ^ {\mu} := \frac {(\mathbb {E} [ (\mu_ {i} - \mu_ {0}) ^ {2} ]) ^ {2}}{\mathbb {E} [ (\mu_ {i} - \mu_ {0}) ^ {4} ]}.
$$

If for every $i \in [ N ]$ there exists $\alpha _ { i } > 1$ such that $\theta _ { i } ( \alpha _ { i } ) \leq 1$ , then

$$
P _ {H} ^ {\delta} > \prod_ {i = 1} ^ {N} (P _ {i} K _ {i} ^ {\mu}),
$$

where $P _ { H } ^ { \delta }$ denotes the probability that the optimal estimator $A ^ { * } ~ \delta$ -hallucinates at $X$ (equivalently, $A ^ { * } ( X ) \notin U _ { i } ^ { \delta }$ for all $i \in [ N ]$ , with $U _ { i } ^ { \delta }$ as in Definition 6.3).

Proof. By Lemma B.5, for every $i \in [ N ]$ , we have

$$
\operatorname * {P r} \big [ \| \mu_ {0} - A ^ {*} (X) \| _ {2} ^ {2} \geq d _ {i} ^ {2} \big ] \leq \frac {(\sum_ {i = 1} ^ {N} p _ {i} ^ {2}) \sigma^ {d}}{d _ {i} ^ {2}}.
$$

This means

$$
\Pr \left[ \| \mu_ {0} - A ^ {*} (X) \| _ {2} ^ {2} \leq d _ {1} ^ {2} \right] \geq 1 - \frac {\left(\sum_ {i = 1} ^ {N} p _ {i} ^ {2}\right) \sigma^ {d}}{d _ {1} ^ {2}}. \tag {B.9}
$$

By Lemma B.6, we have, for every $i \in [ n ]$

$$
\Pr \left[ \| \mu_ {i} - \mu_ {0} \| _ {2} ^ {2} \geq \theta_ {i} \sigma_ {i} ^ {d} \right] \geq (1 - \theta_ {i}) ^ {2} K _ {i} ^ {\mu}. \tag {B.10}
$$

Then, Definition B.3, the probability for $A ^ { * }$ to fall out of the region with a conditioned probability of $A ( X ; Z _ { i } )$ no less than $\delta$ is at least

$$
\begin{array}{l} \operatorname * {P r} \left[ A ^ {*} (X) \notin U _ {i} ^ {\delta} \right] \geq \operatorname * {P r} \left[ A ^ {*} (X) \notin B _ {i} ^ {\delta} (r _ {i}) \right] \\ \geq \operatorname * {P r} [ \| A ^ {*} (X) - \mu_ {0} \| _ {2} \leq d _ {i} ] \cdot \operatorname * {P r} [ \| \mu_ {i} - \mu_ {0} \| _ {2} \geq d _ {i} + r _ {x} ] \\ \geq (1 - \frac {(\sum_ {i = 1} ^ {N} p _ {i} ^ {2}) \sigma^ {d}}{d _ {i} ^ {2}}) ((1 - \theta_ {i}) ^ {2} K _ {i} ^ {\mu}) \qquad \mathrm {(B y (B . 9) a n d (B . 1 0))} \\ = (1 - \frac {1}{\alpha_ {i} ^ {2}}) (1 - \theta_ {i}) ^ {2} K _ {i} ^ {\mu}. \\ \end{array}
$$

Set $\alpha _ { i }$ to maximize

$$
(1 - \frac {1}{\alpha_ {i} ^ {2}}) (1 - \theta_ {i}) ^ {2},
$$

which is equivalent to maximizing $P _ { i }$

Then we have

$$
\Pr \left[ A ^ {*} (X) \notin U _ {i} ^ {\delta} \right] \geq P _ {i} K _ {i} ^ {\mu}.
$$

Given $d _ { i } ^ { \mu } , i \in [ N ]$ are independent to each other, we have

$$
\Pr \left[ A ^ {*} (X) \notin U _ {i} ^ {\delta}, i \in [ N ] \right] \geq \prod_ {i = 1} ^ {N} P _ {i} K _ {i} ^ {\mu}.
$$

The left-hand side is equivalent to $P _ { h } ^ { \delta }$ (see Definition B.2 and Remark B.1).

This completes the proof.

# C Derivation to Cross-Entropy Loss

In this section, we derive the cross-entropy loss version of our results in Section 5.

Definition C.1 (Cross-Entropy Loss). For an input $X$ and an according possible output $a \in { \mathcal { A } }$ , given a target probability density $q _ { X } ^ { a } \in [ 0 , 1 ] ^ { C }$ and a model-estimated distribution $p _ { X } \in [ 0 , 1 ] ^ { C }$ over $C$ classes, let $q _ { X } ^ { a } ( t )$ and $p _ { X } ( t )$ denote their $t$ -th entry respectively. The cross-entropy loss at $X$ is defined as

$$
\mathcal {L} \left(q _ {X} ^ {a}, p _ {X}\right) = - \sum_ {t \in [ C ]} q _ {X} ^ {a} (t) \log p _ {X} (t),
$$

where $\begin{array} { r } { q _ { X } ^ { a } ( t ) \geq 0 , \sum _ { t \in [ C ] } q _ { X } ^ { a } ( t ) = 1 } \end{array}$ , $p _ { X } ( t ) \geq 0$ , and $\begin{array} { r } { \sum _ { t \in [ C ] } p _ { X } ( t ) = 1 } \end{array}$ .

We define the total loss at $X$ as the expectation of loss over $A$ at all $a$ , that is

$$
E _ {a} \left(\mathcal {L} \left(q _ {X} ^ {a}, p _ {X}\right)\right).
$$

Comparing to the notation in Section 5, the predictor $A ^ { * }$ at input $X$ outputs the predicted probabilities $A ^ { * } ( X )$ , which can be noted here as

$$
[ A ^ {*} (X) ] (t) := p _ {X} (t), t \in [ C ],
$$

We now prove the existence of $\delta$ -hallucination under cross-entropy loss.

Theorem C.1 (Existence of $\delta$ -Hallucination under Cross-Entropy Loss). For an input $X$ , there exists infinitely many target distributions $A ( X )$ such that the $A ^ { * }$ minimizing the cross-entropy loss defined in Definition C.1 at $X ~ \delta$ -hallucinates.

Proof. We first calculate the loss minimizing $A ^ { * }$ at $X$ .

$$
\begin{array}{l} E _ {a} (\mathcal {L} (q _ {X} ^ {a}, p _ {X})) \\ = \int_ {\mathcal {A}} p (a) \left[ - \sum_ {t \in [ C ]} q _ {X} ^ {a} (t) \log p _ {X} (t) \right] d a \\ = \sum_ {t \in [ C ]} (- \log p _ {X} (t)) \left[ \int_ {\mathcal {A}} q _ {X} ^ {a} (t) d a \right] \\ = \sum_ {t \in [ C ]} (- \log p _ {X} (t)) E _ {a} q _ {X} ^ {a} (t). \\ \end{array}
$$

Thus by Gibbs Inequality, we have the loss minimizing $p _ { X } ( t )$ of $E _ { a } ( \mathcal { L } ( q _ { X } ^ { a } , p _ { X } ) )$ is

$$
p _ {X} (t) = E _ {a} q _ {X} ^ {a} (t), t \in [ C ].
$$

We then construct the latents that induce the $\delta$ -hallucination at $X$ .

Define the probability distribution under each $Z _ { i }$ as

$$
A (q _ {X} ^ {a} | Z = Z _ {i}) \sim \mathcal {N} (q _ {i}, d), i \in [ N ],
$$

in which $q _ { i }$ is

$$
q _ {i} (t) := e _ {t} ^ {(C)},
$$

and

$$
d \leq - \frac {N - 1}{N \ln (\delta^ {2})}.
$$

Then let $P ( Z _ { i } ) = 1 / N$ , we have $p _ { X }$ equals

$$
p _ {X} := \frac {\sum_ {i = 1} ^ {N} e _ {i} ^ {(C)}}{N}.
$$

Then

$$
\begin{array}{l} P (p _ {x} | Z = Z _ {i}) \\ = \frac {1}{\sqrt {2 \pi d}} \exp \left(- \frac {\left(p _ {X} - q _ {i}\right) ^ {2}}{2 d}\right) \\ = \frac {1}{\sqrt {2 \pi d}} \exp \left(- \frac {N - 1}{2 d N}\right) \\ \leq \frac {1}{\sqrt {- 2 \pi \frac {N - 1}{N \ln (\delta^ {2})}}} \exp \left(- \frac {N - 1}{- 2 \frac {N - 1}{N \ln (\delta^ {2})} N}\right) \\ \leq \frac {1}{\sqrt {- \pi \frac {1}{\ln (\delta^ {2})}}} \frac {\delta^ {2}}{2} \\ \leq \frac {\delta^ {2} \ln (\delta^ {- 1})}{\sqrt {2 \pi}} \\ \leq \frac {\delta^ {2} (\delta^ {- 1} - 1)}{\sqrt {2 \pi}} \\ \leq \delta , \\ \end{array}
$$

for every $i$

This completes the proof.

# References

Sumukh K Aithal, Pratyush Maini, Zachary Lipton, and J Zico Kolter. Understanding hallucinations in diffusion models through mode interpolation. Advances in Neural Information Processing Systems, 37:134614–134644, 2024.   
Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930, 2024.   
Sourav Banerjee, Ayushi Agarwal, and Saloni Singla. Llms will always hallucinate, and we need to live with this. arXiv preprint arXiv:2409.05746, 2024.   
Michele Caprio, David Stutz, Shuo Li, and Arnaud Doucet. Conformalized credal regions for classification with ambiguous ground truth. arXiv preprint arXiv:2411.04852, 2024.   
Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020.   
Matthew Dahl, Varun Magesh, Mirac Suzgun, and Daniel E Ho. Large legal fictions: Profiling legal hallucinations in large language models. Journal of Legal Analysis, 16(1):64–93, 2024.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.   
Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.   
Andreas F Haselsteiner, Jan-Hendrik Ohlendorf, Werner Wosniok, and Klaus-Dieter Thoben. Deriving environmental contours from highest density regions. Coastal Engineering, 123:42–51, 2017.   
Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems, 43(2):1–55, 2025.   
Rob J Hyndman. Computing and graphing highest density regions. The American Statistician, 50 (2):120–126, 1996.   
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM computing surveys, 55(12):1–38, 2023.

Xuhui Jiang, Yuxing Tian, Fengrui Hua, Chengjin Xu, Yuanzhuo Wang, and Jian Guo. A survey on large language model hallucination via a creativity perspective. arXiv preprint arXiv:2402.06647, 2024.   
Adam Tauman Kalai and Santosh S Vempala. Calibrated language models must hallucinate. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, pages 160–171, 2024.   
Adam Tauman Kalai, Ofir Nachum, Santosh S Vempala, and Edwin Zhang. Why language models hallucinate. arXiv preprint arXiv:2509.04664, 2025.   
Junyi Li, Jie Chen, Ruiyang Ren, Xiaoxue Cheng, Wayne Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. The dawn after the dark: An empirical study on factuality hallucination in large language models. arXiv preprint arXiv:2401.03205, 2024.   
Stephanie Lin, Jacob Hilton, and Owain Evans. Truthfulqa: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958, 2021.   
Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253, 2024.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.   
RJ Samworth and MP Wand. Asymptotics and optimal bandwidth selection for highest density region estimation. 2010.   
Yuji Tachikawa. On gauging finite subgroups. SciPost Physics, 8(1):015, 2020.   
Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. Hallucination is inevitable: An innate limitation of large language models. arXiv preprint arXiv:2401.11817, 2024.   
Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, et al. Siren’s song in the ai ocean: a survey on hallucination in large language models. arXiv preprint arXiv:2309.01219, 2023.