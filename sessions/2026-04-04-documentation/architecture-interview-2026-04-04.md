# Architecture Interview - 2026-04-04

**Project**: Barevision  
**Interviewer**: AI Assistant  
**Interviewee**: Project Author  

---

## Big Picture

**Q: What problem is barevision solving? What does "non-semantic perception" mean?**

**A:** I mean "non-semantic" because whenever someone talks about computer vision for robots, or computer vision in general, the most salient idea is "object recognition"—or roughly anything that converts raw vision to something intelligible by human reason. This is not that. This is a lower-level kind of vision, more akin to what animals in general need in order to move around: understand the space around you so that you can know if you can move safely, know where you are, know how to get somewhere, what's the shape of something in front of you, how to inspect an object, and so on.

**Q: Is this approach better than alternatives?**

**A:** I don't know if this is "better" because I also don't know about the alternatives. I think most robotics computer vision efforts are geared towards higher-level reasoning, or in the case of Tesla, my project would be only the bottom layer of perception, upon which they derive higher and higher concepts. To some extent this aims to be a dumbed-down version of what a Tesla car does at its bottom layer.

The restrictions are: a single camera, cheap hardware, little power.

The relaxations to the problem: we can move slow and we can iteratively understand our surroundings better.

We're not a 2-ton beast moving at 60 miles/hour. We're a 1-pound sloth doing low-cost maintenance stuff (this is just an exaggerated example).

So I guess what makes us "different" rather than better is that we're positioning ourselves at the cheap end of the spectrum. I'm convinced that we can derive a lot with a camera, time, and a cheap CPU—and that we might not need more than that.

If you look at this from an economics perspective: a $30k robot will likely not do tasks that are below a certain value because it wouldn't be financially convenient (it must at least pay for its own value over time). I think a $100 robot has a wide range of tasks it can do, and it would not have competition from the more expensive products. This creates an opportunity to focus on something that's not the lion's share of the robotics market that is starting to exist.

**Q: What's the core insight behind using hierarchical embeddings for optical flow?**

**A:** The core insight for using hierarchical embeddings is: regardless of embeddings, for determining optical flow we want to match a given feature or patch of an image to another image (the next frame) and find out how it was transformed—hopefully mostly shifted.

So these "features" need to be:
1. Unique within the frame so they identify themselves
2. Not so brittle that the next frame—which has a slightly different pose, illumination, or the object slightly moved—no longer contains the same feature

So we say "embeddings" just to mean: the projection of the image into a feature vector. We work with dense feature vectors.

Why hierarchical? A couple of reasons.

First, if we need to find a patch in another frame, it's computationally expensive because the search area grows with the square of the distance, and we'd want to find each pixel in the origin frame. So for a given frame radius, we'd have r² × r² comparisons (squared source items × squared search spaces). This scales awfully. If a patch moved too much, in order to capture it we'd need a big R, but that's a no-go because of the cost constraint.

But given that objects are lumped together, we can use a hierarchical model so that in a very blurred / zoomed-out image, you can see very large movements as small shifts in the image. And we can take that coarse flow as a prior for a more fine-grained flow. So this opens up a chance for flow to always become a "small lookup" as long as we reposition the frames based on the prior flow. This yields a kind of "residual flow estimation" which is part of the model.

**Q: What's the current implementation status?**

**A:**
- **dataset/**: I have a custom set of videos of mostly static scenery. Static is good for this stage since it makes it easier to estimate flow (only depth and pose changes).
- **embeddings/**: Given a frame, we calculate a dense grid of embeddings. These are quite unique within the frame and findable in the other frame. Not unique as in "single match," but rather a blob centered around each position that is usually shifted in the other frame based on flow. We like the performance of this so far.
- **flow/**: We had an implementation here but I deleted it. I made the wrong call of training jointly (just because there was some shared processing). I'll do this again now that we're in a good place on embeddings. It's the next major goal. I have a bunch of notes on this if relevant, but I don't know if we want to document the future.

**Q: What's the end-to-end pipeline once complete?**

**A:** The project right now focuses only on getting good embeddings for flow (unique and matchable). There are many stages that we haven't implemented (well, we have, but I rolled them back after it became a mess; now I have a clearer idea of how to do it). Even after flow, we want to estimate object depth, pose changes, object movements. With that, we'll want to maintain maps so we can locate ourselves and so on. But we'll do one bit at a time.

The input is video frames from a single camera, likely low resolution (up to 512×512 or so). The processing goes as described above: embeddings → flow estimation → (joint pose + depth estimation) → (SLAM, likely many stages and components). The robot moves with this information or understands how to plan actions based on this. It's the low-level vision API.

---

## Design Decisions & Architecture

**Q: How does the hierarchical embedding pyramid work? How many levels?**

**A:** The pyramid expects to have at least 2 and up to any number of levels. Realistically I expect no more than 5-6 levels. There's a tight relationship between the resolution increase at each level (roughly 2× per axis) so this and the "lookup window size" and the max image resolution dictates how many levels we have. And all of this is subject to available computation. The more levels, the more accurate flow is.

The spatial reduction is roughly 2×, but there's a gotcha: I decided not to pad the convolutions (3×3) with SAME padding or any other kind of padding. I think that introduces too many artifacts to the embeddings, and I don't want to spend cycles on bad data. So we're reducing a bit more than that (the code/module has the math on the reduction). The first level, being fed a raw image, makes two depthwise convolutions instead of one, so the reduction is a tad larger (4 lost pixels instead of 2). At some point I think I'll unify both stem and standard block for simplicity.

**Q: What is "decoupled cascade with symmetric mean subtraction"?**

**A:** "Decoupled cascade" is a fancy name some model came up with. I'm not sure if I like it. The main part is "mean subtraction".

See, the main limitation of the embeddings model is that it has a fixed and small set of convolution filters, so it's not amazing at adapting to different scenery. Roughly, you want the embedding generator to find unique patches. The raw convolution is a good generalist pattern finder. On the other end of the spectrum, we can leverage the fact that we are within a specific frame and we care about uniqueness only in that frame (and the next, which is 99% the same).

So we thought about "removing the DC component" of the pre-processed image because that component contributes nothing or little to the uniqueness. So what we do is train another depthwise convolution (this time it's padded) and initialize it so it has a larger chance of learning to extract the "average" image. We use that convolution output in two ways:
1. We pass this down to the next level as the input (we downsample it by slicing it with stride=2 before that)
2. We subtract it from the calculated embeddings before normalization. This helps boost the "unique" part of the signal around a small neighborhood.

The results have varied, but last time I experimented, we got a 10–15% better loss, so it's pretty good. The double usage of the convolution is likely me just trying to reuse every computation that we consume. Perhaps we don't need it, but again, it works well.

**Q: What does the spatial variance loss do? Why is attention concentration important?**

**A:** So we needed to formalize this "uniqueness" of the embeddings. Early in the process, I realized that we wouldn't ever produce a single bright spot in the image identifying our source pixel/patch. Instead, we'd find a smooth peak around the right position. Not even a round shape, but rather a "segmentation" of the pixels that look like it, which are most usually nearby.

We originally used a different loss for this. We were minimizing entropy. Our thought was that a small entropy would be a small spot, so it incentivizes the right embedding. But as we trained and tried to improve performance, the model started predicting very sharp but scattered spots. Weird, but it breaks the entropy → single-spot assumption. This is because entropy doesn't care about the position of each element's value.

So we turned it into a position-aware metric: let's calculate the "center of mass" of each blob, and calculate the weighted difference between each target score against this center. This turns out to be the definition of variance, only it's "spatial" because we multiply each coordinate in the lookup window by the match intensity, so it's a "weighted coordinate" when you add them up.

Attention concentration is very useful because it lets us match a small spatial region of one image to a small region of the other one. Larger areas will undergo more varied transformations (because they are at a different pose), so it'd be hard to deduce how they moved. The closer we are to a "single point identified," the better, because points won't be deformed but rather translated. So this is why we search for focused areas (individual points are not feasible/brittle). The tradeoff is that we need some neighborhood so the structure has enough information to be found, but only as little as possible so it's small.

I tried variants in the past, but they all come down to: search for this patch in the other frame and give me the location. Everything else is heuristics to make the process cheaper/faster/more robust/etc. For example, two cameras (stereo) reduce the search space at the cost of another camera and hardware complications. Still, the model could work well on stereo gear.

**Q: Why self-supervised learning vs. supervised with labeled flow data?**

**A:** I'm interested in self-supervised because:
1. It's the most intuitive way for me to think about models
2. I don't have resources to label a lot of data
3. Most of the datasets for flow are academic. I don't trust that they'll map well to real life
4. I think that in the future, only self-supervised models will drive most of AI. I think there's enough redundancy in video and consistency in the universe to find the rules that dictate how pixels ought to behave over time, in general

**Q: What does the data pipeline look like during training?**

**A:** We have a bunch of videos; we count their frames. We split between training/validation (we even keep a couple of entire videos out). We take frame pairs within a boundary (max frame distance), and those become our examples. We use all the combinations. We have up to 8k combinations in our training dataset. It seems small, but each frame contains many thousand embedding examples, so it's not that few. In the future, I think we'll be able to train with any kind of videos—for example, YouTube ones.

**Q: How do you validate that embeddings are "good" without ground truth flow labels?**

**A:** Let's discuss our loss function and where it could be wrong. We got a good loss value: embeddings in one frame are quite unique among themselves. Also, when we search each one in the other frame, the matching scores are highly concentrated in a single small cluster. Given that the frames are pretty much the same but the pose slightly changed, or an object moved (not yet), or illumination changed, it's very hard to—I haven't yet found a single example where—we have this scenario and we matched the wrong parts of the image.

We do have limitations, such as identical patterns → our loss is not great because we match in many places. The hierarchical nature helps us here a bunch because it limits the search space. But it's not perfect.

**Q: What are the target hardware constraints?**

**A:** I'm working on an NPU that's $10 USD. It can do 0.5 TOPS in INT8. I don't want you to stress about this too much. It is an extreme setup, but in the past it led you to justifying designs that were not great. I would say this: we are conscious about performance. But not all FLOPs are the same. Memory alignment and other things end up taking more time than just doing a million extra ops. I don't yet have the right intuitions about how tensor cores work, but we'll get there.

The actual limitations are:
- We work with a reduced set of ONNX instructions. Typically the cheap ones are OK. One notable example is that we can't do GatherND at inference time (but we do at training time!). This is problematic, and I changed the hierarchical design approach based on this (not yet implemented; we'll talk a lot about this).

Thing is, I don't have a specific memory or latency budget. I have rough ideas: a $100 robot gives me a $25 budget for computing, including a camera. So as barebones as possible. I want the robot to receive signals pretty fast, around 10–15 frames per second. But I'm sure it won't be easy, and I'll likely have a slower rate or more expensive hardware. At this point it's: let's do the cheapest thing that actually runs and is useful, and we'll take it from there.

The constraints affect the architectural choices all the time. It's the primary driver, but I won't budge easily on trade-offs. It needs to be good enough.

**Q: Why JAX/Flax over PyTorch?**

**A:** I found myself more at ease with JAX. I'm new to machine learning, so I figured if I started from scratch, I'd rather pick the more advanced / conceptual framework. I love how JAX works, but I have had some difficulty with their "pure" way of doing stuff. Also, I wanted to avoid the noise of "batteries included" PyTorch. I want to make a model, not stitch together pieces that already work. I'm focusing on an area where most components won't suit me because of performance. And more than anything, I'm doing this to *understand*.

**Q: Why VALID padding specifically?**

**A:** Besides what I already explained (not wasting CPU on tainted information), it opens up something interesting: since two frames that "flowed" will have some of their borders out of frame (some pixels of f1 will land outside f2, typically near borders), I don't need to compute flow on the entire image. I can perfectly concentrate on the "center".

Moreover, we're planning to do this hierarchical flow treated as a prior for the finer-grained levels. And in order to leverage this (without access to GatherND), we "shift" the entire frame by the predicted flow so we maximize the chance that pixels flow nearby. But every time we flow an entire frame, we need to deal with the borders.

With VALID padding, if you think about the pixels lost: every time you're walking from coarse-grained to fine-grained, you'll find that your "flowed / upscaled frame portion" is a subset of the fine-grained frame. Like a crop of a region. So we can shift it by considering a different crop of the other frame. I'm not explaining this well, but consider taking two frames, semi-transparent, one on top of each other. You'd reposition each one so that most of the image aligns. That's what we do, and the wasted pixels work in our favor because they would be part of a wasted border anyway.

**Q: Why L2-normalized outputs?**

**A:** This is where I just followed your (some model's) recommendation. I'm not that good at linear algebra or experienced in ML. I think it has to do with: if we don't normalize them, the loss function (includes a softmax for attention) will want sharper inputs, and this would drive convolutions to become "extreme" so that they can easily win over their neighbors. And this produced some random patterns that didn't work. But this was before we moved to spatial variance, so perhaps it's no longer necessary!

With regards to normalization, here's the situation in general: I don't have the intuition to know exactly where we need them (I could argue in both directions), and you or other models—typically regressing to the mean (or perhaps understanding this way better than me)—suggest normalizing, and one of the arguments is that "everyone does it, hard-won lesson, etc." I don't fully trust the argument, but in this case (at that time) it solved a problem. I really dislike this situation because I never know when it's a good time to remove them, and I want to constantly prune the model from artificial things that make it expensive and complex.

---

## Implementation & Entry Points

**Q: How do you train the embeddings model?**

**A:** We train by running the embeddings.training module (`python -m barevision.embeddings.training`). That's the main entry point. It takes a bunch of options through the command line. I removed the inference script because there was too much complexity to maintain. I'm fine with the validation runs within training so far. We'll add this back, though.

**Q: How do you run tests?**

**A:** We run tests through `pytest src/`. We also have a smoke-test run of the training loop because most errors are integration errors and not always caught in a unit test. So we have an alternative entry point: `python -m barevision.embeddings.smoke_test`. That loads smoke-test settings that set up a small model, minimal dataset, and trains for only a couple of steps, a couple of epochs. Its purpose is to run through checkpointing, logging, visualization generation, loss functions, gradient updates, epoch limits, etc.

**Q: What does GatherND do that you need?**

**A:** GatherND allows me to do a "variable slice". Think of GatherND as the basis for doing a "warp" transformation, where you have a matrix full of offsets, so GatherND would give you back a similar matrix but with those variable offsets (it's more general than this). This would be super useful in that when we apply the prior flow on a frame, we'd really like to apply the flow not to the entire frame but to smaller regions (so we can capture more than one set of movements). Given that we can't, we do the entire frame. At some point we may consider having a "bifocal" or "multifocal" algorithm where we chase more than one flow pattern when we have a multi-modal flow distribution. But at this point, that's costly, and I don't know if it's realistic.

I don't know if this is documented in our code. It definitely should be in the architecture doc. This limitation mostly drives how the hierarchical model behaves (this and the padding choice).

**Q: Can you walk through what happens in a single training step?**

**A:** A training step gets a pair of frames from the dataset. It runs the (embeddings) model once per frame. It estimates the spatial variance loss in two ways: it does it for "self-attention" on the first frame (I think), and then it does the cross-attention spatial variance between frame 1 and frame 2. This happens in a hierarchy (an image is a pyramid, actually). And it mostly works with lookup windows (attention happens in small, non-overlapping areas forming a grid). It adds up both losses (has a weight hyperparameter). This generates gradients that flow back.

If you want to know what the model actually does when called, let me know. I think we covered some of this elsewhere.

The losses (so far) at the embedding model between levels are just averaged per level and then weight-added across levels. In the end, we have the loss equivalent of a single lookup window (that went through this two-step averaging/weighting). I'm not entirely sure this is a good strategy, though, because it dilutes the signals of the fine-grained levels by a lot.

Another thing that I don't like—and this drove my intent to train this jointly with flow: at the fine-grained levels, the frames moved by a lot, so the embeddings are less reliable (cross-attention will sometimes find nothing). So far we have a decay for fine-grained levels, but it's too global. We likely need to make this better. When we implement flow, the "frame shifting" approach would re-align more pixels of each frame, making the loss more relevant at those levels, provided the coarser levels did a good job.

**Q: What parts of the current implementation are you least confident about?**

**A:**
1. **Frame shifting and embeddings loss significance** when no match is findable. What I said above.
2. **Textureless areas**. Not so fragile, but I worry that this is hurting the training procedure by trying to make the loss better in these cases (too many pixels in a textureless area → high influence).
3. **Not enough training data**. Not very worried, though. As we make progress, we should be able to incorporate tons of it.
4. **Limited capacity of the few convolutions we run**. I implemented a massive MoE approach, but didn't get good results. I mostly blame that it got too complex and I couldn't debug it properly.
5. **Mean subtraction is too local**. I wish it had a larger field of view, but it's only a single 3×3. I don't want to spend more on this, though. It's worse on the finer-grained levels.

**Q: What hyperparameters matter vs. ones that don't?**

**A:** Softmax temperature drives most of the results. Low temperatures make the results sharper (good). Too sharp, and it starts misbehaving, though. Sweet spot seems to be at 0.2 (perhaps this is 0.25 or 1/√16)? It'd be good to know.

The level decay is also a bit relevant because fine-grained levels usually have worse results due to lack of shifting, so they should influence less. Haven't played much with it. Decay usually at 1.1 or 1.2.

I haven't played much with the other parameters, or they didn't make much of a difference. Perhaps they do! It'd be interesting to run a grid search. I don't expect any miracles, though.

**Q: What was wrong with the joint training approach for flow?**

**A:** There's nothing wrong with joint estimation per se. It worked to some extent. Only that the code became too complex, and when things didn't work, I had a hard time debugging and figuring out what was wrong. Sometimes it was a visualization, sometimes it just wasn't learning, sometimes the models fought each other. I need to have ways of training in isolation, at least to have a more controlled environment.

The main sin of the joint approach is that I fought too hard to reuse the attention mechanism that the embeddings loss functions use. They're part of training (not inference) for embeddings, but they're the fundamental matching function (inference) for flow. I wanted to calculate this once, so I mixed up training with inference of the other model. That wasn't a good call. I even think JAX may be able to figure out that we're calculating the same thing in two different parts of the graph and simplify it. Even if it doesn't, I can waste more time in training insofar as I don't make the system as complex again.

The plan for flow is pretty much the same as what we had before, but do it a single step at a time. My main concern is not to explode in complexity: two sets of settings? Two training scripts? How about logging? Duplicating brings in one kind of problem. But unifying brings another kind of problem. Hard to decide. I need to move more slowly here and consult you more.

---

## Project Structure

**Q: What goes in runs/ vs checkpoints/?**

**A:**
- `runs/` contains TensorBoard logs. When we train, we put logs in here and TensorBoard picks them up. I don't commit this.
- `checkpoints/` well, training checkpoints. Also not committed.

**Q: What's in datasets/ vs src/barevision/dataset/?**

**A:** (Implicit from context) `datasets/` contains actual video data. `src/barevision/dataset/` contains the data loading code.

**Q: What's the agents/ directory for?**

**A:** (Not discussed in detail)

**Q: What's in ideas/?**

**A:** Right now, just a place where I can put summaries of conversations with you that don't need to affect our current workflow. Things to remember that I should come back to at some point.

---

## Gotchas

**Q: Any gotchas in the codebase that aren't obvious?**

**A:** Most JAX-adjacent libraries (Orbax, Flax) have *terrible* documentation. They usually make breaking changes and don't document them properly. Also, because they're not popular, you don't find very good examples on the web (or in your training data, so you also don't know much about those). The only thing that works is: when something doesn't work or we don't know how it works, write a full standalone app/script that lets us know what the behavior is. This is how we solved issues with checkpointing, with Flax's JIT wrappers, and so on.

---

*Interview conducted 2026-04-04 for the purpose of generating ARCHITECTURE.md*
