# Sketch_2_Image

[![Demo](https://img.shields.io/badge/🤗%20Hugging%20Face%20Space-Sketch2Img-blue)](https://huggingface.co/spaces/pilotj/sketch_2_img)




Helps you convert architecture sketches to realistic images - based on stable diffusion.

Generations on given sample
<img width="985" height="292" alt="image" src="https://github.com/user-attachments/assets/10683e32-b99c-460b-be64-a56bbc657949" />


Example generations on training samples
<img width="865" height="446" alt="image" src="https://github.com/user-attachments/assets/3af0a04d-f6c2-4703-a420-ef6d0bb26af3" />

_Note: Trying some hyperparam tuning and improvements (read below)_


# Resources
Gradio App deployed on Hugging Face Spaces -[Sketch_2_Img](https://huggingface.co/spaces/pilotj/sketch_2_img). Please wait for ~700 sec for the generation.




### How to Reproduce Results
Please use kaggle to run notebooks as data sources are uploaded there. If you face any issue please feel free to contact.
- Trained weights - [controlnet_lora_finetuned](https://www.kaggle.com/datasets/mldtype/lora-weights-full) , use lora_v2 with num_inference_steps <=50.
- Training dataset  - [sketch_2_image](https://www.kaggle.com/datasets/mldtype/sketch-2-image-dataset)
- Training and inference Jupyter notebooks are provided.  

# Building Sketch to Refined Image - v1

<details>
<summary><strong>Task</strong></summary>

Develop, fine-tune, or strategically prompt a small AI model that transforms a predefined hand-drawn building sketch into a refined, structurally consistent image. 

</details>

---

<details>
<summary><strong>Dataset Curation</strong></summary>

Before creating the dataset, I defined a set of requirements to ensure coverage of both structural and stylistic diversity. These requirements were:

- Inclusion of different architectural styles of varying proportions — Modern, Neoclassical, Gothic, American, and Rural  
- Diversity in camera angles/points of view — right, left, and center perspectives.  
- Representation of different materials and, if possible, colors — concrete, steel, brick, stone, glass, and wood.  
- Samples of houses with both angular and flat roofs.  
- Inclusion of some curved structures to avoid overfitting toward straight edges.  
- Examples of glass structures with reflection and transparency to help the model identify glass windows under natural light.  

Based on these requirements, I collected a total of 35 images, sourced partly from the internet and partly from the [Kaggle architecture dataset](https://www.kaggle.com/datasets/wwymak/architecture-dataset). These were colored images of real structures. Initially, I explored whether any construction sketch datasets existed. Although I found one paper on similar work ([Using structure-aware diffusion model to generate renderings of school buildings](https://arxiv.org/pdf/2503.03090)), the dataset was not available, and most publicly available sketch datasets contained only plain outlines. In contrast, architectural diagrams are much more regular and complex in their drawing.  

To address this, I used ChatGPT to generate sketches from the collected real images. 
Each sketch was paired with a small guiding prompt following the syntax: `{structure} of {color} made of {material}`

These prompts were manually created to enforce specific features during generation. Some images were left without prompts to test performance without conditioning. I performed sampling on both prompt-free and prompt-based generation and found that text conditioning worked significantly better in our case.  

For v1, no augmentations were applied apart from these text prompts. I experimented with BLIP2 and Flamingo for automated image captioning to generate such guidance prompts, but their performance was poor, so manual creation was preferred.

</details>

---

<details>
<summary><strong>Update [12/08/25] – Method used is not exactly LoRA</strong></summary>
 

While reviewing my training code, I discovered that I had **not** actually used conventional LoRA as initially intended. My implmentation is slightly different and has infact produced better outcome.

LoRALinear returns _W₀ + ΔW₀_, where: _W₀_ is the original matrix (usually kept frozen), _ΔW₀_ is the required weight update, provided by two low-rank matrices (A, B). Normally,  **only ΔW₀** is trainable, since it’s the product of newly initialized matrices _A_ and _B_ (new `nn.Parameter` tensors in PyTorch are trainable unless you disable their gradients).

However, I spotted an anomaly in my implementation:  
 I explicitly set `requires_grad = True` for `lora_layers` at the end of the `init_lora_layers` function, due to which **both** W₀ and ΔW₀ were being trained.



`def init_lora_attn(model, lora_rank=4, alpha=1.0):`
    
    lora_layers = {}
    # Freeze all original parameters first
    for param in model.parameters():
        param.requires_grad = False
    for name, module in model.named_modules():
        if name.endswith("attn1") or name.endswith("attn2"):
            attn = module

            # Replace and register LoRA for to_q, to_k, to_v
            for proj_name in ['to_q', 'to_k', 'to_v']:
                orig_layer = getattr(attn, proj_name)
                lora_layer = LoRALinear(orig_layer, rank=lora_rank, alpha=alpha)
                setattr(attn, proj_name, lora_layer)
                lora_layers[f"{name}.{proj_name}"] = lora_layer

            # to_out usually ModuleList or Sequential, replace first layer
            if isinstance(attn.to_out, (nn.ModuleList, nn.Sequential)):
                orig_out = attn.to_out[0]
                lora_out = LoRALinear(orig_out, rank=lora_rank, alpha=alpha)
                attn.to_out[0] = lora_out
                lora_layers[f"{name}.to_out.0"] = lora_out

    # Unfreeze all LoRA parameters only   ----> this part
    for lora_layer in lora_layers.values():
        for param in lora_layer.parameters():
            param.requires_grad = True

    return lora_layers

  For proper LoRA fine-tuning I should have trained **only ΔW₀**, keep **W₀** frozen. That means the highlighted part should be removed.

  I retried the experiment with conventional LoRA fine-tuning, but results were underwhelming(artificial looking) 
  <img width="650" height="400" alt="image" src="https://github.com/user-attachments/assets/dd109978-d400-42e7-bc81-8b32b140c08a" />

  **Why did the modified method work for us**  
  - When training **W₀ + ΔW₀** for the first few epochs, W₀ adapts significantly to the target generation style (pure LoRA generations tend to look more “artificial”).
  - In later epochs, ΔW₀ captures finer details with almost no updates to W0

  So, this unintentional 'mistake' worked in my favour. This is the primary method used to produce reported results. In the following sections, this method is referred to as **_modified LoRA_**. The reported results are correct, just wanted to clarify this bit.
  </details>

-----

<details>
<summary><strong>Model & Methodology Details</strong></summary>
   
*Method described here is v1 due to time and compute constraints. Possible extensions and better approaches are also discussed.*

As a baseline, Stable Diffusion was tested both with and without text conditioning. In these trials, many outputs degenerated into random noise, while a few managed to produce basic outlines with some color. However, these results proved to be highly sensitive to the specific prompt used.

Our objective is to perform inpainting as well as capture structure details. Since structural consistency is a need, I chose to go with ControlNet (Canny Edge version). ControlNet takes as input the edge map of a provided image and generates a realistic colored image. In our case, the base version of ControlNet could just produce outlines — this is because architecture sketches have multiple pencil strokes and shades to capture not only the structure but also depth and exposure to the light source. Edge maps are mostly binary and have almost no explicit indication of depth or natural light (pixel intensity sometimes captures light information). Due to compute constraints, I chose to go with a 300M parameter base model of ControlNet. I created a custom dataset of 35 images with guiding prompts. I used _modified LoRA_ to finetune the model.

**Challenge** — Initially I started with Hugging Face's diffusers library to use LoRA; however, a lot of their functions are now deprecated (documentation not updated) and some are under upgradation (discovered this by reading their source code). For ControlNet from a custom checkpoint, there was no direct method — hence I decided to implement custom LoRA and generation pipeline. MSE was used as a loss. 

**Setups tried** —  
- _modified LoRA_ with rank 4 (weak details) — 40M params  
- _modified LoRA_ with rank 256 (most cost effective) — 60M params  
- _modified LoRA_ with rank 512 (most detailed) — 90M params  

It was seen that text conditioning hugely improves generation quality. So the final training happens for `{sketch, prompt}` pairs. We use small prompts with a semi defined structure as discussed above. Best model was decided among a set of models with low loss and good visually aligned generation of provided validation images.

**Improvement Suggested (WIP)** — I have not explicitly enforced number of windows or structural regularity as a condition. One way around this is to generate  edge maps, count the number of rectangles — on target image and generation. A new regularisation term can be added to the loss: `LPIPS(edge_map_tg, edge_map_gen) + K * abs(#rect_tg - #rect_gen)`
where `K` will be a hyper parameter (~0.1). The number of rectangles is directly correlated to the number of windows and to structure as well, hence should improve generation. However, this is highly dependent on the algorithm to count rectangles as for structures made of stone/brick, the number of rectangles will be very large. Hence, a small value of `K` is suggested.

---

### Training Setup

| **Parameter**              | **Details** |
|----------------------------|-------------|
| **Compute**                | Kaggle P100 GPU |
| **Base Models**            | Hugging Face Diffusers (pretrained models) |
| **Frameworks**             | PyTorch (custom DataLoader and LoRA implementation) |
| **Dataset Size (training)**| 32 samples |
| **Batch Size**              | 4 |
| **Total Epochs**            | 250 (early stopped at ~200) |
| **Learning Rate Schedule and Optimiser** | Cosine, AdamW |
| **Initial Learning Rate**   | 5 × 10⁻⁵ |
| **LoRA Rank**               | 512 |

</details>

---

<details><summary><strong>Current Problems and way to v2</strong></summary>

1. **Sensitivity to `num_inference_steps`** : Generation quality and consistency vary significantly across image categories. Emperical observation - It is related to sketch qualtiy and complexity. Unpredictable behavior makes it difficult to generalize the number of steps needed.

2.  **Poor performance without guidance prompt** : Model relies heavily on guidance prompts. Indicates the need for a larger and more diverse training dataset.

3. **Sketch background introduces noise** : Background elements degrade model understanding of the core sketch. This impacts both generation quality and training signal.
4. **Unable to detect balconies** : Our method fails to generate balconies. Pure LoRA is able to produce balcony outlines (see samples below) but is unable to generate good images.
   
   <img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/b640622a-e4af-4ef8-ab23-ff8151aa4dba" /><img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/0405ff91-a782-43e2-8b87-d56d740af339" /><img width="300" height="300" alt="image" src="https://github.com/user-attachments/assets/0a22e3e6-b3e8-4b8a-9c1e-2802c43bdbbe" />




### Proposed Way Forward

1. **Remove sketch background** before feeding into the model.
2. **Incorporate edge loss** term during training to enforce structural alignment.
3. **Use ResNet to encode sketch + edge map**
   - Feed both sketch and its edge map through ResNet layers (multi - controlNet)
   - Finetune first few ResNet layers alongside LoRA.
4. **Introduce material-based color/texture maps**
   - Fixed set of maps based on material and texture. Will be fused with initial input. Provide richer guidance for surface properties.
5. **Use multiple randomized prompts per training sample**
   - Replaces current fixed prompts.
   - Reduces overfitting to specific prompt wording.
   - Encourages the model to learn from image maps (e.g., edges, texture) rather than relying heavily on text.
6. **Use two step finetuning**. Start with current model (_modified LoRA_) and finetune it using conventional LoRA. We should be able to generate finer structure like glass balconies then.
7. **Evaluate model using LPIPS**

</details>

---
## TODOs

- [x] Write function to directly inject LoRA adapters into model layers (due to naming conflicts current method requires full model instantiation)
- [x] Rewrite inference pipeline - unoptimised
- [X] Fix gradio "generation" error (unable to initialise from finetuned wts).
- [ ] Create YAML config for training loop
- [ ] Enable WandB tracking
- [ ] Enable batch-wise validation during training




