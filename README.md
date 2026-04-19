# Diffusion from scratch

In this repo, I implement the Diffusion models from scratch, from the first-principles.

Papers I'm implementing:

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/pdf/2006.11239)

There might be elements from other papers that I have learnt in my process of understanding Diffusion models. I will try my best to incorporate those papers in later commits, but for now, my implementation will be largely faithful to the original paper.

## Steps to setup

I am using uv as the package manager in this project

```bash
```

## Denoising Diffusion Probabilistic Models

Planned folder structure:

### Noise Schedulers

- In this repo, I have implemented a single class that can call the following noise schedulers: linear, log linear, cosine and sigmoid.

