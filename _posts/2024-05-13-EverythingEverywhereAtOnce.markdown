---
layout: post
title: Eveything Everywhere at once - A Distributed Training Saga...
date:   2024-05-13 12:57:49 +0000
categories: jekyll update
excerpt: my notes on distributed data parallelism(DDP)
mathjax: true
---

Training large models on vast datasets used to be a test of patience for me. Hours would pass as I kept an eye on my machine, ensuring it didn't idle out, all while feeling the frustration mount. It was evident that if I could harness the power of multiple GPUs simultaneously, training would be significantly faster. So, I decided to go deep into Distributed Training.

Initially, I was skeptical. I thought mastering it required an in-depth understanding of GPU internals and computer networking. However, it turns out, you can get by just fine without diving into those complexities. But before we dive deeper, let's address the fundamental question: **Why Distributed Training?**

Firstly, it saves time. This much is obvious. Secondly, it increases the amount of compute power we can throw at the problem, ultimately helping us train our models faster. As models grow larger, it becomes increasingly difficult to put them to a single GPU and train. So this is where distributed data parallel comes in, but 
#### What does Distributed Data Parallel (DDP) do exactly?
In traditional single-GPU training, the model resides on one GPU, receiving input batches from a data loader. It performs a forward pass to calculate the loss, then a backward pass to compute parameter gradients. These gradients are then used by the optimizer to update the model. But what if we distributed this process across multiple GPUs? With DDP, one process is launched per GPU. Each process holds a local copy of the model and optimizer. These replicas are kept identical, with DDP ensuring synchronization throughout the training process.
<div class="imgcap">
<img src="/assets/ddp/ddp.jpeg" width="500" style="border: none;">
<figcaption>Figure 1: Process : DDP </figcaption>
</div>
Each GPU process works on different data, but how do we ensure this? Enter the distributed sampler, paired with the data loader. It ensures that each GPU process receives different inputs—a technique known as data parallelism. This allows us to train multiple data instances concurrently with the same batch size. But With each process model recieves different input, loacally run forward and backward pass and beacuse the inputs were different then the gradient accumulated now are also different. Running the optimizer at this stage would yield distinct parameters across devices, resulting in separate models. DDP circumvents this issue by initiating a synchronization step. Gradients from all replicas are aggregated using a bucketed ring all-reduce algorithm. This process overlaps gradient computation with communication, ensuring GPUs are never idle. 
The synchronization step doesn't wait for all gradients to be computed. Instead, it starts communication along the ring while the backward pass is ongoing. This keeps the GPUs busy and optimizes efficiency. Now each model replica has the same gradients,and now running the optimizer step now will update  all the replicas parameters to same value. After each training epoch, all replica models remain in sync, thanks to this coordinated effort.

