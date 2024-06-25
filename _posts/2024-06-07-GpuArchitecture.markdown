---
layout: post
title: How to train your Cuda? Going deep to GPU Architecture!
date:   2024-06-07 12:57:49 +0000
categories: jekyll update
excerpt: The overall Architecture of the GPU and some bottlenecks!
mathjax: true
---

Let's talk about GPU architecture! GPU consists of multiple Streaming Multiprocessor (SMs), each consisting of multiple cores with shared control and memory, these cores can execute some arithematic operations and cores in SM share some kind of control and they also share some kind of memory. And all the SMs have access to the same global memory. So how do we take these grids, blocks and these threads and run them on GPU. Threads are assigned to SMs at block granularity (all threads in a block are assigned to the same SM). We can have multiple thread blocks assigned to same SM, however we cannot have threads inside of a thread block assigned to different SMs. Threads/ blocks require resources to execute like registers and memory, so SMs can accomodate a limited number of threads/blocks at once.

if launch a grid that has more block theb we can run simultanousely, the remaining blocks wait for other blocks to finish before they can be assigned to an SM.

# Synchornization


Threads in the same bloc can collaborate in ways that thread in different blocks cannot :
- Barrier synchornization: __syncthreads()
-- wait for all threads in the block to reach the barrie before any thread can proceed, this help them cordinate work with each othe

-- the another way threads in a same block can collaborate is using some kind of shared memory that threads on a same block can access and threads on the different blocks cannot access


This is the motivation for assigning threads to SMs at block granularity. So when we assign threads in same block to the same SMs this make it easier for us to support collaboration between them efficient to support. All threads in block assigned to SM simultaneously means a block cannot be assigned to an SM until it secures enough resources for all its threads to execute. and if we don't assign all threads to SM at the smae time then some thread may reach a barrier and others cannot exceute, the system could deadlock.

Threads in different blocks do not synchornize with each other this allows block to execute in any order and this allows blocks to exceute in parallel with each other or sequentially with respect to each other.
This enables transparent scalability meaning same code can run on different devices with different amounts of hardware parallelism meaning : Exceute blocks sequentially if device has few SMs, execute blocks in parallel in device has many SMs.

A thread blocks in SMs execute until its done and because of this restriction we do not write code that tries to synchronizes across blocks cause deadlocks if blocks are not scheduled simultanousely.

Now that we have seen how we can schuleded blocks on different SMs, let see how threads are scheduled on one SM.
Threads assigned to an SM run concurrently, the scheduler manages their execution, blocks assigned to an SM are furthur divider into warps which are the unit of scheduling.

Lets talk a little bit about warps, warps are the unit of scheduling in an SM, the size of warps is device-specific, but has always been 32 threads to date.
if I create a thread block that is 1024 threads these 1024 threads are going to be grouped into 32 warps that are 32 thrads each.

Thread in a warp are scheduled together and executed folowing the SIMD model.

# Single Instruction Multiple Data
 means when threads are bound to each other by SIMD model, one instructions for all these threads and all threads will exceute same instructions but each of them processing it on different data, the advantage of that is only one instruction is needs to be fetched, we decode the instruction once only, and decoding the instruction requires some kind of control and if we fetch for each thread then we will 
rerquire to execute lot of control logic, thus this is the advantage of SIMD, that share the same instruction fetch and dispatch  unit across multiple execution units i.e amortizes the cost of control across more execution units.

The disadvantage of SIMD is different threads taking different exceution paths results in control divergence. In this situation, warp does a pass over each unique exceution path, in each pass, threads taking the path execute while others are disabled,example if reach if else statement some threads takes the true branch and some take the false branch what happens is all threads are gonna go together to the true branch only some of them will exceute and some of them gonna false branch and only the others will exceute and what that means is that at samne point during the divergence only some threads will active and the other will be remain inactive.


# Latency Hiding
When a Warp needs to wait for a high latency operation, another warp that is selected and scheduled for execution.
