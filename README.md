# Waveformer: A Frequency-Aware Transformer-CNN Hybrid for Hyperspectral Image Demosaicing

This repository contains the official implementation of **Waveformer: A Frequency-Aware Transformer-CNN Hybrid for Hyperspectral Image Demosaicing**.

## Introduction

Hyperspectral image demosaicing aims to reconstruct full-resolution hyperspectral images from mosaiced observations.  
In this work, we propose **Waveformer**, a frequency-aware Transformer-CNN hybrid framework designed for hyperspectral image demosaicing. The model combines the local representation ability of CNNs with the global modeling capability of Transformers, while explicitly considering frequency information to improve reconstruction quality.

## Code Release

This repository provides the official code implementation of **Waveformer**.

## Evaluation

All quantitative metrics reported in this repository are computed using:

```bash
ssim_compute_xjw.m
