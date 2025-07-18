# Experiment: FFT Loss for DreamBooth LoRA

This project explores integrating **FFT (Fast Fourier Transform) loss** into the fine-tuning of a LoRA adapter for DreamBooth.

## Loss Functions

* **Diffusion Loss**
  Standard latent-space loss: Mean Squared Error (MSE) between predicted and true noise.

* **FFT Loss**
  Pixel-space loss: After decoding latents with a VAE, both generated and target images are transformed to the frequency domain (via FFT), and the difference between their spectra is penalized.

**Training Objective:**
At each diffusion step *t*, we optimize:

$$
L_\text{total} = L_\text{diffusion}(z_t, \epsilon) + \alpha \cdot L_\text{FFT}(x_\text{decoded}, x_\text{target})
$$

Where:

* \$\mathcal{L}\_\text{diffusion}\$: MSE between predicted and target noise in latent space
* \$\mathcal{L}\_\text{FFT}\$: L1 or L2 distance between FFT spectra (magnitude) of decoded and target images
* \$\alpha\$: FFT loss weight (typically 0–0.001)
* \$z\_t\$: latent at step \$t\$
* \$\epsilon\$: ground-truth noise
* \$x\_\text{decoded}\$: model output after VAE decoding
* \$x\_\text{target}\$: ground-truth image

This combined loss is applied at every diffusion step, guiding the model to match both noise structure (latent) and frequency content (pixel), encouraging better preservation of both fine details and overall structure.

## FFT Loss Illustration

FFT loss is calculated as the spectral distance between the generated and target images.

![FFT Loss Diagram](fft_example.png)

*Left: Original image & FFT spectrum. Right: Generated image & FFT spectrum. FFT loss compares these spectra to improve detail and realism.*

## Visual Results

Comparison between baseline and FFT loss-enhanced DreamBooth LoRA fine-tuning:

![Lora-500-steps Results Overview](results.png)

*Left: Default. Right: With FFT loss. Zoomed crops show improved sharpness and fidelity.*

**Benefits:**

* Sharper high-frequency details
* Better object feature preservation
* More realistic outputs
* Reduced blurriness

---

**Model:** [RazinAleks/stable-diffusion-xl-base-dreambooth-lora-fft-loss](https://huggingface.co/RazinAleks/stable-diffusion-xl-base-dreambooth-lora-fft-loss)
