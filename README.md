The implementation of the stable diffusion Web UI is more efficient than diffusers. I like its rich features, but the library mixes UI, API, and other logic, making it very complex. For learning purposes, I want to extract its core generation logic and separate it from the UI and API components.

I didn't use the Web UI directly; instead, I used Web UI Forge. This is because I wanted to understand its patcher methodology (maybe from ComfyUI), which appears to provide better extensibility for the models.

For test:

``` bash
python ./simplediffusion/__init__.py
```

Coverage:

``` bash
coverage run  .\simplediffusion\__init__.py; coverage xml
```

Roadmap:

- [x] Stable Diffusion XL Support
- [x] Basic text to image
- [x] Basic image to image
- [x] Reference only
- [x] ControlNet
- [ ] Other forge built-in extensions
- [ ] Stable Diffusion 3 Support
