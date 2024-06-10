The implementation of the stable diffusion Web UI is more efficient than diffusers. I like its rich features, but the library mixes UI, API, and other logic, making it very complex. For learning purposes, I want to extract its core generation logic and separate it from the UI and API components.

I didn’t use the Web UI directly; instead, I used Web UI Forge. This is because I wanted to understand its patcher methodology, which appears to provide better extensibility for the models.
