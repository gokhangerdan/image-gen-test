import time
import uuid

t = time.time()
from diffusers import DiffusionPipeline
print("diffusers module loaded in:", time.time()-t, "s")

t = time.time()
pipeline = DiffusionPipeline.from_pretrained("midjourney-community/midjourney-mini")
print("pipeline loaded in:", time.time()-t, "s")

t = time.time()
image = pipeline(
	"stained glass of darth vader, backlight, centered composition, masterpiece, photorealistic, 8k"
).images[0]
print("image generated in:", time.time()-t, "s")

t = time.time()
image.save('images/image_'+str(uuid.uuid4())+'.png')
print("image saved in:", time.time()-t, "s")
