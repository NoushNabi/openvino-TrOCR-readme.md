## Environment Setup
```
sudo apt update
sudo apt install git-lfs -y

python3 -m venv python-env
source python-env/bin/activate
pip install update --upgrade

python -m pip install "optimum-intel[openvino]"@git+https://github.com/huggingface/optimum-intel.git
```


## Sample TrOCR pipeline with OpenVINO (FP32)

[Optimum Intel](https://huggingface.co/docs/optimum/intel/inference) can be used to load optimized models from the Hugging Face Hub and create pipelines to run inference with OpenVINO Runtime without rewriting your APIs.
For Transformer models, just replace the `AutoModelForXxx` class with the corresponding `OVModelForXxx` class. Click [HERE](https://huggingface.co/docs/optimum/intel/inference) for more details.
For TrOCR, since it is a vision to sequence model, we use `OVModelForVision2Seq` instead of using `AutoModelForVision2Seq`. 


```
from transformers import TrOCRProcessor
from optimum.intel.openvino import OVModelForVision2Seq
from PIL import Image
import requests
 
# load image from the IAM database
url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
 
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
model = OVModelForVision2Seq.from_pretrained('microsoft/trocr-small-handwritten', export=True)
pixel_values = processor(images=image, return_tensors="pt").pixel_values
 
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
``` 

You can also apply 8-bit quantization on your model’s weight when loading your model by setting the `load_in_8bit=True` argument when calling the `from_pretrained()` method.
```
model = OVModelForVision2Seq.from_pretrained('microsoft/trocr-small-handwritten', load_in_8bit=True, export=True)
```
**NOTE**: `load_in_8bit` is enabled by default for the models larger than 1 billion parameters. You can disable it with `load_in_8bit=False`.



## Export INT8 and INT4 model using optimum-cli
[Optimum Intel](https://github.com/huggingface/optimum-intel?tab=readme-ov-file#openvino) CLI provides a way to export a model from HuggingFace to the OpenVINO IR format.

```
optimum-cli export openvino --model MODEL_ID --weight-format WEIGHT_FORMAT --output EXPORT_PATH
```

Replace the placeholders with the appropriate values:
- `MODEL_ID`: The ID of the HuggingFace model you wish to export.
- `WEIGHT_FORMAT`: The desired weight format for compression. Options include "int4", "int8", or "fp16". See [HERE](https://huggingface.co/docs/optimum/intel/optimization_ov#weight-only-quantization) for more details.
- `EXPORT_PATH`: The directory path where the exported model will be saved.

## Export FP16, INT8 and INT4 models using optimum-cli and Python SDK.
Below is how you can use optimum-cli to apply FP16, INT8, or INT4 quantization on your model’s weight. 
```
optimum-cli export openvino --model microsoft/trocr-base-printed --weight-format fp16 ov_model_sixt

optimum-cli export openvino --model microsoft/trocr-base-printed --weight-format int8 ov_model_int8

optimum-cli export openvino --model microsoft/trocr-base-printed --weight-format int4 ov_model_int4
```
After this conversion, you can pass the converted model path as **model_id** argument when calling the `from_pretrained()` method. Also, you can determine your target device (CPU, GPU, or MULTI:CPU,GPU) as **device** argument in that method. 
```
from transformers import TrOCRProcessor
from optimum.intel.openvino import OVModelForVision2Seq
from PIL import Image
import requests

url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = OVModelForVision2Seq.from_pretrained(model_id='./ov_model_int8', device="CPU", ov_config={"PERFORMANCE_HINT": "LATENCY", "CACHE_DIR":"./ovcache"}, export=False)
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
```
Please note that instead of using optimum-cli for int8 quantization, you can directly use `OVModelForVision2Seq` as mentioned above. 
