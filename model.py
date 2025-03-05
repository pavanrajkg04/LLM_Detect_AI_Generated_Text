from transformers import AutoModelForCausalLM, CodeGenTokenizer
import torch

device = "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

llm_tokenizer = CodeGenTokenizer.from_pretrained("microsoft/phi-2", add_bos_token=True, trust_remote_code=True)
if llm_tokenizer.pad_token is None:
    llm_tokenizer.pad_token = llm_tokenizer.eos_token

llm_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/phi-2",
    torch_dtype=torch_dtype,
    device_map=device,
    trust_remote_code=True
)
max_length = 2048