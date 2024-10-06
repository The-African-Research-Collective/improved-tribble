from typing import Optional, List

import torch
from transformers import AutoConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str,
                             device: str,
                             tokenizer_name: Optional[str] = None,
                             model_kwargs: Optional[dict] = None,
                             ) -> tuple:
    """
    Loads a pre-trained model and tokenizer.
    Args:
        model_name (str): The name or path of the pre-trained model.
        device (str): The device to load the model onto (e.g., 'cpu', 'cuda').
        dtype (Optional[torch.dtype]): The data type to load the model with (e.g., torch.float32).
        tokenizer_name (Optional[str]): The name or path of the tokenizer. If None, defaults to model_name.
        model_kwargs (Optional[dict]): Additional keyword arguments to pass to the model.
    Returns:
        Tuple[PreTrainedModel, PreTrainedTokenizer]: The loaded model and tokenizer.
    """
    
    model_configuration = AutoConfig.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        **model_kwargs if model_kwargs else {},)
    
    tokenizer = AutoTokenizer.from_pretrained( tokenizer_name if tokenizer_name else model_name)

    return model, tokenizer, model_configuration

def inference_model(system_prompt: str | List[str],
                    input_sample: str | List[str],
                    model: AutoModelForCausalLM,
                    tokenizer: AutoTokenizer,
                    model_configuration: AutoConfig,
                    few_shot_examples: Optional[List[str]] = None,
                    temperature: Optional[float] = 0.0,
                    num_tokens_to_generate: Optional[int] = 100,
                    output_attentions: Optional[bool] = False,
                    ) :
    
    system_prompt_tokens = tokenizer(system_prompt, return_tensors='pt')['input_ids']
    input_sample_tokens = tokenizer(input_sample, return_tensors='pt')['input_ids']

    if few_shot_examples:
        few_shot_examples_tokens = tokenizer(few_shot_examples, return_tensors='pt')['input_ids']

    with torch.no_grad():
        model.eval()

        if few_shot_examples:
            input_tokens = torch.cat([system_prompt_tokens, few_shot_examples_tokens, input_sample_tokens], dim=-1) 
        else:
            input_tokens = torch.cat([system_prompt_tokens, input_sample_tokens], dim=-1)
        
        device = model.device
        input_tokens = input_tokens.to(device)
        
        for i in range(num_tokens_to_generate):
            output = model(input_ids=input_tokens,
                           output_attentions=output_attentions,
                           return_dict=True,
                           )
            print(len(input_tokens[0]))
            print(len(output))
            print(output['attentions'][0][:,:,:,-1].shape)

            print(len(output['attentions']))

            break

        print(model_configuration)
        
        




if __name__ == "__main__":
    model_name = "microsoft/Phi-3.5-mini-instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, model_configuration = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
    )

    system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."
    input_sample = "The assistant helps the user to book a flight ticket."

    inference_model(system_prompt, input_sample, model, tokenizer, model_configuration, output_attentions=True)