from typing import Optional, List

import torch
from transformers import AutoConfig, StoppingCriteria
from transformers import AutoModelForCausalLM, AutoTokenizer

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence: List[int]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

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
                    argmax: Optional[bool] = True,
                    max_new_tokens: Optional[int] = 100,
                    min_new_tokens: Optional[int] = 5,
                    do_sample: Optional[bool] = False,
                    penalty_alpha: Optional[float] = None,
                    no_repeat_ngram_size: Optional[int] = None,
                    ) :
    
    system_prompt_tokens = tokenizer(system_prompt, return_tensors='pt')['input_ids']
    input_sample_tokens = tokenizer(input_sample, return_tensors='pt', add_special_tokens=False)['input_ids']

    if few_shot_examples:
        few_shot_examples_tokens = tokenizer(few_shot_examples, return_tensors='pt')['input_ids']

    eos_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        model.eval()

        if few_shot_examples:
            input_tokens = torch.cat([system_prompt_tokens, few_shot_examples_tokens, input_sample_tokens], dim=-1) 
        else:
            input_tokens = torch.cat([system_prompt_tokens, input_sample_tokens], dim=-1)
        
        device = model.device
        input_tokens = input_tokens.to(device)

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": min_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample, 
            "penalty_alpha": penalty_alpha,
            "no_repeat_ngram_size": no_repeat_ngram_size,
        }

        generated = model.generate(
            input_tokens,
            **generate_kwargs,
            return_dict_in_generate=True,
            output_attentions=True,
            stopping_criteria=[EosListStoppingCriteria(eos_sequence=[eos_id] + model_configuration.eos_token_id)],
        )

        print(len(generated['attentions']))

        #generated tokens is the last 10 characters
        generated_tokens = generated['sequences'][0].tolist()[len(input_tokens[0]):]

        for i in range(0, len(generated['attentions'])):

            if i ==0:
                pass
            else:

                print("=====================================")
                layer_id = 0

                print(f'Number of input tokens: {len(input_tokens[0])}')



                print(len(generated['attentions'][i][0]))

                print(generated['attentions'][i][0].shape)

                print(generated['attentions'][i][0][:,layer_id,:,:].shape)
                # print(generated['attentions'][i][0][:,layer_id,:,:].squeeze(0).squeeze(0))

                grouped_attention = torch.mean(generated['attentions'][i][0], dim=1)
                print(grouped_attention.shape)

                print(torch.argmax(generated['attentions'][i][0][:,layer_id,:,:].squeeze(0).squeeze(0)))


                topk_values, topk_indices = torch.topk(grouped_attention.squeeze(0).squeeze(0), 50)
                # print("Top 10 values: ", topk_values)
                print("Top 10 indices: ", topk_indices)

                print(tokenizer.convert_ids_to_tokens(generated['sequences'][0].tolist()[index] for index in topk_indices.tolist()))
                print(f"Generated token: {tokenizer.convert_ids_to_tokens(generated_tokens[i])}")


                topk_values, topk_indices = torch.topk(generated['attentions'][i][0][:,layer_id,:,:].squeeze(0).squeeze(0), 50)
                # print("Top 10 values: ", topk_values)
                print("Top 10 indices: ", topk_indices)

                print(tokenizer.convert_ids_to_tokens(generated['sequences'][0].tolist()[index] for index in topk_indices.tolist()))
                print(f"Generated token: {tokenizer.convert_ids_to_tokens(generated_tokens[i])}")

                print("=====================================")

        
        print(tokenizer.decode(generated['sequences'][0].tolist()))
        
       
        
        




if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer, model_configuration = load_model_and_tokenizer(
        model_name=model_name,
        device=device,
    )

    system_prompt = """<|start_header_id|>system<|end_header_id|>You are with performing topic classification on the following English text. For each input, classify the topic as science/technology, travel, politics, sports, health, entertainment, or geography. Use the following guidelines: 
science/technology: The text discusses scientific discoveries, technological advancements, or related topics. 
travel: The text describes travel experiences, destinations, or related topics. 
politics: The text covers political events, policies, or related topics. 
sports: The text talks about sports events, athletes, or related topics. 
health: The text addresses health issues, medical advancements, or related topics. 
entertainment: The text pertains to movies, music, celebrities, or related topics. 
geography: The text involves geographical information, locations, or related topics. 

If the text contains multiple topics, choose the dominant topic. For ambiguous or unclear topics, select the category that best reflects the overall content. Please provide a single classification for each input.<|eot_id|>"""

    input_sample = "<|start_header_id|>user<|end_header_id|>text: Police said Lo Piccolo had the upper hand because he had been Provenzano's right-hand man in Palermo and his greater experience won him the respect of the older generation of bosses as they pursued Provenzano's policy of keeping as low as possible while strengthening their power network.\n\ncategory:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    inference_model(system_prompt, input_sample, model, tokenizer, model_configuration, output_attentions=True)