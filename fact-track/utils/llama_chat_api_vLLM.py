from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams

from typing import List

# Code is from here: https://docs.mystic.ai/docs/llama-2-with-vllm-7b-13b-multi-gpu-70b
class LlamaPipeline:
    def __init__(self, model_name = "7B-chat", max_gen_len = 40960, temperature = 0, top_p = 1, presence_penalty = 1):
        from vllm import LLM
        self.llm = LLM(f"/shared/yangk/llama2_hf/{model_name}", tensor_parallel_size=1, gpu_memory_utilization = 0.7)
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p
        self.sampling_param = SamplingParams(top_p=self.top_p, temperature=self.temperature, max_tokens=self.max_gen_len)
        self.presence_penalty = presence_penalty
        self.tokenizer = self.llm.get_tokenizer()

    def inference(self, dialogs: list) -> List[str]:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "SYS\n", "\n<</SYS>>\n\n"

        # if kwargs is None:
        #     kwargs = ModelKwargs()
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_gen_len,
            presence_penalty = self.presence_penalty
        )

        prompt_tokens = []
        for dialog in dialogs:
            if dialog[0]["role"] != "system":
                dialog = [
                             {
                                 "role": "system",
                                 "content": """You are a helpful, respectful and honest assistant.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful,
unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.""",
                             }
                         ] + dialog
            dialog = [
                         {
                             "role": dialog[1]["role"],
                             "content": B_SYS
                                        + dialog[0]["content"]
                                        + E_SYS
                                        + dialog[1]["content"],
                         }
                     ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )

            dialog_tokens = sum(
                [
                    [
                        [self.tokenizer.bos_token_id]
                        + self.tokenizer(
                            f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} "
                        ).input_ids
                        + [self.tokenizer.eos_token_id]
                    ]
                    for prompt, answer in zip(
                    dialog[::2],
                    dialog[1::2],
                )
                ],
                [],
            )
            assert (
                    dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += [
                [self.tokenizer.bos_token_id]
                + self.tokenizer(
                    f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
                ).input_ids
            ]

            prompt_tokens.append(dialog_tokens)
        generation_tokens = []
        input_num_tokens = []
        for prompt_tok in prompt_tokens:
            prompt_tok = [[item for sublist in prompt_tok for item in sublist]]
            # if kwargs.max_new_tokens == -1:
            #     sampling_params.max_new_tokens = self.tokenizer.model_max_length - len(
            #         prompt_tok[0]
            #     )
            generation_tokens.append(
                self.llm.generate(
                    prompt_token_ids=prompt_tok,
                    sampling_params=sampling_params,
                )
            )
            input_num_tokens.append(len(prompt_tok[0]))

        # for i, t in enumerate(generation_tokens):
        #     print(i, t)
        return [
            {
                "role": "assistant",
                "content": t[0].outputs[0].text,
            }
            for i, t in enumerate(generation_tokens)
        ]

if __name__ == "__main__":
    model = LlamaPipeline()
    dialogs = [
        [
            {
                "role": "user",
                "content":  """Premise: An ordinary high school student discovers that they possess an extraordinary ability to manipulate reality through their dreams.    As they struggle to control this power and keep it hidden from those who would exploit it, they are drawn into a dangerous conflict between secret organizations vying for control over the fate of the world.

Outline:

Point 2.1.2
Main plot: Alex struggles to control their power
Begin Event: Alex accidentally manipulates reality in their dream
End Event: Alex seeks guidance from Mr. Lee to control their power
Characters: Alex, Mr. Lee


Can you break down point 2.1.2 into less than 3 independent, chronological and same-scaled outline points? Also, assign each character a name. Please use the following template with "Main Plot", "Begin Event". "End Event" and "Characters". Do not answer anything else.

Point 2.1.2.1
Main plot: [TODO]
Begin Event: [TODO]
End Event: [TODO]
Characters: [TODO]

Point 2.1.2.2
Main plot: [TODO]
Begin Event: [TODO]
End Event: [TODO]
Characters: [TODO]

...
""",
            },
        ]
    ]
    print(model.inference(dialogs))
