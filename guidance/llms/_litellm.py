from litellm import acompletion
import os
from ._openai import OpenAI, prompt_to_messages, add_text_to_chat_mode
import litellm

class LiteLLM(OpenAI):
    def __init__(self, endpoint=None, service=None, chat_mode=True):
        super().__init__(model="gpt-4", chat_mode=chat_mode)
        self.endpoint = endpoint
        self.service = service
        
    async def _library_call(self, **kwargs):
        """ Call the LLM APIs using LiteLLM: https://github.com/BerriAI/litellm/"""

        if self.chat_mode:
            kwargs['messages'] = prompt_to_messages(kwargs['prompt'])
            del kwargs['prompt']
            del kwargs['echo']
            del kwargs['logprobs']
            del kwargs['model']
            del kwargs['stream']
            # print(kwargs)
            out = await acompletion(model=self.endpoint,
                                    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", None),
                                    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", None),
                                    aws_region_name=os.environ.get("REGION", None),
                                    custom_llm_provider=self.service,
                                    stream=True,
                                    **kwargs)
            out = add_text_to_chat_mode(out)
        else:
            out = await acompletion(**kwargs)
            
        return out
