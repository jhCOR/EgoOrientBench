

class Claude:
    def __init__(self, API_key, setting):
        self.load_library()
        self.client = self.anthropic.Anthropic(api_key=API_key)
        self.setting = setting
    
    def load_library(self):
        import anthropic
        self.anthropic = anthropic
    def setSetting(self, **kwargs):
        for key in kwargs:
            self.setting[key] = kwargs[key]
    
    def inference(self,prompt, image_object):
        response = self.client.messages.create(
            model=self.setting.get("model_version","claude-3-5-sonnet-20241022"),
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt,
                    },
                    {
                    "type": "image",
                    "source":{
                        "type": "base64",
                        "media_type":image_object.get("media_type", "image/jpeg"),
                        "data": image_object
                    }
                    },
                ]
                }
            ],
            temperature=self.setting.get("temperature", 0.2),
            max_tokens=self.setting.get("max_new_tokens", 150),
        )

        if response.type == 'error':
            print('claude error')
            print(response)
            return ''
        else:
            return response.content[0].text

