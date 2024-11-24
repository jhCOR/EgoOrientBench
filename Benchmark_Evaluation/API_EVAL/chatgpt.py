

class ChatGPT:
    def __init__(self, API_key, setting):
        self.load_library()
        self.client = self.OpenAI(api_key=API_key)
        self.setting = setting

    def load_library(self):
        from openai import OpenAI
        self.OpenAI = OpenAI
        
    def setSetting(self, **kwargs):
        for key in kwargs:
            self.setting[key] = kwargs[key]
    
    def inference(self, prompt, image_object):

        response = self.client.chat.completions.create(
            model= self.setting.get("model_version", "gpt-4o-2024-08-06"),
            messages=[
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt,
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:image/jpeg;base64,{image_object['image']}"
                    },
                    },
                ],
                }
            ],
            temperature=self.setting.get("temperature", 0.2),
            max_tokens=self.setting.get("max_new_tokens", 150),
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        return response.choices[0].message.content.strip()