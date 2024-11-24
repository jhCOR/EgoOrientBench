

class Gemini:
    def __init__(self, API_key, setting):
        self.load_library()
        self.genai.configure(api_key=API_key)
        self.setting = setting
    def load_library(self):
        import genai
        self.genai = genai
    def setSetting(self, **kwargs):
        for key in kwargs:
            self.setting[key] = kwargs[key]

    def inference(self, prompt, image_object):
        self.generation_config = self.genai.GenerationConfig(
            temperature= self.setting.get("temperature", 0.2),
            max_output_tokens= self.setting.get("max_new_tokens", 150)
        )
        
        model = self.genai.GenerativeModel(self.setting.get("model_version", "models/gemini-1.5-flash-8b"))
        response = model.generate_content([prompt, image_object['image']], generation_config = self.generation_config)
        return response.text